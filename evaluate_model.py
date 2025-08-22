#!/usr/bin/env python3
r"""
Evaluate a trained (or base) math model on available test datasets.

Computes simple exact-match (EM) by extracting the final numeric/text token
from the model output and gold labels. Outputs a compact JSON report.

Usage (Windows, with repo venv):
  .\\.venv_app\\Scripts\\python evaluate_model.py \
    --model_dir ./trained_math_model_qwen_run2 \
    --limit 200 \
    --output eval_report.json
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd  # type: ignore
from tqdm import tqdm
from symbolic_math_ai import SymbolicMathProcessor

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_loader import MathDatasetLoader


# -------------------------
# Utilities
# -------------------------

def extract_final_answer(text: str) -> str:
    """Extract a simple final answer string from text.

    Heuristic: pick the last number in the text; if none, return a lowercased
    trimmed version of the text. Mirrors logic used during training EM.
    """
    matches = re.findall(r"[-+]?[0-9]*\.?[0-9]+", text)
    return matches[-1] if matches else text.strip().lower()


def safe_json_loads(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


# -------------------------
# Model loading (adapter-aware)
# -------------------------

def load_model_or_adapter(model_dir: str) -> Tuple[Any, Any]:
    """Load a HF model or a PEFT adapter directory seamlessly.

    Returns (tokenizer, model).
    """
    from peft import PeftModel  # local import to avoid hard dep

    model_path = Path(model_dir)
    training_cfg = model_path / "training_config.json"
    is_adapter = training_cfg.exists()

    if is_adapter:
        with open(training_cfg, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        base_model_name = cfg.get("model_name", "Qwen/Qwen2.5-Math-1.5B")

        # Prefer tokenizer saved alongside adapter; else fallback to base
        if (model_path / "tokenizer.json").exists() or (
            model_path / "tokenizer_config.json"
        ).exists():
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
        else:
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=(
                torch.float16 if torch.cuda.is_available() else torch.float32
            ),
            device_map="auto" if torch.cuda.is_available() else None,
        )
        try:
            model = PeftModel.from_pretrained(base_model, model_dir)
        except RuntimeError:
            # Handle occasional vocab-size mismatches by resizing embeddings
            base_model.resize_token_embeddings(len(tokenizer))
            model = PeftModel.from_pretrained(base_model, model_dir)
        return tokenizer, model

    # Full model directory
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=(
            torch.float16 if torch.cuda.is_available() else torch.float32
        ),
        device_map="auto" if torch.cuda.is_available() else None,
        ignore_mismatched_sizes=True,
    )
    input_embeds = model.get_input_embeddings()
    num_rows = (
        input_embeds.weight.shape[0]
        if hasattr(input_embeds, "weight")
        else None
    )
    if num_rows is not None and num_rows != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model


@dataclass
class EvalConfig:
    model_dir: str
    limit: int
    max_new_tokens: int
    output: str
    datasets: Optional[List[str]]
    metrics: Optional[List[str]]
    reasoning_metrics: bool = False


def build_prompt(example: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """Construct a prompt and extract the gold answer from a raw example.

    Returns (prompt, gold_answer or None).
    """
    # If a normalized 'answer' (lower) or 'Answer' (upper) field exists,
    # try common question keys, including preprocessed SVAMP variants
    if ("answer" in example) or ("Answer" in example):
        q_keys = [
            "question_concat",  # preprocessed SVAMP combined text
            "problem_text",     # fallback combined text
            "Question",         # capitalized
            "question",         # lower
            "sQuestion",        # SVAMP raw
            "Problem",          # MathQA
            "problem",
            "Body",             # SVAMP body (may need to concat with Question)
            "prompt",
            "text",
        ]
        question_text: Optional[str] = None
        for q_key in q_keys:
            if q_key in example and isinstance(example[q_key], str) and example[q_key].strip():
                question_text = example[q_key]
                break
        # Special-case: if we only have Body + Question fields, combine
        if question_text is None and ("Body" in example and "Question" in example):
            try:
                question_text = f"{example['Body']} {example['Question']}"
            except Exception:
                question_text = example.get("Question") or example.get("Body")
        if question_text is not None:
            prompt = f"Question: {question_text}\nAnswer:"
            ans_val = example.get("answer", example.get("Answer"))
            return prompt, None if ans_val is None else str(ans_val)

    # GSM8K
    if (
        "question" in example
        and ("answer" in example or "solution" in example)
    ):
        prompt = f"Question: {example['question']}\nAnswer:"
        gold = example.get("answer")
        return prompt, gold

    # MathQA (include options if present to guide the model)
    if (
        "Problem" in example
        and ("correct" in example or "Rationale" in example)
    ):
        base = f"Problem: {example['Problem']}"
        if "options" in example and isinstance(example["options"], str):
            base += f"\nOptions: {example['options']}"
        prompt = base + "\nSolution:"
        gold = example.get("correct")
        return prompt, gold

    # SVAMP
    if (
        "sQuestion" in example
        and ("lSolutions" in example or "lEquations" in example)
    ):
        prompt = f"Question: {example['sQuestion']}\nAnswer:"
        gold_field = example.get("lSolutions")
        if isinstance(gold_field, str):
            gold_field = safe_json_loads(gold_field)
        gold = None
        if isinstance(gold_field, list) and gold_field:
            gold = str(gold_field[0])
        return prompt, gold

    # MATH-500
    if (
        "problem" in example and ("answer" in example or "solution" in example)
    ):
        prompt = f"Problem: {example['problem']}\nSolution:"
        gold = example.get("answer")
        return prompt, gold

    # Fallback
    prompt = f"Problem: {example.get('problem', example)}\nAnswer:"
    gold = example.get("answer") if isinstance(example, dict) else None
    return prompt, gold


def generate_answer(tokenizer, model, prompt: str, max_new_tokens: int) -> str:
    target_device = model.device if hasattr(model, "device") else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt").to(target_device)
    with torch.no_grad():
        prev_use_cache = getattr(model.config, "use_cache", True)
        if hasattr(model, "config"):
            model.config.use_cache = True
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )
        if hasattr(model, "config"):
            model.config.use_cache = prev_use_cache
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def evaluate_dataset(
    name: str,
    rows: List[Dict[str, Any]],
    tokenizer,
    model,
    limit: int,
    max_new_tokens: int,
    math_proc: Optional[SymbolicMathProcessor] = None,
    compute_reasoning: bool = False,
    tol: float = 1e-6,
) -> Dict[str, Any]:
    total = 0
    correct = 0
    preds_raw: List[str] = []
    refs_raw: List[str] = []
    # Reasoning counters
    rs_equations: int = 0
    rs_solvable: int = 0
    rs_examples_with_any_eq: int = 0
    rs_consistent: int = 0

    n = min(limit, len(rows))
    for i in tqdm(range(n), desc=f"Eval {name}", leave=False):
        example = rows[i]
        prompt, gold = build_prompt(example)
        if gold is None:
            continue
        out = generate_answer(tokenizer, model, prompt, max_new_tokens)
        pred = extract_final_answer(out)
        gold_norm = extract_final_answer(str(gold))
        correct += int(pred == gold_norm)
        total += 1
        preds_raw.append(out.strip())
        refs_raw.append(str(gold).strip())

        if compute_reasoning and math_proc is not None:
            # Extract equations across lines of output
            lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
            eqs: List[str] = []
            for ln in lines:
                try:
                    eqs.extend(math_proc.extract_equations(ln))
                except Exception:
                    continue
            eqs = list(set(eqs))
            if eqs:
                rs_examples_with_any_eq += 1
            rs_equations += len(eqs)

            # Count solvable and check consistency with gold numeric
            gold_num: Optional[float] = None
            try:
                gold_num = float(gold_norm)
            except Exception:
                gold_num = None

            solv_here = 0
            consistent_here = False
            for eq in eqs:
                try:
                    sols = math_proc.solve_equation(eq)
                    if sols:
                        solv_here += 1
                        if gold_num is not None:
                            # Compare any solution numerically
                            for s in sols:
                                try:
                                    val = float(s)
                                    if abs(val - gold_num) <= tol:
                                        consistent_here = True
                                        break
                                except Exception:
                                    pass
                        if consistent_here:
                            break
                except Exception:
                    continue
            rs_solvable += solv_here
            if consistent_here:
                rs_consistent += 1

    em = (correct / max(total, 1))
    return {
        "dataset": name,
        "exact_match": em,
        "total": total,
        "correct": correct,
        "preds_raw": preds_raw,
        "refs_raw": refs_raw,
        "rs_equations": rs_equations,
        "rs_solvable": rs_solvable,
        "rs_examples_with_any_eq": rs_examples_with_any_eq,
        "rs_consistent": rs_consistent,
    }


def collect_datasets(
    loader: MathDatasetLoader, wanted: Optional[List[str]]
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    selected: List[str] = wanted or ["gsm8k", "mathqa", "svamp", "math500"]
    result: List[Tuple[str, List[Dict[str, Any]]]] = []

    preprocessed_map = {
        "gsm8k": "preprocessed_gsm8k_test.csv",
        "mathqa": "preprocessed_mathqa_test.csv",
        "svamp": "preprocessed_svamp_test.csv",
        "math500": "preprocessed_math500_test.csv",
    }

    for name in selected:
        df = None
        source_label = ""
        # Prefer preprocessed file if present
        pp_name = preprocessed_map.get(name)
        if pp_name is not None:
            pp_path = loader.data_dir / pp_name
            if pp_path.exists():
                try:
                    df = pd.read_csv(pp_path)
                    source_label = str(pp_path)
                except Exception:
                    df = None

        def _load_fallback() -> Tuple[Optional[pd.DataFrame], str]:
            try:
                if name == "gsm8k":
                    return loader.load_gsm8k("test"), "loader(gsm8k)"
                if name == "mathqa":
                    return loader.load_mathqa("test"), "loader(mathqa)"
                if name == "svamp":
                    return loader.load_svamp("test"), "loader(svamp)"
                if name == "math500":
                    return loader.load_math500("test"), "loader(math500)"
            except Exception:
                pass
            return None, ""

        # If preprocessed present but yields 0 valid golds, fallback
        if df is not None:
            rows_all = df.to_dict("records")
            valid = [r for r in rows_all if build_prompt(r)[1] is not None]
            if not valid:
                df, source_label = _load_fallback()
                if df is not None:
                    rows_all = df.to_dict("records")
                    valid = [
                        r for r in rows_all if build_prompt(r)[1] is not None
                    ]
        else:
            df, source_label = _load_fallback()
            valid = []
            if df is not None:
                rows_all = df.to_dict("records")
                valid = [r for r in rows_all if build_prompt(r)[1] is not None]

        if df is not None and valid:
            result.append((name, valid))

    # Print which files/sources are used (prefer actual sources resolved above)
    print("Datasets used:")
    for name, rows in result:
        # Determine whether rows originated from preprocessed CSV or loader
        pp_name = preprocessed_map.get(name)
        pp_path = loader.data_dir / pp_name if pp_name else None
        origin = (
            str(pp_path) if (pp_path and pp_path.exists()) else f"loader({name})"
        )
        print(" -", f"{name}: {origin} - {len(rows)} usable examples")

    return result


def parse_args() -> EvalConfig:
    p = argparse.ArgumentParser(
        description="Evaluate model EM on common math datasets"
    )
    p.add_argument(
        "--model_dir",
        type=str,
        default="./trained_math_model_qwen_run2",
    )
    p.add_argument(
        "--limit", type=int, default=200, help="Max examples per dataset"
    )
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--output", type=str, default="eval_report.json")
    p.add_argument(
        "--reasoning_metrics",
        action="store_true",
        help="Compute SymPy-based reasoning metrics",
    )
    p.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Subset of datasets: gsm8k mathqa svamp math500 "
            "(default: auto/all)"
        ),
    )
    p.add_argument(
        "--metrics",
        type=str,
        nargs="*",
        default=["bleu", "meteor", "rouge"],
        help="Which metrics to compute: bleu meteor rouge",
    )
    args = p.parse_args()
    return EvalConfig(
        model_dir=args.model_dir,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        output=args.output,
        datasets=args.datasets,
        metrics=args.metrics,
        reasoning_metrics=args.reasoning_metrics,
    )


def main() -> None:
    cfg = parse_args()

    tokenizer, model = load_model_or_adapter(cfg.model_dir)
    model.eval()

    loader = MathDatasetLoader(data_dir="Dataset")
    datasets = collect_datasets(loader, cfg.datasets)

    reports: List[Dict[str, Any]] = []
    total_correct = 0
    total_seen = 0
    all_preds: List[str] = []
    all_refs: List[str] = []
    math_proc: Optional[SymbolicMathProcessor] = None
    if cfg.reasoning_metrics:
        math_proc = SymbolicMathProcessor()

    for name, rows in datasets:
        res = evaluate_dataset(
            name=name,
            rows=rows,
            tokenizer=tokenizer,
            model=model,
            limit=cfg.limit,
            max_new_tokens=cfg.max_new_tokens,
            math_proc=math_proc,
            compute_reasoning=bool(math_proc is not None),
        )
        reports.append(res)
        total_correct += int(res["correct"])
        total_seen += int(res["total"])
        all_preds.extend(res.get("preds_raw", []))
        all_refs.extend(res.get("refs_raw", []))

    overall_em = (total_correct / max(total_seen, 1))
    per_dataset: Dict[str, Dict[str, Any]] = {}
    # Overall reasoning metric totals
    rs_tot_equations = 0
    rs_tot_examples = 0
    rs_tot_solvable = 0
    rs_tot_consistent = 0

    for r in reports:
        entry: Dict[str, Any] = {
            "exact_match": r["exact_match"],
            "total": r["total"],
            "correct": r["correct"],
        }
        # Attach reasoning metrics when present
        for k in (
            "rs_equations",
            "rs_examples_with_any_eq",
            "rs_solvable",
            "rs_consistent",
        ):
            if k in r:
                entry[k] = r[k]
        per_dataset[r["dataset"]] = entry

        # Accumulate overall reasoning metrics
        rs_tot_equations += int(r.get("rs_equations", 0))
        rs_tot_examples += int(r.get("rs_examples_with_any_eq", 0))
        rs_tot_solvable += int(r.get("rs_solvable", 0))
        rs_tot_consistent += int(r.get("rs_consistent", 0))

    # Optional text-similarity metrics
    overall_metrics: Dict[str, Any] = {}
    want_bleu = (cfg.metrics is None) or ("bleu" in cfg.metrics)
    want_meteor = (cfg.metrics is None) or ("meteor" in cfg.metrics)
    want_rouge = (cfg.metrics is None) or ("rouge" in cfg.metrics)

    if want_bleu:
        try:
            import sacrebleu as sbleu  # type: ignore

            bleu = (
                sbleu.corpus_bleu(all_preds, [all_refs]).score
                if all_preds
                else 0.0
            )
            overall_metrics["bleu"] = bleu
            # Per-dataset BLEU
            for r in reports:
                if r.get("preds_raw"):
                    r["bleu"] = sbleu.corpus_bleu(
                        r["preds_raw"],
                        [r["refs_raw"]],
                    ).score
                else:
                    r["bleu"] = 0.0
        except Exception as e:
            overall_metrics["bleu"] = None
            overall_metrics["bleu_error"] = str(e)

    if want_meteor:
        try:
            from nltk.translate.meteor_score import meteor_score  # type: ignore
            try:
                from nltk.tokenize import word_tokenize  # type: ignore
                def _tok(s: str) -> List[str]:
                    return word_tokenize(s)
            except Exception:
                def _tok(s: str) -> List[str]:
                    return s.split()

            def avg_meteor(
                preds: List[str], refs: List[str]
            ) -> float:
                if not preds:
                    return 0.0
                scores: List[float] = []
                for i in range(len(preds)):
                    hyp = _tok(preds[i])
                    ref = _tok(refs[i])
                    scores.append(meteor_score([ref], hyp))
                return float(sum(scores) / max(len(scores), 1))

            overall_metrics["meteor"] = avg_meteor(all_preds, all_refs)
            for r in reports:
                r["meteor"] = avg_meteor(
                    r.get("preds_raw", []), r.get("refs_raw", [])
                )
        except Exception as e:
            overall_metrics["meteor"] = None
            overall_metrics["meteor_error"] = str(e)

    if want_rouge:
        try:
            from rouge_score import rouge_scorer  # type: ignore

            scorer = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"], use_stemmer=True
            )

            def avg_rouge(
                preds: List[str], refs: List[str]
            ) -> Dict[str, float]:
                if not preds:
                    return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
                sums = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
                for i in range(len(preds)):
                    s = scorer.score(refs[i], preds[i])
                    for k in sums:
                        sums[k] += float(s[k].fmeasure)
                n = float(len(preds))
                return {k: (v / n) for k, v in sums.items()}

            overall_r = avg_rouge(all_preds, all_refs)
            overall_metrics.update(overall_r)
            for r in reports:
                r_r = avg_rouge(
                    r.get("preds_raw", []),
                    r.get("refs_raw", []),
                )
                r.update(r_r)
        except Exception as e:
            overall_metrics["rouge1"] = None
            overall_metrics["rouge2"] = None
            overall_metrics["rougeL"] = None
            overall_metrics["rouge_error"] = str(e)

    summary = {
        "model_dir": cfg.model_dir,
        "overall_exact_match": overall_em,
        "total": total_seen,
        "correct": total_correct,
        "overall_metrics": overall_metrics,
        "overall_reasoning": {
            "rs_equations": rs_tot_equations,
            "rs_examples_with_any_eq": rs_tot_examples,
            "rs_solvable": rs_tot_solvable,
            "rs_consistent": rs_tot_consistent,
        },
        "per_dataset": per_dataset,
    }

    out_path = Path(cfg.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved evaluation report to:", out_path)
    if reports:
        print("Overall EM:", f"{overall_em:.4f}")
        if overall_metrics:
            print("Overall metrics:")
            for k, v in overall_metrics.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
        for r in reports:
            line = (
                f"  {r['dataset']}: EM={r['exact_match']:.4f} "
                f"(n={r['total']})"
            )
            # Append metric snippets if present
            for k in ("bleu", "meteor", "rouge1", "rouge2", "rougeL"):
                if k in r and isinstance(r[k], float):
                    line += f", {k}={r[k]:.4f}"
            # Append reasoning snippets if present
            if "rs_equations" in r:
                line += (
                    f", rs_eq={int(r.get('rs_equations', 0))}"
                    f", rs_has_eq={int(r.get('rs_examples_with_any_eq', 0))}"
                    f", rs_solv={int(r.get('rs_solvable', 0))}"
                    f", rs_cons={int(r.get('rs_consistent', 0))}"
                )
            print(line)


if __name__ == "__main__":
    main()
