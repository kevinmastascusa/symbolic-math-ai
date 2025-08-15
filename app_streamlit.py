from pathlib import Path
from typing import List, Dict, Any
import json
import os

import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import Tuple

from data_loader import MathDatasetLoader
from symbolic_math_ai import (
    SymbolicMathProcessor,
    TreeOfThoughtsGenerator,
)


@st.cache_resource(show_spinner=False)
def load_hf_model(model_dir: str):
    # Try to detect if model_dir is a PEFT adapter folder (our training output)
    training_cfg_path = Path(model_dir) / "training_config.json"
    is_adapter_dir = training_cfg_path.exists()

    if is_adapter_dir:
        # Load base model name from training config
        with open(training_cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        base_model_name = cfg.get("model_name", "Qwen/Qwen2.5-Math-1.5B")

        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32),
            device_map="auto" if torch.cuda.is_available() else None,
        )
        try:
            model = PeftModel.from_pretrained(base_model, model_dir)
        except RuntimeError as e:
            import re
            msg = str(e)
            # Try to extract expected embedding rows from the error message
            m = re.search(r"copying a param with shape torch.Size\(\[(\d+),\s*\d+\]\).*shape in current model is torch.Size\(\[(\d+),", msg)
            if m:
                expected_rows = int(m.group(1))
                current_rows = int(m.group(2))
                if expected_rows != current_rows:
                    # Resize base model embeddings to match adapter expectation
                    base_model.resize_token_embeddings(expected_rows)
                    model = PeftModel.from_pretrained(base_model, model_dir)
            else:
                raise
        return tokenizer, model
    else:
        # Fall back: model_dir is a full HF model directory
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32),
            device_map="auto" if torch.cuda.is_available() else None,
        )
        return tokenizer, model


def load_onnx_model(_: str) -> Tuple[None, None]:
    st.error("ONNX Runtime was removed. Please use the HF (PyTorch) backend.")
    return None, None


def run_decoding(
    tokenizer, model, prompt: str, max_new_tokens: int = 128
) -> str:
    target_device = model.device if hasattr(model, "device") else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt").to(target_device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def run_shap_explanation(
    tokenizer, model, question: str, num_samples: int = 20
) -> Dict[str, Any]:
    try:
        import shap
    except Exception:
        return {
            "error": (
                "SHAP not installed. Install 'shap' to enable explanations."
            )
        }

    def f(inputs: List[str]):
        scores: List[float] = []
        for text in inputs:
            out = run_decoding(
                tokenizer,
                model,
                f"Question: {text}\nAnswer:",
            )
            # Simple numeric extraction heuristic for a scalar score
            digits = ''.join(
                [c for c in out if (c.isdigit() or c in ['.', '-'])]
            )
            try:
                scores.append(float(digits) if digits not in ("", "-") else 0.0)
            except Exception:
                scores.append(0.0)
        return np.array(scores)

    explainer = shap.Explainer(
        f,
        shap.maskers.Text(tokenizer.sep_token or ' '),
    )
    shap_values = explainer([question], max_evals=num_samples)
    return {
        "base_values": getattr(shap_values, "base_values", None),
        "values": getattr(shap_values, "values", None),
        "data": getattr(shap_values, "data", None),
        "output_names": getattr(shap_values, "output_names", None),
    }


def main():
    st.set_page_config(page_title="Symbolic Math AI - ToT + SHAP")
    st.title("Symbolic Math AI")
    st.caption("Training pipeline with ToT, SymPy, and SHAP")

    with st.sidebar:
        st.header("Model (PyTorch)")
        hf_model_dir = st.text_input(
            "HF model dir or adapter dir", value="./trained_math_model_qwen_run2"
        )
        max_depth = st.slider("ToT max depth", 1, 6, 3)
        max_children = st.slider("ToT max children", 1, 5, 2)
        st.checkbox("Enable SHAP explanation (slow)", value=False)

    # Load model (PyTorch-only)
    tokenizer, model = load_hf_model(hf_model_dir)

    # Processors
    math_processor = SymbolicMathProcessor()
    tot = TreeOfThoughtsGenerator(
        model=model,
        tokenizer=tokenizer,
        math_processor=math_processor,
        max_depth=max_depth,
        max_children=max_children,
    )

    tab_solve, tab_dataset = st.tabs(["Solve", "Datasets"])

    with tab_solve:
        st.subheader("Interactive Solver")
        question = st.text_area(
            "Enter a math problem", "Solve for x: 2x + 5 = 13"
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Solve with ToT"):
                with st.spinner("Reasoning with Tree-of-Thoughts..."):
                    answer = tot.generate(question)
                st.success("Answer")
                st.write(answer)
        with c2:
            if st.button("Explain with SHAP"):
                with st.spinner("Running SHAP explanation..."):
                    shap_result = run_shap_explanation(tokenizer, model, question)
                if "error" in shap_result:
                    st.error(shap_result["error"])
                else:
                    st.write("Base values:", shap_result.get("base_values"))
                    shape_val = (
                        None
                        if shap_result.get("values") is None
                        else np.array(shap_result["values"]).shape
                    )
                    st.write("Values shape:", shape_val)

        st.divider()
        st.subheader("SymPy Extraction")
        eqs = math_processor.extract_equations(question)
        if eqs:
            st.write("Detected equations:", eqs[:5])
            sols = []
            for eq in eqs[:3]:
                sol = math_processor.solve_equation(eq)
                if sol:
                    sols.append({"equation": eq, "solutions": [str(s) for s in sol]})
            if sols:
                st.json(sols)
        else:
            st.info("No equations detected.")

    with tab_dataset:
        st.subheader("Preprocessed Datasets")
        loader = MathDatasetLoader(data_dir="Dataset")
        with st.expander("Summary", expanded=False):
            datasets = loader.get_all_datasets()
            summary = {k: v.shape for k, v in datasets.items()}
            st.json({
                k: {"rows": v[0], "cols": v[1]}
                for k, v in summary.items()
            })

        st.write("Preview of preprocessed CSVs (if present)")
        for fname in [
            "preprocessed_gsm8k_train.csv",
            "preprocessed_mathqa_train.csv",
            "preprocessed_svamp_train.csv",
            "preprocessed_math500_train.csv",
        ]:
            p = Path("Dataset") / fname
            if p.exists():
                st.caption(str(p))
                try:
                    import pandas as pd
                    df_preview = pd.read_csv(p).head(5)
                    st.dataframe(df_preview)
                except Exception as e:
                    st.warning(f"Could not read {p}: {e}")
            else:
                st.caption(f"{p} (missing)")


if __name__ == "__main__":
    main()
