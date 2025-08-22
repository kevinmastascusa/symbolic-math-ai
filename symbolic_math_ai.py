#!/usr/bin/env python3
"""
Symbolic Math AI Training Script

This script trains a language model for symbolic mathematical reasoning using:
- Hugging Face Transformers
- Multiple math datasets (GSM8K, MathQA, SVAMP, MATH-500)
- Tree of Thoughts reasoning methodology
- SymPy for symbolic math processing
- Model fine-tuning and evaluation

Author: Symbolic Math AI Project
"""

import os
import sys
import json
import logging
import warnings
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Core imports
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
)
from transformers.trainer_callback import EarlyStoppingCallback
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.utils.quantization_config import BitsAndBytesConfig
from datasets import Dataset as HFDataset
import sympy as sp
from sympy import symbols, Eq, solve, simplify, sympify
from sympy.parsing.sympy_parser import parse_expr
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Import local modules
from data_loader import MathDatasetLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TreeOfThoughtsGenerator:
    """Generates solutions using a Tree of Thoughts approach."""
    
    def __init__(self, model, tokenizer, math_processor, max_depth=3, max_children=2):
        self.model = model
        self.tokenizer = tokenizer
        self.math_processor = math_processor
        self.max_depth = max_depth
        self.max_children = max_children
        # Introspective state for visualization
        self.last_tree = None
        self.last_path = None

    def generate(self, problem: str) -> str:
        """Generate a solution using ToT."""
        logger.info(f"Starting ToT generation for: '{problem}'")
        tree = {'problem': problem, 'children': [], 'score': 0, 'text': ""}
        
        # Explore the tree
        solution_path = self._explore_node(tree, 0)
        # Persist for downstream visualization
        self.last_tree = tree
        self.last_path = solution_path
        
        if not solution_path:
            logger.warning("ToT could not find a valid solution path.")
            return "Could not determine a solution path."
            
        # Combine the steps for the final answer
        final_answer = "\n".join([node['text'] for node in solution_path])
        # Try to compute a concrete symbolic solution and override any contradictory text
        try:
            equations = self.math_processor.extract_equations(problem)
            sympy_solutions = []
            if equations:
                sols = self.math_processor.solve_equation(equations[0])
                if sols:
                    sympy_solutions = [str(s) for s in sols]

            if sympy_solutions:
                import re as _re
                import sympy as _sp

                # Helper: numeric comparison with tolerance for various formats
                def _values_match(v_text: str, sol_texts: list[str]) -> bool:
                    try:
                        v_expr = _sp.sympify(v_text)
                        for s in sol_texts:
                            s_expr = _sp.sympify(s)
                            if _sp.simplify(v_expr - s_expr) == 0:
                                return True
                    except Exception:
                        pass
                    # Fallback float compare
                    try:
                        v_float = float(_sp.N(v_expr))  # type: ignore[name-defined]
                        for s in sol_texts:
                            try:
                                s_float = float(_sp.N(_sp.sympify(s)))
                                if abs(v_float - s_float) < 1e-6:
                                    return True
                            except Exception:
                                continue
                    except Exception:
                        pass
                    return False

                # Remove any lines that assert an x = value inconsistent with SymPy
                cleaned_lines: list[str] = []
                for line in final_answer.splitlines():
                    m_line = _re.search(r"\bx\s*=\s*([^\s,;]+)", line)
                    if m_line:
                        if not _values_match(m_line.group(1), sympy_solutions):
                            continue
                    cleaned_lines.append(line)
                final_answer = "\n".join(cleaned_lines)

                # Append canonical final answer from SymPy
                sol_str = ", ".join(sympy_solutions)
                final_answer = f"{final_answer}\nFinal answer (SymPy): x = {sol_str}"
        except Exception:
            pass

        logger.info(f"ToT Final Answer: {final_answer}")
        
        return final_answer

    def _explore_node(self, node, depth):
        if depth >= self.max_depth:
            return [node]

        # Generate thoughts (potential next steps)
        thoughts = self._generate_thoughts(node)
        
        if not thoughts:
            return [node]

        # Evaluate and score thoughts
        for thought in thoughts:
            thought['score'] = self._evaluate_thought(thought)
        
        # Select the best thoughts to expand
        best_thoughts = sorted(thoughts, key=lambda x: x['score'], reverse=True)[:self.max_children]

        # Recursively explore the best thoughts
        best_path = []
        # Attach children for visualization
        node['children'] = best_thoughts
        for thought in best_thoughts:
            path = self._explore_node(thought, depth + 1)
            if path:
                if not best_path or sum(n['score'] for n in path) > sum(n['score'] for n in best_path):
                    best_path = path
        
        return [node] + best_path if best_path else [node]

    def _generate_thoughts(self, node) -> List[Dict]:
        """Generate a number of next-step thoughts from the current state."""
        # Build a chat-style prompt when supported (Qwen)
        if hasattr(self.tokenizer, "apply_chat_template"):
            user_content = (
                f"Problem: {node['problem']}\n"
                + (f"Current thought: {node['text']}\n" if node.get('text') else "")
                + "What is the next logical step?"
            )
            messages = [
                {"role": "system", "content": "You are a helpful math tutor. Reason step by step."},
                {"role": "user", "content": user_content},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        else:
            prompt = f"Problem: {node['problem']}\n"
            if node.get('text'):
                prompt += f"Current thought: {node['text']}\n"
            prompt += "What is the next logical step?"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            # Use a single beam for single-child ToT to avoid duplicate variants
            num_beams = 1 if self.max_children <= 1 else min(4, self.max_children * 2)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=48,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                do_sample=False,
                no_repeat_ngram_size=4,
                repetition_penalty=1.15,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only generated tokens to avoid echoing prompt/template
        try:
            input_len = inputs["input_ids"].shape[-1]
            gen_only = outputs[:, input_len:]
            decoded = self.tokenizer.batch_decode(gen_only, skip_special_tokens=True)
        except Exception:
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        thoughts = []
        for text in decoded:
            # Clean up the generated text
            cleaned_text = text.replace(prompt, "").strip()
            for role in ("system", "user", "assistant"):
                if cleaned_text.lower().startswith(role + ":"):
                    cleaned_text = cleaned_text.split(":", 1)[-1].strip()
            # Strip spurious artifacts like repeated 'Asphalt' tokens
            try:
                cleaned_text = re.sub(r"(?:\bAsphalt\b[\s\-·,.;:]*)+", "", cleaned_text, flags=re.IGNORECASE)
                cleaned_text = re.sub(r"\n{2,}", "\n", cleaned_text).strip()
            except Exception:
                pass
            if cleaned_text:
                thoughts.append({'problem': node['problem'], 'text': cleaned_text, 'children': [], 'score': 0})
        
        return thoughts

    def _evaluate_thought(self, thought: Dict) -> float:
        """Evaluate a thought based on heuristics (e.g., mathematical validity)."""
        text = thought['text']
        score = 0.0

        # Heuristic 1: Presence of numbers from the problem
        problem_numbers = re.findall(r'\d+', thought['problem'])
        thought_numbers = re.findall(r'\d+', text)
        if any(n in thought_numbers for n in problem_numbers):
            score += 0.2

        # Heuristic 2: Contains a solvable equation
        equations = self.math_processor.extract_equations(text)
        if equations:
            score += 0.5
            for eq in equations:
                if self.math_processor.solve_equation(eq):
                    score += 0.3 # Bonus for solvable equations

        # Heuristic 3: Avoids repetitive or nonsensical phrases
        if "The total number is" in text or "The area is" in text:
            score -= 0.3
            
        return score


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Hugging Face settings
    hf_token: Optional[str] = os.getenv("HUGGING_FACE_HUB_TOKEN")
    model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # Powerful model for reasoning
    output_dir: str = "./trained_model_mixtral"
    
    # Training parameters
    batch_size: int = 1  # Reduced for larger model
    learning_rate: float = 2e-5
    num_epochs: int = 1  # Start with 1 epoch for large models
    max_length: int = 256
    warmup_steps: int = 50
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    
    # Checkpointing and early stopping
    evaluation_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    early_stopping_patience: int = 3
    
    # Data parameters
    train_split: float = 0.8
    max_samples_per_dataset: int = 1000
    
    # Model parameters
    use_quantization: bool = True
    use_gradient_checkpointing: bool = True
    # PEFT LoRA settings (enable when quantizing to allow fine-tuning)
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    # Common target modules for LLaMA/Qwen-style architectures
    lora_target_modules: Optional[List[str]] = None
    
    # Tree of Thoughts parameters
    tot_max_depth: int = 3
    tot_max_children: int = 2
    
    def __post_init__(self):
        """Validate configuration."""
        if not self.hf_token:
            logger.warning("Hugging Face token not found. Model loading may fail.")
        
        # Create output directory
        Path(self.output_dir).mkdir(exist_ok=True)
        
        # Save config
        config_path = Path(self.output_dir) / "training_config.json"
        config_dict = self.__dict__.copy()
        config_dict.pop('hf_token', None)  # Exclude token from saved config
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

class SymbolicMathProcessor:
    """SymPy-based symbolic math processing for math word problems."""
    
    def __init__(self):
        self.x, self.y, self.z = symbols('x y z')
        self.variables = symbols('a b c d e f g h i j k l m n o p q r s t u v w x y z')
        
    def extract_equations(self, text: str) -> List[str]:
        """Extract mathematical equations from text."""
        import re
        # More permissive equality pattern to capture full LHS and RHS expressions
        eq_pattern = r'([A-Za-z0-9\.\s\+\-\*\/\(\)\^]+?)\s*=\s*([A-Za-z0-9\.\s\+\-\*\/\(\)\^]+)'
        other_patterns = [
            r'(\d+[\+\-\*\/]\d+)',
            r'(\d+\s*[\+\-\*\/]\s*\d+)',
            r'([\d\.]+\s*[\+\-\*\/]\s*[\d\.]+)'
        ]

        equations: List[str] = []

        # Handle equations with '=' first, join both sides
        for lhs, rhs in re.findall(eq_pattern, text):
            lhs_clean = lhs.strip()
            rhs_clean = rhs.strip()
            if lhs_clean and rhs_clean:
                equations.append(f"{lhs_clean} = {rhs_clean}")

        # Handle simple arithmetic expressions as fallbacks
        for pattern in other_patterns:
            matches = re.findall(pattern, text)
            for m in matches:
                equations.append(m if isinstance(m, str) else m[0])

        return list(set(equations))
    
    def parse_expression(self, expr_str: str) -> Optional[sp.Expr]:
        """Safely parse a mathematical expression."""
        try:
            expr_str = expr_str.replace('^', '**')
            expr_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr_str)
            return parse_expr(expr_str)
        except:
            return None
    
    def solve_equation(self, equation: str, variable: str = 'x') -> List[sp.Expr]:
        """Solve an equation for a given variable."""
        try:
            var = symbols(variable)
            if '=' in equation:
                lhs, rhs = equation.split('=')
                lhs_expr = self.parse_expression(lhs.strip())
                rhs_expr = self.parse_expression(rhs.strip())
                if lhs_expr is not None and rhs_expr is not None:
                    eq = Eq(lhs_expr, rhs_expr)
                    return solve(eq, var)
            else:
                expr = self.parse_expression(equation)
                if expr is None:
                    return []
                return solve(expr, var)
        except:
            return []

class MathDataset(Dataset):
    """Custom dataset for math problems."""
    
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format the input text
        if 'question' in item:
            # GSM8K format
            input_text = f"Question: {item['question']}\nAnswer:"
            target_text = f" {item['answer']}"
        elif 'Problem' in item:
            # MathQA format
            input_text = f"Problem: {item['Problem']}\nSolution:"
            target_text = f" {item['correct']}"
        elif 'sQuestion' in item:
            # SVAMP format
            input_text = f"Question: {item['sQuestion']}\nAnswer:"
            target_text = f" {item['lSolutions'][0] if item['lSolutions'] else 'Unknown'}"
        elif 'problem' in item and 'solution' in item:
            # MATH-500 format
            input_text = f"Problem: {item['problem']}\nSolution:"
            target_text = f" {item['answer']}"
        else:
            # Generic format
            input_text = f"Problem: {item.get('problem', 'Unknown')}\nAnswer:"
            target_text = f" {item.get('answer', 'Unknown')}"
        
        # Combine input and target
        full_text = input_text + target_text
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Create labels (same as input_ids for causal LM)
        encoding['labels'] = encoding['input_ids'].clone()
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['labels'].squeeze()
        }

class MathModelTrainer:
    """Main trainer class for the symbolic math model."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.math_processor = SymbolicMathProcessor()
        self.data_loader = MathDatasetLoader(data_dir="Dataset")
        
        # Set up Hugging Face token
        if self.config.hf_token:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = self.config.hf_token
        else:
            logger.warning("Hugging Face token not found. Model loading may fail.")
        
        logger.info("Initializing Math Model Trainer...")
        
    def setup_model_and_tokenizer(self):
        """Initialize the model and tokenizer."""
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            use_auth_token=self.config.hf_token
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization if enabled
        # Enable TF32 on matmul for better performance without extra memory
        if torch.cuda.is_available():
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
            except Exception:
                pass

        model_kwargs = {
            "use_auth_token": self.config.hf_token or True,
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            "low_cpu_mem_usage": True,
        }

        # Allow automatic device mapping to avoid OOM during load on small GPUs
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
        
        if self.config.use_quantization and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = quantization_config
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )

        # If quantized, prepare for k-bit training and attach LoRA adapters
        if self.config.use_quantization and self.config.use_lora and torch.cuda.is_available():
            try:
                self.model = prepare_model_for_kbit_training(self.model)
            except Exception:
                pass

            target_modules = self.config.lora_target_modules
            if target_modules is None:
                # Default for LLaMA/Qwen-like models
                target_modules = [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ]

            lora_cfg = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules,
            )
            self.model = get_peft_model(self.model, lora_cfg)
        
        # Enable gradient checkpointing after model loading
        if self.config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            # Disable KV cache during training to reduce memory usage
            if hasattr(self.model, "config") and hasattr(self.model.config, "use_cache"):
                self.model.config.use_cache = False
        
        # Resize token embeddings if needed
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        logger.info("Model and tokenizer loaded successfully")
        
    def prepare_datasets(self) -> Tuple[MathDataset, MathDataset]:
        """Prepare training and validation datasets from preprocessed files."""
        logger.info("Loading preprocessed datasets...")
        
        preprocessed_files = {
            'gsm8k_train': 'preprocessed_gsm8k_train.csv',
            'mathqa_train': 'preprocessed_mathqa_train.csv',
            'svamp_train': 'preprocessed_svamp_train.csv',
            'math500_train': 'preprocessed_math500_train.csv'
        }
        
        all_train_data = []
        all_val_data = []

        for name, filename in preprocessed_files.items():
            file_path = self.data_loader.data_dir / filename
            if file_path.exists():
                df = pd.read_csv(file_path)
                df = df.head(self.config.max_samples_per_dataset)
                data = df.to_dict('records')
                
                split_idx = int(len(data) * self.config.train_split)
                all_train_data.extend(data[:split_idx])
                all_val_data.extend(data[split_idx:])
                
                logger.info(f"Loaded {len(data)} samples from {filename}")
            else:
                logger.warning(f"Preprocessed file not found: {filename}")

        if not all_train_data:
            raise FileNotFoundError("No preprocessed training data found. Please run preprocessing first.")

        train_dataset = MathDataset(all_train_data, self.tokenizer, self.config.max_length)
        val_dataset = MathDataset(all_val_data, self.tokenizer, self.config.max_length)
        
        logger.info(f"Total training samples: {len(train_dataset)}")
        logger.info(f"Total validation samples: {len(val_dataset)}")
        
        return train_dataset, val_dataset

    def preprocess_logits_for_metrics(self, logits, labels):
        """Preprocesses logits to save memory during evaluation."""
        if isinstance(logits, tuple):
            logits = logits[0]
        return torch.argmax(logits, dim=-1)

    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation. Predictions are already argmax'd."""
        predictions, labels = eval_pred
        # Ignore padding tokens (-100)
        mask = labels != -100
        predictions = predictions[mask]
        labels = labels[mask]
        accuracy = np.mean(predictions == labels) if len(labels) > 0 else 0.0
        return {"accuracy": accuracy}

    def train_model(self):
        """Train the model using Hugging Face Trainer."""
        logger.info("Starting model training with Trainer...")
        
        self.setup_model_and_tokenizer()
        train_dataset, val_dataset = self.prepare_datasets()

        if self.model is None or self.tokenizer is None:
            raise ValueError("Model or tokenizer not initialized")

        # Mixed precision selection
        use_bf16 = torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        use_fp16 = torch.cuda.is_available() and not use_bf16

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_dir='./logs',
            logging_steps=10,
            eval_strategy=self.config.evaluation_strategy,
            eval_steps=self.config.eval_steps,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            report_to="none",  # Disable wandb/tensorboard for simplicity
            fp16=use_fp16,
            bf16=use_bf16,
            eval_accumulation_steps=1,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            label_names=["labels"]
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False, pad_to_multiple_of=8),
            preprocess_logits_for_metrics=self.preprocess_logits_for_metrics,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.config.early_stopping_patience)]
        )

        trainer.train()
        
        logger.info(f"Training completed! Model saved to {self.config.output_dir}")
        trainer.save_model()
        # Ensure tokenizer and config (including vocab_size) are saved with the adapter
        try:
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(self.config.output_dir)
        except Exception:
            pass
        try:
            if hasattr(self.model, "config"):
                self.model.config.vocab_size = len(self.tokenizer) if self.tokenizer else getattr(self.model.config, "vocab_size", None)
                # Persist updated config
                self.model.config.save_pretrained(self.config.output_dir)
        except Exception:
            pass
        
        return self.model
    
    def evaluate_model(self, model):
        """Evaluate the trained model."""
        logger.info("Evaluating model...")
        
        if self.model is None:
            raise ValueError("Model not initialized")
        
        _, val_dataset = self.prepare_datasets()

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_eval_batch_size=self.config.batch_size,
            report_to="none",
            dataloader_num_workers=0,
            label_names=["labels"]
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            eval_dataset=val_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False) if self.tokenizer else None,
            preprocess_logits_for_metrics=self.preprocess_logits_for_metrics,
            compute_metrics=self.compute_metrics
        )
        
        eval_results = trainer.evaluate()
        eval_loss = eval_results.get("eval_loss")
        if eval_loss is not None:
            try:
                eval_results["perplexity"] = float(np.exp(eval_loss))
            except Exception:
                pass
        
        # Log results
        logger.info("Evaluation Results:")
        for key, value in eval_results.items():
            logger.info(f"  {key}: {value:.4f}")
        
        # Save evaluation results
        eval_path = Path(self.config.output_dir) / "evaluation_results.json"
        with open(eval_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        return eval_results
    
    def test_model(self, test_problems: List[str]):
        """Test the trained model on specific problems using Tree of Thoughts."""
        logger.info("Testing model on sample problems with ToT...")
        
        if not self.model or not self.tokenizer:
            logger.error("Model not loaded. Run setup_model_and_tokenizer() first.")
            return

        self.model.eval()
        
        tot_generator = TreeOfThoughtsGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            math_processor=self.math_processor,
            max_depth=self.config.tot_max_depth,
            max_children=self.config.tot_max_children
        )
        
        results = []
        for problem in test_problems:
            answer = tot_generator.generate(problem)
            
            results.append({
                'problem': problem,
                'answer': answer
            })
            
            logger.info(f"Problem: {problem}")
            logger.info(f"Answer: {answer}")
            logger.info("-" * 50)
        
        return results

def main():
    """Main training function."""
    logger.info("🚀 Starting Symbolic Math AI Training")
    
    # Initialize configuration
    config = TrainingConfig()
    
    # Initialize trainer
    trainer = MathModelTrainer(config)
    
    try:
        # Train the model
        trained_model = trainer.train_model()
        
        # Evaluate the model
        eval_results = trainer.evaluate_model(trained_model)
        
        # Test on sample problems
        test_problems = [
            "What is 15 + 27?",
            "If a train travels 120 miles in 2 hours, what is its speed?",
            "Solve for x: 2x + 5 = 13",
            "What is the area of a rectangle with length 8 and width 6?"
        ]
        
        test_results = trainer.test_model(test_problems)
        
        # Save test results
        test_path = Path(config.output_dir) / "test_results.json"
        with open(test_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        logger.info("✅ Training completed successfully!")
        logger.info(f"📁 Model saved to: {config.output_dir}")
        logger.info(f"📊 Evaluation results: {eval_results}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
