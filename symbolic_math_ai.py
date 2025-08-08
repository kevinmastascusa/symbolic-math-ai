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

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Hugging Face settings
    hf_token: Optional[str] = os.getenv("HUGGING_FACE_HUB_TOKEN")
    model_name: str = "microsoft/DialoGPT-small"  # Smaller model for faster training
    output_dir: str = "./trained_model"
    
    # Training parameters
    batch_size: int = 2
    learning_rate: float = 5e-5
    num_epochs: int = 3  # Increase epochs for early stopping
    max_length: int = 256
    warmup_steps: int = 50
    weight_decay: float = 0.01
    
    # Checkpointing and early stopping
    evaluation_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "loss"
    greater_is_better: bool = False
    early_stopping_patience: int = 3
    
    # Data parameters
    train_split: float = 0.8
    max_samples_per_dataset: int = 1000
    
    # Model parameters
    use_quantization: bool = True
    use_gradient_checkpointing: bool = True
    
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
        patterns = [
            r'\b(\w+)\s*=\s*([\w\s\+\-\*\/\(\)\d\.]+)',
            r'(\d+[\+\-\*\/]\d+)',
            r'(\d+\s*[\+\-\*\/]\s*\d+)',
            r'([\d\.]+\s*[\+\-\*\/]\s*[\d\.]+)'
        ]
        
        equations = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            equations.extend([match if isinstance(match, str) else match[0] for match in matches])
        
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
            eq = self.parse_expression(equation)
            if eq is None:
                return []
            
            var = symbols(variable)
            if '=' in equation:
                lhs, rhs = equation.split('=')
                lhs_expr = self.parse_expression(lhs.strip())
                rhs_expr = self.parse_expression(rhs.strip())
                if lhs_expr is not None and rhs_expr is not None:
                    eq = Eq(lhs_expr, rhs_expr)
                    return solve(eq, var)
            
            return solve(eq, var)
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
        model_kwargs = {
            "use_auth_token": self.config.hf_token or True,
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32
        }
        
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
        
        # Enable gradient checkpointing after model loading
        if self.config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # Resize token embeddings if needed
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        logger.info("Model and tokenizer loaded successfully")
        
    def prepare_datasets(self) -> Tuple[MathDataset, MathDataset]:
        """Prepare training and validation datasets."""
        logger.info("Loading and preparing datasets...")
        
        # Load all datasets
        datasets = {
            'gsm8k_train': self.data_loader.load_gsm8k('train'),
            'gsm8k_test': self.data_loader.load_gsm8k('test'),
            'mathqa_train': self.data_loader.load_mathqa('train'),
            'mathqa_test': self.data_loader.load_mathqa('test'),
            'svamp_train': self.data_loader.load_svamp('train'),
            'svamp_test': self.data_loader.load_svamp('test'),
            'math500_train': self.data_loader.load_math500('train'),
            'math500_test': self.data_loader.load_math500('test')
        }
        
        # Combine all training data
        all_train_data = []
        all_val_data = []
        
        for name, df in datasets.items():
            if 'train' in name:
                # Limit samples per dataset
                df = df.head(self.config.max_samples_per_dataset)
                
                # Convert to list of dicts
                data = df.to_dict('records')
                
                # Split into train/val
                split_idx = int(len(data) * self.config.train_split)
                all_train_data.extend(data[:split_idx])
                all_val_data.extend(data[split_idx:])
                
                logger.info(f"Added {len(data)} samples from {name}")
        
        # Create datasets
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

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
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
            report_to="none"  # Disable wandb/tensorboard for simplicity
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False),
            preprocess_logits_for_metrics=self.preprocess_logits_for_metrics,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.config.early_stopping_patience)]
        )

        trainer.train()
        
        logger.info(f"Training completed! Model saved to {self.config.output_dir}")
        trainer.save_model()
        
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
            report_to="none"
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
        """Test the trained model on specific problems."""
        logger.info("Testing model on sample problems...")
        
        if not self.model or not self.tokenizer:
            logger.error("Model not loaded. Run setup_model_and_tokenizer() first.")
            return
        
        self.model.eval()
        
        results = []
        for problem in test_problems:
            # Format input
            input_text = f"Question: {problem}\nAnswer:"
            
            # Tokenize
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length
            ).to(self.model.device)  # Move inputs to model's device
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    num_beams=3,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract answer
            answer = response.split("Answer:")[-1].strip()
            
            results.append({
                'problem': problem,
                'response': response,
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
