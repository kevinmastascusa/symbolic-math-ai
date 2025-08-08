#!/usr/bin/env python3
"""
MATH-500 Dataset Preprocessing Script

This script preprocesses the MATH-500 dataset for training:
- Cleans LaTeX formatting
- Standardizes problem and solution formats
- Removes problematic entries
- Creates training-ready datasets

Author: Symbolic Math AI Project
"""

import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Math500Preprocessor:
    """Preprocessor for MATH-500 dataset."""
    
    def __init__(self, data_dir: str = "Dataset"):
        self.data_dir = Path(data_dir)
        
    def clean_latex(self, text: str) -> str:
        """Clean LaTeX formatting from text."""
        if not isinstance(text, str):
            return str(text)
        
        # Remove LaTeX commands but keep content
        text = re.sub(r'\\\\[a-zA-Z]+', '', text)
        
        # Replace common LaTeX symbols
        replacements = {
            r'\\boxed{([^}]*)}': r'\1',  # Remove \boxed{} but keep content
            r'\\frac{([^}]*)}{([^}]*)}': r'(\1)/(\2)',  # Convert fractions
            r'\\sqrt{([^}]*)}': r'sqrt(\1)',  # Convert square roots
            r'\\^': '^',  # Convert superscript
            r'\\\\': '',  # Remove backslashes
            r'\\{': '(',  # Convert braces to parentheses
            r'\\}': ')',
            r'\\left': '',  # Remove left/right
            r'\\right': '',
            r'\\text{([^}]*)}': r'\1',  # Remove \text{} but keep content
            r'\\mathrm{([^}]*)}': r'\1',  # Remove \mathrm{} but keep content
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def clean_problem(self, problem: str) -> str:
        """Clean problem text."""
        # Remove ASY code blocks
        problem = re.sub(r'\\\\[.*?\\\\]', '', problem)
        problem = re.sub(r'\\\\asy\\].*?\\\\/asy\\\\]', '', problem, flags=re.DOTALL)
        
        # Clean LaTeX
        problem = self.clean_latex(problem)
        
        return problem
    
    def clean_solution(self, solution: str) -> str:
        """Clean solution text."""
        # Clean LaTeX
        solution = self.clean_latex(solution)
        
        return solution
    
    def clean_answer(self, answer: str) -> str:
        """Clean answer text."""
        # Clean LaTeX
        answer = self.clean_latex(answer)
        
        # Remove boxed formatting
        answer = re.sub(r'\\boxed{([^}]*)}', r'\1', answer)
        
        return answer
    
    def load_and_clean_data(self, split: str) -> pd.DataFrame:
        """Load and clean MATH-500 data."""
        logger.info(f"Loading MATH-500 {split} dataset...")
        
        # Try different file formats
        csv_path = self.data_dir / f"math500_{split}.csv"
        json_path = self.data_dir / f"math500_{split}.json"
        
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} samples from CSV")
        elif json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            logger.info(f"Loaded {len(df)} samples from JSON")
        else:
            raise FileNotFoundError(f"MATH-500 {split} dataset not found")
        
        # Clean the data
        logger.info("Cleaning data...")
        
        # Clean problem, solution, and answer columns
        if 'problem' in df.columns:
            df['problem_clean'] = df['problem'].apply(self.clean_problem)
        if 'solution' in df.columns:
            df['solution_clean'] = df['solution'].apply(self.clean_solution)
        if 'answer' in df.columns:
            df['answer_clean'] = df['answer'].apply(self.clean_answer)
        
        # Remove rows with empty or very short problems
        if 'problem_clean' in df.columns:
            df = df[df['problem_clean'].str.len() > 10]
        
        # Remove rows with empty answers
        if 'answer_clean' in df.columns:
            df = df[df['answer_clean'].str.len() > 0]
        
        logger.info(f"Cleaned dataset: {len(df)} samples")
        
        return df
    
    def create_training_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert to training format."""
        logger.info("Converting to training format...")
        
        # Create training-ready format
        training_data = []
        
        for _, row in df.iterrows():
            # Use cleaned versions if available, otherwise original
            problem = row.get('problem_clean', row.get('problem', ''))
            solution = row.get('solution_clean', row.get('solution', ''))
            answer = row.get('answer_clean', row.get('answer', ''))
            
            if not problem or not answer:
                continue
            
            # Create training entry
            entry = {
                'problem': problem,
                'solution': solution,
                'answer': answer,
                'subject': row.get('subject', 'Unknown'),
                'level': row.get('level', 1),
                'dataset': 'math500',
                'split': 'train' if 'train' in str(row.name) else 'test'
            }
            
            training_data.append(entry)
        
        training_df = pd.DataFrame(training_data)
        logger.info(f"Created training format: {len(training_df)} samples")
        
        return training_df
    
    def save_preprocessed_data(self, df: pd.DataFrame, split: str):
        """Save preprocessed data."""
        output_path = self.data_dir / f"preprocessed_math500_{split}.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved preprocessed data to: {output_path}")
        
        # Also save as JSON for compatibility
        json_path = self.data_dir / f"preprocessed_math500_{split}.json"
        df.to_json(json_path, orient='records', indent=2)
        logger.info(f"Saved preprocessed data to: {json_path}")
    
    def analyze_dataset(self, df: pd.DataFrame, split: str):
        """Analyze the dataset statistics."""
        logger.info(f"\n📊 MATH-500 {split} Dataset Analysis:")
        logger.info(f"  Total samples: {len(df)}")
        
        if 'subject' in df.columns:
            subject_counts = df['subject'].value_counts()
            logger.info(f"  Subjects: {dict(subject_counts)}")
        
        if 'level' in df.columns:
            level_counts = df['level'].value_counts().sort_index()
            logger.info(f"  Difficulty levels: {dict(level_counts)}")
        
        # Analyze problem lengths
        if 'problem' in df.columns:
            problem_lengths = df['problem'].str.len()
            logger.info(f"  Problem length - Mean: {problem_lengths.mean():.1f}, "
                       f"Min: {problem_lengths.min()}, Max: {problem_lengths.max()}")
        
        # Analyze answer lengths
        if 'answer' in df.columns:
            answer_lengths = df['answer'].str.len()
            logger.info(f"  Answer length - Mean: {answer_lengths.mean():.1f}, "
                       f"Min: {answer_lengths.min()}, Max: {answer_lengths.max()}")

def main():
    """Main preprocessing function."""
    logger.info("🚀 Starting MATH-500 Dataset Preprocessing")
    
    preprocessor = Math500Preprocessor()
    
    # Process train and test splits
    for split in ['train', 'test']:
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing {split} split...")
            
            # Load and clean data
            df = preprocessor.load_and_clean_data(split)
            
            # Analyze original data
            preprocessor.analyze_dataset(df, split)
            
            # Convert to training format
            training_df = preprocessor.create_training_format(df)
            
            # Analyze processed data
            preprocessor.analyze_dataset(training_df, split)
            
            # Save preprocessed data
            preprocessor.save_preprocessed_data(training_df, split)
            
            logger.info(f"✅ {split} split processed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error processing {split} split: {e}")
    
    logger.info("\n🎉 MATH-500 preprocessing completed!")

if __name__ == "__main__":
    main()
