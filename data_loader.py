"""
Data Loader for Symbolic Math Reasoning Assistant (SMRA)

This module provides functions to load and prepare various math datasets
commonly used in symbolic reasoning research.
"""

import pandas as pd
import numpy as np
import json
import requests
import os
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class MathDatasetLoader:
    """
    A comprehensive loader for math datasets used in symbolic reasoning research.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory to store/load datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def load_gsm8k(self, split: str = "train") -> pd.DataFrame:
        """
        Load GSM8K (Grade School Math 8K) dataset.
        
        Args:
            split: 'train' or 'test'
            
        Returns:
            DataFrame with math problems and solutions
        """
        try:
            # Try to load from local file first
            file_path = self.data_dir / f"gsm8k_{split}.csv"
            if file_path.exists():
                df = pd.read_csv(file_path)
                print(f"Loaded GSM8K {split} dataset from local file")
                return df
            
            # If not available locally, create sample data
            print("GSM8K dataset not found locally. Creating sample data...")
            return self._create_gsm8k_sample(split)
            
        except Exception as e:
            print(f"Error loading GSM8K: {e}")
            return self._create_gsm8k_sample(split)
    
    def load_mathqa(self, split: str = "train") -> pd.DataFrame:
        """
        Load MathQA dataset.
        
        Args:
            split: 'train', 'dev', or 'test'
            
        Returns:
            DataFrame with math problems and solutions
        """
        try:
            file_path = self.data_dir / f"mathqa_{split}.csv"
            if file_path.exists():
                df = pd.read_csv(file_path)
                print(f"Loaded MathQA {split} dataset from local file")
                return df
            
            print("MathQA dataset not found locally. Creating sample data...")
            return self._create_mathqa_sample(split)
            
        except Exception as e:
            print(f"Error loading MathQA: {e}")
            return self._create_mathqa_sample(split)
    
    def load_mawps(self, split: str = "train") -> pd.DataFrame:
        """
        Load MAWPS (Math Word Problem Solving) dataset.
        
        Args:
            split: 'train' or 'test'
            
        Returns:
            DataFrame with math word problems
        """
        try:
            file_path = self.data_dir / f"mawps_{split}.csv"
            if file_path.exists():
                df = pd.read_csv(file_path)
                print(f"Loaded MAWPS {split} dataset from local file")
                return df
            
            print("MAWPS dataset not found locally. Creating sample data...")
            return self._create_mawps_sample(split)
            
        except Exception as e:
            print(f"Error loading MAWPS: {e}")
            return self._create_mawps_sample(split)
    
    def load_custom_math_dataset(self) -> pd.DataFrame:
        """
        Load custom math dataset for symbolic reasoning.
        
        Returns:
            DataFrame with custom math problems
        """
        try:
            file_path = self.data_dir / "custom_math_dataset.csv"
            if file_path.exists():
                df = pd.read_csv(file_path)
                print("Loaded custom math dataset from local file")
                return df
            
            print("Custom dataset not found. Creating sample data...")
            return self._create_custom_sample()
            
        except Exception as e:
            print(f"Error loading custom dataset: {e}")
            return self._create_custom_sample()
    
    def _create_gsm8k_sample(self, split: str) -> pd.DataFrame:
        """Create sample GSM8K data."""
        sample_data = {
            'question': [
                "Janet's dogs eat 2 pounds of dog food per day. How many pounds of dog food do her dogs eat in 3 days?",
                "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
                "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
                "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
                "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?"
            ],
            'answer': [
                '6 pounds',
                '6 trees',
                '39 chocolates',
                '5 cars',
                '6 trees'
            ],
            'solution': [
                "Janet's dogs eat 2 pounds of dog food per day. In 3 days, they will eat 2 * 3 = 6 pounds of dog food.",
                "There are 15 trees initially. After planting, there will be 21 trees. So the workers planted 21 - 15 = 6 trees.",
                "Leah had 32 chocolates and her sister had 42. Together they had 32 + 42 = 74 chocolates. After eating 35, they have 74 - 35 = 39 chocolates left.",
                "There are 3 cars initially. 2 more cars arrive. So there are 3 + 2 = 5 cars in the parking lot.",
                "There are 15 trees initially. After planting, there will be 21 trees. So the workers planted 21 - 15 = 6 trees."
            ],
            'difficulty': ['basic', 'basic', 'intermediate', 'basic', 'basic'],
            'category': ['word_problem', 'word_problem', 'word_problem', 'word_problem', 'word_problem']
        }
        
        df = pd.DataFrame(sample_data)
        df['dataset'] = 'gsm8k'
        df['split'] = split
        return df
    
    def _create_mathqa_sample(self, split: str) -> pd.DataFrame:
        """Create sample MathQA data."""
        sample_data = {
            'Problem': [
                'What is the value of x if 2x + 5 = 13?',
                'Find the derivative of f(x) = x^2 + 3x + 1',
                'Calculate the area of a circle with radius 5',
                'Solve the equation: 3x - 7 = 8',
                'What is the slope of the line y = 2x + 3?'
            ],
            'Rationale': [
                'Subtract 5 from both sides: 2x = 8. Then divide by 2: x = 4',
                'Apply power rule: d/dx(x^2) = 2x. Apply constant rule: d/dx(3x) = 3. So f\'(x) = 2x + 3',
                'Area = πr^2 = π(5)^2 = 25π',
                'Add 7 to both sides: 3x = 15. Then divide by 3: x = 5',
                'The slope is the coefficient of x, which is 2'
            ],
            'correct': [
                '4',
                '2x + 3',
                '25π',
                '5',
                '2'
            ],
            'options': [
                'a) 3 b) 4 c) 5 d) 6',
                'a) x + 3 b) 2x + 3 c) 2x d) x^2 + 3',
                'a) 10π b) 25π c) 50π d) 100π',
                'a) 3 b) 4 c) 5 d) 6',
                'a) 1 b) 2 c) 3 d) 4'
            ],
            'category': ['algebra', 'calculus', 'geometry', 'algebra', 'algebra']
        }
        
        df = pd.DataFrame(sample_data)
        df['dataset'] = 'mathqa'
        df['split'] = split
        return df
    
    def _create_mawps_sample(self, split: str) -> pd.DataFrame:
        """Create sample MAWPS data."""
        sample_data = {
            'sQuestion': [
                'A train travels 120 miles in 2 hours. What is its speed in miles per hour?',
                'John has 5 apples. He gives 2 to Mary. How many apples does John have now?',
                'A rectangle has length 8 and width 6. What is its area?',
                'If a car travels 60 miles per hour for 3 hours, how far does it travel?',
                'There are 20 students in a class. 12 are girls. How many are boys?'
            ],
            'lSolutions': [
                ['60'],
                ['3'],
                ['48'],
                ['180'],
                ['8']
            ],
            'lEquations': [
                ['120/2'],
                ['5-2'],
                ['8*6'],
                ['60*3'],
                ['20-12']
            ],
            'iIndex': [1, 2, 3, 4, 5],
            'category': ['speed', 'subtraction', 'area', 'distance', 'subtraction']
        }
        
        df = pd.DataFrame(sample_data)
        df['dataset'] = 'mawps'
        df['split'] = split
        return df
    
    def _create_custom_sample(self) -> pd.DataFrame:
        """Create custom math dataset for symbolic reasoning."""
        sample_data = {
            'problem_id': ['P001', 'P002', 'P003', 'P004', 'P005'],
            'problem_text': [
                'Solve for x: 2x + 5 = 13',
                'Find the derivative of f(x) = x^2 + 3x + 1',
                'Calculate the area under the curve y = x^2 from x=0 to x=2',
                'Prove that the sum of two even numbers is even',
                'Find the limit as x approaches 0 of (sin(x))/x'
            ],
            'difficulty_level': ['basic', 'intermediate', 'advanced', 'intermediate', 'advanced'],
            'subject': ['algebra', 'calculus', 'calculus', 'number_theory', 'calculus'],
            'solution_steps': [
                ['Subtract 5 from both sides: 2x = 8', 'Divide by 2: x = 4'],
                ['Apply power rule: d/dx(x^2) = 2x', 'Apply constant rule: d/dx(3x) = 3', 'Result: f\'(x) = 2x + 3'],
                ['Set up integral: ∫₀² x² dx', 'Apply power rule: [x³/3]₀²', 'Evaluate: 8/3 - 0 = 8/3'],
                ['Let a = 2k, b = 2m for integers k,m', 'a + b = 2k + 2m = 2(k + m)', 'Since k+m is integer, sum is even'],
                ['Use L\'Hôpital\'s rule', 'd/dx(sin(x)) = cos(x), d/dx(x) = 1', 'Limit = cos(0)/1 = 1']
            ],
            'final_answer': ['x = 4', 'f\'(x) = 2x + 3', 'Area = 8/3', 'The sum is even', 'Limit = 1'],
            'symbolic_complexity': [2, 4, 6, 5, 7],
            'step_count': [2, 3, 3, 3, 3]
        }
        
        df = pd.DataFrame(sample_data)
        df['dataset'] = 'custom'
        return df
    
    def get_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all available datasets.
        
        Returns:
            Dictionary with dataset names as keys and DataFrames as values
        """
        datasets = {}
        
        # Load different datasets
        datasets['gsm8k_train'] = self.load_gsm8k('train')
        datasets['gsm8k_test'] = self.load_gsm8k('test')
        datasets['mathqa_train'] = self.load_mathqa('train')
        datasets['mathqa_test'] = self.load_mathqa('test')
        datasets['mawps_train'] = self.load_mawps('train')
        datasets['mawps_test'] = self.load_mawps('test')
        datasets['custom'] = self.load_custom_math_dataset()
        
        return datasets
    
    def save_datasets(self, datasets: Dict[str, pd.DataFrame]):
        """
        Save datasets to CSV files.
        
        Args:
            datasets: Dictionary of datasets to save
        """
        for name, df in datasets.items():
            file_path = self.data_dir / f"{name}.csv"
            df.to_csv(file_path, index=False)
            print(f"Saved {name} to {file_path}")

def main():
    """Main function to demonstrate data loading."""
    loader = MathDatasetLoader()
    
    print("Loading all datasets...")
    datasets = loader.get_all_datasets()
    
    print("\nDataset Summary:")
    for name, df in datasets.items():
        print(f"{name}: {df.shape[0]} samples, {df.shape[1]} features")
    
    # Save datasets
    loader.save_datasets(datasets)
    
    return datasets

if __name__ == "__main__":
    datasets = main() 