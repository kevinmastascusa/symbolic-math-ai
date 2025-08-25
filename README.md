# Symbolic Math Reasoning Assistant (SMRA)

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![SymPy](https://img.shields.io/badge/SymPy-1.13+-3B5526)
![Transformers](https://img.shields.io/badge/Transformers-4.53+-FFBF00)
![License](https://img.shields.io/badge/License-Research-blue)
![Status](https://img.shields.io/badge/Status-Working%20prototype-brightgreen)

A Symbolic Math Reasoning Assistant designed to improve the accuracy and transparency of multi-step mathematical problem-solving in generative AI models. Built to enable robust symbolic reasoning, reduce hallucinations, and provide clear step-by-step explanations for mathematical problems.

## Project Overview

The Symbolic Math Reasoning Assistant (SMRA) is a research tool focused on enhancing the mathematical reasoning capabilities of AI models. By providing structured datasets, preprocessing pipelines, and analysis tools, SMRA enables researchers and developers to build more reliable and transparent math-solving AI systems.

The project addresses key challenges in AI mathematical reasoning:
- **Hallucination reduction**: Ensuring AI models provide accurate mathematical solutions
- **Step-by-step transparency**: Breaking down complex problems into clear, logical steps
- **Symbolic reasoning**: Supporting formal mathematical reasoning processes
- **Multi-step problem solving**: Handling complex mathematical problems requiring multiple solution steps

## Features

### 🔢 **Comprehensive Dataset Support**
- **GSM8K**: Grade School Math 8K dataset for elementary mathematical reasoning
- **MathQA**: Mathematical question-answering dataset with diverse problem types
- **SVAMP (Simple Variations on Arithmetic Math word Problems)**: A challenge set for elementary-level Math Word Problems (MWP). An MWP consists of a short Natural Language narrative that describes a state of the world and poses a question about some unknown quantities.
- **Custom datasets**: Support for user-defined mathematical problem datasets

### 📊 **Data Processing & Analysis**
- Automated data loading and preprocessing pipelines
- Data validation and quality checks
- Exploratory data analysis tools
- Feature engineering for mathematical reasoning tasks

### 🧠 **Symbolic Reasoning Focus**
- Step-by-step solution tracking
- Difficulty level classification
- Problem complexity analysis
- Solution verification mechanisms

### 📈 **Research-Ready Tools**
- Jupyter notebook workflows for data exploration
- Standardized data formats for model training
- Sample data generation for testing
- Comprehensive dataset statistics and insights

## Quickstart

### Run the app (Windows)

```bat
\.venv_app\Scripts\python -m streamlit run app_streamlit.py --server.headless=true
```

If you prefer a clean setup:

```bat
python -m venv .venv_app
\.venv_app\Scripts\pip install -r requirements.txt
\.venv_app\Scripts\python -m streamlit run app_streamlit.py
```

Figures are exported to `trained_math_model_qwen_run2/` by `make_evaluation_diagram.py`.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/kevinmastascusa/symbolic-math-ai.git
   cd symbolic-math-ai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python -c "from data_loader import MathDatasetLoader; print('Installation successful!')"
   ```

### Dependencies

The project requires the following key packages:
- `pandas>=1.5.0` - Data manipulation and analysis
- `numpy>=1.21.0` - Numerical computing
- `matplotlib>=3.5.0` - Data visualization
- `scikit-learn>=1.1.0` - Machine learning utilities
- `jupyter>=1.0.0` - Interactive notebook environment

See `requirements.txt` for the complete list of dependencies.

## Results (current)

- Overall EM: **29.4%**
- Per‑dataset EM: GSM8K **32.5%**, SVAMP **62.5%**, MATH‑500 **16.0%**, MathQA **0.0%** (format/schema mismatch).
- Reasoning diagnostics (examples with any equation): GSM8K **72.5%**, MathQA **60.0%**, SVAMP **85.0%**, MATH‑500 **69.0%**.

Generated figures (see folder `trained_math_model_qwen_run2/`):
- `eval_scoreboard.(svg|png)` — overall metrics table
- `eval_datasets.(svg|png)` — per‑dataset EM bars
- `eval_reasoning.(svg|png)` — rs_* reasoning stats

## Usage Examples

### Basic Dataset Loading

```python
from data_loader import MathDatasetLoader

# Initialize the data loader
loader = MathDatasetLoader()

# Load all available datasets
datasets = loader.get_all_datasets()

# Display dataset information
for name, df in datasets.items():
    print(f"{name}: {df.shape[0]} samples, {df.shape[1]} features")
```

### Working with Specific Datasets

```python
# Load GSM8K training data
gsm8k_train = loader.load_gsm8k(split="train")
print(f"GSM8K Training: {len(gsm8k_train)} problems")

# Load MathQA test data
mathqa_test = loader.load_mathqa(split="test")
print(f"MathQA Test: {len(mathqa_test)} problems")

# Examine sample problems
print("\nSample GSM8K problem:")
print(gsm8k_train['question'].iloc[0])
print(f"Answer: {gsm8k_train['answer'].iloc[0]}")

# For custom dataset with different column names
custom_data = loader.load_custom_math_dataset()
print(f"\nSample Custom problem:")
print(custom_data['problem_text'].iloc[0])
print(f"Answer: {custom_data['final_answer'].iloc[0]}")
```

### Jupyter Notebook Workflows

The project includes pre-built Jupyter notebooks for common workflows:

1. **Data Preprocessing** (`01_data_preprocessing.ipynb`)
   - Load and clean datasets
   - Perform data validation
   - Generate sample data for testing

2. **Exploratory Data Analysis** (`02_exploratory_data_analysis.ipynb`)
   - Analyze dataset characteristics
   - Visualize problem distributions
   - Generate insights for model development

To run the notebooks:
```bash
jupyter notebook 01_data_preprocessing.ipynb
```

### Custom Dataset Integration

```python
# Create custom math dataset
custom_data = loader.load_custom_math_dataset()

# Save processed datasets
loader.save_datasets(datasets)
```

## Contributing

We welcome contributions to improve the Symbolic Math Reasoning Assistant! Here's how to get started:

### Development Setup

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/symbolic-math-ai.git
   cd symbolic-math-ai
   ```

3. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Install development dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Making Changes

1. **Make your changes** in the feature branch
2. **Test your changes** thoroughly:
   ```bash
   python -c "from data_loader import MathDatasetLoader; loader = MathDatasetLoader(); loader.get_all_datasets()"
   ```
3. **Run the Jupyter notebooks** to ensure they work correctly
4. **Update documentation** if needed

### Submitting Changes

1. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add your descriptive commit message"
   ```

2. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a Pull Request** on GitHub with:
   - Clear description of changes
   - Any relevant issue references
   - Testing instructions

### Contribution Guidelines

- Follow existing code style and conventions
- Add appropriate comments and documentation
- Test your changes with sample datasets
- Update README.md if adding new features
- Ensure backward compatibility when possible

## License

This project is currently under development. Please check back for license information or contact the project maintainers for usage permissions.

## Contact & Support

### Getting Help

- **Issues**: Report bugs or request features via [GitHub Issues](https://github.com/kevinmastascusa/symbolic-math-ai/issues)
- **Questions**: For general questions about usage, please open a GitHub issue with the "question" label

### Project Maintainer

- **Kevin Mastascusa** - [@kevinmastascusa](https://github.com/kevinmastascusa)

### Contributors

- **Kevin Mastascusa**
- **Jiky Dong**
- **Ian Powrie**

### Support the Project

If you find this project helpful for your research or development work, please consider:
- ⭐ Starring the repository
- 🍴 Contributing improvements
- 📢 Sharing with the community

---

**Note**: This is a research project focused on improving mathematical reasoning in AI systems. The datasets and tools are intended for academic and research purposes.
