# BioPredict - NHANES Data Analysis Project

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

BioPredict is a comprehensive data analysis project that leverages NHANES (National Health and Nutrition Examination Survey) data to build predictive models and extract valuable insights. This project follows modern software engineering practices and includes robust data processing, machine learning, and visualization capabilities.

## Features

- 📊 Advanced data processing and cleaning pipelines
- 🤖 Machine learning models with MLflow tracking
- 📈 Interactive visualizations and reports
- 🛠️ Automated testing and code quality checks
- 📦 Modular and maintainable codebase

## Project Structure

```
BioPredict/
├── configs/                    # Configuration files for models and experiments
├── data/                      # Data directory
│   ├── external/              # Data from third-party sources
│   ├── interim/               # Intermediate data that has been transformed
│   ├── processed/             # The final, canonical datasets for modeling
│   └── raw/                   # The original, immutable data dump
├── docs/                      # Documentation
├── logs/                      # Log files
├── mlruns/                    # MLflow experiment tracking
├── notebooks/                 # Jupyter notebooks for interactive analysis
├── outputs/                   # Model outputs and predictions
├── references/                # Data dictionaries, manuals, and explanatory materials
├── reports/                   # Generated analysis reports
│   └── figures/               # Generated graphics and figures
├── src/                       # Source code
│   ├── data/                  # Data processing scripts
│   ├── models/                # Model training and prediction scripts
│   ├── evaluation/            # Model evaluation scripts
│   ├── visualization/         # Visualization scripts
│   └── utils/                 # Utility functions and helpers
└── tests/                     # Test files
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- MLflow
- Jupyter Notebook (optional)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/biopredict.git
   cd biopredict
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

### Usage

1. Run data processing:
   ```bash
   python src/data/data_conversion.py
   python src/data/data_merging.py
   python src/data/data_cleaning.py
   ```

2. Train models:
   ```bash
   python src/models/models_training.py
   ```

3. Analyze feature importance:
   ```bash
   python src/models/feature_importance.py
   ```

4. Generate visualizations:
   ```bash
   python src/visualization/plots.py
   ```

### Using MLflow

Track experiments and compare model runs using MLflow:

```bash
mlflow ui
```

Then open your web browser to `http://localhost:5000` to view the MLflow UI.

## Testing

Run the test suite:

```bash
pytest --cov=src --cov-report=term-missing
```

## Code Style and Quality

The project uses several tools to maintain code quality:

- Black for code formatting
- isort for import sorting
- flake8 for style checking
- mypy for type checking

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NHANES (National Health and Nutrition Examination Survey)
- MLflow
- All contributors who have helped shape this project

*FOR GRIFOLS EMPLOYEES: uv sync --trusted-host github.com --trusted-host pypi.org --trusted-host files.pythonhosted.org*