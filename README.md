# BioPredict - Data-Driven Modeling of Serological Markers using NHANES Data

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE) 
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) 

## Project Overview

BioPredict is a Python project developed within Grifols Bio Supplies to optimize the identification of plasma donors with specific serological biomarkers (e.g., Hepatitis B Surface Antigen - HBsAg). Confirming these biomarkers through traditional lab tests across large donor populations is often costly and time-consuming. This project leverages Machine Learning and Data Science techniques, specifically using publicly available NHANES data as a proxy, to build a predictive system.

The primary objective is to develop an AI model that uses readily available donor characteristics (demographics, clinical history, lifestyle factors) to predict the likelihood of a donor possessing a specific biomarker. This allows for prioritizing confirmatory testing on donors with a higher probability, thereby reducing costs, improving turnaround times, and increasing the availability of specialized biological materials for research and diagnostic purposes.

This repository contains a fully automated and modular pipeline, built following software engineering best practices. It handles the entire workflow from data ingestion and preprocessing to model training, evaluation, and interpretation using PyCaret.

## Features

-   **Automated Data Pipeline:** Converts NHANES `.xpt` files to `.csv`, merges datasets based on a common key, cleans data (handles missing values based on configurable strategies, drops irrelevant/low-variance columns), and preprocesses features (renaming, target variable mapping).
-   **Configuration-Driven:** Uses a central `config.yaml` file to manage all paths, parameters, and settings, allowing easy modification of datasets, targets, and pipeline behavior without code changes.
-   **Automated ML with PyCaret:** Sets up classification experiments, compares various ML models, performs hyperparameter tuning, and selects the best-performing model based on specified metrics.
-   **Comprehensive Evaluation:** Evaluates the final model on a hold-out test set, saves predictions, and generates standard classification metrics and plots.
-   **Model Interpretability:** Integrates SHAP and other techniques to generate plots (Feature Importance, Summary Plots, Dependence Plots) for understanding model predictions.
-   **Fairness Analysis:** Includes checks for model fairness across predefined sensitive demographic features.
-   **Reproducibility:** Creates timestamped output directories for each run, storing logs, configuration snapshots, results, saved models, and all generated plots/reports.
-   **Clean Code & Documentation:** Emphasizes well-documented, clean code stored in a publicly accessible repository.

## Project Structure

```
BioPredict-Data-Driven-Modeling-of-Serological-Markers/
├── configs/                    # Configuration files (config.yaml)
├── data/                       # Data directory
│   ├── extra/                  # Extra data files (e.g., merged data)
│   ├── interim/                # Intermediate data (e.g., converted CSVs)
│   ├── processed/              # Final, cleaned datasets for modeling
│   └── raw/                    # Original, immutable data dump (e.g., .xpt files)
├── notebooks/                  # Jupyter notebooks for exploration (e.g., eda.ipynb)
├── outputs/                    # Generated outputs from runs
│   └── run_{timestamp}/        # Timestamped directory for a specific run
│       ├── analysis.txt        # Log file for the run
│       ├── fairness_metrics.csv # Fairness analysis results
│       ├── model_comparison_results.csv # PyCaret model comparison scores
│       ├── test_predictions.csv # Predictions on the test set
│       ├── interpretability/   # Model interpretation plots/reports
│       ├── models/             # Saved model artifacts (e.g., best_model.pkl)
│       └── plots/              # Generated plots (correlation, evaluation)
├── src/                        # Source code
│   ├── data/                   # Data processing scripts (conversion, merging, cleaning)
│   ├── models/                 # Model training, evaluation, interpretation scripts
│   ├── utils/                  # Utility functions (config loading, file ops)
│   └── visualizations/         # Visualization scripts
│   └── main.py                 # Main script to run the pipeline
├── requirements.txt            # Project dependencies
├── README.md                   # This file
└── .gitignore                  # Git ignore file
```

## Getting Started

### Prerequisites

-   Python (version specified in `requirements.txt`, likely 3.11+)
-   Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd BioPredict-Data-Driven-Modeling-of-Serological-Markers
    ```

2.  **Create and activate a virtual environment:** (Recommended)
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The entire pipeline is executed via the main script `src/main.py`, driven by the settings in `configs/config.yaml`.

1.  **Prepare Data:**
    -   Place your raw NHANES `.xpt` data files (or other compatible data) into the directory specified by `paths.raw_data` in `config.yaml`.

2.  **Configure the Pipeline:**
    -   Modify `configs/config.yaml` extensively to control the pipeline:
        -   Define input/output paths (`paths`).
        -   Set data cleaning parameters (`data_cleaning`: missing value thresholds, imputation strategies, columns to drop).
        -   Specify feature renaming and selection (`data_preprocessing`: `column_mapping`).
        -   Configure modeling (`modeling`: target variable, train/test split, PyCaret `setup()` parameters like `fix_imbalance`, target value mapping, sensitive features for fairness, interpretation flags).
        -   Choose visualizations (`visualizations`: list of PyCaret plots).
        -   Adjust logging settings (`logging`).

3.  **Run the Pipeline:**
    ```bash
    python src/main.py
    ```

    This single command triggers the complete, automated workflow:
    -   Data Conversion & Merging
    -   Data Cleaning & Preprocessing
    -   PyCaret Experiment Setup
    -   Model Training, Tuning & Selection
    -   Model Evaluation & Prediction Saving
    -   Model Interpretation & Fairness Checks (if enabled)
    -   Visualization Generation
    -   Saving all artifacts to a timestamped output directory.

## Configuration (`configs/config.yaml`)

This file is central to the project's flexibility:

-   `paths`: Manages all file system locations. Uses `{timestamp}` for unique run outputs.
-   `data_cleaning`: Controls how raw data is cleaned (missing thresholds, variance thresholds, explicit drops, imputation mapping).
-   `data_preprocessing`: Defines feature renaming via `column_mapping`.
-   `modeling`: Governs the ML process - target definition, data splitting, PyCaret setup args (`session_id`, `fix_imbalance`, etc.), target value re-mapping, sensitive features, interpretation options.
-   `visualizations`: Specifies which PyCaret plots to generate.
-   `logging`: Configures log level, format, and output file.

## Output Structure

Each execution generates a unique `outputs/run_{timestamp}` directory containing:

-   `analysis.txt`: Comprehensive run log.
-   `*.csv`: Result files (model comparison scores, test predictions, fairness metrics).
-   `interpretability/`: Interpretation plots/reports (HTML/images).
-   `models/`: Saved final model pipeline (`.pkl`).
-   `plots/`: Generated plots (correlation, evaluation metrics, SHAP plots etc.).

## Contributing

(Standard contribution guidelines - keep if relevant)

1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/your-feature`).
3.  Commit your changes (`git commit -m 'Add some feature'`).
4.  Push to the branch (`git push origin feature/your-feature`).
5.  Open a Pull Request.

## Acknowledgments

-   Grifols Bio Supplies for the project context and support.
-   Grifols Team

