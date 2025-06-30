# BioPredict - Data-Driven Modeling of Serological Markers using NHANES Data


[![Python](https://img.shields.io/badge/python-3.9--3.11.*-blue.svg)](https://www.python.org/downloads/)
![State](https://img.shields.io/badge/State-Active-success)
![Version](https://img.shields.io/badge/Version-1.0.0-informational)

## Project Overview

BioPredict is a Python project developed within Grifols Bio Supplies to optimize the identification of plasma donors with specific serological biomarkers (e.g., Hepatitis B Surface Antigen - HBsAg). Confirming these biomarkers through traditional lab tests across large donor populations is often costly and time-consuming. This project leverages Machine Learning and Data Science techniques, specifically using publicly available NHANES data as a proxy, to build a predictive system.

The primary objective is to develop an AI model that uses readily available donor characteristics (demographics, clinical history, lifestyle factors) to predict the likelihood of a donor possessing a specific biomarker. This allows for prioritizing confirmatory testing on donors with a higher probability, thereby reducing costs, improving turnaround times, and increasing the availability of specialized biological materials for research and diagnostic purposes.

This repository contains a fully automated and modular pipeline, built following software engineering best practices. It handles the entire workflow from data ingestion and preprocessing to model training, evaluation, and interpretation using PyCaret.

## Features

-   **Automated Data Pipeline:** Converts NHANES `.xpt` files to `.csv`, merges datasets based on a common key, cleans data (handles missing values based on configurable strategies, drops irrelevant/low-variance columns), and preprocesses features (renaming, target variable mapping).
-   **Configuration-Driven:** Uses a central `config.yaml` file to manage all paths, parameters, and settings, allowing easy modification of datasets, targets, and pipeline behavior without code changes.
-   **Automated ML with PyCaret:** Sets up classification experiments, compares various ML models, performs hyperparameter tuning with Optuna integration, and selects the best-performing model based on specified metrics.
-   **Comprehensive Evaluation:** Evaluates the final model on a hold-out test set, saves predictions, and generates standard classification metrics and plots.
-   **Model Interpretability:** Integrates SHAP and other techniques to generate plots (Feature Importance, Summary Plots, Dependence Plots) for understanding model predictions.
-   **Fairness Analysis:** Includes checks for model fairness across predefined sensitive demographic features.
-   **Interactive Demo Application:** Streamlit-based web interface for testing trained models with real-time predictions and risk assessment.
-   **Multiple Model Support:** Supports both demographic-only and full-feature models for different use cases and data availability scenarios.
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
│   ├── deployment/             # Streamlit demo application and deployment scripts
│   ├── models/                 # Model training, evaluation, interpretation scripts
│   ├── utils/                  # Utility functions (config loading, file ops)
│   ├── visualizations/         # Visualization scripts
│   └── main.py                 # Main script to run the pipeline
├── requirements.txt            # Project dependencies
├── README.md                   # This file
└── .gitignore                  # Git ignore file
```

## Getting Started

### Prerequisites

-   Python 3.9-3.11 (as specified in requirements.txt)
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
    
    **Note:** The requirements include Streamlit for the interactive demo application. If you only need the core pipeline functionality, you can install a minimal subset, but the full requirements are recommended for the complete experience.

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

## Interactive Demo Application

After training your models using the main pipeline, you can test them interactively using the built-in Streamlit demo application.

### Demo Features

-   **Two Model Types:**
    -   **Demographics Model:** Quick assessment using only basic demographic information (Gender, Age, Race/Ethnicity)
    -   **Full Model:** Comprehensive risk assessment using all available clinical and lifestyle factors

-   **Interactive Interface:**
    -   User-friendly web interface with intuitive input forms
    -   Real-time prediction results with probability scores
    -   Color-coded risk levels (Low, Moderate, High)
    -   Prediction history tracking

### Running the Demo

1.  **Ensure models are trained:** Run the main pipeline first to generate the required model files
2.  **Launch the demo application:**
    ```bash
    # Option 1: Using the launcher script
    python src/deployment/launch_demo.py
    
    # Option 2: Direct Streamlit command
    cd src/deployment
    streamlit run streamlit_demo.py
    ```
3.  **Access the application:** Open your browser to `http://localhost:8501`

### Demo Requirements

The demo application requires the following model files in the `src/deployment/` directory:
-   `hepatitis_b_xgboost_model_demo.pkl` (Demographics model)
-   `hepatitis_b_scaler_demo.pkl` (Demographics scaler)
-   `hepatitis_b_xgboost_model.pkl` (Full model)
-   `hepatitis_b_scaler.pkl` (Full scaler)

These files are automatically generated when you run the main pipeline with appropriate configuration settings.

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

## Project Outputs & Results

### Generated Artifacts

Each pipeline run creates a comprehensive set of outputs in the `outputs/run_{timestamp}/` directory:

#### Analysis & Metrics
-   `analysis.txt`: Detailed execution log with performance metrics
-   `model_comparison_results.csv`: Comparison scores across different ML algorithms
-   `test_predictions.csv`: Final model predictions on the test set
-   `fairness_metrics.csv`: Bias and fairness analysis across demographic groups

#### Model Artifacts
-   `models/best_model.pkl`: Trained and optimized final model
-   `models/model.ipynb`: Jupyter notebook with training details (if generated)

#### Visualizations
-   `plots/`: Comprehensive collection of evaluation plots including:
    -   ROC curves and AUC analysis
    -   Confusion matrices and classification reports
    -   Feature importance rankings
    -   Calibration curves and reliability diagrams
    -   Learning curves and performance trends
    -   Correlation heatmaps

#### Interpretability Reports
-   `interpretability/`: Model explanation artifacts:
    -   SHAP summary plots and dependence plots
    -   Feature importance analysis
    -   HTML reports for interactive exploration

### Model Performance

The pipeline automatically evaluates models using multiple metrics:
-   **Classification Metrics:** Accuracy, Precision, Recall, F1-Score, AUC-ROC
-   **Calibration:** Brier Score, Calibration plots
-   **Fairness:** Demographic parity, Equalized odds across sensitive features
-   **Interpretability:** SHAP values, feature importance rankings

## Key Technologies & Dependencies

This project leverages several powerful libraries and frameworks:

-   **Data Processing:** pandas, numpy for data manipulation and analysis
-   **Machine Learning:** PyCaret for automated ML pipelines, XGBoost for gradient boosting
-   **Hyperparameter Optimization:** Optuna for efficient hyperparameter tuning
-   **Model Interpretation:** SHAP, interpret-community for explainable AI
-   **Visualization:** matplotlib for plotting and data visualization
-   **Web Interface:** Streamlit for interactive demo applications
-   **Configuration:** PyYAML for configuration management
-   **Environment:** Python 3.9-3.11 supported

For a complete list of dependencies with versions, see `requirements.txt`.

## Contributing

1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/your-feature`).
3.  Commit your changes (`git commit -m 'Add some feature'`).
4.  Push to the branch (`git push origin feature/your-feature`).
5.  Open a Pull Request.

## Acknowledgments

-   Grifols Bio Supplies for the project context and support.
-   Grifols Team

## Troubleshooting

### Common Issues

1.  **Python Version Compatibility:**
    -   Ensure you're using Python 3.9-3.11. Later versions may have compatibility issues with some dependencies
    -   Use `python --version` to check your current version

2.  **Missing Model Files for Demo:**
    -   Run the main pipeline first: `python src/main.py`
    -   Ensure the configuration includes model saving settings
    -   Check that model files are generated in the `src/deployment/` directory

3.  **Memory Issues:**
    -   Large datasets may require significant memory for processing
    -   Consider reducing dataset size or using a machine with more RAM
    -   Monitor memory usage during model training phases

4.  **Configuration Errors:**
    -   Verify `configs/config.yaml` syntax is valid YAML
    -   Ensure all required paths exist or can be created
    -   Check that file paths use forward slashes or proper OS-specific separators

5.  **Package Installation Issues:**
    -   Try upgrading pip: `pip install --upgrade pip`
    -   Use virtual environments to avoid conflicts
    -   On Windows, you may need Visual C++ Build Tools for some packages

### Getting Help

-   Check the logs in the `outputs/run_{timestamp}/analysis.txt` file for detailed error information
-   Review the configuration file for any obvious misconfigurations
-   Ensure all required data files are present in the specified directories

