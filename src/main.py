"""
Main Script

This is the entry point of the NHANES data analysis and modeling pipeline.
It orchestrates data conversion, merging, cleaning, data loading, preprocessing,
visualization, model training, evaluation, interpretability, and final reporting,
using parameters loaded from a configuration file.
"""

import os
import sys
from datetime import datetime
import pandas as pd
import yaml
import logging
import logging.config

# Import configuration loader first
from utils.config_loader import load_config, setup_logging

# Import modules from the package
from utils.info_renamedata import load_data, rename_and_select_data, display_data_info
from visualizations.plots import plot_correlation_heatmap, generate_visualizations
from utils.utils import move_plots, move_html, create_timestamp_dir_structure, map_target_values  # Import the new mapping function
from models.models_training import (
    setup_experiment,
    compare_and_select_model,
)
from models.model_evaluation import refine_and_save_models, evaluate_model
from models.model_interpretation import (
    generate_all_interpretation_plots,
    check_model_fairness
)

# Additional modules for conversion, merging, and cleaning
from data.data_conversion import convert_xpt_to_csv
from data.data_merging import merge_nhanes_data
from data.data_cleaning import clean_data, impute_missing_values

# Get a logger for this module
logger = logging.getLogger(__name__)


def main():
    # Load configuration
    config = load_config()

    # Initial logging setup for console
    setup_logging(config)

    paths_config = config['paths']
    data_cleaning_config = config['data_cleaning']
    data_preprocessing_config = config['data_preprocessing']
    modeling_config = config['modeling']
    visualization_config = config.get('visualizations', {}) # Get visualization config, default to empty dict

    # Create output directory for logs and results using config path
    output_dir = paths_config['output']
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamped directory structure for this run
    paths_config = create_timestamp_dir_structure(paths_config)
    
    # Update the run-specific directories in config
    run_dir = paths_config.get('run_dir')
    plots_dir = paths_config['plots_dir']
    interpretability_dir = paths_config['interpretability_dir']
    models_dir = paths_config['models_dir']
    
    # Update the config with new path values for logging reconfiguration
    config['paths'] = paths_config
    
    # Reconfigure logging with the timestamp-specific log file
    setup_logging(config)

    logger.info(f"Analysis started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Results will be saved in: {run_dir}")
    logger.info("Using configuration:")
    config_dump = yaml.dump(config, default_flow_style=False)
    for line in config_dump.splitlines():
        logger.info(line)
    logger.info("=" * 80)

    try:
        # Define paths from config
        raw_dir = paths_config['raw_data']
        interim_dir = paths_config['interim_data']
        merged_file = paths_config['merged_file']
        cleaned_file = paths_config['cleaned_file']
        run_dir = paths_config['run_dir']

        # ---------------------------
        # 1. Data Conversion (Optional - uncomment if needed)
        # ---------------------------
        logger.info("Starting Data Conversion Process")
        num_converted = convert_xpt_to_csv(raw_dir, interim_dir)
        logger.info(f"Data Conversion Completed: {num_converted} files converted.")

        # ---------------------------
        # 2. Data Merging
        # ---------------------------
        logger.info("Starting Data Merging Process")
        os.makedirs(os.path.dirname(merged_file), exist_ok=True)
        merge_nhanes_data(interim_dir, merged_file)
        logger.info("Data Merging Completed.")

        # ---------------------------
        # 3. Data Cleaning
        # ---------------------------
        logger.info("Starting Data Cleaning Process")
        clean_data(merged_file, cleaned_file, config)
        logger.info("Data Cleaning Completed.")

        # ---------------------------
        # 4. Data Loading and Preprocessing for Analysis
        # ---------------------------
        logger.info("Starting Data Loading and Preprocessing for Analysis")
        df = load_data(cleaned_file)

        logger.info("Checking for any remaining missing values after cleaning (should be minimal/none)")
        df = impute_missing_values(df, data_cleaning_config['imputation_mapping'])

        logger.info("Renaming and Selecting Features based on config")
        column_mapping = data_preprocessing_config['column_mapping']
        df = rename_and_select_data(df, column_mapping)

        info_str, missing_values = display_data_info(df)
        logger.info("Processed Data Info (Post-Renaming/Selection):")
        for line in info_str.splitlines():
            logger.info(line)
        logger.info("Percentage of Missing Values (Post-Renaming/Selection - should be 0%):")
        missing_vals_str = str(missing_values[missing_values > 0])
        if missing_vals_str and missing_vals_str != "Series([], )":
            logger.warning(f"Remaining missing values found:\n{missing_vals_str}")
        else:
            logger.info("No remaining missing values found.")

        os.makedirs(plots_dir, exist_ok=True)
        plot_path = os.path.join(plots_dir, "correlation_heatmap.png")
        plot_correlation_heatmap(df, save_path=plot_path)
        logger.info(f"Correlation heatmap saved to {plot_path}")
        #Save df to data/processed/processed_data.csv
        processed_data_path = os.path.join(paths_config['processed_data'], "processed_data.csv")
        os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
        df.to_csv(processed_data_path, index=False)
        logger.info(f"Processed data saved to {processed_data_path}")
        

        # ---------------------------
        # 5. Model Setup and Training
        # ---------------------------
        modeling_df = df.copy()
        target_column = modeling_config['target_column']
        target_mapping = modeling_config.get('target_mapping')  # Get the mapping from config

        # Apply the target mapping using our utility function
        if target_mapping:
            logger.info("Applying target mapping from configuration...")
            modeling_df = map_target_values(modeling_df, target_column, target_mapping)
            # save the df with mapped target to a new file
            mapped_target_file = os.path.join(paths_config['processed_data'], "mapped_target_data.csv")
            os.makedirs(os.path.dirname(mapped_target_file), exist_ok=True)
            modeling_df.to_csv(mapped_target_file, index=False)
            logger.info(f"Mapped target data saved to {mapped_target_file}")

        # Set up PyCaret experiment with the DataFrame that has the target already mapped
        exp = setup_experiment(modeling_df, config)          # cfg now parsed inside
        logger.info("Comparing and selecting best model...")
        best_model, model_results = compare_and_select_model(exp)
        logger.info(f"Best model selected: {best_model}")
        logger.info("Model comparison results:")
        logger.info(model_results.to_string())

        # ---------------------------
        # 6. Evaluation
        # ---------------------------
        logger.info("Evaluating best model...")
        predictions = evaluate_model(exp, best_model, run_dir)
        logger.info("Predictions (first 5 rows):")
        logger.info(predictions.head().to_string())

        move_plots(os.getcwd(), plots_dir)

        # ---------------------------
        # 7. Model Interpretability
        # ---------------------------
        logger.info("Starting Model Interpretability Analysis")

        generate_all_interpretation_plots(exp, best_model, run_dir)

        sensitive_features = modeling_config['sensitive_features']
        valid_sensitive_features = [f for f in sensitive_features if f in modeling_df.columns]
        if len(valid_sensitive_features) < len(sensitive_features):
            missing_sens_features = set(sensitive_features) - set(valid_sensitive_features)
            logger.warning(f"Some sensitive features not found in modeling data: {missing_sens_features}")
        if valid_sensitive_features:
            logger.info(f"Checking model fairness for features: {valid_sensitive_features}")
            check_model_fairness(exp, best_model, valid_sensitive_features, run_dir)
        else:
            logger.warning("Skipping fairness check as no valid sensitive features were found in the data.")

        logger.info("Model Interpretability Analysis Completed.")
        move_html(os.getcwd(), interpretability_dir)

        # ---------------------------
        # 8. Visualization 
        # ---------------------------
        logger.info("Generating PyCaret model evaluation plots...")
        # Get the list of plots from the config
        pycaret_plots_list = visualization_config.get('pycaret_plots', []) # Default to empty list if not found
        generate_visualizations(exp, best_model, pycaret_plots_list) # Pass the list
        move_plots(os.getcwd(), plots_dir)  # Because PyCaret saves plots to cwd

        # ---------------------------
        # 9. Model Refinement and Saving
        # ---------------------------
        logger.info("Refining and saving models...")
        os.makedirs(models_dir, exist_ok=True)
        refine_and_save_models(exp, best_model, models_dir)
        logger.info(f"Models saved to {models_dir}")

        # ---------------------------
        # 10. Final Summary
        # ---------------------------
        logger.info("=" * 80)
        logger.info(f"Analysis completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Results saved in {run_dir}")

    except Exception as e:
        logger.exception(f"An error occurred during the analysis: {e}", exc_info=True)
    finally:
        log_file_path = config.get('logging', {}).get('log_file', 'console')
        if log_file_path != 'console':
            if not os.path.isabs(log_file_path):
                project_root = config.get('paths', {}).get('project_root', os.getcwd())
                log_file_path = os.path.join(project_root, log_file_path)
            logger.info(f"Log file saved to {log_file_path}")
        else:
            logger.info("Logging was directed to console.")
        logging.shutdown()

if __name__ == '__main__':
    main()
