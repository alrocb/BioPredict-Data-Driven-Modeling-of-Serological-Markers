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
# Update the import name
from models.model_evaluation import tune_model, evaluate_model
from models.model_interpretation import (
    generate_all_interpretation_plots,
    check_model_fairness
)

# Additional modules for conversion, merging, and cleaning
from data.data_conversion import convert_xpt_to_csv
from data.data_merging import merge_nhanes_data
# Import specific cleaning functions needed
from data.data_cleaning import (
    impute_missing_values,
    remove_high_missing_columns,
    remove_low_variance_features,
    drop_specified_columns # Assuming this function exists or we implement the logic
)


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
    logger.info("=" * 80)

    try:
        # Define paths from config
        raw_dir = paths_config['raw_data']
        interim_dir = paths_config['interim_data']
        merged_file = paths_config['merged_file']
        # Let's define the final processed file path clearly
        processed_data_path = os.path.join(paths_config['processed_data'], "processed_data.csv")
        mapped_target_file = os.path.join(paths_config['processed_data'], "mapped_target_data.csv")
        # run_dir is already defined above

        # ---------------------------
        # 1. Data Conversion (Optional)
        # ---------------------------
        # logger.info("Starting Data Conversion Process")
        # num_converted = convert_xpt_to_csv(raw_dir, interim_dir)
        # logger.info(f"Data Conversion Completed: {num_converted} files converted.")

        # ---------------------------
        # 2. Data Merging
        # ---------------------------
        logger.info("Starting Data Merging Process")
        os.makedirs(os.path.dirname(merged_file), exist_ok=True)
        merge_nhanes_data(interim_dir, merged_file)
        logger.info("Data Merging Completed.")

        # ---------------------------
        # 3. Load Merged Data
        # ---------------------------
        logger.info(f"Loading merged data from {merged_file}")
        try:
            df = pd.read_csv(merged_file)
            logger.info(f"Loaded merged data with shape: {df.shape}")
        except FileNotFoundError:
            logger.error(f"Merged file not found at {merged_file}. Exiting.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading merged file {merged_file}: {e}")
            sys.exit(1)

        # ---------------------------
        # 4. Drop Rows with Missing Raw Target
        # ---------------------------
        raw_target_col = data_cleaning_config.get('target_column_raw')
        if raw_target_col and raw_target_col in df.columns:
            initial_rows = len(df)
            df.dropna(subset=[raw_target_col], inplace=True)
            rows_dropped = initial_rows - len(df)
            logger.info(f"Dropped {rows_dropped} rows with missing raw target ('{raw_target_col}'). Shape is now: {df.shape}")
        else:
            logger.warning(f"Raw target column '{raw_target_col}' not specified or not found in merged data. Skipping dropna.")

        # ---------------------------
        # 5. Drop Explicit Columns (e.g., SEQN)
        # ---------------------------
        columns_to_drop = data_cleaning_config.get('columns_to_drop', [])
        if columns_to_drop:
            # Use the imported drop_specified_columns function
            df = drop_specified_columns(df, columns_to_drop)
            logger.info(f"Data shape after dropping specified columns: {df.shape}")

        # ---------------------------
        # 7. Impute Missing Values
        # ---------------------------
        logger.info("Imputing missing values using mapping (ensure mapping uses RENAMED columns)")
        imputation_map = data_cleaning_config.get('imputation_mapping', {})
        df = impute_missing_values(df, imputation_map)
        # Verify imputation
        missing_after_impute = df.isnull().sum().sum()
        if missing_after_impute > 0:
             logger.warning(f"There are still {missing_after_impute} missing values after imputation. Check mapping and data.")
        else:
             logger.info("Imputation completed. No missing values remaining before high-missing/low-variance checks.")


        # ---------------------------
        # 7. Rename and Select Features
        # ---------------------------
        logger.info("Renaming and Selecting Features based on config")
        column_mapping = data_preprocessing_config['column_mapping']
        df = rename_and_select_data(df, column_mapping)
        logger.info(f"Data shape after renaming and selection: {df.shape}")
        logger.info(f"Columns remaining: {df.columns.tolist()}")





        # ---------------------------
        # 8. Clean: Remove High Missing & Low Variance Columns
        # ---------------------------
        logger.info("Applying further cleaning: Removing high-missing and low-variance columns")
        missing_threshold = data_cleaning_config.get('missing_threshold')
        variance_threshold = data_cleaning_config.get('variance_threshold')

        if missing_threshold is not None:
            df = remove_high_missing_columns(df, missing_threshold)
            logger.info(f"Data shape after removing high-missing columns: {df.shape}")

        if variance_threshold is not None:
            # Ensure target column is not accidentally removed if it has low variance
            target_col_renamed = modeling_config.get('target_column')
            target_series = None # Initialize target_series

            # Check if target exists and store it temporarily
            if target_col_renamed and target_col_renamed in df.columns:
                logger.info(f"Temporarily separating target column '{target_col_renamed}' before low variance check.")
                target_series = df[target_col_renamed]
                # Remove target column from DataFrame for variance check
                df_features = df.drop(columns=[target_col_renamed])
            else:
                logger.warning(f"Target column '{target_col_renamed}' not found or not specified. Applying variance check to all columns.")
                df_features = df # Apply to the whole df if target is not there

            # Apply remove_low_variance_features only to the features
            df_features_filtered = remove_low_variance_features(df_features, variance_threshold)
            logger.info(f"Shape after removing low-variance features (excluding target): {df_features_filtered.shape}")

            # Re-add the target column if it was separated
            if target_series is not None:
                df = pd.concat([df_features_filtered, target_series], axis=1)
                logger.info(f"Re-added target column '{target_col_renamed}'.")
            else:
                df = df_features_filtered # Update df if target was not separated

            logger.info(f"Data shape after removing low-variance features and re-adding target (if applicable): {df.shape}")

        # ---------------------------
        # 9. Display Info, Save Processed Data & Correlation Heatmap
        # ---------------------------
        info_str, missing_values = display_data_info(df)
        logger.info("Final Processed Data Info:")
        for line in info_str.splitlines():
            logger.info(line)
        logger.info("Percentage of Missing Values (should be 0%):")
        missing_vals_str = str(missing_values[missing_values > 0])
        if missing_vals_str and missing_vals_str != "Series([], )":
            logger.warning(f"Remaining missing values found unexpectedly:\n{missing_vals_str}")
        else:
            logger.info("No remaining missing values found.")

        os.makedirs(plots_dir, exist_ok=True)
        plot_path = os.path.join(plots_dir, "correlation_heatmap.png")
        plot_correlation_heatmap(df, save_path=plot_path)
        logger.info(f"Correlation heatmap saved to {plot_path}")

        os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
        df.to_csv(processed_data_path, index=False)
        logger.info(f"Final processed data saved to {processed_data_path}")


        # ---------------------------
        # 10. Model Comparison and Initial Selection
        # ---------------------------
        modeling_df = df.copy() # Use the fully processed df
        target_column = modeling_config['target_column'] # Renamed target column
        target_mapping = modeling_config.get('target_mapping')

        # Apply the target mapping using our utility function
        if target_mapping:
            logger.info("Applying target mapping from configuration...")
            modeling_df = map_target_values(modeling_df, target_column, target_mapping)
            # save the df with mapped target to a new file
            os.makedirs(os.path.dirname(mapped_target_file), exist_ok=True)
            modeling_df.to_csv(mapped_target_file, index=False)
            logger.info(f"Mapped target data saved to {mapped_target_file}")
        else:
             logger.warning("No target mapping specified in configuration.")


        # Set up PyCaret experiment
        exp = setup_experiment(modeling_df, config)
        logger.info("Comparing and selecting initial best model...")
        best_model_initial, model_results = compare_and_select_model(exp)
        logger.info(f"Initial best model selected: {best_model_initial}")
        logger.info("Model comparison results:")

        # Save the model results to a CSV file
        model_results_path = os.path.join(run_dir, "model_comparison_results.csv")
        model_results.to_csv(model_results_path, index=False)
        logger.info(f"Model comparison results saved to {model_results_path}")


        # (Optional: Save the initial best model if needed)
        # initial_model_path = os.path.join(models_dir, "initial_best_model_pipeline.pkl")
        # os.makedirs(models_dir, exist_ok=True)
        # exp.save_model(best_model_initial, initial_model_path)
        # logger.info(f"Initial best model saved to {initial_model_path}")


        # ---------------------------
        # 11. Model Tuning, Ensembling, and Saving
        # ---------------------------
        logger.info(f"Attempting to tune and ensemble the initial best model: {best_model_initial}...")
        os.makedirs(models_dir, exist_ok=True) # Ensure models dir exists
        # Call the updated function, passing the initial best model
        final_model = tune_model(exp, best_model_initial, models_dir)

        if final_model is None:
            logger.error("Model tuning/ensembling failed.")
            # Fallback to the initial best model
            logger.warning("Falling back to the initial best model for evaluation and interpretation.")
            final_model = best_model_initial
            # Save the initial model if it wasn't saved and tuning failed
            initial_model_path = os.path.join(models_dir, "initial_best_model_pipeline.pkl")
            if not os.path.exists(initial_model_path):
                 try:
                     exp.save_model(best_model_initial, initial_model_path)
                     logger.info(f"Initial best model saved to {initial_model_path} as fallback.")
                 except Exception as save_err:
                     logger.error(f"Failed to save fallback initial model: {save_err}")

        logger.info(f"Proceeding with final model: {final_model}")


        # ---------------------------
        # 12. Evaluation (Using Final Model)
        # ---------------------------
        logger.info("Evaluating final model...")
        # Use final_model here
        predictions = evaluate_model(exp, final_model, run_dir)
        if predictions is not None:
            logger.info("Evaluation completed. Predictions saved.") # Added confirmation log
        else:
            logger.error("Evaluation failed.")

        move_plots(os.getcwd(), plots_dir)


        # ---------------------------
        # 13. Model Interpretability (Using Final Model)
        # ---------------------------
        run_interpretation = modeling_config.get('run_interpretation', True)

        if run_interpretation:
            logger.info("Starting Model Interpretability Analysis for final model")
            # Use final_model here
            generate_all_interpretation_plots(exp, final_model, config, run_dir)

            sensitive_features = modeling_config['sensitive_features']
            valid_sensitive_features = [f for f in sensitive_features if f in modeling_df.columns]
            if len(valid_sensitive_features) < len(sensitive_features):
                missing_sens_features = set(sensitive_features) - set(valid_sensitive_features)
                logger.warning(f"Some sensitive features not found in modeling data: {missing_sens_features}")
            if valid_sensitive_features:
                logger.info(f"Checking model fairness for features: {valid_sensitive_features}")
                 # Use final_model here
                check_model_fairness(exp, final_model, valid_sensitive_features, run_dir)
            else:
                logger.warning("Skipping fairness check as no valid sensitive features were found in the data.")

            logger.info("Model Interpretability Analysis Completed.")
            move_html(os.getcwd(), interpretability_dir)
        else:
            logger.info("Skipping Model Interpretability Analysis as per configuration.")


        # ---------------------------
        # 14. Visualization (Using Final Model)
        # ---------------------------
        logger.info("Generating PyCaret model evaluation plots for final model...")
        pycaret_plots_list = visualization_config.get('pycaret_plots', [])
        # Use final_model here
        generate_visualizations(exp, final_model, pycaret_plots_list)
        move_plots(os.getcwd(), plots_dir)


        # ---------------------------
        # 15. Final Summary
        # ---------------------------
        # Model saving is now handled within refine_gbc_and_save or the fallback logic
        logger.info("=" * 80)
        logger.info(f"Analysis completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Results saved in {run_dir}")

    except Exception as e:
        logger.exception(f"An error occurred during the analysis: {e}", exc_info=True)
    finally:
        log_file_path = config.get('logging', {}).get('log_file', 'console')
        if log_file_path != 'console':
            final_log_path = paths_config.get('log_file_template', 'analysis.txt')
            logger.info(f"Log file saved to {final_log_path}")
        else:
            logger.info("Logging was directed to console.")
        logging.shutdown()

if __name__ == '__main__':
    main()
