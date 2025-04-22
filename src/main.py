"""
Main Script

This is the entry point of the NHANES data analysis and modeling pipeline.
It orchestrates data conversion, merging, cleaning, data loading, preprocessing,
visualization, model training, evaluation, interpretability, and final reporting.
"""

import os
import sys
from datetime import datetime

# Import modules from the package
from utils.output_capture import OutputCapture
from utils.info_renamedata import load_data, rename_and_select_data, display_data_info
from visualizations.plots import plot_correlation_heatmap, move_plots, move_html
from models.models_training import (
    setup_experiment,
    compare_and_select_model,
    generate_visualizations,
    evaluate_model,
    generate_classification_report,
    refine_and_save_models
)
from models.model_interpretation import (
    interpret_model,
    generate_all_interpretation_plots,
    generate_shap_plots,
    check_model_fairness
)

# Additional modules for conversion, merging, and cleaning
from data.data_conversion import convert_xpt_to_csv
from data.data_merging import merge_nhanes_data
from data.data_cleaning import clean_data, impute_missing_values

def main():
    # Create output directory for logs and results
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up output capture for logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"analysis_{timestamp}.txt")
    output_capture = OutputCapture(output_file)
    sys.stdout = output_capture
    
    print(f"Analysis started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    try:
        # Define project root and data directories
        #parent_project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
        project_root = os.getcwd()
        print(project_root)
        raw_dir = os.path.join(project_root, "data", "raw")
        interim_dir = os.path.join(project_root, "data", "interim")
        extra_dir = os.path.join(project_root, "data", "extra")
        processed_dir = os.path.join(project_root, "data", "processed")
        
        # ---------------------------
        # 1. Data Conversion
        # ---------------------------
        #print("Starting Data Conversion Process")
        #num_converted = convert_xpt_to_csv(raw_dir, interim_dir)
        #print(f"Data Conversion Completed: {num_converted} files converted.")
        
        # ---------------------------
        # 2. Data Merging
        # ---------------------------
        merged_file = os.path.join(extra_dir, "merged.csv")
        print("Starting Data Merging Process")
        merge_nhanes_data(interim_dir, merged_file)
        print("Data Merging Completed.")
        
        # ---------------------------
        # 3. Data Cleaning
        # ---------------------------
        cleaned_file = os.path.join(processed_dir, "merged_cleaned.csv")
        target_column = 'LBXHBS'
        print("Starting Data Cleaning Process")
        clean_data(merged_file, cleaned_file, target_column)
        print("Data Cleaning Completed.")
        
        # ---------------------------
        # 4. Data Loading and Preprocessing for Analysis
        # ---------------------------
        print("Starting Data Loading and Preprocessing for Analysis")
        df = load_data(cleaned_file)
        
        # Make sure any remaining missing values are imputed
        print("Imputing any remaining missing values")
        df = impute_missing_values(df)
        
        print("Renaming and Selecting Selected Features")
        # Rename and select relevant columns
        # This function also returns a string with the info of the DataFrame
        df = rename_and_select_data(df)
        info_str, missing_values = display_data_info(df)
        print("Renamed and Selected Data Info:")
        print(info_str)
        print("Percentage of Missing Values after Renaming:")
        print(missing_values)
        
        
        # Plot correlation heatmap
        plot_path = os.path.join(output_dir, "plots", "correlation_heatmap.png")
        plot_correlation_heatmap(df, save_path=plot_path)
        
        # ---------------------------
        # 5. Model Setup and Training
        # ---------------------------
        modeling_df = df.copy()
        exp = setup_experiment(modeling_df, target_column='HBsAg')
        best_model, model_results = compare_and_select_model(exp)

        
        # ---------------------------
        # 6. Visualization and Evaluation
        # ---------------------------
        generate_visualizations(exp, best_model, output_dir)
        predictions = evaluate_model(exp, best_model, output_dir)
        print("Predictions (first 5 rows):")
        print(predictions.head())
        #report = generate_classification_report(predictions)  --> This could be removed
        #print("Classification Report:")
        #print(report)
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        move_plots(os.getcwd(), plots_dir)
        # ---------------------------
        # 7. Model Interpretability 
        # ---------------------------
        print("Starting Model Interpretability Analysis")
 
        # Create output directories

        # Generate model interpretation plots (SHAP, PDP, etc.)
        generate_all_interpretation_plots(exp, best_model, output_dir)
        
        # Check model fairness across demographic features
        sensitive_features = ['Sex', 'Race_Ethnicity', 'Age']
        check_model_fairness(exp, best_model, sensitive_features, output_dir)
        
        print("Model Interpretability Analysis Completed.")
        interpretability_dir = os.path.join(output_dir, "interpretability")
        os.makedirs(interpretability_dir, exist_ok=True)
        move_html(os.getcwd(), interpretability_dir)
        # ---------------------------
        # 8. Model Refinement and Saving
        # ---------------------------
        refine_and_save_models(exp, best_model, output_dir)
        
        # Optionally, move any remaining plots from working directory to plots_dir
        
        
        # ---------------------------
        # 9. Final Summary
        # ---------------------------
        print("=" * 80)
        print(f"Analysis completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Results saved in {output_dir}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Restore standard output and close log file
        sys.stdout = output_capture.terminal
        output_capture.close()
        print(f"Logs saved to {output_file}")

if __name__ == '__main__':
    main()
