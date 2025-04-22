"""
Model Interpretation Module

This module provides interpretability techniques for machine learning models
using PyCaret's tools with a fallback to the SHAP library.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import shap
from pycaret.classification import ClassificationExperiment

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def interpret_model(exp, model, plot_type='summary', feature=None, save=True, output_dir=None):
    """
    Generate interpretability plots for the given model using PyCaret.
    
    Parameters
    ----------
    exp : ClassificationExperiment
        The PyCaret experiment.
    model : object
        The trained model to interpret.
    plot_type : str, optional
        Type of interpretability plot (default is 'summary').
        Options: 'summary', 'correlation', 'pdp', 'msa', 'pfi'
    feature : str, optional
        Feature to focus on for certain plots (default is None).
    save : bool, optional
        Whether to save the plots (default is True).
    output_dir : str, optional
        Directory to save plots (default is None).
        
    Returns
    -------
    object
        Plot object or None if saving only.
    """
    logger.info(f"Generating {plot_type} interpretation plot")
    
    try:
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Call PyCaret's interpret_model
        plot = exp.interpret_model(model, plot=plot_type, feature=feature, save=save)
        
        if save and output_dir:
            # Move the plot if it was saved in the current directory
            plot_name = f"{plot_type}_plot"
            if feature:
                plot_name += f"_{feature.replace(' ', '_')}"
            plot_name += ".png"
            
            # Check if plot exists in current directory
            if os.path.exists(plot_name):
                target_path = os.path.join(output_dir, plot_name)
                os.rename(plot_name, target_path)
                logger.info(f"Moved plot to {target_path}")
        
        return plot
    except Exception as e:
        logger.error(f"Error generating {plot_type} plot: {str(e)}")
        print(f"Error generating {plot_type} plot: {str(e)}")
        return None

def generate_shap_plots(exp, model, output_dir):
    """
    Generate SHAP plots for model interpretation using the SHAP library directly.
    
    Parameters
    ----------
    exp : ClassificationExperiment
        The PyCaret experiment.
    model : object
        The trained model to interpret.
    output_dir : str
        Directory to save plots.
    """
    logger.info("Generating SHAP plots using SHAP library")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    try:
        # Get the estimator from the pipeline
        estimator = model
        if hasattr(model, 'named_steps') and 'estimator' in model.named_steps:
            estimator = model.named_steps['estimator']
            
        # Get training data
        X_train = exp.get_config('X_train')
        feature_names = X_train.columns.tolist()
        
        # Sample data for SHAP (for performance)
        X_sample = X_train.sample(min(100, len(X_train)), random_state=42)
        
        # Tree explainer for tree-based models, Kernel explainer as fallback
        try:
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(X_sample)
            print("Using TreeExplainer for SHAP")
        except Exception:
            print("TreeExplainer failed, using KernelExplainer")
            # Use KernelExplainer as fallback
            if hasattr(estimator, 'predict_proba'):
                explainer = shap.KernelExplainer(estimator.predict_proba, shap.sample(X_sample, 50))
            else:
                explainer = shap.KernelExplainer(estimator.predict, shap.sample(X_sample, 50))
            shap_values = explainer.shap_values(shap.sample(X_sample, 50))
        
        # For binary classification, use the positive class (class 1)
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values_to_plot = shap_values[1]
        else:
            shap_values_to_plot = shap_values[0] if isinstance(shap_values, list) else shap_values
            
        # Summary plot (beeswarm)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values_to_plot, X_sample, feature_names=feature_names, show=False)
        plt.title('SHAP Summary Plot')
        summary_path = os.path.join(plots_dir, "shap_summary_plot.png")
        plt.savefig(summary_path)
        plt.close()
        print(f"SHAP summary plot saved to {summary_path}")
        
        # Bar plot of feature importance
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values_to_plot, X_sample, plot_type="bar", feature_names=feature_names, show=False)
        plt.title('SHAP Feature Importance')
        bar_path = os.path.join(plots_dir, "shap_importance_plot.png")
        plt.savefig(bar_path)
        plt.close()
        print(f"SHAP feature importance plot saved to {bar_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating SHAP plots: {str(e)}")
        print(f"Error generating SHAP plots: {str(e)}")
        return False

def generate_all_interpretation_plots(exp, model, output_dir):
    """
    Generate all types of interpretation plots for the given model.
    
    Parameters
    ----------
    exp : ClassificationExperiment
        The PyCaret experiment.
    model : object
        The trained model to interpret.
    output_dir : str
        Directory to save plots.
    """
    plots_dir = os.path.join(output_dir, "interpretability")
    os.makedirs(plots_dir, exist_ok=True)
    
    
    # Try direct SHAP implementation first (most robust)
    print("Generating SHAP plots...")
    shap_success = generate_shap_plots(exp, model, output_dir)
    
    # Generate PyCaret plots that work with most models
    print("Generating Partial Dependence plots...")
    interpret_model(exp, model, plot_type='pdp', save=True, output_dir=plots_dir)
    
    print("Generating Morris Sensitivity Analysis...")
    interpret_model(exp, model, plot_type='msa', save=True, output_dir=plots_dir)
    
    print("Generating Permutation Feature Importance...")
    interpret_model(exp, model, plot_type='pfi', save=True, output_dir=plots_dir)
    
    # Try correlation plots - these might fail for non-tree models
    try:
        print("Generating correlation plots...")
        # Get top 3 features
        features = exp.get_config('X_train').columns.tolist()[:3]
        for feature in features:
            interpret_model(exp, model, plot_type='correlation', feature=feature, 
                          save=True, output_dir=plots_dir)
    except Exception:
        print("Correlation plots could not be generated (only works with tree-based models)")

def check_model_fairness(exp, model, sensitive_features, output_dir):
    """
    Check model fairness across sensitive demographic features.
    
    Parameters
    ----------
    exp : ClassificationExperiment
        The PyCaret experiment.
    model : object
        The trained model.
    sensitive_features : list
        List of column names to check for fairness.
    output_dir : str
        Directory to save outputs.
    """
    try:
        logger.info(f"Checking model fairness for features: {sensitive_features}")
        
        # Check if the sensitive features exist in the dataset
        available_features = exp.get_config('X_train').columns.tolist()
        valid_features = [f for f in sensitive_features if f in available_features]
        
        if not valid_features:
            print(f"Warning: None of the specified sensitive features {sensitive_features} exist in dataset")
            return None
        
        # Try PyCaret's fairness check
        fairness_results = exp.check_fairness(estimator=model, sensitive_features=valid_features)
        
        # Save fairness results
        fairness_file = os.path.join(output_dir, "fairness_metrics.csv")
        fairness_results.to_csv(fairness_file, index=False)
        print(f"Fairness metrics saved to {fairness_file}")
        
        return fairness_results
        
    except Exception as e:
        logger.error(f"Error in fairness check: {str(e)}")
        print(f"Error in fairness check: {str(e)}")
        return None

if __name__ == "__main__":
    # This allows the module to be run as a script for testing
    import sys
    from models_training import setup_experiment, compare_and_select_model
    
    # Example usage (similar to what would be in main.py)
    print("Loading data...")
    # Add code to load your dataframe here
    
    # Setup experiment
    #exp = setup_experiment(df, target_column='HBsAg')
    #best_model, _ = compare_and_select_model(exp)
    
    # Generate interpretability plots
    #output_dir = "../outputs"
    #generate_all_interpretation_plots(exp, best_model, output_dir)
    
    # Check fairness if demographic features are available
    #check_model_fairness(exp, best_model, ['Sex', 'Race_Ethnicity', 'Age'], output_dir)