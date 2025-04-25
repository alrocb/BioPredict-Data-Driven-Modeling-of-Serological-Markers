"""
Visualization Module

This module provides functions to generate and save plots for exploratory analysis
and model evaluation (e.g., correlation heatmap, moving plot files).
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import logging  # Import logging

# Get logger for this module
logger = logging.getLogger(__name__)

def plot_correlation_heatmap(df, save_path="plots/Correlation_Heatmap.png"):
    """
    Plot and save a correlation heatmap for numeric features in the DataFrame.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing features.
    save_path : str, optional
        Path to save the heatmap image (default is "plots/Correlation_Heatmap.png").
    """
    # Select only numeric columns for correlation analysis
    numeric_df = df.select_dtypes(include=['number'])
    corr = numeric_df.corr()
    
    plt.figure(figsize=(12, 10))
    plt.imshow(corr, cmap='coolwarm', interpolation='none', aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(corr)), corr.columns, rotation=90)
    plt.yticks(range(len(corr)), corr.columns)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Correlation heatmap saved as {save_path}")


def generate_visualizations(exp, best_model, plots):
    """
    Generate and save model evaluation plots using PyCaret.
    
    Parameters
    ----------
    exp : ClassificationExperiment
        The PyCaret experiment.
    best_model : object
        The best performing model.
    output_dir : str
        Directory where plots will be saved.
    """
    for plot in plots:
        try:  # Add try-except block for robustness
            exp.plot_model(best_model, plot=plot, save=True)
            logger.info(f"{plot.capitalize()} plot saved")
        except Exception as e:
            logger.warning(f"Could not generate or save {plot} plot: {e}")
    
    # Optionally plot feature importance if supported
    try:
        exp.plot_model(best_model, plot='feature', save=True)
        logger.info("Feature importance plot saved")
    except Exception as e:
        logger.warning(f"Feature importance plot not supported or failed: {e}")