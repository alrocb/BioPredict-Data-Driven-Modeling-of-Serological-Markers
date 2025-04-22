"""
Visualization Module

This module provides functions to generate and save plots for exploratory analysis
and model evaluation (e.g., correlation heatmap, moving plot files).
"""

import os
import matplotlib.pyplot as plt
import pandas as pd

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
    print(f"Correlation heatmap saved as {save_path}")

def move_plots(source_dir, dest_dir):
    """
    Move PNG plot files from a source directory to a destination directory.
    
    Parameters
    ----------
    source_dir : str
        Directory to search for .png files.
    dest_dir : str
        Directory where the plot files will be moved.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    plot_files = [f for f in os.listdir(source_dir) if f.endswith('.png')]
    for file in plot_files:
        source_path = os.path.join(source_dir, file)
        dest_path = os.path.join(dest_dir, file)
        os.rename(source_path, dest_path)
        print(f"Moved {file} to {dest_dir}")


def move_html(source_dir, dest_dir):
    """
    Move HTML files from a source directory to a destination directory.
    
    Parameters
    ----------
    source_dir : str
        Directory to search for .html files.
    dest_dir : str
        Directory where the plot files will be moved.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    html_files = [f for f in os.listdir(source_dir) if f.endswith('.html')]
    for file in html_files:
        source_path = os.path.join(source_dir, file)
        dest_path = os.path.join(dest_dir, file)
        os.rename(source_path, dest_path)
        print(f"Moved {file} to {dest_dir}")
