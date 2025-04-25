"""
Data Cleaning Module

This module provides functions to clean and preprocess NHANES data.
It handles missing values, removes columns with high missing percentages,
filters out low-variance features, and drops specified columns.
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
import logging
from utils.config_loader import load_config  # Import the centralized config loader

# Get logger for this module
logger = logging.getLogger(__name__)

def analyze_missing_values(df, target_column):
    """
    Analyze missing values in the dataset, particularly for the target column.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    target_column : str
        The name of the target column
        
    Returns:
    --------
    dict
        Dictionary containing missing value statistics
    """
    # Count missing values for the target column
    missing_values = df[target_column].isnull().sum()
    total_values = df[target_column].shape[0]
    
    # Calculate the percentage of missing values
    percentage_missing = missing_values / total_values * 100
    
    # Count missing values for all columns
    missing_counts = df.isnull().sum()
    missing_percentage = missing_counts / len(df) * 100
    
    return {
        'target_missing': missing_values,
        'target_total': total_values,
        'target_missing_percentage': percentage_missing,
        'all_missing_counts': missing_counts,
        'all_missing_percentage': missing_percentage
    }

def remove_high_missing_columns(df, threshold=95):
    """
    Remove columns with a high percentage of missing values.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    threshold : float, default=95
        The percentage threshold above which columns will be dropped
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with high-missing columns removed
    """
    # Calculate percentage of missing values for each column
    missing_percentage = df.isnull().sum() / len(df) * 100
    
    # Get columns to drop
    cols_to_drop = missing_percentage[missing_percentage > threshold].index
    
    logger.info(f"Removing {len(cols_to_drop)} columns with more than {threshold}% missing values")
    
    # Drop the columns
    return df.drop(columns=cols_to_drop)

def drop_specified_columns(df, columns_to_drop):
    """
    Drops columns specified in the configuration.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe.
    columns_to_drop : list
        A list of column names to drop.

    Returns:
    --------
    pandas.DataFrame
        DataFrame with specified columns removed.
    """
    existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
    if existing_cols_to_drop:
        logger.info(f"Dropping specified columns: {existing_cols_to_drop}")
        df = df.drop(columns=existing_cols_to_drop)
    else:
        logger.info("No specified columns to drop found in the DataFrame.")
    return df

def impute_missing_values(df, impute_mapping):
    """
    Impute missing values based on a provided mapping.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe.
    impute_mapping : dict
        Dictionary mapping column names to imputation values.

    Returns:
    --------
    pandas.DataFrame
        DataFrame with missing values imputed according to the provided mapping.
    """
    logger.info("Imputing missing values based on configuration mapping.")
    # Apply imputation only for columns present in the DataFrame
    imputed_count = 0
    for col, fill_value in impute_mapping.items():
        if col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(fill_value)
                imputed_count += 1
    logger.info(f"Imputed missing values in {imputed_count} columns based on mapping.")
    return df

def remove_low_variance_features(df, threshold=0.01):
    """
    Remove features with low variance.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    threshold : float, default=0.01
        The variance threshold below which features will be removed
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with low-variance features removed
    """
    # Select only numeric columns
    df_numeric = df.select_dtypes(include=['float64', 'int64'])
    
    # Impute missing values with 0 for variance calculation
    df_numeric_imputed = df_numeric.fillna(0)
    
    # Apply the VarianceThreshold method
    vt = VarianceThreshold(threshold=threshold)
    vt.fit(df_numeric_imputed)
    
    # Get columns that pass the variance threshold
    cols_variance = df_numeric.columns[vt.get_support()]
    
    logger.info(f"Removing {len(df_numeric.columns) - len(cols_variance)} low-variance features")
    
    # Keep only the selected columns from the original dataframe
    return df[cols_variance]

def clean_data(input_path, output_path, config):
    """
    Main function to clean the data using configuration parameters.

    Parameters:
    -----------
    input_path : str
        Path to the input CSV file.
    output_path : str
        Path where the cleaned data will be saved.
    config : dict
        Dictionary containing configuration parameters.

    Returns:
    --------
    pandas.DataFrame
        The cleaned dataframe.
    """
    # Extract parameters from config
    data_cleaning_config = config['data_cleaning']
    target_column = data_cleaning_config['target_column_raw']
    missing_threshold = data_cleaning_config['missing_threshold']
    variance_threshold = data_cleaning_config['variance_threshold']
    impute_mapping = data_cleaning_config['imputation_mapping']
    columns_to_drop = data_cleaning_config.get('columns_to_drop', [])  # Use .get for optional keys

    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    df_rows, df_columns = df.shape

    # Analyze missing values for the raw target column
    missing_stats = analyze_missing_values(df, target_column)
    logger.info(f"Target column '{target_column}' has {missing_stats['target_missing_percentage']:.2f}% missing values before cleaning.")

    # Remove rows with missing target values
    initial_rows = len(df)
    df = df.dropna(subset=[target_column])
    rows_dropped = initial_rows - len(df)
    logger.info(f"Removed {rows_dropped} rows with missing target values ('{target_column}').")

    # Drop specified columns (e.g., 'SEQN') BEFORE other cleaning steps
    df = drop_specified_columns(df, columns_to_drop)

    # Remove columns with high missing values
    df_reduced = remove_high_missing_columns(df, threshold=missing_threshold)

    # Impute remaining missing values based on config mapping
    df_imputed = impute_missing_values(df_reduced, impute_mapping)

    # Remove low-variance features
    df_cleaned = remove_low_variance_features(df_imputed, threshold=variance_threshold)

    # Save the cleaned data
    logger.info(f"Saving cleaned data to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure dir exists
    df_cleaned.to_csv(output_path, index=False)

    logger.info(f"Data cleaning complete. Original shape: {(df_rows, df_columns)}, Cleaned shape: {df_cleaned.shape}")

    return df_cleaned

def main():
    """Main function to execute the data cleaning process using config."""
    # Load configuration
    config = load_config()  # Load the main config

    # Get paths from config
    paths_config = config['paths']
    input_path = paths_config['merged_file']
    output_path = paths_config['cleaned_file']

    logger.info("Starting data cleaning process")
    # Pass the whole config to clean_data
    clean_data(input_path, output_path, config)
    logger.info("Data cleaning completed successfully")

if __name__ == "__main__":
    main()
