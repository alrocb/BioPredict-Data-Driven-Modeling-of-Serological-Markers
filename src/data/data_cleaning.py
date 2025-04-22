"""
Data Cleaning Module

This module provides functions to clean and preprocess NHANES data.
It handles missing values, removes columns with high missing percentages,
and filters out low-variance features.
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
import logging
import yaml

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_project_root():
    """Get the absolute path to the project root directory."""
    # Assuming this script is in src/data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(current_dir)
    project_dir = os.path.dirname(src_dir)
    return project_dir

def load_config():
    """Load configuration from the model config file."""
    project_root = get_project_root()
    config_path = os.path.join(project_root, 'configs', 'model1.yaml')
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config

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

def impute_missing_values(df):
    """
    Impute missing values based on variable-specific 'don't know' codes.
    
    For each column, missing values are filled with the corresponding code:
      - If a specific "don't know" code is provided, that value is used.
      - For variables with no explicit mapping, missing values are imputed with 0.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with missing values imputed according to predefined mapping.
    """
    impute_mapping = {
        'DMDBORN4': 99,       # Country_of_Birth: don't know = 99
        'INDFMPIR': 0,        # Income_to_Poverty_Ratio: no "don't know" provided → 0
        'RIAGENDR': 0,        # Gender: no "don't know" provided → 0
        'RIDAGEYR': 0,        # Age: no mapping provided → 0
        'DMDHRBR4': 99,       # Household_Reference_Country: don't know = 99
        'DMDEDUC2': 9,        # Education_Level: don't know = 9
        'DMDMARTL': 99,       # Marital_Status: don't know = 99
        'DMDHHSIZ': 0,        # Household Size: no mapping provided → 0
        'INDFMIN2': 99,       # Family_Income: don't know = 99
        'RIDRETH1': 0,        # Race_Ethnicity: no "don't know" provided → 0
        'DUQ370': 9,          # Injected_Drugs_Ever: don't know = 9
        'IMQ020': 9,          # HepatitisB_Vaccinated: don't know = 7
        'ALQ120Q': 999,       # Alcohol_Frequency_12m: don't know = 999
        'OHXIMP': 0,          # Dental_Implant: no mapping provided → 0
        'SXQ251': 9,          # Unprotected_Sex_12m: don't know = 9
        'HIQ031A': 99,        # Private_Insurance: impute as 77 per instruction
        'BMXBMI': 0,          # Body_Mass_Index: no mapping provided → 0
        'BMXWAIST': 0,        # Waist_Circumference: no mapping provided → 0
        'LBXHBS': 0,          # Additional variable: no mapping provided → 0
    }
    # Apply imputation only for columns present in the DataFrame
    for col, fill_value in impute_mapping.items():
        if col in df.columns:
            df[col] = df[col].fillna(fill_value)
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

def clean_data(input_path, output_path, target_column, missing_threshold=95, variance_threshold=0.01):
    """
    Main function to clean the data.
    
    Parameters:
    -----------
    input_path : str
        Path to the input CSV file
    output_path : str
        Path where the cleaned data will be saved
    target_column : str
        Name of the target column
    missing_threshold : float, default=95
        Threshold for removing columns with high missing values
    variance_threshold : float, default=0.01
        Threshold for removing low-variance features
        
    Returns:
    --------
    pandas.DataFrame
        The cleaned dataframe
    """
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    # Get df rows and columns for logging
    df_rows = df.shape[0]
    df_columns = df.shape[1]
    
    # Analyze missing values
    missing_stats = analyze_missing_values(df, target_column)
    logger.info(f"Target column '{target_column}' has {missing_stats['target_missing_percentage']:.2f}% missing values")
    
    # Remove rows with missing target values
    
    df = df.dropna(subset=[target_column])

    # Remove columns with high missing values
    df_reduced = remove_high_missing_columns(df, threshold=missing_threshold)
    
    # Impute missing values based on specific 'don't know' codes
    df_imputed = impute_missing_values(df_reduced)
    
    # Remove low-variance features
    df_cleaned = remove_low_variance_features(df_imputed, threshold=variance_threshold)
    
    # Save the cleaned data
    logger.info(f"Saving cleaned data to {output_path}")
    df_cleaned.to_csv(output_path, index=False)
    
    logger.info(f"Data cleaning complete. Original shape: {df_rows, df_columns}, Cleaned shape: {df_cleaned.shape}")
    
    return df_cleaned

def main():
    """Main function to execute the data cleaning process."""
    # Load configuration
    config = load_config()
    target_variable = config['data']['target_variable']
    missing_threshold = config['data']['missing_threshold']
    variance_threshold = config['data']['variance_threshold']
    
    project_root = get_project_root()
    
    # Define input and output paths
    input_path = os.path.join(project_root, "data", "extra", "merged.csv")
    output_path = os.path.join(project_root, "data", "processed", "merged_cleaned.csv")
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    logger.info("Starting data cleaning process")
    clean_data(input_path, output_path, target_variable, missing_threshold, variance_threshold)
    logger.info("Data cleaning completed successfully")

if __name__ == "__main__":
    main()
