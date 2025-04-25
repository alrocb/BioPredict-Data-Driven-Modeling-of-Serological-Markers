"""
Data Preprocessing Module

This module contains functions to load and preprocess the NHANES data.
It handles tasks such as column selection and renaming based on configuration.
"""

import os
import pandas as pd
from io import StringIO
import logging  # Import logging

# Get logger for this module
logger = logging.getLogger(__name__)

def load_data(file_path):
    """
    Load CSV data from a specified file path.
    
    Parameters
    ----------
    file_path : str
        Path to the CSV file.
        
    Returns
    -------
    pandas.DataFrame
        Loaded DataFrame.
        
    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at {file_path}")
    df = pd.read_csv(file_path)
    return df

def rename_and_select_data(df, column_mapping):
    """
    Rename columns for clarity and select only the relevant ones based on the provided mapping.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame (should contain columns specified as keys in column_mapping).
    column_mapping : dict
        Dictionary mapping original column names (keys) to new names (values).

    Returns
    -------
    pandas.DataFrame
        DataFrame with selected and renamed columns.

    Raises
    ------
    KeyError
        If any column specified in the mapping keys is not found in the DataFrame.
    """
    logger.info("Renaming and selecting columns based on configuration...")

    # Check if all keys in the mapping exist in the DataFrame columns
    missing_cols = [col for col in column_mapping.keys() if col not in df.columns]
    if missing_cols:
        raise KeyError(f"The following columns specified in the mapping were not found in the DataFrame: {missing_cols}")

    # Select only the columns that are keys in the mapping
    df_selected = df[list(column_mapping.keys())]

    # Rename the selected columns
    df_renamed = df_selected.rename(columns=column_mapping)

    logger.info(f"Selected {len(df_renamed.columns)} columns and renamed them.")
    return df_renamed

def display_data_info(df):
    """
    Generate basic information and missing value summary for a DataFrame.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to analyze.
        
    Returns
    -------
    tuple
        A tuple containing the DataFrame info string and missing values (in percentage).
    """
    buffer = StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    missing_values = df.isnull().mean() * 100
    return info_str, missing_values
