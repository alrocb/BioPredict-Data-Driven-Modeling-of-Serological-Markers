"""
Data Preprocessing Module

This module contains functions to load and preprocess the NHANES data.
It handles tasks such as missing value imputation, column selection, and renaming.
"""

import os
import pandas as pd
from io import StringIO

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

def preprocess_data(df):
    """
    Preprocess the DataFrame: select columns, impute missing values, and rename columns.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Raw input DataFrame.
        
    Returns
    -------
    pandas.DataFrame
        Preprocessed DataFrame.
    """
    # Select columns of interest
    cols = ['SEQN', 'LBXHBS', 'DMDBORN4', 'INDFMPIR', 'RIAGENDR', 'RIDAGEYR', 'DMDHRBR4',
            'DMDEDUC2', 'DMDMARTL', 'DMDHHSIZ', 'INDFMIN2', 'RIDRETH1', 'DUQ370',
            'IMQ020', 'ALQ120Q', 'OHXIMP', 'SXQ251', 'HIQ031A', 'BMXBMI', 'BMXWAIST']
    df = df[cols]
    
    # Impute missing values for numeric columns with the median
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    # Impute missing values for categorical columns with the mode
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # Rename columns for clarity
    df = df.rename(columns={
        "DMDBORN4": "Country_of_Birth",
        "INDFMPIR": "Income_to_Poverty_Ratio",
        "RIAGENDR": "Gender",
        "RIDAGEYR": "Age",
        "DMDHRBR4": "Household_Reference_Country",
        "DMDEDUC2": "Education_Level",
        "DMDMARTL": "Marital_Status",
        "DMDHHSIZ": "Household_Size",
        "INDFMIN2": "Family_Income",
        "RIDRETH1": "Race_Ethnicity",
        "DUQ370": "Injected_Drugs_Ever",
        "IMQ020": "HepatitisB_Vaccinated",
        "ALQ120Q": "Alcohol_Frequency_12m",
        "OHXIMP": "Dental_Implant",
        "SXQ251": "Unprotected_Sex_12m",
        "HIQ031A": "Private_Insurance",
        "BMXBMI": "Body_Mass_Index",
        "BMXWAIST": "Waist_Circumference",
        "LBXHBS": "HBsAg"
    })
    return df

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
