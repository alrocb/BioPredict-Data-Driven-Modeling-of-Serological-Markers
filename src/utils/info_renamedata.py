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

def rename_and_select_data(df):
    """
    Rename columns for clarity and select only the relevant ones.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Raw input DataFrame.
        
    Returns
    -------
    pandas.DataFrame
        Cleaned DataFrame with selected and renamed columns.
    """
    column_mapping = {
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
    }

    # Select and rename columns
    df = df[list(column_mapping.keys())].rename(columns=column_mapping)
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
