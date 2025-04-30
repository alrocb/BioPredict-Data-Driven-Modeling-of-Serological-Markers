"""
Data Merging Module

This module provides functions to merge all CSV files in the interim directory into a single dataset.
It handles the merging process while ensuring proper handling of common identifiers.
"""

import os
import pandas as pd
import logging
import glob

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PRIMARY_KEY = "SEQN" 

def get_project_root():
    """Returns the absolute path to the project root directory."""
    return r"C:\Users\Alex\Desktop\GRIFOLS\TFG\Código\BioPredict"

def read_csv_file(filepath, key=PRIMARY_KEY):
    """
    Reads a CSV file and checks for the primary key.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    key : str
        Primary key column to check
    
    Returns:
    --------
    pandas.DataFrame or None
        Returns the DataFrame if successful; otherwise, returns None
    """
    try:
        df = pd.read_csv(filepath)
        if key not in df.columns:
            logger.warning(f"Primary key '{key}' not found in {os.path.basename(filepath)}. Skipping.")
            return None
        logger.debug(f"Loaded {os.path.basename(filepath)} with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error reading {os.path.basename(filepath)}: {e}")
        return None

def merge_dataframes_on_key(dataframes, key=PRIMARY_KEY):
    """
    Merges a list of DataFrames on the specified key using an inner join.
    
    Parameters:
    -----------
    dataframes : list of pandas.DataFrame
        List of DataFrames to merge
    key : str
        Column name to merge on
    
    Returns:
    --------
    pandas.DataFrame
        The merged DataFrame
    """
    merged_df = dataframes[0]
    for df in dataframes[1:]:
        merged_df = pd.merge(merged_df, df, on=key, how="inner")
        logger.debug(f"Merged shape is now {merged_df.shape}")
    return merged_df

def merge_nhanes_data(input_dir, output_file, key=PRIMARY_KEY):
    """
    Finds all CSV files in the input directory, reads them,
    merges them on the primary key, and saves the merged DataFrame.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing CSV files
    output_file : str
        Path where the merged CSV will be saved
    key : str
        Primary key for merging
    """
    logger.info(f"Looking for CSV files in {input_dir}")
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    
    if not csv_files:
        logger.error("No CSV files found in the input directory.")
        return
    
    dataframes = []
    for file_path in csv_files:
        logger.debug(f"Processing file: {os.path.basename(file_path)}")
        df = read_csv_file(file_path, key)
        if df is not None:
            dataframes.append(df)
    
    if not dataframes:
        logger.error("No valid DataFrames loaded. Exiting merge process.")
        return
    
    merged_df = merge_dataframes_on_key(dataframes, key)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    merged_df.to_csv(output_file, index=False)
    logger.info(f"Merged data saved to {output_file}")

def main():
    """Main function to execute the data merging process."""
    project_root = get_project_root()
    
    # Define input directory and output file paths
    input_dir = os.path.join(project_root, "BioPredict","data", "interim")
    output_file = os.path.join(project_root,"BioPredict", "data", "extra", "merged.csv")
    
    logger.info("Starting NHANES data merging process")
    merge_nhanes_data(input_dir, output_file)
    logger.info("Data merging completed")

if __name__ == "__main__":
    main()
