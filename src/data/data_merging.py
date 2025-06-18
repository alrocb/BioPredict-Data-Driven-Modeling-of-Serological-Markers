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

def concatenate_nhanes_data(input_dir, output_file, key=PRIMARY_KEY):
    """
    Finds all CSV files in the input directory, reads them,
    and concatenates them vertically (union operation) without joins.
    This preserves all data from all cycles.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing CSV files
    output_file : str
        Path where the concatenated CSV will be saved
    key : str
        Primary key to check for existence (not used for merging)
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
            logger.info(f"Loaded {os.path.basename(file_path)} with {len(df)} rows and {len(df.columns)} columns")
            dataframes.append(df)
    
    if not dataframes:
        logger.error("No valid DataFrames loaded. Exiting concatenation process.")
        return
    
    # Concatenate all dataframes vertically, filling missing columns with NaN
    concatenated_df = pd.concat(dataframes, ignore_index=True, sort=False)
    
    logger.info(f"Concatenated {len(dataframes)} files into dataset with {len(concatenated_df)} rows and {len(concatenated_df.columns)} columns")
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    concatenated_df.to_csv(output_file, index=False)
    logger.info(f"Concatenated data saved to {output_file}")

def merge_by_cycles_nhanes_data(input_dir, output_file, key=PRIMARY_KEY):
    """
    Groups CSV files by NHANES cycle, merges files within each cycle on SEQN,
    then concatenates the merged cycles.
    
    Strategy:
    1. Merge 2023-2024 demographics + hepatitis files (inner join on SEQN)
    2. Merge 2017-2020 demographics + hepatitis files (inner join on SEQN)  
    3. Concatenate the two merged datasets
    
    Parameters:
    -----------
    input_dir : str
        Directory containing CSV files
    output_file : str
        Path where the final merged CSV will be saved
    key : str
        Primary key for merging (SEQN)
    """
    logger.info(f"Looking for CSV files in {input_dir}")
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    
    if not csv_files:
        logger.error("No CSV files found in the input directory.")
        return
    
    # Group files by cycle
    cycle_2023_2024 = []
    cycle_2017_2020 = []
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        if "2023_2024" in filename:
            cycle_2023_2024.append(file_path)
        elif "2017_2020" in filename or filename.startswith("P_"):
            cycle_2017_2020.append(file_path)
        else:
            logger.warning(f"Could not classify file {filename} into a cycle. Skipping.")
    
    logger.info(f"2023-2024 cycle: {len(cycle_2023_2024)} files")
    logger.info(f"2017-2020 cycle: {len(cycle_2017_2020)} files")
    
    merged_cycles = []
    
    # Process 2023-2024 cycle
    if cycle_2023_2024:
        logger.info("Processing 2023-2024 cycle...")
        cycle_dataframes = []
        for file_path in cycle_2023_2024:
            df = read_csv_file(file_path, key)
            if df is not None:
                logger.info(f"Loaded {os.path.basename(file_path)} with {len(df)} rows")
                cycle_dataframes.append(df)
        
        if len(cycle_dataframes) >= 2:
            # Merge within cycle using inner join
            merged_2023_2024 = merge_dataframes_on_key(cycle_dataframes, key)
            logger.info(f"Merged 2023-2024 cycle: {merged_2023_2024.shape[0]} rows, {merged_2023_2024.shape[1]} columns")
            merged_cycles.append(merged_2023_2024)
        elif len(cycle_dataframes) == 1:
            logger.info(f"Only one file in 2023-2024 cycle, using as-is")
            merged_cycles.append(cycle_dataframes[0])
    
    # Process 2017-2020 cycle
    if cycle_2017_2020:
        logger.info("Processing 2017-2020 cycle...")
        cycle_dataframes = []
        for file_path in cycle_2017_2020:
            df = read_csv_file(file_path, key)
            if df is not None:
                logger.info(f"Loaded {os.path.basename(file_path)} with {len(df)} rows")
                cycle_dataframes.append(df)
        
        if len(cycle_dataframes) >= 2:
            # Merge within cycle using inner join
            merged_2017_2020 = merge_dataframes_on_key(cycle_dataframes, key)
            logger.info(f"Merged 2017-2020 cycle: {merged_2017_2020.shape[0]} rows, {merged_2017_2020.shape[1]} columns")
            merged_cycles.append(merged_2017_2020)
        elif len(cycle_dataframes) == 1:
            logger.info(f"Only one file in 2017-2020 cycle, using as-is")
            merged_cycles.append(cycle_dataframes[0])
    
    if not merged_cycles:
        logger.error("No valid merged cycles. Exiting merge process.")
        return
    
    # Concatenate the merged cycles
    if len(merged_cycles) == 1:
        final_merged = merged_cycles[0]
        logger.info(f"Only one cycle available, using as final dataset")
    else:
        final_merged = pd.concat(merged_cycles, ignore_index=True, sort=False)
        logger.info(f"Concatenated {len(merged_cycles)} cycles")
    
    logger.info(f"Final merged dataset: {final_merged.shape[0]} rows, {final_merged.shape[1]} columns")
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    final_merged.to_csv(output_file, index=False)
    logger.info(f"Merged data saved to {output_file}")

def main():
    """Main function to execute the data merging process by cycles."""
    project_root = get_project_root()
    
    # Define input directory and output file paths
    input_dir = os.path.join(project_root, "BioPredict","data", "interim")
    output_file = os.path.join(project_root,"BioPredict", "data", "extra", "merged.csv")
    
    logger.info("Starting NHANES data merging by cycles process")
    merge_by_cycles_nhanes_data(input_dir, output_file)
    logger.info("Data merging by cycles completed")

if __name__ == "__main__":
    main()
