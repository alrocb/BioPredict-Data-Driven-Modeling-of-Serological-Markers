"""
NHANES Data Conversion Module

This module converts NHANES data files from .xpt format to .csv format.
It reads files from the raw data directory and saves them to the processed directory.
"""

import os
import pandas as pd
import logging

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

def convert_xpt_to_csv(input_folder, output_folder):
    """
    Convert all .xpt files in the input folder to .csv files in the output folder.
    
    Parameters:
    -----------
    input_folder : str
        Path to the folder containing .xpt files
    output_folder : str
        Path to the folder where .csv files will be saved
    """
    # Check if the input folder exists
    if not os.path.exists(input_folder):
        logger.warning(f"Input directory not found: {input_folder}. Skipping conversion.")
        return 0 # Return 0 files processed

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        logger.info(f"Created output directory: {output_folder}")
    
    # Count the number of files processed
    files_processed = 0
    
    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.xpt'):
            xpt_file = os.path.join(input_folder, filename)
            logger.debug(f"Processing {xpt_file}")

            try:
                # Read the .xpt file into a DataFrame
                df = pd.read_sas(xpt_file, format='xport')
                
                # Create the CSV filename (same base name, but with .csv extension)
                csv_filename = os.path.splitext(filename)[0] + '.csv'
                csv_file = os.path.join(output_folder, csv_filename)
                
                # Save the DataFrame to CSV
                df.to_csv(csv_file, index=False)
                logger.debug(f"Saved to {csv_file}")
                files_processed += 1
                
            except Exception as e:
                logger.error(f"Error processing {xpt_file}: {str(e)}")
    
    logger.info(f"Conversion complete. Processed {files_processed} files.")
    return files_processed

def main():
    """Main function to execute the conversion process."""
    project_root = get_project_root()
    
    # Define input and output folders
    input_folder = os.path.join(project_root, "data", "raw")
    output_folder = os.path.join(project_root, "data", "interim")
    
    logger.info("Starting NHANES data conversion process")
    num_files = convert_xpt_to_csv(input_folder, output_folder)
    logger.info(f"Conversion completed. Converted {num_files} files.")

if __name__ == "__main__":
    main()
