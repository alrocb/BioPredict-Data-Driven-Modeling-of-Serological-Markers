"""
Utils Module

"""

import os
import logging  # Import logging
from datetime import datetime

# Get a logger for this module
logger = logging.getLogger(__name__)

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
        try:  # Add try-except for robustness
            os.rename(source_path, dest_path)
            logger.info(f"Moved plot {file} to {dest_dir}")
        except OSError as e:
            logger.error(f"Error moving plot {file} from {source_path} to {dest_path}: {e}")


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
        try:  # Add try-except for robustness
            os.rename(source_path, dest_path)
            logger.info(f"Moved HTML {file} to {dest_dir}")
        except OSError as e:
            logger.error(f"Error moving HTML {file} from {source_path} to {dest_path}: {e}")


def create_timestamp_dir_structure(paths_config):
    """
    Create a timestamped directory structure for the current run and update paths.
    
    Parameters
    ----------
    paths_config : dict
        Dictionary containing path configurations.
        
    Returns
    -------
    dict
        Updated paths config with timestamp-based paths.
    """
    # Generate timestamp string for the current run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Update all paths containing {timestamp} placeholder
    updated_paths = {}
    for key, path in paths_config.items():
        if isinstance(path, str) and '{timestamp}' in path:
            updated_paths[key] = path.replace('{timestamp}', timestamp)
        else:
            updated_paths[key] = path
    
    # Create the main run directory
    run_dir = updated_paths.get('run_dir')
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        logger.info(f"Created timestamped run directory: {run_dir}")
    
    return updated_paths


def map_target_values(df, target_column, target_mapping=None):
    """
    Maps the values in the target column according to the provided mapping.
    
    This is particularly useful for ensuring PyCaret interprets classes correctly
    (e.g., mapping class 1.0 to positive class 1, and class 2.0 to negative class 0).
    
    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the target column.
    target_column : str
        The name of the target column.
    target_mapping : dict or None, optional
        Dictionary mapping original values to new values (from config).
        If None, no mapping is applied.
        
    Returns
    -------
    pandas.DataFrame
        A new DataFrame with the target column mapped according to the mapping.
        
    Raises
    ------
    ValueError
        If the mapping results in NaN values (meaning some original values were not in the mapping).
    KeyError
        If the target column doesn't exist in the DataFrame.
    """
    if target_mapping is None:
        logger.info(f"No target mapping provided for column '{target_column}'. Using original values.")
        return df.copy()
    
    # Create a copy to avoid modifying the original DataFrame
    df_mapped = df.copy()
    
    # Check if target column exists
    if target_column not in df_mapped.columns:
        error_msg = f"Target column '{target_column}' not found in DataFrame."
        logger.error(error_msg)
        raise KeyError(error_msg)
    
    logger.info(f"Applying target mapping: {target_mapping} to column '{target_column}'")
    
    # Ensure keys from YAML are correctly typed (e.g., float)
    try:
        # Attempt conversion assuming keys might be strings like '1.0'
        typed_mapping = {float(k) if isinstance(k, str) else k: v for k, v in target_mapping.items()}
        logger.info(f"Converted mapping keys to appropriate types: {typed_mapping}")
    except ValueError:
        logger.warning("Could not convert target_mapping keys. Using as is.")
        typed_mapping = target_mapping
    
    # Apply the mapping
    original_values = df_mapped[target_column].value_counts()
    logger.info(f"Original {target_column} values distribution:\n{original_values}")
    
    # Calculate and log percentage distribution
    percentage_dist = (original_values / original_values.sum() * 100)
    logger.info(f"Percentage distribution:\n{percentage_dist}")
    
    # Apply mapping
    df_mapped[target_column] = df_mapped[target_column].map(typed_mapping)
    
    # Check for NaN values that might have been introduced by mapping
    if df_mapped[target_column].isnull().any():
        unmapped_values = set(df[target_column].unique()) - set(typed_mapping.keys())
        error_msg = f"Target mapping resulted in NaN values. These values were not in mapping: {unmapped_values}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Log the mapped distribution
    mapped_values = df_mapped[target_column].value_counts()
    logger.info(f"Mapped {target_column} values distribution:\n{mapped_values}")
    logger.info(f"Mapping complete. Unique values in target column: {sorted(df_mapped[target_column].unique())}")
    
    return df_mapped