"""
Configuration Loading Module

This module provides functions to load configuration settings from YAML files.
"""

import os
import yaml
import logging
import logging.config  # Import logging.config
import sys

def get_project_root():
    """Get the absolute path to the project root directory."""
    # Assumes this script is in src/utils
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(current_dir)
    project_dir = os.path.dirname(src_dir)
    return project_dir

def load_config(config_name="config.yaml"):
    """Load configuration from a YAML file in the configs directory."""
    project_root = get_project_root()
    config_path = os.path.join(project_root, 'configs', config_name)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Resolve relative paths based on project root
    if 'paths' in config:
        for key, path_val in config['paths'].items():
            if isinstance(path_val, str) and not os.path.isabs(path_val) and key != 'project_root':
                 # Keep project_root as is, resolve others relative to it
                 config['paths'][key] = os.path.join(project_root, path_val)
            elif key == 'project_root': # Ensure project_root itself is absolute if provided relatively
                 config['paths']['project_root'] = project_root

    return config

def setup_logging(config, force_console=False):
    """Configure logging based on the loaded configuration.

    Args:
        config (dict): The loaded configuration dictionary.
        force_console (bool): If True, only configure console logging.
    """
    log_config = config.get('logging')
    if not log_config:
        # Use basicConfig if no logging config is found, but log a warning
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.warning("Logging configuration not found in config file. Using basic console logging.")
        return

    # Determine log file path
    log_file_path = None
    if not force_console:
        paths_config = config.get('paths', {})
        log_file_path_config = log_config.get('log_file')
        log_file_template = paths_config.get('log_file_template')

        if log_file_template:
            log_file_path = log_file_template # Use the already resolved template path
        elif log_file_path_config:
            log_file_path = log_file_path_config # Use the general log file path

        # Resolve log file path relative to project root if needed and ensure dir exists
        if log_file_path and not os.path.isabs(log_file_path):
            project_root = paths_config.get('project_root', get_project_root())
            log_file_path = os.path.join(project_root, log_file_path)

        # IMPORTANT: Ensure the directory exists *before* configuring the handler
        if log_file_path:
            log_dir = os.path.dirname(log_file_path)
            try:
                os.makedirs(log_dir, exist_ok=True)
            except OSError as e:
                # Handle potential race conditions or permission issues if needed
                logging.error(f"Failed to create log directory {log_dir}: {e}")
                log_file_path = None # Fallback to console if dir creation fails

    # Use dictConfig for more robust configuration based on the yaml structure
    try:
        # Make sure the config structure matches what dictConfig expects
        log_config_dict = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
                    'datefmt': log_config.get('date_format', '%Y-%m-%d %H:%M:%S')
                },
            },
            'handlers': {
                'console': {
                    'level': log_config.get('level', 'INFO').upper(),
                    'formatter': 'standard',
                    'class': 'logging.StreamHandler',
                    'stream': 'ext://sys.stdout',  # Default to stdout
                },
            },
            'root': {
                'level': log_config.get('level', 'INFO').upper(),
                'handlers': ['console'] # Start with console handler
            }
        }

        # Add file handler *only* if log_file_path is valid and not forcing console
        if log_file_path and not force_console:
             log_config_dict['handlers']['file'] = {
                    'level': log_config.get('level', 'INFO').upper(),
                    'formatter': 'standard',
                    'class': 'logging.FileHandler',
                    'filename': log_file_path,
                    'mode': 'a', # Append mode
             }
             # Ensure 'file' handler is added only once if reconfiguring
             if 'file' not in log_config_dict['root']['handlers']:
                 log_config_dict['root']['handlers'].append('file')

        # If forcing console, ensure only console handler is present
        elif force_console:
            log_config_dict['root']['handlers'] = ['console']

        logging.config.dictConfig(log_config_dict)
        logging.info("Logging configured using dictConfig.")
        if log_file_path and not force_console:
            logging.info(f"Logging to file: {log_file_path}")
        else:
            logging.info("Logging to console.")

    except Exception as e:
        # Fallback to basic config if dictConfig fails
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.exception("Error configuring logging from dict. Falling back to basic config.", exc_info=True)
