"""
Model Training Module

This module handles the setup, training, evaluation, and saving of models using PyCaret.
It also integrates MLflow for tracking experiments.
"""

import os
import pandas as pd
from pycaret.classification import ClassificationExperiment
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import logging

# Get logger for this module
logger = logging.getLogger(__name__)

def setup_experiment(modeling_df, cfg):

    """
    Configure and launch a PyCaret *ClassificationExperiment*  
    with **zero-code** control over every ``setup`` knob.

    Parameters
    ----------
    modeling_df : pandas.DataFrame
        The feature-matrix already containing the mapped target column.
    cfg : dict
        Full configuration dictionary (parsed from *config.yaml*).  
        The function expects:
        ::
            cfg['modeling'] = {
                'target_column' : <str>,
                'train_size'    : <float>,
                'session_id'    : <int>,
                'setup_params'  : { … optional kwargs … }
            }

    Returns
    -------
    pycaret.classification.ClassificationExperiment
        A fully initialised experiment ready for
        ``compare_models``, ``create_model`` … etc.
    """

    modeling_cfg   = cfg['modeling']
    target_column  = modeling_cfg['target_column']
    train_size     = modeling_cfg['train_size']
    session_id     = modeling_cfg['session_id']
    setup_kwargs   = modeling_cfg.get('setup_params', {})   #  dynamic part
    log_experiment = modeling_cfg.get('log_experiment', False)

    logger.info("Setting up PyCaret experiment with kwargs: %s", setup_kwargs)

    exp = ClassificationExperiment()
    exp.setup(
        data=modeling_df,
        target=target_column,
        train_size=train_size,
        session_id=session_id,
        log_experiment=log_experiment,
        **setup_kwargs           #  forward everything from YAML
    )
    return exp


def compare_and_select_model(exp):
    """
    Compare multiple models and select the best performing one.
    
    Parameters
    ----------
    exp : ClassificationExperiment
        A PyCaret classification experiment.
        
    Returns
    -------
    tuple
        Best model and a DataFrame with model comparison results.
    """
    logger.info("Comparing models...")
    best_model = exp.compare_models()
    model_results = exp.pull()
    return best_model, model_results

