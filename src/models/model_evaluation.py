"""
Model Evaluation Module

This module handles the evaluation and refining of the trained models, and saving of models using PyCaret.
"""

import os
import logging  # Import logging
import numpy as np
from pycaret.classification import ClassificationExperiment

# Get logger for this module
logger = logging.getLogger(__name__)


def refine_and_save_models(exp, best_model, output_dir):
    """
    Refine the model using gradient boosting, ensemble it, and save the pipelines.
    
    Parameters
    ----------
    exp : ClassificationExperiment
        The PyCaret experiment.
    best_model : object
        The best performing model.
    output_dir : str
        Directory where the model pipelines will be saved.
    """
    # Create, tune, and ensemble a Gradient Boosting Classifier (GBC)
    logger.info("Refining model with Gradient Boosting...")
    gbc = exp.create_model('gbc', fold=30)
    tuned_gbc = exp.tune_model(gbc, choose_better = True)
    bagged_gbc = exp.ensemble_model(tuned_gbc, choose_better = True)
    
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    best_model_path = os.path.join(model_dir, "best_model_pipeline")
    exp.save_model(best_model, best_model_path)
    logger.info(f"Best model saved to {best_model_path}")
    





def evaluate_model(exp, best_model, output_dir):
    """
    Evaluate the best model on the test set and save predictions.
    
    Parameters
    ----------
    exp : ClassificationExperiment
        The PyCaret experiment.
    best_model : object
        The selected best model.
    output_dir : str
        Directory to save test set predictions.
        
    Returns
    -------
    pandas.DataFrame
        Predictions on the test set.
    """
    logger.info("Evaluating model on test set...")
    
    # Make sure to save predictions to the run directory
    predictions = exp.predict_model(best_model)
    predictions_file = os.path.join(output_dir, "test_predictions.csv")
    predictions.to_csv(predictions_file, index=False)
    logger.info(f"Predictions saved to {predictions_file}")
    return predictions


