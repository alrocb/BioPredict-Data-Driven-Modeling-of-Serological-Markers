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

# Rename and modify the function to accept the model to refine
def tune_model(exp, model_to_refine, models_dir):
    """
    Tunes the hyperparameters of the provided model, creates a Bagging ensemble
    of the tuned model, saves the final ensembled pipeline, and returns the
    ensembled model object.

    Parameters
    ----------
    exp : ClassificationExperiment
        The PyCaret experiment.
    model_to_refine : object
        The model object (e.g., the initial best model) to tune and ensemble.
    models_dir : str
        Directory to save the final model pipeline.

    Returns
    -------
    object
        The final ensembled model object or None if failed.
    """
    try:
        model_name = type(model_to_refine).__name__
        logger.info(f"Tuning hyperparameters for model: {model_name}...")
        # Tune the passed model
        tuned_model = exp.tune_model(model_to_refine, choose_better=True) # Keep the better one

        # logger.info(f"Ensembling tuned model ({model_name}) using Bagging...")
        # Ensemble the tuned model
        #ensembled_model = exp.ensemble_model(tuned_model, choose_better=True) # Keep the better one

        # Ensure models directory exists
        os.makedirs(models_dir, exist_ok=True)

        # Define path for the final model
        final_model_filename = "best_model.pkl"
        final_model_path = os.path.join(models_dir, final_model_filename)

        # Save the final ensembled model pipeline
        logger.info(f"Saving final tuned and ensembled model pipeline to {final_model_path}")
        exp.save_model(tuned_model, final_model_path)

        return tuned_model # Return the final model object

    except Exception as e:
        logger.error(f"Error during model tuning and ensembling: {e}", exc_info=True)
        return None # Return None if refinement fails


def evaluate_model(exp, model, output_dir):
    """
    Evaluate the given model on the test set and save predictions.
    
    Parameters
    ----------
    exp : ClassificationExperiment
        The PyCaret experiment.
    model : object
        The model to evaluate (e.g., initial best or refined).
    output_dir : str
        Directory to save test set predictions.
        
    Returns
    -------
    pandas.DataFrame or None
        Predictions on the test set, or None if evaluation failed.
    """
    logger.info(f"Evaluating model on test set: {model}")
    try:
        predictions = exp.predict_model(model) # Use the passed model
        # Save predictions
        predictions_path = os.path.join(output_dir, "test_predictions.csv")
        predictions.to_csv(predictions_path, index=False)
        logger.info(f"Test set predictions saved to {predictions_path}")
        return predictions
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}", exc_info=True)
        return None


