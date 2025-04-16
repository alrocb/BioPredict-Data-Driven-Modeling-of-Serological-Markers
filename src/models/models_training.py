"""
Model Training Module

This module handles the setup, training, evaluation, and saving of models using PyCaret.
It also integrates MLflow for tracking experiments.
"""

import os
import mlflow
import pandas as pd
from pycaret.classification import ClassificationExperiment
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def setup_experiment(modeling_df, target_column='HBsAg', train_size=0.7, session_id=123):
    """
    Set up a PyCaret classification experiment.
    
    Parameters
    ----------
    modeling_df : pandas.DataFrame
        DataFrame containing features and target.
    target_column : str, optional
        Name of the target column (default is 'HBsAg').
    train_size : float, optional
        Proportion of data used for training (default is 0.7).
    session_id : int, optional
        Random seed for reproducibility (default is 123).
        
    Returns
    -------
    ClassificationExperiment
        Configured PyCaret experiment.
    """
    # End any active MLflow run
    if mlflow.active_run():
        mlflow.end_run()
    mlflow.set_experiment("PyCaret_Classification_Experiment")
    
    # Drop identifier column if present
    if 'SEQN' in modeling_df.columns:
        modeling_df = modeling_df.drop('SEQN', axis=1)
    
    exp = ClassificationExperiment()
    exp.setup(
        data=modeling_df,
        target=target_column,
        train_size=train_size,
        session_id=session_id,
        log_experiment=False  # Avoid MLflow logging issues
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
    print("Comparing models...")
    best_model = exp.compare_models()
    model_results = exp.pull()
    return best_model, model_results

def generate_visualizations(exp, best_model, output_dir):
    """
    Generate and save model evaluation plots using PyCaret.
    
    Parameters
    ----------
    exp : ClassificationExperiment
        The PyCaret experiment.
    best_model : object
        The best performing model.
    output_dir : str
        Directory where plots will be saved.
    """
    plots = ['auc', 'confusion_matrix', 'learning', 'calibration', 'pr']
    for plot in plots:
        exp.plot_model(best_model, plot=plot, save=True)
        print(f"{plot.capitalize()} plot saved")
    
    # Optionally plot feature importance if supported
    try:
        exp.plot_model(best_model, plot='feature', save=True)
        print("Feature importance plot saved")
    except Exception as e:
        print("Feature importance plot not supported:", e)

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
    print("Evaluating model on test set...")
    predictions = exp.predict_model(best_model)
    predictions_file = os.path.join(output_dir, "test_predictions.csv")
    predictions.to_csv(predictions_file, index=False)
    print(f"Predictions saved to {predictions_file}")
    return predictions

def generate_classification_report(predictions):
    """
    Generate a text classification report from predictions.
    
    Parameters
    ----------
    predictions : pandas.DataFrame
        DataFrame containing true and predicted values.
        
    Returns
    -------
    str
        Classification report as a string.
    """
    print("Generating classification report...")
    pred_col = 'prediction_label' if 'prediction_label' in predictions.columns else 'Label'
    report = classification_report(predictions['HBsAg'], predictions[pred_col])
    return report

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
    print("Refining model with Gradient Boosting...")
    gbc = exp.create_model('gbc', fold=10)
    tuned_gbc = exp.tune_model(gbc)
    bagged_gbc = exp.ensemble_model(tuned_gbc)
    
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    best_model_path = os.path.join(model_dir, "best_model_pipeline")
    exp.save_model(best_model, best_model_path)
    print(f"Best model saved to {best_model_path}")
    
    refined_model_path = os.path.join(model_dir, "refined_model_pipeline")
    exp.save_model(bagged_gbc, refined_model_path)
    print(f"Refined model saved to {refined_model_path}")
