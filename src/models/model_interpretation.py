"""
Model Interpretation Module

This module provides functions for generating model interpretability plots
(e.g., SHAP or LIME) and analyzing feature importance.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd

def interpret_model(exp, model, output_dir, method='shap'):
    """
    Generate interpretability plots using the specified method.
    
    Parameters
    ----------
    exp : ClassificationExperiment
        PyCaret experiment.
    model : object
        Model to interpret.
    output_dir : str
        Directory where plots will be saved.
    method : str, optional
        Interpretation method ('shap' or 'lime'). Default is 'shap'.
    """
    try:
        if method == 'shap':
            exp.interpret_model(model, plot='summary', save=True)
            print("SHAP summary plot saved")
        elif method == 'lime':
            test_predictions = exp.predict_model(model)
            # Select a sample observation for LIME explanation
            observation_idx = test_predictions[test_predictions['HBsAg'] == 1.0].index[0]
            exp.interpret_model(model, plot='lime', observation=observation_idx, save=True)
            print("LIME explanation plot saved")
    except Exception as e:
        print(f"{method.capitalize()} interpretation plot could not be generated: {e}")
        if method == 'lime':
            try:
                exp.plot_model(model, plot='pd', save=True)
                print("Partial dependence plot saved as alternative to LIME")
            except Exception as e2:
                print(f"Partial dependence plot also failed: {e2}")

def analyze_feature_importance(exp, best_model, output_dir, plots_dir):
    """
    Analyze and save feature importance using PyCaret or an alternative approach.
    
    Parameters
    ----------
    exp : ClassificationExperiment
        PyCaret experiment.
    best_model : object
        The best performing model.
    output_dir : str
        Directory to save CSV results.
    plots_dir : str
        Directory to save feature importance plots.
    """
    try:
        importance = exp.get_feature_importance(best_model)
        print("\nFeature Importance:")
        print(importance)
        importance_file = os.path.join(output_dir, "feature_importance.csv")
        importance.to_csv(importance_file, index=False)
        print(f"Feature importance saved to {importance_file}")
        
        # Create a custom plot for top 15 features
        plt.figure(figsize=(10, 8))
        plt.barh(importance['Feature'][:15], importance['Value'][:15])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Top 15 Features by Importance')
        plt.tight_layout()
        importance_plot_path = os.path.join(plots_dir, "custom_feature_importance.png")
        plt.savefig(importance_plot_path)
        plt.close()
        print(f"Custom feature importance plot saved to {importance_plot_path}")
    except Exception as e:
        print(f"Error getting feature importance: {e}")
        print("Trying alternative approach using model's feature_importances_...")
        try:
            if hasattr(best_model, 'feature_importances_'):
                feature_names = exp.get_config('X_train').columns.tolist()
                fi_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Value': best_model.feature_importances_
                }).sort_values(by='Value', ascending=False)
                print("\nAlternative Feature Importance:")
                print(fi_df)
                alt_importance_file = os.path.join(output_dir, "feature_importance_alt.csv")
                fi_df.to_csv(alt_importance_file, index=False)
                print(f"Alternative feature importance saved to {alt_importance_file}")
                
                plt.figure(figsize=(10, 8))
                plt.barh(fi_df['Feature'][:15], fi_df['Value'][:15])
                plt.xlabel('Importance')
                plt.ylabel('Feature')
                plt.title('Top 15 Features by Importance (Alternative)')
                plt.tight_layout()
                alt_importance_plot_path = os.path.join(plots_dir, "alt_feature_importance.png")
                plt.savefig(alt_importance_plot_path)
                plt.close()
                print(f"Alternative feature importance plot saved to {alt_importance_plot_path}")
            else:
                print("Model does not have feature_importances_ attribute.")
        except Exception as e2:
            print(f"Alternative feature importance approach also failed: {e2}")
