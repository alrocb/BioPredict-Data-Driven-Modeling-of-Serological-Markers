"""
Deep Model Analysis Script

This script provides comprehensive analysis of saved models including:
1. Model structure and parameters
2. Individual tree visualization from Gradient Boosting
3. Feature importance analysis
4. Decision path analysis
5. Model interpretation and visualization
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib

# Tree visualization imports
from sklearn.tree import export_graphviz, export_text, plot_tree
from sklearn.ensemble import GradientBoostingClassifier
import graphviz
from IPython.display import Image, display

# PyCaret imports
from pycaret.classification import ClassificationExperiment, load_model

# Additional analysis imports
import shap
from sklearn.inspection import permutation_importance
from sklearn.model_selection import validation_curve

# Logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelDeepAnalyzer:
    """
    Comprehensive model analyzer for Gradient Boosting models
    """
    
    def __init__(self, model_path, data_path=None):
        """
        Initialize the analyzer
        
        Parameters
        ----------
        model_path : str
            Path to the saved model (PyCaret pipeline)
        data_path : str, optional
            Path to the data used for training
        """
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.data = None
        self.feature_names = None
        self.target_name = None
        
    def load_model_and_data(self):
        """Load the saved model and associated data"""
        try:
            # Load PyCaret model
            logger.info(f"Loading PyCaret model from {self.model_path}")
            self.model = load_model(self.model_path)
            logger.info(f"Model loaded successfully: {type(self.model)}")
            
            # Extract the actual estimator from PyCaret pipeline
            if hasattr(self.model, 'named_steps'):
                # It's a pipeline, get the final estimator
                estimator_names = list(self.model.named_steps.keys())
                final_estimator_name = estimator_names[-1]
                self.estimator = self.model.named_steps[final_estimator_name]
                logger.info(f"Extracted estimator: {type(self.estimator)}")
            else:
                self.estimator = self.model
                
            # Load data if provided
            if self.data_path and os.path.exists(self.data_path):
                logger.info(f"Loading data from {self.data_path}")
                self.data = pd.read_csv(self.data_path)
                self.feature_names = [col for col in self.data.columns if col != 'Hepatitis_B']
                self.target_name = 'Hepatitis_B'
                logger.info(f"Data loaded with shape: {self.data.shape}")
                logger.info(f"Features: {self.feature_names}")
            
        except Exception as e:
            logger.error(f"Error loading model or data: {e}")
            raise
            
    def analyze_model_structure(self):
        """Analyze and display model structure and parameters"""
        print("="*80)
        print("MODEL STRUCTURE ANALYSIS")
        print("="*80)
        
        if hasattr(self.model, 'named_steps'):
            print("\n🔍 PIPELINE STRUCTURE:")
            for step_name, step_obj in self.model.named_steps.items():
                print(f"  └── {step_name}: {type(step_obj).__name__}")
                
        print(f"\n🎯 FINAL ESTIMATOR: {type(self.estimator).__name__}")
        
        if isinstance(self.estimator, GradientBoostingClassifier):
            params = self.estimator.get_params()
            print(f"\n📊 GRADIENT BOOSTING PARAMETERS:")
            key_params = [
                'learning_rate', 'n_estimators', 'max_depth', 'min_samples_split',
                'min_samples_leaf', 'subsample', 'max_features', 'criterion'
            ]
            
            for param in key_params:
                if param in params:
                    print(f"  ├── {param}: {params[param]}")
                    
            print(f"\n🌳 TREE ENSEMBLE INFORMATION:")
            print(f"  ├── Number of trees: {self.estimator.n_estimators}")
            print(f"  ├── Number of features: {self.estimator.n_features_in_}")
            if hasattr(self.estimator, 'feature_names_in_'):
                print(f"  ├── Feature names available: Yes")
            else:
                print(f"  ├── Feature names available: No")
                
            # Training information
            if hasattr(self.estimator, 'train_score_'):
                print(f"  ├── Training scores available: Yes ({len(self.estimator.train_score_)} iterations)")
                print(f"  └── Final training score: {self.estimator.train_score_[-1]:.4f}")
                
    def visualize_training_progress(self, save_dir=None):
        """Visualize training progress"""
        if not isinstance(self.estimator, GradientBoostingClassifier):
            print("Training progress visualization only available for GradientBoostingClassifier")
            return
            
        if not hasattr(self.estimator, 'train_score_'):
            print("Training scores not available in this model")
            return
        print("\n📈 TRAINING PROGRESS VISUALIZATION")
        print("-"*50)
        
        # Create larger figure with better spacing
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Training score over iterations
        iterations = range(1, len(self.estimator.train_score_) + 1)
        ax1.plot(iterations, self.estimator.train_score_, 'b-', linewidth=3, label='Training Score')
        ax1.set_xlabel('Boosting Iterations', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Training Score', fontsize=12, fontweight='bold')
        ax1.set_title('Training Score vs Iterations', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)
        ax1.tick_params(labelsize=10)
        
        # Feature importance with better formatting
        if hasattr(self.estimator, 'feature_importances_'):
            importances = self.estimator.feature_importances_
            if self.feature_names:
                feature_names = [name[:15] + '...' if len(name) > 15 else name for name in self.feature_names]
            else:
                feature_names = [f'Feature_{i}' for i in range(len(importances))]
                  # Sort features by importance - show top 12 for better visibility
            indices = np.argsort(importances)[::-1][:12]  # Top 12
            
            bars = ax2.barh(range(len(indices)), importances[indices], color='steelblue', alpha=0.7)
            ax2.set_yticks(range(len(indices)))
            ax2.set_yticklabels([feature_names[i] for i in indices], fontsize=10)
            ax2.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
            ax2.set_title('Top 12 Feature Importances', fontsize=14, fontweight='bold')
            ax2.invert_yaxis()
            ax2.grid(True, alpha=0.3, axis='x')
            ax2.tick_params(labelsize=10)
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax2.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha='left', va='center', fontsize=9)
            
        # Better layout with more spacing
        plt.tight_layout(pad=3.0)
        
        if save_dir:
            save_path = os.path.join(save_dir, 'model_training_analysis.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Training analysis saved to: {save_path}")
            
        plt.show()
        
    def visualize_individual_trees(self, tree_indices=[0, 1, 2], save_dir=None):
        """Visualize individual trees from the Gradient Boosting ensemble with improved layout"""
        if not isinstance(self.estimator, GradientBoostingClassifier):
            print("Tree visualization only available for GradientBoostingClassifier")
            return
            
        print(f"\n🌳 INDIVIDUAL TREE VISUALIZATION")
        print("-"*50)
        
        # Create separate figures for each tree to avoid overlap
        for tree_idx in tree_indices:
            if tree_idx >= self.estimator.n_estimators:
                print(f"Tree index {tree_idx} exceeds number of estimators ({self.estimator.n_estimators})")
                continue
                
            # Create individual figure for each tree
            fig, ax = plt.subplots(1, 1, figsize=(20, 16))
            
            # Get the specific tree
            tree = self.estimator.estimators_[tree_idx, 0]  # [tree_idx, class_idx]
            
            # Feature names (truncate long names)
            if self.feature_names:
                feature_names = [name[:15] + '...' if len(name) > 15 else name for name in self.feature_names]
            else:
                feature_names = [f'F_{i}' for i in range(self.estimator.n_features_in_)]
                
            # Plot tree with better formatting
            plot_tree(tree, 
                     ax=ax,
                     feature_names=feature_names,
                     class_names=['Negative', 'Positive'],
                     filled=True,
                     rounded=True,
                     fontsize=12,  # Larger font
                     max_depth=3,  # Limit depth for readability
                     impurity=False,  # Remove impurity to reduce clutter
                     proportion=True)  # Show proportions instead of raw counts
            
            ax.set_title(f'Tree {tree_idx} (Boosting Iteration {tree_idx})', 
                        fontsize=18, fontweight='bold', pad=20)
            
            # Adjust layout with more spacing
            plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9)
            
            if save_dir:
                save_path = os.path.join(save_dir, f'tree_{tree_idx}_visualization.png')
                plt.savefig(save_path, dpi=200, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                print(f"Tree {tree_idx} visualization saved to: {save_path}")
                
            plt.show()
            plt.close()  # Close figure to free memory
        
    def export_tree_to_text(self, tree_idx=0, max_depth=5):
        """Export tree structure to text format"""
        if not isinstance(self.estimator, GradientBoostingClassifier):
            print("Tree export only available for GradientBoostingClassifier")
            return
            
        if tree_idx >= self.estimator.n_estimators:
            print(f"Tree index {tree_idx} exceeds number of estimators ({self.estimator.n_estimators})")
            return
            
        print(f"\n📝 TREE {tree_idx} TEXT REPRESENTATION")
        print("-"*50)
        
        tree = self.estimator.estimators_[tree_idx, 0]
        
        if self.feature_names:
            feature_names = self.feature_names
        else:
            feature_names = [f'Feature_{i}' for i in range(self.estimator.n_features_in_)]
            
        tree_text = export_text(tree, 
                               feature_names=feature_names,
                               max_depth=max_depth,
                               spacing=3,
                               decimals=3,
                               show_weights=True)
        
        print(tree_text)
        return tree_text
        
    def create_simplified_decision_tree(self, save_dir=None):
        """Create a simplified single decision tree to approximate the GBM"""
        if self.data is None:
            print("Data required for simplified tree creation")
            return
            
        print(f"\n🌲 SIMPLIFIED DECISION TREE APPROXIMATION")
        print("-"*50)
        
        from sklearn.tree import DecisionTreeClassifier
        
        # Prepare data
        X = self.data[self.feature_names]
        y = self.data[self.target_name]
        
        # Create simplified tree with limited depth
        simple_tree = DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=50,
            min_samples_leaf=20,
            random_state=42
        )
        
        simple_tree.fit(X, y)
        
        # Calculate accuracy comparison
        gbm_predictions = self.model.predict(self.data.drop(columns=[self.target_name]))
        tree_predictions = simple_tree.predict(X)
        from sklearn.metrics import accuracy_score, classification_report
        
        gbm_accuracy = accuracy_score(y, gbm_predictions)
        tree_accuracy = accuracy_score(y, tree_predictions)
        
        print(f"Original Model Accuracy: {gbm_accuracy:.4f}")
        print(f"Simplified Tree Accuracy: {tree_accuracy:.4f}")
        print(f"Accuracy Loss: {gbm_accuracy - tree_accuracy:.4f}")
        
        # Visualize simplified tree with improved layout
        fig, ax = plt.subplots(1, 1, figsize=(28, 20))  # Increased size for better spacing
        
        # Truncate feature names more aggressively for better readability
        truncated_feature_names = [name[:12] + '...' if len(name) > 12 else name for name in self.feature_names]
        
        plot_tree(simple_tree,
                 ax=ax,
                 feature_names=truncated_feature_names,
                 class_names=['Negative', 'Positive'],
                 filled=True,
                 rounded=True,
                 fontsize=9,  # Smaller font to fit better
                 impurity=False,  # Remove impurity to reduce clutter
                 proportion=True,  # Show proportions instead of raw counts
                 precision=2,  # Limit decimal places
                 max_depth=4)  # Limit depth to reduce overlapping
        
        # Set title with better spacing
        ax.set_title(f'Simplified Decision Tree Approximation\n'
                    f'(Accuracy: {tree_accuracy:.4f}, Original Model: {gbm_accuracy:.4f})',
                    fontsize=18, fontweight='bold', pad=40)
        
        # Remove axis for cleaner look
        ax.axis('off')
        
        # Adjust layout with optimal spacing
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, bottom=0.05, left=0.02, right=0.98)
        
        if save_dir:
            save_path = os.path.join(save_dir, 'simplified_decision_tree.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none',
                       pad_inches=0.5)  # Add padding around the figure
            print(f"Simplified tree saved to: {save_path}")
            
        plt.show()
        
        return simple_tree
        
    def create_improved_tree_layout(self, save_dir=None):
        """Create a tree with custom layout to avoid overlapping"""
        if self.data is None:
            print("Data required for improved tree creation")
            return
            
        print(f"\n🎨 IMPROVED TREE LAYOUT")
        print("-"*50)
        
        from sklearn.tree import DecisionTreeClassifier
        import matplotlib.patches as patches
        
        # Prepare data
        X = self.data[self.feature_names]
        y = self.data[self.target_name]
        
        # Create even simpler tree to avoid overlapping
        simple_tree = DecisionTreeClassifier(
            max_depth=3,  # Reduced depth for better visibility
            min_samples_split=100,  # More restrictive to create simpler tree
            min_samples_leaf=50,
            random_state=42
        )
        
        simple_tree.fit(X, y)
        
        # Calculate accuracy
        gbm_predictions = self.model.predict(self.data.drop(columns=[self.target_name]))
        tree_predictions = simple_tree.predict(X)
        
        from sklearn.metrics import accuracy_score
        gbm_accuracy = accuracy_score(y, gbm_predictions)
        tree_accuracy = accuracy_score(y, tree_predictions)
        
        # Create custom visualization with better spacing
        fig, ax = plt.subplots(1, 1, figsize=(24, 16))
        
        # Get tree structure
        tree = simple_tree.tree_
        
        def draw_node(x, y, width, height, text, color, text_color='black'):
            """Draw a node with custom styling"""
            rect = patches.FancyBboxPatch(
                (x - width/2, y - height/2), width, height,
                boxstyle="round,pad=0.02",
                facecolor=color,
                edgecolor='black',
                linewidth=1.5
            )
            ax.add_patch(rect)
            
            # Add text with word wrapping
            lines = text.split('\n')
            line_height = height / (len(lines) + 1)
            for i, line in enumerate(lines):
                ax.text(x, y + height/2 - (i+1)*line_height, line,
                       ha='center', va='center', fontsize=10, 
                       fontweight='bold', color=text_color)
        
        def draw_edge(x1, y1, x2, y2, label=""):
            """Draw edge between nodes"""
            ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2, alpha=0.7)
            if label:
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                ax.text(mid_x, mid_y, label, ha='center', va='center',
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        # Define positions for nodes (manual layout for perfect spacing)
        positions = {}
        node_width = 3.5
        node_height = 1.2
        level_height = 3.5
        
        # Level 0 (root)
        positions[0] = (12, 12)
        
        # Level 1
        positions[tree.children_left[0]] = (6, 12 - level_height)
        positions[tree.children_right[0]] = (18, 12 - level_height)
        
        # Level 2
        if tree.children_left[tree.children_left[0]] != -1:
            positions[tree.children_left[tree.children_left[0]]] = (3, 12 - 2*level_height)
        if tree.children_right[tree.children_left[0]] != -1:
            positions[tree.children_right[tree.children_left[0]]] = (9, 12 - 2*level_height)
        if tree.children_left[tree.children_right[0]] != -1:
            positions[tree.children_left[tree.children_right[0]]] = (15, 12 - 2*level_height)
        if tree.children_right[tree.children_right[0]] != -1:
            positions[tree.children_right[tree.children_right[0]]] = (21, 12 - 2*level_height)
        
        # Draw nodes
        def draw_tree_recursive(node_id, depth=0):
            if node_id not in positions:
                return
                
            x, y = positions[node_id]
            
            if tree.children_left[node_id] == tree.children_right[node_id]:
                # Leaf node
                samples = tree.n_node_samples[node_id]
                value = tree.value[node_id][0]
                class_pred = "Positive" if value[1] > value[0] else "Negative"
                confidence = max(value) / sum(value)
                
                text = f"{class_pred}\nConfidence: {confidence:.2f}\nSamples: {samples}"
                color = '#90EE90' if class_pred == "Positive" else '#FFB6C1'
                
                draw_node(x, y, node_width, node_height, text, color)
            else:
                # Internal node
                feature_name = self.feature_names[tree.feature[node_id]]
                threshold = tree.threshold[node_id]
                samples = tree.n_node_samples[node_id]
                
                # Truncate long feature names
                if len(feature_name) > 15:
                    feature_name = feature_name[:12] + "..."
                
                text = f"{feature_name}\n≤ {threshold:.2f}\nSamples: {samples}"
                color = '#ADD8E6'
                
                draw_node(x, y, node_width, node_height, text, color)
                
                # Draw edges to children
                left_child = tree.children_left[node_id]
                right_child = tree.children_right[node_id]
                
                if left_child in positions:
                    left_x, left_y = positions[left_child]
                    draw_edge(x, y - node_height/2, left_x, left_y + node_height/2, "Yes")
                    draw_tree_recursive(left_child, depth + 1)
                
                if right_child in positions:
                    right_x, right_y = positions[right_child]
                    draw_edge(x, y - node_height/2, right_x, right_y + node_height/2, "No")
                    draw_tree_recursive(right_child, depth + 1)
        
        # Draw the tree
        draw_tree_recursive(0)
        
        # Set plot properties
        ax.set_xlim(0, 24)
        ax.set_ylim(0, 15)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Title
        plt.suptitle(f'Improved Decision Tree Layout\n'
                    f'Accuracy: {tree_accuracy:.4f} | Original Model: {gbm_accuracy:.4f}',
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Legend
        legend_elements = [
            patches.Patch(color='#ADD8E6', label='Decision Node'),
            patches.Patch(color='#90EE90', label='Positive Prediction'),
            patches.Patch(color='#FFB6C1', label='Negative Prediction')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        
        if save_dir:
            save_path = os.path.join(save_dir, 'improved_tree_layout.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"Improved tree layout saved to: {save_path}")
            
        plt.show()
        
        return simple_tree
        
    def create_interpretable_binary_tree(self, save_dir=None):
        """Create a clean, interpretable binary tree visualization similar to medical decision trees"""
        if self.data is None:
            print("Data required for interpretable tree creation")
            return
            
        print(f"\n🌳 INTERPRETABLE BINARY TREE")
        print("-"*50)
        
        from sklearn.tree import DecisionTreeClassifier
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
        
        # Prepare data
        X = self.data[self.feature_names]
        y = self.data[self.target_name]
        
        # Create a simple, interpretable tree
        interpretable_tree = DecisionTreeClassifier(
            max_depth=4,  # Moderate depth for interpretability
            min_samples_split=50,  # Ensure statistical significance
            min_samples_leaf=25,   # Ensure meaningful leaf nodes
            random_state=42,
            criterion='gini'
        )
        
        interpretable_tree.fit(X, y)
        
        # Calculate accuracy comparison
        gbm_predictions = self.model.predict(self.data.drop(columns=[self.target_name]))
        tree_predictions = interpretable_tree.predict(X)
        
        from sklearn.metrics import accuracy_score, classification_report
        gbm_accuracy = accuracy_score(y, gbm_predictions)
        tree_accuracy = accuracy_score(y, tree_predictions)
        
        print(f"Original Model Accuracy: {gbm_accuracy:.4f}")
        print(f"Interpretable Tree Accuracy: {tree_accuracy:.4f}")
        print(f"Interpretability vs Accuracy Trade-off: {gbm_accuracy - tree_accuracy:.4f}")
        
        # Create the binary tree visualization
        fig, ax = plt.subplots(1, 1, figsize=(20, 14))
        
        # Get tree structure
        tree = interpretable_tree.tree_
        
        # Define visual parameters
        node_width = 2.8
        node_height = 1.0
        level_spacing = 2.5
        
        # Color scheme for medical interpretation
        decision_color = '#4A90E2'  # Blue for decision nodes
        positive_color = '#E74C3C'  # Red for positive (mutation/disease)
        negative_color = '#27AE60'  # Green for negative (no mutation/healthy)
        
        def get_node_info(node_id):
            """Extract information for a tree node"""
            samples = tree.n_node_samples[node_id]
            value = tree.value[node_id][0]
            
            if tree.children_left[node_id] == tree.children_right[node_id]:
                # Leaf node
                class_pred = "YES" if value[1] > value[0] else "NO"
                confidence = max(value) / sum(value) * 100
                positive_samples = int(value[1])
                negative_samples = int(value[0])
                
                return {
                    'type': 'leaf',
                    'prediction': class_pred,
                    'confidence': confidence,
                    'samples': samples,
                    'positive': positive_samples,
                    'negative': negative_samples
                }
            else:
                # Decision node
                feature_idx = tree.feature[node_id]
                threshold = tree.threshold[node_id]
                feature_name = self.feature_names[feature_idx]
                
                return {
                    'type': 'decision',
                    'feature': feature_name,
                    'threshold': threshold,
                    'samples': samples
                }
        
        def draw_medical_node(x, y, node_info, node_id):
            """Draw a node in medical decision tree style"""
            if node_info['type'] == 'leaf':
                # Leaf node - prediction box
                color = positive_color if node_info['prediction'] == "YES" else negative_color
                
                # Main prediction box
                rect = patches.FancyBboxPatch(
                    (x - node_width/2, y - node_height/2), 
                    node_width, node_height,
                    boxstyle="round,pad=0.1",
                    facecolor=color,
                    edgecolor='black',
                    linewidth=2,
                    alpha=0.8
                )
                ax.add_patch(rect)
                
                # Prediction text
                pred_text = f"{node_info['prediction']} ({node_info['samples']})"
                ax.text(x, y + 0.1, pred_text, 
                       ha='center', va='center', 
                       fontsize=12, fontweight='bold', color='white')
                
                # Details text
                details = f"Hepatitis B {node_info['positive']}\nNo Hepatitis B {node_info['negative']}"
                ax.text(x, y - 0.2, details,
                       ha='center', va='center',
                       fontsize=9, color='white')
                
            else:
                # Decision node - condition box
                rect = patches.FancyBboxPatch(
                    (x - node_width/2, y - node_height/2),
                    node_width, node_height,
                    boxstyle="round,pad=0.1",
                    facecolor=decision_color,
                    edgecolor='black',
                    linewidth=2,
                    alpha=0.9
                )
                ax.add_patch(rect)
                
                # Feature name (truncated if too long)
                feature_display = node_info['feature']
                if len(feature_display) > 12:
                    feature_display = feature_display[:12] + "..."
                
                # Decision text
                if node_info['threshold'] == int(node_info['threshold']):
                    threshold_text = f"> {int(node_info['threshold'])}"
                else:
                    threshold_text = f"> {node_info['threshold']:.1f}"
                    
                condition_text = f"{feature_display}\n{threshold_text}"
                ax.text(x, y, condition_text,
                       ha='center', va='center',
                       fontsize=11, fontweight='bold', color='white')
        
        def draw_connection(x1, y1, x2, y2, label, is_yes=True):
            """Draw connection between nodes with labels"""
            # Connection line
            ax.plot([x1, x2], [y1 - node_height/2, y2 + node_height/2], 
                   'k-', linewidth=2, alpha=0.7)
            
            # Label background
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            label_color = '#2ECC71' if is_yes else '#E67E22'  # Green for YES, Orange for NO
            
            ax.text(mid_x, mid_y, label,
                   ha='center', va='center',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", 
                           facecolor=label_color, 
                           edgecolor='black',
                           alpha=0.9),
                   color='white')
        
        # Calculate node positions using breadth-first traversal
        positions = {}
        queue = [(0, 10, 0)]  # (node_id, x, y)
        level_width = {0: 20}  # Track width needed for each level
        
        # First pass: calculate positions
        while queue:
            node_id, x, y = queue.pop(0)
            positions[node_id] = (x, y)
            
            left_child = tree.children_left[node_id]
            right_child = tree.children_right[node_id]
            
            if left_child != -1:  # Has children
                child_y = y - level_spacing
                child_spacing = level_width.get(y, 10) / 4  # Adaptive spacing
                
                left_x = x - child_spacing
                right_x = x + child_spacing
                
                queue.append((left_child, left_x, child_y))
                queue.append((right_child, right_x, child_y))
                
                level_width[child_y] = max(level_width.get(child_y, 0), abs(right_x - left_x) + 4)
        
        # Second pass: draw the tree
        def draw_tree_recursive(node_id):
            if node_id not in positions:
                return
                
            x, y = positions[node_id]
            node_info = get_node_info(node_id)
            
            # Draw current node
            draw_medical_node(x, y, node_info, node_id)
            
            # Draw children and connections
            left_child = tree.children_left[node_id]
            right_child = tree.children_right[node_id]
            
            if left_child != -1:  # Has children
                left_x, left_y = positions[left_child]
                right_x, right_y = positions[right_child]
                
                # Draw connections with labels
                draw_connection(x, y, left_x, left_y, "NO", False)
                draw_connection(x, y, right_x, right_y, "YES", True)
                
                # Recursively draw children
                draw_tree_recursive(left_child)
                draw_tree_recursive(right_child)
        
        # Draw the complete tree
        draw_tree_recursive(0)
        
        # Set plot properties
        ax.set_xlim(-2, 22)
        ax.set_ylim(-12, 12)
        ax.set_aspect('equal')
        ax.axis('off')
          # Add title and legend
        plt.suptitle(f'Interpretable Binary Decision Tree for Hepatitis B Prediction\n'
                    f'Accuracy: {tree_accuracy:.3f} | Original Model: {gbm_accuracy:.3f} | Samples: {len(X)}',
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Create legend
        legend_elements = [
            patches.Patch(color=decision_color, label='Decision Node'),
            patches.Patch(color=positive_color, label='Hepatitis B Positive'),
            patches.Patch(color=negative_color, label='Hepatitis B Negative'),
            patches.Patch(color='#2ECC71', label='YES Branch'),
            patches.Patch(color='#E67E22', label='NO Branch')
        ]
        ax.legend(handles=legend_elements, loc='upper right', 
                 bbox_to_anchor=(1.0, 0.98), fontsize=11)
        
        # Add interpretation guide
        guide_text = ("Reading the Tree:\n"
                     "• Start at the top (root) node\n"
                     "• Follow YES/NO branches based on conditions\n"
                     "• End at prediction (YES = Hepatitis B, NO = Healthy)\n"
                     "• Numbers show sample counts at each node")
        
        ax.text(-1.5, -10, guide_text,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if save_dir:
            save_path = os.path.join(save_dir, 'interpretable_binary_tree.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"Interpretable binary tree saved to: {save_path}")
            
        plt.show()
        
        # Print tree rules in text format
        self.print_tree_rules(interpretable_tree)
        
        return interpretable_tree
    
    def print_tree_rules(self, tree_model):
        """Print human-readable decision rules from the tree"""
        print(f"\n📋 DECISION RULES (Human Readable)")
        print("-"*50)
        
        tree = tree_model.tree_
        
        def get_rules(node_id, depth=0, condition=""):
            """Recursively extract rules from tree"""
            indent = "  " * depth
            
            if tree.children_left[node_id] == tree.children_right[node_id]:
                # Leaf node
                samples = tree.n_node_samples[node_id]
                value = tree.value[node_id][0]
                prediction = "Hepatitis B POSITIVE" if value[1] > value[0] else "Hepatitis B NEGATIVE"
                confidence = max(value) / sum(value) * 100
                
                print(f"{indent}🎯 PREDICTION: {prediction}")
                print(f"{indent}   Confidence: {confidence:.1f}%")
                print(f"{indent}   Samples: {samples}")
                print(f"{indent}   Condition: {condition}")
                print()
            else:
                # Decision node
                feature_name = self.feature_names[tree.feature[node_id]]
                threshold = tree.threshold[node_id]
                samples = tree.n_node_samples[node_id]
                
                print(f"{indent}🔍 Decision Point: {feature_name}")
                print(f"{indent}   Threshold: {threshold:.2f}")
                print(f"{indent}   Samples: {samples}")
                print()
                
                # Left child (condition not met)
                left_condition = f"{condition} AND {feature_name} ≤ {threshold:.2f}" if condition else f"{feature_name} ≤ {threshold:.2f}"
                print(f"{indent}├─ NO Branch:")
                get_rules(tree.children_left[node_id], depth + 1, left_condition)
                
                # Right child (condition met)  
                right_condition = f"{condition} AND {feature_name} > {threshold:.2f}" if condition else f"{feature_name} > {threshold:.2f}"
                print(f"{indent}└─ YES Branch:")
                get_rules(tree.children_right[node_id], depth + 1, right_condition)
        
        get_rules(0)

    def analyze_decision_paths(self, sample_indices=[0, 1, 2]):
        """Analyze decision paths for specific samples"""
        if self.data is None:
            print("Data required for decision path analysis")
            return
            
        print(f"\n🛤️ DECISION PATH ANALYSIS")
        print("-"*50)
        
        X = self.data[self.feature_names]
        
        for sample_idx in sample_indices:
            if sample_idx >= len(self.data):
                continue
                
            sample = X.iloc[sample_idx:sample_idx+1]
            actual_label = self.data[self.target_name].iloc[sample_idx]
            predicted_proba = self.model.predict_proba(sample)[0]
            predicted_label = self.model.predict(sample)[0]
            
            print(f"\n📊 SAMPLE {sample_idx}:")
            print(f"  ├── Actual Label: {actual_label}")
            print(f"  ├── Predicted Label: {predicted_label}")
            print(f"  ├── Predicted Probability: {predicted_proba}")
            print(f"  └── Feature Values:")
            
            for feature, value in sample.iloc[0].items():
                print(f"      ├── {feature}: {value}")
                
    def comprehensive_analysis(self, save_dir=None):
        """Run comprehensive model analysis"""
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        print("🔬 STARTING COMPREHENSIVE MODEL ANALYSIS")
        print("="*80)
        
        # 1. Model structure
        self.analyze_model_structure()
        
        # 2. Training progress
        self.visualize_training_progress(save_dir)
        
        # 3. Individual trees
        self.visualize_individual_trees([0, 1, 2, -1], save_dir)  # First 3 and last tree
        
        # 4. Text representation
        self.export_tree_to_text(tree_idx=0)
          # 5. Simplified tree (original)
        if self.data is not None:
            simplified_tree = self.create_simplified_decision_tree(save_dir)
              # 6. Improved tree layout (new)
        if self.data is not None:
            improved_tree = self.create_improved_tree_layout(save_dir)
            
        # # 7. Interpretable binary tree (medical-style)
        # if self.data is not None:
        #     interpretable_tree = self.create_interpretable_binary_tree(save_dir)
            
        # 8. Decision paths
        if self.data is not None:
            self.analyze_decision_paths([0, 10, 50])
            
        print("\n✅ COMPREHENSIVE ANALYSIS COMPLETE")


def main():
    """Main function to run the analysis"""
    # Configuration
    project_root = r"c:\Users\bntmm\Desktop\GRIFOLS\BioPredict-Data-Driven-Modeling-of-Serological-Markers"
    
    # Find the most recent run directory
    outputs_dir = os.path.join(project_root, "outputs")
    run_dirs = [d for d in os.listdir(outputs_dir) if d.startswith("run_")]
    if not run_dirs:
        print("No run directories found!")
        return
        
    latest_run = sorted(run_dirs)[-1]

    print(f"Analyzing latest run: {latest_run}")
    
    # Paths
    model_path = os.path.join(outputs_dir, latest_run, "models", "best_model")
    data_path = os.path.join(project_root, "data", "processed", "mapped_target_data.csv")
    analysis_save_dir = os.path.join(outputs_dir, latest_run, "deep_analysis")
    
    # Check if model exists
    if not os.path.exists(model_path + ".pkl"):
        print(f"Model not found at {model_path}")
        return
        
    # Initialize analyzer
    analyzer = ModelDeepAnalyzer(model_path, data_path)
    
    try:
        # Load model and data
        analyzer.load_model_and_data()
        
        # Run comprehensive analysis
        analyzer.comprehensive_analysis(analysis_save_dir)
        
        print(f"\n📁 Analysis results saved to: {analysis_save_dir}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
