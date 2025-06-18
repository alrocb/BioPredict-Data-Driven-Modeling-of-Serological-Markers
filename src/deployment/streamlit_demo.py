"""
Streamlit Demo Application for Hepatitis B Prediction Models

This application provides an interactive interface to test two models:
1. Demographics Model - uses only basic demographic variables
2. Full Model - uses all available variables including clinical and lifestyle factors

The models predict the likelihood of Hepatitis B Surface Antigen (HBsAg) positivity.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
from pathlib import Path

# Add the src directory to the Python path
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.append(str(src_dir))

# Import project utilities
from utils.config_loader import load_config
import logging
from sklearn.preprocessing import StandardScaler

# Configure logging for streamlit
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="BioPredict - Hepatitis B Prediction Demo",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration
@st.cache_data
def load_app_config():
    """Load the project configuration."""
    try:
        config = load_config()
        return config
    except Exception as e:
        st.error(f"Error loading configuration: {e}")
        return None

# Load models
@st.cache_resource
def load_models():
    """Load the trained XGBoost models and their corresponding scalers."""
    models = {}
    model_dir = Path(__file__).parent
    
    # Demographics model files
    demo_model_path = model_dir / "hepatitis_b_xgboost_model_demo.pkl"
    demo_scaler_path = model_dir / "hepatitis_b_scaler_demo.pkl"
    
    # Full model files
    full_model_path = model_dir / "hepatitis_b_xgboost_model.pkl"
    full_scaler_path = model_dir / "hepatitis_b_scaler.pkl"
    
    try:        # Load demographics model and scaler
        if demo_model_path.exists() and demo_scaler_path.exists():
            with open(demo_model_path, 'rb') as f:
                demo_model = pickle.load(f)
            with open(demo_scaler_path, 'rb') as f:
                demo_scaler = pickle.load(f)
            
            # Debug: Check what type the scaler is
            logger.info(f"Demo scaler type: {type(demo_scaler)}")
            if hasattr(demo_scaler, 'transform'):
                logger.info("Demo scaler has transform method")
            elif isinstance(demo_scaler, np.ndarray):
                logger.info(f"Demo scaler is numpy array with shape: {demo_scaler.shape}")
            
            models['demo'] = {'model': demo_model, 'scaler': demo_scaler}
            st.success("Demographics model and scaler loaded successfully")
        else:
            missing_files = []
            if not demo_model_path.exists():
                missing_files.append(str(demo_model_path))
            if not demo_scaler_path.exists():
                missing_files.append(str(demo_scaler_path))
            st.error(f"Demographics model files not found: {', '.join(missing_files)}")
              # Load full model and scaler
        if full_model_path.exists() and full_scaler_path.exists():
            with open(full_model_path, 'rb') as f:
                full_model = pickle.load(f)
            with open(full_scaler_path, 'rb') as f:
                full_scaler = pickle.load(f)
            
            # Debug: Check what type the scaler is
            logger.info(f"Full scaler type: {type(full_scaler)}")
            if hasattr(full_scaler, 'transform'):
                logger.info("Full scaler has transform method")
            elif isinstance(full_scaler, np.ndarray):
                logger.info(f"Full scaler is numpy array with shape: {full_scaler.shape}")
            
            models['full'] = {'model': full_model, 'scaler': full_scaler}
            st.success("Full model and scaler loaded successfully")
        else:
            missing_files = []
            if not full_model_path.exists():
                missing_files.append(str(full_model_path))
            if not full_scaler_path.exists():
                missing_files.append(str(full_scaler_path))
            st.error(f"Full model files not found: {', '.join(missing_files)}")
            
    except Exception as e:
        st.error(f"Error loading models: {e}")
        
    return models

def get_feature_definitions():
    """Define the features and their possible values based on the actual processed data."""
    return {
        # Demographics features (common to both models)
        'Gender': {
            'description': 'Gender of the individual',
            'type': 'categorical',
            'options': {1: 'Male', 2: 'Female'},
            'default': 1
        },
        'Age': {
            'description': 'Age in years',
            'type': 'slider',
            'min': 0,
            'max': 80,
            'default': 35,
            'step': 1
        },
        'Race_Ethnicity': {
            'description': 'Race and ethnicity background',
            'type': 'categorical',
            'options': {
                1: 'Mexican American',
                2: 'Other Hispanic', 
                3: 'Non-Hispanic White',
                4: 'Non-Hispanic Black',
                5: 'Other Race'
            },
            'default': 3
        },
        
        # Additional features for full model
        'Blood_Pressure': {
            'description': 'Systolic blood pressure (mmHg)',
            'type': 'slider',
            'min': 80,
            'max': 200,
            'default': 120,
            'step': 5
        },
        'Waist_Circumference': {
            'description': 'Waist circumference (cm)',
            'type': 'slider',
            'min': 50,
            'max': 150,
            'default': 85,
            'step': 1
        },
        'Education_Level': {
            'description': 'Highest level of education completed',
            'type': 'categorical',
            'options': {
                1: 'Less than 9th grade',
                2: '9-11th grade',
                3: 'High school graduate',
                4: 'Some college or AA degree',
                5: 'College graduate or above',
                7: 'Refused',
                9: 'Don\'t know'
            },
            'default': 4
        },
        'Dental_Visit_Reason': {
            'description': 'Main reason for last dental visit',
            'type': 'categorical',
            'options': {
                0: 'Check-up/examination/cleaning',
                1: 'Called by dentist for check-up',
                2: 'Something was wrong/bothering',
                3: 'Treatment of discovered condition',
                4: 'Other',
                5: 'Never been to dentist',
                99: 'Missing/Don\'t know'
            },
            'default': 1
        },
        'Smoking_Status': {
            'description': 'Cigarettes smoked per day (past 30 days)',
            'type': 'categorical',
            'options': {
                0: 'Not at all',
                1: 'Less than 1 per day',
                2: '1 per day',
                3: '2-5 per day',
                4: '6-10 per day',
                5: '11-20 per day',
                6: 'More than 20 per day',
                999: 'Don\'t know'
            },
            'default': 0
        },
        'Country_of_Birth': {
            'description': 'Country where born',
            'type': 'categorical',
            'options': {
                1: 'Born in 50 US states or Washington, DC',
                2: 'Born elsewhere',
                99: 'Don\'t know/Missing'
            },
            'default': 1
        },
        'Income_to_Poverty_Ratio': {
            'description': 'Family income to poverty threshold ratio',
            'type': 'slider',
            'min': 0.0,
            'max': 5.0,
            'default': 2.5,
            'step': 0.1
        },
        'Marital_Status': {
            'description': 'Current marital status',
            'type': 'categorical',
            'options': {
                1: 'Married',
                2: 'Widowed',
                3: 'Divorced',
                4: 'Separated',
                5: 'Never married',
                6: 'Living with partner',
                99: 'Don\'t know/Refused'
            },
            'default': 1
        },
        'Injected_Drugs_Ever': {
            'description': 'Ever used illegal injection drugs',
            'type': 'categorical',
            'options': {
                1: 'Yes',
                2: 'No',
                9: 'Don\'t know/Refused'
            },
            'default': 2
        },
        'Alcohol_Frequency_12m': {
            'description': 'Alcohol drinking frequency in past 12 months',
            'type': 'categorical',
            'options': {
                0: 'Never',
                1: '1-2 times in the last year',
                2: '3-6 times in the last year',
                3: '7-11 times in the last year',
                4: 'Once a month',
                5: '2-3 times a month',
                6: 'Once a week',
                7: '2 times a week',
                8: '3-4 times a week',
                9: 'Nearly every day',
                10: 'Every day',
                999: 'Don\'t know/Refused'
            },
            'default': 0
        },
        'Private_Insurance': {
            'description': 'Covered by private insurance',
            'type': 'categorical',
            'options': {
                14: 'Yes',
                99: 'No/Don\'t know'
            },
            'default': 14
        }
    }

def create_input_form(model_type, feature_definitions):
    """Create input form based on model type."""
    
    # Define which features each model uses
    demo_features = ['Gender', 'Age', 'Race_Ethnicity']
    full_features = [
        'Gender', 'Age', 'Race_Ethnicity', 'Blood_Pressure', 'Waist_Circumference',
        'Education_Level', 'Dental_Visit_Reason', 'Smoking_Status', 'Country_of_Birth',
        'Income_to_Poverty_Ratio', 'Marital_Status', 'Injected_Drugs_Ever',
        'Alcohol_Frequency_12m', 'Private_Insurance'
    ]
    
    features_to_use = demo_features if model_type == 'demo' else full_features
    user_inputs = {}
    
    st.subheader(f"📊 Input Features for {'Demographics' if model_type == 'demo' else 'Full'} Model")
    
    # Create tabs for better organization in full model
    if model_type == 'full':
        tab1, tab2, tab3 = st.tabs(["👤 Demographics", "🏥 Health Metrics", "🏠 Lifestyle & Social"])
        
        # Demographics tab
        with tab1:
            st.markdown("**Basic demographic information**")
            col1, col2 = st.columns(2)
            demo_features_full = ['Gender', 'Age', 'Race_Ethnicity']
            for i, feature in enumerate(demo_features_full):
                feature_def = feature_definitions[feature]
                with col1 if i % 2 == 0 else col2:
                    user_inputs[feature] = create_input_widget(feature, feature_def, model_type)
        
        # Health metrics tab
        with tab2:
            st.markdown("**Physical health measurements**")
            col1, col2 = st.columns(2)
            health_features = ['Blood_Pressure', 'Waist_Circumference', 'Dental_Visit_Reason']
            for i, feature in enumerate(health_features):
                feature_def = feature_definitions[feature]
                with col1 if i % 2 == 0 else col2:
                    user_inputs[feature] = create_input_widget(feature, feature_def, model_type)
        
        # Lifestyle tab
        with tab3:
            st.markdown("**Lifestyle and social factors**")
            col1, col2 = st.columns(2)
            lifestyle_features = ['Education_Level', 'Smoking_Status', 'Country_of_Birth', 
                                  'Income_to_Poverty_Ratio', 'Marital_Status', 'Injected_Drugs_Ever',
                                  'Alcohol_Frequency_12m', 'Private_Insurance']
            for i, feature in enumerate(lifestyle_features):
                feature_def = feature_definitions[feature]
                with col1 if i % 2 == 0 else col2:
                    user_inputs[feature] = create_input_widget(feature, feature_def, model_type)
    
    else:
        # Simple layout for demographics model
        col1, col2 = st.columns(2)
        for i, feature in enumerate(features_to_use):
            feature_def = feature_definitions[feature]
            with col1 if i % 2 == 0 else col2:
                user_inputs[feature] = create_input_widget(feature, feature_def, model_type)
    
    return user_inputs

def create_input_widget(feature, feature_def, model_type):
    """Create the appropriate input widget based on feature type."""
    
    if feature_def['type'] == 'categorical':
        # Create selectbox with descriptive labels
        options = list(feature_def['options'].keys())
        labels = [f"{v}" for k, v in feature_def['options'].items()]
        
        selected_idx = st.selectbox(
            f"**{feature.replace('_', ' ')}**",
            range(len(options)),
            format_func=lambda x: labels[x],
            index=options.index(feature_def['default']) if feature_def['default'] in options else 0,
            help=feature_def['description'],
            key=f"{model_type}_{feature}"
        )
        return options[selected_idx]
        
    elif feature_def['type'] == 'slider':
        return st.slider(
            f"**{feature.replace('_', ' ')}**",
            min_value=feature_def['min'],
            max_value=feature_def['max'],
            value=feature_def['default'],
            step=feature_def.get('step', 1),
            help=feature_def['description'],
            key=f"{model_type}_{feature}"
        )
    
    elif feature_def['type'] == 'numeric':
        return st.number_input(
            f"**{feature.replace('_', ' ')}**",
            min_value=float(feature_def['min']),
            max_value=float(feature_def['max']),
            value=float(feature_def['default']),
            help=feature_def['description'],
            key=f"{model_type}_{feature}"
        )

def make_prediction(model_dict, inputs, model_type):
    """Make prediction using XGBoost model and scaler."""
    try:
        # Extract model and scaler from dictionary
        model = model_dict['model']
        scaler = model_dict['scaler']
        
        # Debug: Check what type the scaler is
        logger.info(f"Scaler type: {type(scaler)}")
        logger.info(f"Scaler object: {scaler}")
        
        # Convert inputs to DataFrame
        input_df = pd.DataFrame([inputs])
        
        # Define the expected feature order for each model
        if model_type == 'demo':
            # Demographics model features
            expected_features = ['Gender', 'Age', 'Race_Ethnicity']
        else:
            # Full model features (based on your feature definitions)
            expected_features = [
                'Gender', 'Age', 'Race_Ethnicity', 'Blood_Pressure', 'Waist_Circumference',
                'Education_Level', 'Dental_Visit_Reason', 'Smoking_Status', 'Country_of_Birth',
                'Income_to_Poverty_Ratio', 'Marital_Status', 'Injected_Drugs_Ever',
                'Alcohol_Frequency_12m', 'Private_Insurance'
            ]
        
        # Reorder columns to match expected feature order
        input_df = input_df[expected_features]
        
        # Check if scaler has transform method (scikit-learn scaler)
        if hasattr(scaler, 'transform'):
            # Use scikit-learn scaler
            input_scaled = scaler.transform(input_df)
        elif isinstance(scaler, np.ndarray):
            # If scaler is a numpy array, it might be mean/std values
            # Try manual standardization: (x - mean) / std
            logger.info(f"Scaler is numpy array with shape: {scaler.shape}")
            
            # If it's a 2D array with shape (2, n_features), assume first row is mean, second is std
            if scaler.ndim == 2 and scaler.shape[0] == 2:
                mean_values = scaler[0]
                std_values = scaler[1]
                input_scaled = (input_df.values - mean_values) / std_values
            # If it's a 1D array, might need different handling
            elif scaler.ndim == 1:
                # Could be concatenated mean and std values
                n_features = len(expected_features)
                if len(scaler) == 2 * n_features:
                    mean_values = scaler[:n_features]
                    std_values = scaler[n_features:]
                    input_scaled = (input_df.values - mean_values) / std_values
                else:
                    # Fallback: use raw values
                    logger.warning("Cannot interpret scaler format, using raw values")
                    input_scaled = input_df.values
            else:
                # Fallback: use raw values
                logger.warning("Cannot interpret scaler format, using raw values")
                input_scaled = input_df.values
        else:
            # No scaling available, use raw values
            logger.warning("No valid scaler found, using raw values")
            input_scaled = input_df.values
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Get prediction probabilities
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Extract probabilities for each class
        negative_proba = prediction_proba[0]  # Probability of class 0 (negative)
        positive_proba = prediction_proba[1]  # Probability of class 1 (positive)
        
        # Calculate confidence as the maximum probability
        confidence = max(negative_proba, positive_proba)
        
        logger.info(f"Model: {model_type}, Prediction: {prediction}, Positive Prob: {positive_proba:.4f}, Confidence: {confidence:.4f}")
        
        return {
            'prediction': int(prediction),
            'probability': float(positive_proba),
            'confidence': float(confidence),
            'negative_probability': float(negative_proba)
        }
            
    except Exception as e:
        st.error(f"Error making prediction with {model_type} model: {e}")
        logger.error(f"Prediction error: {e}", exc_info=True)
        return None

def display_prediction_results(result, model_name):
    """Display prediction results in a formatted way."""
    if result is None:
        return
        
    st.subheader(f"🔬 {model_name} Model Results")
    
    # Safety check for probability
    probability = result.get('probability')
    confidence = result.get('confidence')
    
    # Handle None probability case
    if probability is None:
        st.error("⚠️ Could not extract probability from model prediction. Showing prediction only.")
        probability = 0.5 if result['prediction'] == 1 else 0.1  # Default fallback
    
    if confidence is None:
        confidence = probability
    
    # Create columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        prediction_text = "Positive ⚠️" if result['prediction'] == 1 else "Negative ✅"
        prediction_color = "red" if result['prediction'] == 1 else "green"
        st.markdown(f"**Prediction:**")
        st.markdown(f":{prediction_color}[{prediction_text}]")
    
    with col2:
        st.markdown(f"**Positive Class Probability:**")
        st.markdown(f"**{probability:.1%}**")
        st.caption("Probability of being HBsAg positive")
    
    with col3:
        st.markdown(f"**Model Confidence:**")
        st.markdown(f"**{confidence:.1%}**")
        st.caption("Overall prediction confidence")
    
    # Enhanced probability explanation
    st.markdown("---")
    st.markdown("### 📊 Detailed Probability Analysis")
    
    # Create a visual probability bar
    col_neg, col_pos = st.columns(2)
    
    negative_prob = 1 - probability
    
    with col_neg:
        st.metric(
            label="🟢 Negative Probability", 
            value=f"{negative_prob:.1%}",
            help="Probability of being HBsAg negative"
        )
        
    with col_pos:
        st.metric(
            label="🔴 Positive Probability", 
            value=f"{probability:.1%}",
            help="Probability of being HBsAg positive"
        )
    
    # Progress bars for visual representation
    st.markdown("**Visual Probability Distribution:**")
    col1, col2 = st.columns([negative_prob, probability])
    
    with col1:
        if negative_prob > 0:
            st.success(f"Negative: {negative_prob:.1%}")
    with col2:
        if probability > 0:
            st.error(f"Positive: {probability:.1%}")
    
    # Risk level interpretation
    if probability < 0.1:
        risk_level = "Very Low 🟢"
        risk_color = "green"
        risk_desc = "Very unlikely to be HBsAg positive"
    elif probability < 0.3:
        risk_level = "Low 🟢"
        risk_color = "green" 
        risk_desc = "Low probability of being HBsAg positive"
    elif probability < 0.5:
        risk_level = "Moderate-Low 🟡"
        risk_color = "orange"
        risk_desc = "Moderate-low probability of being HBsAg positive"
    elif probability < 0.7:
        risk_level = "Moderate-High 🟡"
        risk_color = "orange"
        risk_desc = "Moderate-high probability of being HBsAg positive"
    elif probability < 0.9:
        risk_level = "High 🔴"
        risk_color = "red"
        risk_desc = "High probability of being HBsAg positive"
    else:
        risk_level = "Very High 🔴"
        risk_color = "red"
        risk_desc = "Very high probability of being HBsAg positive"
        
    st.markdown(f"**Risk Assessment:** :{risk_color}[{risk_level}]")
    st.caption(risk_desc)
        
    # Interpretation
    st.markdown("---")
    st.markdown("### 🎯 Clinical Interpretation")
    
    if result['prediction'] == 1:
        st.warning(f"""
        **⚠️ POSITIVE Prediction (Probability: {probability:.1%})**
        
        This individual has a **{probability:.1%} probability** of testing positive for Hepatitis B Surface Antigen (HBsAg).
        
        **Recommended Actions:**
        - Prioritize confirmatory testing
        - Follow enhanced screening protocols
        - Consider additional hepatitis markers testing
        - Implement appropriate follow-up procedures
        """)
    else:
        negative_certainty = (1 - probability) * 100
        st.success(f"""
        **✅ NEGATIVE Prediction (Probability: {probability:.1%})**
        
        This individual has a **{negative_certainty:.1f}% probability** of testing negative for Hepatitis B Surface Antigen (HBsAg).
        
        **Recommended Actions:**
        - Standard screening protocols may be followed
        - Regular monitoring as per guidelines
        - Low priority for immediate confirmatory testing
        """)

def main():
    """Main Streamlit application."""
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #667eea;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧬 BioPredict - Hepatitis B Prediction Demo</h1>
        <p>AI-Powered Plasma Donor Screening</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    This application demonstrates two machine learning models for predicting Hepatitis B Surface Antigen (HBsAg) positivity:
    
    - **📊 Demographics Model**: Uses basic demographic information (Gender, Age, Race/Ethnicity)
    - **🔬 Full Model**: Uses comprehensive features including demographics, clinical measurements, and lifestyle factors
    
    The models were trained on NHANES data and can help prioritize confirmatory testing for plasma donors.
    """)
      # Load configuration and models
    config = load_app_config()
    models = load_models()
    
    if not models:
        st.error("❌ Models could not be loaded. Please check if model files exist in the deployment folder.")
        return
    
    # Sidebar for model selection
    st.sidebar.markdown("# 🎯 Model Selection")
    
    available_models = []
    if 'demo' in models:
        available_models.append(("📊 Demographics Model", "demo"))
    if 'full' in models:
        available_models.append(("🔬 Full Model", "full"))
    
    if not available_models:
        st.error("❌ No models available for prediction.")
        return
    
    model_choice = st.sidebar.selectbox(
        "Choose a prediction model:",
        available_models,
        format_func=lambda x: x[0]
    )
    
    selected_model_name, selected_model_type = model_choice
    selected_model_dict = models[selected_model_type]
    
    # Information about selected model
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Model Information")
    
    if selected_model_type == 'demo':
        st.sidebar.info("""
        **📊 Demographics Model Features:**
        - 👤 Gender
        - 📅 Age  
        - 🌍 Race/Ethnicity
        
        **Use Case:** Quick assessment based on basic demographic information. Ideal for initial screening.
        """)
    else:
        st.sidebar.info("""
        **🔬 Full Model Features:**
        - All demographic features
        - 🩺 Blood pressure & waist circumference
        - 🎓 Education level
        - 🦷 Dental visit patterns
        - 🚬 Smoking status
        - 🌍 Country of birth
        - 💰 Income level & insurance
        - 👥 Marital status
        - 💉 Drug use & alcohol history
        
        **Use Case:** Comprehensive assessment using multiple risk factors for enhanced accuracy.
        """)
    
    # Add model performance info if available
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Model Performance")
    if selected_model_type == 'demo':
        st.sidebar.metric("Features Used", "3")
        st.sidebar.metric("Model Type", "Tree-based")
    else:
        st.sidebar.metric("Features Used", "14")
        st.sidebar.metric("Model Type", "Tree-based")
    
    # Feature definitions
    feature_definitions = get_feature_definitions()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input form
        user_inputs = create_input_form(selected_model_type, feature_definitions)
        
        # Prediction button
        if st.button("Make Prediction", type="primary"):
            with st.spinner("Making prediction..."):
                result = make_prediction(selected_model_dict, user_inputs, selected_model_type)
                
                if result:
                    display_prediction_results(result, selected_model_name)
                    
                    # Store result in session state for comparison
                    if 'prediction_history' not in st.session_state:
                        st.session_state.prediction_history = []
                    
                    st.session_state.prediction_history.append({
                        'model': selected_model_name,
                        'inputs': user_inputs.copy(),
                        'result': result,
                        'timestamp': pd.Timestamp.now()
                    })
    with col2:
        # Display input summary
        st.subheader("📋 Input Summary")
        
        # Create a nice summary box
        summary_data = []
        for feature, value in user_inputs.items():
            feature_def = feature_definitions[feature]
            
            if feature_def['type'] == 'categorical' and value in feature_def['options']:
                display_value = feature_def['options'][value]
            else:
                if feature_def['type'] == 'slider' and feature in ['Income_to_Poverty_Ratio']:
                    display_value = f"{value:.1f}"
                else:
                    display_value = str(value)
            
            summary_data.append({
                'Feature': feature.replace('_', ' '),
                'Value': display_value
            })
        
        # Display as a clean table
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
      # Prediction history
    if 'prediction_history' in st.session_state and st.session_state.prediction_history:
        st.markdown("---")
        st.subheader("📈 Prediction History")
        
        # Convert to DataFrame for display
        history_data = []
        for entry in st.session_state.prediction_history[-5:]:  # Show last 5 predictions
            probability = entry['result'].get('probability')
            prob_display = f"{probability:.1%}" if probability is not None else "N/A"
            
            row = {
                'Timestamp': entry['timestamp'].strftime('%H:%M:%S'),
                'Model': entry['model'],
                'Prediction': 'Positive ⚠️' if entry['result']['prediction'] == 1 else 'Negative ✅',
                'Positive Probability': prob_display
            }
            history_data.append(row)
        
        if history_data:
            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df, use_container_width=True, hide_index=True)
            
            # Clear history button
            if st.button("🗑️ Clear History"):
                st.session_state.prediction_history = []
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Disclaimer:** This tool is for research and demonstration purposes only. 
    It should not be used for actual medical diagnosis or treatment decisions. 
    Always consult with healthcare professionals for medical advice.
    
    **About:** This application is part of the BioPredict project developed for Grifols Bio Supplies 
    to optimize plasma donor screening using machine learning techniques.
    """)

if __name__ == "__main__":
    main()
