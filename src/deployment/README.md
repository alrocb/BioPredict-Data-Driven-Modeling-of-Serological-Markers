# BioPredict Streamlit Demo

## Overview
This is a Streamlit demonstration application for the BioPredict project that allows interactive testing of two trained machine learning models for Hepatitis B Surface Antigen (HBsAg) prediction.

## Models Available

### 1. Demographics Model (`best_model_demo.pkl`)
- **Features**: Gender, Age, Race/Ethnicity
- **Purpose**: Quick assessment based on basic demographic information
- **Use Case**: Initial screening with minimal data requirements

### 2. Full Model (`best_model_full.pkl`)
- **Features**: All demographic features plus clinical and lifestyle factors
- **Purpose**: Comprehensive risk assessment
- **Use Case**: Detailed evaluation when complete patient information is available

## Features

### Model Comparison
- Side-by-side comparison of both models
- Interactive input forms with user-friendly interfaces
- Real-time prediction results with probability scores

### Risk Assessment
- Color-coded risk levels (Low, Moderate, High)
- Confidence scores for predictions
- Detailed probability outputs

### User Experience
- Intuitive sidebar for model selection
- Organized input forms with helpful descriptions
- Prediction history tracking
- Input summary display

## Installation and Setup

### Prerequisites
Make sure you have the project requirements installed:
```bash
pip install -r requirements.txt
pip install streamlit
```

### Running the Application

1. Navigate to the deployment directory:
```bash
cd src/deployment
```

2. Run the Streamlit application:
```bash
streamlit run streamlit_demo.py
```

3. The application will open in your default web browser at `http://localhost:8501`

## Usage Guide

### Step 1: Select Model
- Use the sidebar to choose between Demographics Model or Full Model
- Review the model information displayed in the sidebar

### Step 2: Input Features
- Fill in the required fields in the input form
- Each field has helpful descriptions and appropriate input types
- Categorical variables show descriptive labels

### Step 3: Make Prediction
- Click "Make Prediction" to get results
- View the prediction result (Positive/Negative)
- Check the probability score and confidence level
- Review the risk level assessment

### Step 4: Review History
- View your recent predictions in the history section
- Compare results across different inputs
- Clear history when needed

## Model Features Explanation

### Demographics Model Features:
- **Gender**: Male (1) or Female (2)
- **Age**: Age in years (18-80)
- **Race_Ethnicity**: Ethnic background categories

### Full Model Additional Features:
- **Blood_Pressure**: Systolic blood pressure (mmHg)
- **Waist_Circumference**: Waist measurement (cm)
- **Education_Level**: Highest education completed
- **Dental_Visit_Reason**: Main reason for last dental visit
- **Smoking_Status**: Current smoking habits
- **Country_of_Birth**: Birth country (US vs. others)
- **Income_to_Poverty_Ratio**: Financial status indicator
- **Marital_Status**: Current relationship status
- **Injected_Drugs_Ever**: History of injection drug use
- **Alcohol_Frequency_12m**: Alcohol consumption patterns
- **Private_Insurance**: Insurance coverage status

## Technical Details

### Model Format
- Models are saved as pickle files using PyCaret's save_model function
- Compatible with scikit-learn pipeline format
- Include preprocessing steps and trained algorithms

### Data Processing
- Automatic feature encoding for categorical variables
- Input validation and error handling
- Standardized feature scaling (handled by model pipeline)

### Prediction Output
- Binary classification: 0 (Negative) or 1 (Positive)
- Probability scores for positive class
- Confidence scores based on maximum class probability

## Troubleshooting

### Common Issues

1. **Models not loading**:
   - Ensure `best_model_demo.pkl` and `best_model_full.pkl` are in the deployment folder
   - Check file permissions and paths

2. **Import errors**:
   - Verify all requirements are installed
   - Check Python path configuration
   - Ensure src directory is accessible

3. **Prediction errors**:
   - Validate input data types and ranges
   - Check for missing or invalid values
   - Review model compatibility

### Error Messages
The application includes comprehensive error handling with descriptive messages to help identify and resolve issues.

## Disclaimer
This tool is for research and demonstration purposes only. It should not be used for actual medical diagnosis or treatment decisions. Always consult with healthcare professionals for medical advice.

## Project Context
This application is part of the BioPredict project developed for Grifols Bio Supplies to optimize plasma donor screening using machine learning techniques on NHANES data.
