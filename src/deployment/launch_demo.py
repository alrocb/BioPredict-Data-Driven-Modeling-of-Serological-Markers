#!/usr/bin/env python3
"""
Streamlit Demo Launcher

This script launches the BioPredict Streamlit demonstration application.
Run this script from the project root directory.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Launch the Streamlit demo application."""
    
    # Get the project root directory
    project_root = Path(__file__).parent
    
    # Path to the Streamlit demo
    demo_path = project_root/ "streamlit_demo.py"
    
    # Check if the demo file exists
    if not demo_path.exists():
        print(f"Error: Demo file not found at {demo_path}")
        sys.exit(1)
      # Check if models exist
    model_dir = project_root 
    demo_model = model_dir / "hepatitis_b_xgboost_model_demo.pkl"
    demo_scaler = model_dir / "hepatitis_b_scaler_demo.pkl"
    full_model = model_dir / "hepatitis_b_xgboost_model.pkl"
    full_scaler = model_dir / "hepatitis_b_scaler.pkl"
    
    missing_models = []
    if not demo_model.exists():
        missing_models.append(str(demo_model))
    if not demo_scaler.exists():
        missing_models.append(str(demo_scaler))
    if not full_model.exists():
        missing_models.append(str(full_model))
    if not full_scaler.exists():
        missing_models.append(str(full_scaler))
    
    if missing_models:
        print("Warning: The following model files are missing:")
        for model in missing_models:
            print(f"  - {model}")
        print("\nThe demo may not work properly without these models.")
        print("Please ensure the models are trained and saved in the deployment folder.")
        
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() not in ['y', 'yes']:
            sys.exit(1)
    
    # Change to the deployment directory
    os.chdir(model_dir)
    
    # Launch Streamlit
    try:
        print(f"Launching Streamlit demo from {model_dir}")
        print("The application will open in your default web browser...")
        print("Press Ctrl+C to stop the application.")
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(demo_path), 
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
        
    except KeyboardInterrupt:
        print("\nShutting down Streamlit demo...")
    except FileNotFoundError:
        print("Error: Streamlit is not installed.")
        print("Please install it using: pip install streamlit")
        sys.exit(1)
    except Exception as e:
        print(f"Error launching Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
