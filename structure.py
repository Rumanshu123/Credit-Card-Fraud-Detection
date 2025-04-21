import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

# Project name and file structure
project_name = "credit-card-fraud-detection"

list_of_files = [
    # Data directory
    "data/creditcard.csv",
    
    # Notebooks directory
    "notebooks/fraud_detection.ipynb",
    
    # Scripts directory
    "scripts/__init__.py",
    "scripts/data_preprocessing.py",
    "scripts/supervised_model.py",
    "scripts/unsupervised_model.py",
    
    # Models directory
    "models/xgboost_model.pkl",
    "models/isolation_forest.pkl",
    
    # Results directory
    "results/confusion_matrix.png",
    "results/roc_curve.png",
    "results/feature_importance.png",
    "results/if_confusion_matrix.png",
    "results/pr_curve.png",
    "results/sample_prediction.png",
    
    # Root files
    "requirements.txt",
    "README.md"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    # Create directory if it doesn't exist
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for file: {filename}")

    # Create empty file if it doesn't exist or is empty
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        if filename.endswith('.py'):
            # For Python files, add a basic docstring
            with open(filepath, "w") as f:
                if filename == "__init__.py":
                    f.write('"""Package initialization file."""')
                else:
                    f.write(f'"""Module for {filename.split(".")[0]} functionality."""')
        elif filename.endswith('.ipynb'):
            # For Jupyter notebooks, create a minimal notebook
            import nbformat as nbf
            nb = nbf.v4.new_notebook()
            nb['cells'] = [nbf.v4.new_markdown_cell(f"# {filename.split('.')[0].replace('_', ' ').title()}")]
            with open(filepath, 'w') as f:
                nbf.write(nb, f)
        else:
            # For other files, create empty file
            with open(filepath, "w") as f:
                pass
        logging.info(f"Creating file: {filepath}")
    else:
        logging.info(f"{filename} already exists")

logging.info("Project structure created successfully!")