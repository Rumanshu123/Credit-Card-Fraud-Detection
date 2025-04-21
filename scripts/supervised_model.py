import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
import os
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Simplified data loading with basic preprocessing"""
    df = pd.read_csv(filepath)
    
    # Initialize scalers for Amount and Time
    scaler_amount = StandardScaler()
    scaler_time = StandardScaler()
    
    # Scale Amount and Time
    df['Amount'] = scaler_amount.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['Time'] = scaler_time.fit_transform(df['Time'].values.reshape(-1, 1))
    
    # Save the scalers
    joblib.dump(scaler_amount, 'scalers/scaler_amount.pkl')
    joblib.dump(scaler_time, 'scalers/scaler_time.pkl')
    
    return df, scaler_amount, scaler_time

def train_xgboost(X_train, y_train):
    """Simplified XGBoost training with fewer hyperparameters"""
    # Calculate class weight
    scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
    
    # Use simpler model with fewer tuning parameters
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=1  # Use only 1 CPU core to reduce resource usage
    )
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Basic evaluation with key metrics"""
    y_pred = model.predict(X_test)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # ROC AUC score
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        print(f"\nROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    
    # Simple confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig('results/confusion_matrix.png')
    plt.close()

if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('scalers', exist_ok=True)
    
    try:
        # Load and prepare data
        print("Loading data...")
        df, scaler_amount, scaler_time = load_data('data/creditcard.csv')
        
        # Simple train-test split (no SMOTE to reduce computation)
        X = df.drop('Class', axis=1)
        y = df['Class']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Train model
        print("Training model...")
        model = train_xgboost(X_train, y_train)
        
        # Evaluate
        print("Evaluating model...")
        evaluate_model(model, X_test, y_test)
        
        # Save model
        joblib.dump(model, 'models/xgboost_model.pkl')
        print("Model saved successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
