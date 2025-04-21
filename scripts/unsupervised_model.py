import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def train_isolation_forest(X_train, contamination=0.01):
    """Train Isolation Forest model."""
    iforest = IsolationForest(
        n_estimators=100,
        max_samples='auto',
        contamination=contamination,
        random_state=42,
        verbose=1
    )
    
    iforest.fit(X_train)
    return iforest

def evaluate_anomaly_detection(model, X_test, y_test):
    """Evaluate anomaly detection performance."""
    preds = model.predict(X_test)
    
    # Convert predictions (1=normal, -1=anomaly) to match our labels (0=normal, 1=fraud)
    preds = np.where(preds == 1, 0, 1)
    
    print(classification_report(y_test, preds))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
    plt.title('Isolation Forest Confusion Matrix')
    plt.savefig('results/if_confusion_matrix.png')
    plt.close()

def find_optimal_contamination(model, X_train, y_train, X_val, y_val):
    """Find optimal contamination parameter."""
    contamination_values = np.linspace(0.001, 0.02, 10)
    best_f1 = 0
    best_contamination = 0.01
    
    for cont in contamination_values:
        model.set_params(contamination=cont)
        model.fit(X_train)
        preds = model.predict(X_val)
        preds = np.where(preds == 1, 0, 1)
        
        report = classification_report(y_val, preds, output_dict=True)
        f1 = report['1']['f1-score']
        
        if f1 > best_f1:
            best_f1 = f1
            best_contamination = cont
    
    print(f"Best contamination: {best_contamination}, F1-score: {best_f1}")
    return best_contamination

if __name__ == "__main__":
    # Load data
    from data_preprocessing import load_data, split_data
    
    df = load_data('data/creditcard.csv')
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Train model
    iforest = train_isolation_forest(X_train)
    
    # Evaluate
    evaluate_anomaly_detection(iforest, X_test, y_test)
    
    # Save model
    joblib.dump(iforest, 'models/isolation_forest.pkl')