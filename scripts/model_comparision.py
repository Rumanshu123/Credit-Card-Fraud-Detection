import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, precision_recall_curve, auc,
                           f1_score, recall_score, precision_score, accuracy_score)
import joblib

def load_models():
    """Load trained models from disk"""
    xgb_model = joblib.load('models/xgboost_model.pkl')
    iforest_model = joblib.load('models/isolation_forest.pkl')
    return xgb_model, iforest_model

def evaluate_supervised(model, X_test, y_test):
    """Evaluate supervised model and return metrics"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    
    # Calculate PR AUC
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    metrics['pr_auc'] = auc(recall, precision)
    
    return metrics

def evaluate_unsupervised(model, X_test, y_test):
    """Evaluate unsupervised model and return metrics"""
    preds = model.predict(X_test)
    # Convert predictions (1=normal, -1=anomaly) to (0=normal, 1=fraud)
    y_pred = np.where(preds == 1, 0, 1)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred)  # Note: unsupervised can't produce probabilities
    }
    
    # Calculate PR AUC
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    metrics['pr_auc'] = auc(recall, precision)
    
    return metrics

def plot_metrics_comparison(metrics_df):
    """Plot comparison of metrics between models"""
    plt.figure(figsize=(12, 8))
    
    # Exclude ROC AUC and PR AUC for this plot
    plot_metrics = ['accuracy', 'precision', 'recall', 'f1']
    melted_df = metrics_df[plot_metrics].reset_index().melt(id_vars='index')
    
    ax = sns.barplot(x='variable', y='value', hue='index', data=melted_df)
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.ylim(0, 1.1)
    plt.legend(title='Model')
    
    # Add values on top of bars
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    
    plt.savefig('results/metrics_comparison.png')
    plt.close()

def plot_auc_comparison(metrics_df):
    """Plot comparison of AUC metrics"""
    plt.figure(figsize=(10, 6))
    
    auc_metrics = ['roc_auc', 'pr_auc']
    melted_df = metrics_df[auc_metrics].reset_index().melt(id_vars='index')
    
    ax = sns.barplot(x='variable', y='value', hue='index', data=melted_df)
    plt.title('AUC Metrics Comparison')
    plt.ylabel('Score')
    plt.ylim(0, 1.1)
    plt.legend(title='Model')
    
    # Add values on top of bars
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    
    plt.savefig('results/auc_comparison.png')
    plt.close()

def compare_models(X_test, y_test):
    """Compare performance of both models"""
    # Load models
    xgb_model, iforest_model = load_models()
    
    # Evaluate models
    xgb_metrics = evaluate_supervised(xgb_model, X_test, y_test)
    iforest_metrics = evaluate_unsupervised(iforest_model, X_test, y_test)
    
    # Create comparison dataframe
    metrics_df = pd.DataFrame({
        'XGBoost': xgb_metrics,
        'Isolation Forest': iforest_metrics
    }).T
    
    # Print metrics
    print("Model Performance Comparison:")
    print(metrics_df)
    
    # Save metrics to CSV
    metrics_df.to_csv('results/model_metrics.csv')
    
    # Plot comparisons
    plot_metrics_comparison(metrics_df)
    plot_auc_comparison(metrics_df)
    
    return metrics_df

if __name__ == "__main__":
    # Load test data (assuming you have the preprocessing functions)
    from data_preprocessing import load_data, split_data
    
    df = load_data('data/creditcard.csv')
    _, X_test, _, y_test = split_data(df)
    
    # Compare models
    metrics_df = compare_models(X_test, y_test)