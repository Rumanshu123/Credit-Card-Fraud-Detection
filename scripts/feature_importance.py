import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_feature_importance(model_path, feature_names, top_n=20):
    """Plot and save the top N most important features."""
    # Load trained XGBoost model
    model = joblib.load(model_path)

    # Get feature importances
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(top_n)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title(f'Top {top_n} Feature Importances (XGBoost)')
    plt.tight_layout()

    # Ensure results folder exists
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/feature_importance.png')
    plt.close()
    print("Feature importance diagram saved to 'results/feature_importance.png'.")

if __name__ == "__main__":
    # Example usage: assuming your training used scaled_time, scaled_amount, V1-V28
    features = ['scaled_time', 'scaled_amount'] + [f'V{i}' for i in range(1, 29)]
    plot_feature_importance('models/xgboost_model.pkl', features)
