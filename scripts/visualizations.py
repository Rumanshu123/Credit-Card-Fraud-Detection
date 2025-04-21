import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split

# Configuration
MODEL_PATH = 'models/xgboost_model.pkl'
DATA_PATH = 'data/creditcard.csv'
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data():
    """Load and preprocess data consistently with training"""
    try:
        df = pd.read_csv(DATA_PATH)
        print("✅ Data loaded successfully")
        
        # Load scalers used during training
        scaler_amount = joblib.load('scalers/scaler_amount.pkl')
        scaler_time = joblib.load('scalers/scaler_time.pkl')
        
        # Apply same transformations as training
        df['Amount'] = scaler_amount.transform(df['Amount'].values.reshape(-1, 1))
        df['Time'] = scaler_time.transform(df['Time'].values.reshape(-1, 1))
        
        return df
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        raise

def generate_visualizations(model, X_test, y_test):
    """Generate all evaluation visualizations"""
    try:
        print("\n🔹 Generating predictions...")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # 1. Feature Importance
        print("\n📊 Creating feature importance plot...")
        plt.figure(figsize=(10, 6))
        importance = model.feature_importances_
        features = X_test.columns
        fi_df = pd.DataFrame({'Feature': features, 'Importance': importance})
        fi_df = fi_df.sort_values('Importance', ascending=False)
        sns.barplot(x='Importance', y='Feature', data=fi_df)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/feature_importance.png')
        plt.close()
        print("✅ Feature importance plot saved")
        
        # 2. ROC Curve
        print("\n📈 Creating ROC curve...")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/roc_curve.png')
        plt.close()
        print("✅ ROC curve saved")
        
        # 3. Precision-Recall Curve
        print("\n📉 Creating Precision-Recall curve...")
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        ap_score = average_precision_score(y_test, y_proba)
        plt.figure()
        plt.plot(recall, precision, label=f'PR curve (AP = {ap_score:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/pr_curve.png')
        plt.close()
        print("✅ Precision-Recall curve saved")
        
        # 4. Confusion Matrix
        print("\n🔢 Creating confusion matrix...")
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/confusion_matrix.png')
        plt.close()
        print("✅ Confusion matrix saved")
        
        # 5. Sample Predictions
        print("\n👀 Creating sample predictions visualization...")
        sample_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).head(50)
        plt.figure(figsize=(12, 4))
        plt.plot(sample_df['Actual'], 'o-', label='Actual')
        plt.plot(sample_df['Predicted'], 'x--', label='Predicted')
        plt.title('Sample Predictions (First 50)')
        plt.xlabel('Sample Index')
        plt.ylabel('Class')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/sample_predictions.png')
        plt.close()
        print("✅ Sample predictions visualization saved")
        
    except Exception as e:
        print(f"❌ Error generating visualizations: {str(e)}")
        raise

def evaluate_and_visualize():
    try:
        print("\n🔹 Loading model...")
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully")
        
        print("\n🔹 Loading and preparing data...")
        df = load_data()
        X_train, X_test, y_train, y_test = train_test_split(
            df.drop('Class', axis=1), 
            df['Class'], 
            test_size=0.3, 
            random_state=42, 
            stratify=df['Class']
        )
        print("✅ Data prepared successfully")
        
        generate_visualizations(model, X_test, y_test)
        
        print("\n🎉 All visualizations generated successfully!")
        print(f"📁 Check the '{RESULTS_DIR}' directory for output files")
        
    except Exception as e:
        print(f"\n❌ Error in evaluation: {str(e)}")

if __name__ == '__main__':
    evaluate_and_visualize()