import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath):
    """Load and preprocess the credit card fraud dataset."""
    df = pd.read_csv(filepath)
    
    # Scale 'Time' and 'Amount' features
    df['scaled_amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = StandardScaler().fit_transform(df['Time'].values.reshape(-1, 1))
    
    # Drop original Time and Amount columns
    df.drop(['Time', 'Amount'], axis=1, inplace=True)
    
    return df

def split_data(df, test_size=0.3, random_state=42):
    """Split data into train and test sets."""
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    return X_train, X_test, y_train, y_test

def handle_imbalance(X_train, y_train):
    """Apply SMOTE to handle class imbalance."""
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, y_res

if __name__ == "__main__":
    # Example usage
    df = load_data('data/creditcard.csv')
    X_train, X_test, y_train, y_test = split_data(df)
    X_res, y_res = handle_imbalance(X_train, y_train)
    
    print(f"Original train shape: {X_train.shape}")
    print(f"Resampled train shape: {X_res.shape}")