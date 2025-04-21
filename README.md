# Credit Card Fraud Detection 🕵️‍♂️💳

This project focuses on detecting fraudulent transactions using both supervised and unsupervised machine learning techniques on the popular credit card fraud dataset.

---

## 📁 Project Structure


---

## 📊 Dataset Info

- Source: [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Contains **284,807 transactions**, with **only 492 fraudulent** (highly imbalanced).
- Features are PCA-transformed for confidentiality.



---

## 🧠 Models Used

### Supervised
- **XGBoostClassifier**
  - Used with class weighting
  - Evaluated on accuracy, precision, recall, F1, ROC AUC, and PR AUC

### Unsupervised
- **Isolation Forest**
  - Detects anomalies without labeled fraud examples
  - Evaluated with converted prediction labels

---

## 🛠️ Preprocessing

- `Time` and `Amount` are scaled separately.
- SMOTE is used for class balancing (only for supervised training).
- Stratified train-test split ensures class balance in testing.

---

## 📈 Evaluation Metrics

- **Confusion Matrix**
- **Classification Report**
- **ROC AUC Score**
- **PR AUC Score**
- Comparison plots generated for both models.

---

## 🧪 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
