Credit Score Model

This project trains a Logistic Regression model to predict credit scores (or any binary target) using uploaded datasets.

Steps:
1. Upload Data – CSV or Excel file
2. Preprocess – Removes ID, fills missing values, scales features
3. Train Model – Logistic Regression
4. Evaluate – Shows accuracy and confusion matrix
5. Save Results –
   - credit_score_classifier.joblib → trained model
   - normalisation_coefficients.joblib → scaler
   - model_predictions.csv → predictions

Requirements:
pip install pandas numpy scikit-learn joblib openpyxl

Example Output:
Accuracy: 0.92
Confusion Matrix:
[[85  5]
 [10 100]]

Folder Structure:
model_output/
 ├── credit_score_classifier.joblib
 ├── normalisation_coefficients.joblib
 └── model_predictions.csv
