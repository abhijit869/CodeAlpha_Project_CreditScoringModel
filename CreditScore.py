import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
import joblib
import os
from google.colab import files

print("Upload your dataset (CSV or Excel file):")
uploaded = files.upload()

file_name = list(uploaded.keys())[0]
print(f"File uploaded: {file_name}")

if file_name.endswith('.csv'):
    dataset = pd.read_csv(file_name)
elif file_name.endswith(('.xlsx', '.xls')):
    dataset = pd.read_excel(file_name)
else:
    raise ValueError("Unsupported file format! Please upload a .csv or .xlsx file.")

print(f"Dataset loaded successfully! Shape: {dataset.shape}")
print(dataset.head())

if 'ID' in dataset.columns:
    dataset = dataset.drop('ID', axis=1)

dataset = dataset.fillna(dataset.mean())

y = dataset.iloc[:, 0].values
X = dataset.iloc[:, 1:29].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

output_dir = "model_output"
os.makedirs(output_dir, exist_ok=True)

normalisation_path = os.path.join(output_dir, 'normalisation_coefficients.joblib')
joblib.dump(sc, normalisation_path)
print(f"Saved normalisation coefficients to {normalisation_path}")

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

classifier_path = os.path.join(output_dir, 'credit_score_classifier.joblib')
joblib.dump(classifier, classifier_path)
print(f"Saved classifier model to {classifier_path}")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

predictions = classifier.predict_proba(X_test)
df_prediction_prob = pd.DataFrame(predictions, columns=['prob_0', 'prob_1'])
df_prediction_target = pd.DataFrame(classifier.predict(X_test), columns=['predicted_TARGET'])
df_test_dataset = pd.DataFrame(y_test, columns=['Actual Outcome'])
dfx = pd.concat([df_test_dataset, df_prediction_prob, df_prediction_target], axis=1)

predictions_path = os.path.join(output_dir, 'model_predictions.csv')
dfx.to_csv(predictions_path, index=False)
print(f"Saved predictions to {predictions_path}")
print("First few rows of predictions:")
print(dfx.head())

# Auto-download in Colab
files.download(predictions_path)

