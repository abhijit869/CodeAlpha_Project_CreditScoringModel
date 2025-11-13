# Credit Scoring Model (CodeAlpha Project Task 1)

This project is a Credit Scoring Model implementation as part of CodeAlpha's Data Science Internship, developed using Python and scikit-learn.

## Overview

Credit scoring helps lenders assess the risk of lending to borrowers. This model uses logistic regression to predict the likelihood of default, based on historical credit data.

## Features

- Upload your own dataset (`.csv` or `.xlsx`)
- Data preprocessing: missing value imputation, feature scaling, optional column removal
- Model training and evaluation (confusion matrix, accuracy score)
- Saves trained model and normalization coefficients for reuse
- Outputs predictions with probabilities and downloads them as CSV

## How It Works

1. **Upload Dataset:** You can upload your CSV or Excel file containing credit data.
2. **Preprocessing:** The script handles missing values and scales features.
3. **Training:** Logistic Regression is used for binary classification.
4. **Evaluation:** Provides accuracy and confusion matrix.
5. **Predictions:** Produces a CSV containing actual values, probabilities, and predicted outcomes.

## Usage

Run in Google Colab for best results:
```python
# Upload your dataset and run all cells
# Example (snippet from main.py):
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
 ...
```

## Output Files

- `model_output/credit_score_classifier.joblib` : Saved logistic regression model
- `model_output/normalisation_coefficients.joblib` : Scaler coefficients
- `model_output/model_predictions.csv` : Predictions and probabilities for test data

## Dependencies

- pandas
- numpy
- scikit-learn
- joblib

Install them via pip if running locally:
```sh
pip install pandas numpy scikit-learn joblib
```

## Contributing

Feel free to fork and contribute improvements. Pull requests are welcome!

## License

This project is released under the MIT License.

---

**Developed by [abhijit869](https://github.com/abhijit869) as part of CodeAlpha Internship.**
