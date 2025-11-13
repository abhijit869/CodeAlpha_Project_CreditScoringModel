# Credit Scoring Model

*CodeAlpha Project Task 1 — Data Science Internship*

## Table of Contents

- [Project Overview](#project-overview)
- [Motivation](#motivation)
- [Features](#features)
- [Workflow](#workflow)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Output Files](#output-files)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

## Project Overview

This project implements a **Credit Scoring Model** that uses historical credit data to predict the likelihood of default for borrowers. It's designed for rapid prototyping and experimentation, making it a useful starting point for credit risk analysis.

## Motivation

Credit scoring is essential for lenders to evaluate borrower risk and make informed lending decisions. Automating the scoring process with machine learning helps improve accuracy, reduce manual errors, and enable data-driven policies.

## Features

- **Dataset Upload:** Supports `.csv` and `.xlsx` file formats
- **Data Preprocessing:** Handles missing values, feature scaling, and optional column removal
- **Model Training:** Uses Logistic Regression for binary classification
- **Evaluation Metrics:** Displays confusion matrix and accuracy score
- **Output Generation:** Saves the trained model, scaler coefficients, and prediction results
- **User-Friendly:** Designed for easy use in Google Colab and local environments

## Workflow

1. **Upload Dataset:** Import your credit dataset via the interface or code
2. **Preprocessing:** Clean and scale the data automatically
3. **Model Training:** Fit a logistic regression model to your target variable
4. **Evaluation:** Review performance metrics (accuracy, confusion matrix)
5. **Prediction & Output:** Predict outcomes on test data and export results

## Setup & Installation

### Quick Start (Google Colab)

The project is optimized for Google Colab but can be run locally:
```python
# Upload your dataset as prompted in Colab
# Run main.py or copy relevant code into your notebook
```

### Local Installation

Clone the repository:
```sh
git clone https://github.com/abhijit869/CodeAlpha_Project_CreditScoringModel.git
cd CodeAlpha_Project_CreditScoringModel
```

Install dependencies:
```sh
pip install pandas numpy scikit-learn joblib
```

## Usage

Example usage (from `main.py`):
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# ...see main.py for details
```

1. Upload your dataset when prompted
2. Run all code cells (Colab) or execute `main.py` locally
3. Review output metrics and files in `model_output/`

## Output Files

- `model_output/credit_score_classifier.joblib` — Trained Logistic Regression model
- `model_output/normalisation_coefficients.joblib` — Scaler normalization coefficients
- `model_output/model_predictions.csv` — Test set predictions and probabilities

## Dependencies

- `pandas`
- `numpy`
- `scikit-learn`
- `joblib`

## Contributing

Pull requests, bug reports, and feature suggestions are welcome! Please fork the repository and submit changes via PR.

## License

This project is licensed under the [MIT License](LICENSE).

## Author

Developed by [abhijit869](https://github.com/abhijit869) as part of CodeAlpha Internship.

---

*For questions, open an issue or contact via GitHub.*
