# Heart Disease Prediction

Predicting the presence of heart disease from clinical measurements, using the
[UCI Heart Disease dataset on Kaggle](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data).

> Learning project, not a medical tool.

## Overview
- 920 patients, 12 clinical features used after cleaning
- Target: 0 = no heart disease, 1 = heart disease (any severity)
- Models compared: Logistic Regression, Random Forest, XGBoost
- Tools: Python, pandas, scikit-learn, XGBoost, SHAP, Streamlit

## Workflow
1. **Cleaning:** impossible zeros in `chol` and `trestbps` treated as missing, median imputation for numeric columns, a "missing" category for `thal` and `slope`, `ca` dropped (about 66% missing)
2. **EDA:** class balance, numeric and categorical features vs target, correlation heatmap
3. **Modeling:** one-hot encoding, stratified 80/20 split, 5-fold cross-validation
4. **Evaluation:** recall, ROC-AUC, confusion matrix, threshold analysis
5. **Interpretation:** logistic regression coefficients and SHAP values
6. **App:** a Streamlit app that loads the saved model and estimates risk from patient inputs

## Results

5-fold cross-validation on the training set:

| Model | Accuracy | ROC-AUC | Recall |
|---|---|---|---|
| Logistic Regression | 0.806 | 0.875 | 0.843 |
| Random Forest |