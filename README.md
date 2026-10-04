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
| Random Forest | 0.813 | 0.871 | 0.862 |
| XGBoost | 0.807 | 0.871 | 0.852 |

All three models perform about the same. Differences are smaller than the variation between folds, and the test set has only 184 patients.

**Key findings**
- Strongest predictors: `oldpeak`, `thalch`, `age`, chest pain type, exercise-induced angina, ST slope and `thal`
- `chol` added almost no predictive value (many values were recorded as 0 and imputed)
- Lowering the decision threshold from 0.5 to 0.4 raised recall of logistic regression from 0.882 to 0.902 on the test set, at the cost of more false alarms

## Limitations
- Imputation was done before the train/test split, which may leak a small amount of information
- Missingness differs by hospital; the hospital column was excluded on purpose
- The data contains many more men than women
- The threshold was compared on the test set
- Not validated for clinical use

## Run it
```bash
pip install -r requirements.txt
python -m streamlit run app.py
```
The notebook additionally needs `matplotlib seaborn xgboost shap jupyter`, and the dataset CSV downloaded from Kaggle and placed next to the notebook.