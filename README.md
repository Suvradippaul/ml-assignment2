# Classification Model Analysis

This document details the analysis and evaluation of various machine learning models trained on the Heart Disease dataset.

## a. Problem Statement
The objective is to build and evaluate machine learning models to classify whether a patient has heart disease or not based on a set of medical attributes. This is a binary classification problem (Target: 1 = Disease, 0 = No Disease).

## b. Dataset Description
- **Dataset Name**: Heart Disease Dataset (downloaded via Kaggle)
- **Source**: Kaggle (johnsmith88/heart-disease-dataset)
- **Features**: 13 Numerical Features including age, sex, chest pain type (cp), resting blood pressure (trestbps), cholesterol (chol), fasting blood sugar (fbs), etc.
- **Target**: 'target' (Binary: 0 or 1)
- **Instance Count**: 1025 instances (total)

## c. Models Used and Evaluation Metrics
We implemented 6 classification models: Logistic Regression, Decision Tree, K-Nearest Neighbors (kNN), Naive Bayes (Gaussian), Random Forest (Ensemble), and XGBoost (Ensemble).

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.7951 | 0.8787 | 0.7563 | 0.8738 | 0.8108 | 0.5973 |
| **Decision Tree** | 0.9854 | 0.9854 | 1.0000 | 0.9709 | 0.9852 | 0.9712 |
| **kNN** | 0.8341 | 0.9486 | 0.8000 | 0.8932 | 0.8440 | 0.6727 |
| **Naive Bayes** | 0.8000 | 0.8706 | 0.7541 | 0.8932 | 0.8178 | 0.6102 |
| **Random Forest (Ensemble)** | 0.9854 | 1.0000 | 1.0000 | 0.9709 | 0.9852 | 0.9712 |
| **XGBoost (Ensemble)** | 0.9854 | 0.9894 | 1.0000 | 0.9709 | 0.9852 | 0.9712 |

## d. How to Run

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Streamlit App**:
    ```bash
    streamlit run app.py
    ```
    The application will be accessible at `http://localhost:8501`.

3.  **Train Models (Optional)**:
    If you need to retrain the models:
    ```bash
    python model/ml_models.py
    ```

## Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | The linear model performed moderately with ~79.5% accuracy. It struggled to capture the complex non-linear relationships in the medical data compared to tree-based models. |
| **Decision Tree** | Achieved excellent results (98.5% accuracy), matching the ensemble methods. This suggests the features have strong hierarchical predictive power, though single trees can be prone to overfitting. |
| **kNN** | Performed reasonably well (83.4% accuracy) with a high AUC of 0.949, indicating that patients with similar metrics tend to have similar diagnoses, but it was outperformed by tree-based logic. |
| **Naive Bayes** | Showed comparable performance to Logistic Regression (80% accuracy). The assumption of feature independence may be too strong for correlated physiological metrics. |
| **Random Forest (Ensemble)** | Demonstrated superior performance (98.5% accuracy) with a perfect AUC of 1.0. As an ensemble, it effectively captured complex patterns and is robust. |
| **XGBoost (Ensemble)** | Matched the top-tier performance (98.5% accuracy). Its sophisticated boosting algorithm effectively minimized errors, making it one of the best choices for this dataset. |
