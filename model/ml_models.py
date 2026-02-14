import kagglehub
import glob
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_data():
    """
    Step 1: Dataset Choice
    Dataset: Heart Disease Dataset
    Source: Kaggle (via kagglehub)
    Task: Binary Classification (Heart Disease: Yes/No)
    
    Checks:
    - Feature Size: 13 (Minimum required: 12) -> PASSED
    - Instance Size: 1025 (Minimum required: 500) -> PASSED
    """
    print("Downloading Heart Disease dataset from Kaggle...")
    # Download latest version
    path = kagglehub.dataset_download("johnsmith88/heart-disease-dataset")
    
    print("Path to dataset files:", path)
    
    # Find the csv file
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV file found in the downloaded dataset.")
        
    csv_path = csv_files[0]
    print(f"Loading data from: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Check for text columns and drop them if any (though this dataset should be numeric)
    # This dataset is already numeric, but good practice to ensure
    df = df.select_dtypes(include=[np.number])
    
    # Separate Features and Target
    # Target column is 'target'
    X = df.drop('target', axis=1)
    y = df['target']
    
    print(f"Dataset Loaded: Heart Disease")
    print(f"Feature Size: {X.shape[1]}")
    print(f"Instance Size: {X.shape[0]}")
    print("-" * 30)
    
    return X, y

def get_models():
    """
    Step 2: Initialize the 6 required classification models.
    """
    models = {
        "Logistic_Regression": LogisticRegression(random_state=42, max_iter=2000),
        "Decision_Tree": DecisionTreeClassifier(random_state=42),
        "K_Nearest_Neighbor": KNeighborsClassifier(),
        "Naive_Bayes_Gaussian": GaussianNB(),
        "Random_Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    return models

def evaluate_and_save_model(name, model, X_train, X_test, y_train, y_test):
    # Train
    model.fit(X_train, y_train)
    
    # Save Model
    model_path = os.path.join("model", f"{name}.pkl")
    joblib.dump(model, model_path)
    # print(f"Saved model: {model_path}")

    # Predict
    y_pred = model.predict(X_test)
    
    # Get probabilities for AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred 
        
    # Calculate Metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC Score": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "MCC Score": matthews_corrcoef(y_test, y_pred)
    }
    
    return metrics

def main():
    # Determine model directory relative to this script
    model_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(model_dir, exist_ok=True)
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    # 1. Load Data
    try:
        X, y = load_data()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling (Important for LR and KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler too, as it's part of the pipeline strictly speaking
    joblib.dump(scaler, scaler_path)

    # Models that benefit significantly from scaling
    scale_sensitive_models = ["Logistic_Regression", "K_Nearest_Neighbor"]
    
    # 3. specific models and evaluate
    models = get_models()
    results = []
    
    print("Training and Evaluating Models...\n")
    
    for name, model in models.items():
        print(f"Processing: {name.replace('_', ' ')}")
        
        # Save Model
        model_path = os.path.join(model_dir, f"{name}.pkl")
        
        if name in scale_sensitive_models:
            model.fit(X_train_scaled, y_train)
            joblib.dump(model, model_path)
            y_pred = model.predict(X_test_scaled)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
            else:
                y_prob = y_pred
        else:
            model.fit(X_train, y_train)
            joblib.dump(model, model_path)
            y_pred = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = y_pred
            
        # Calculate Metrics
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC Score": roc_auc_score(y_test, y_prob) if y_prob is not None else 0, # Handle potential None for consistency
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0),
            "MCC Score": matthews_corrcoef(y_test, y_pred)
        }
            
        metrics['Model'] = name.replace('_', ' ')
        results.append(metrics)
        
    # 4. Display Results
    results_df = pd.DataFrame(results)
    
    # Reorder columns
    cols = ['Model', 'Accuracy', 'AUC Score', 'Precision', 'Recall', 'F1 Score', 'MCC Score']
    results_df = results_df[cols]
    
    print("\n" + "="*80)
    print("FINAL EVALUATION RESULTS")
    print("="*80)
    print(results_df.to_string(index=False))
    
    # Save to CSV in parent directory or same dir? Let's keep it in parent so it's visible.
    # Or actually keep it alongside models? The prompt just said "move ml_models.py to model folder".
    # Usually results stay in root or output dir. Let's put it in the same dir as the script for tidiness.
    results_csv_path = os.path.join(model_dir, "classification_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nResults saved to '{results_csv_path}'")
    print(f"Models saved to '{model_dir}' directory")

if __name__ == "__main__":
    main()
