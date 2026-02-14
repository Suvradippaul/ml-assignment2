import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, f1_score, precision_score, recall_score, matthews_corrcoef

# Set page config
st.set_page_config(page_title="Classification Model Evaluation", layout="wide")

st.title("Machine Learning Classification Dashboard")
st.markdown("Upload your test dataset to evaluate the performance of trained models.")

# 1. Sidebar for Model Selection
st.sidebar.header("Settings")
model_dir = "model"

# Get list of available models
if os.path.exists(model_dir):
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pkl") and not f.startswith("scaler")]
    model_files.sort()
else:
    model_files = []

if not model_files:
    st.error("No trained models found in 'model/' directory. Please run 'ml_models.py' first.")
    st.stop()

selected_model_name = st.sidebar.selectbox("Select Model", model_files)

# Load the selected model
model_path = os.path.join(model_dir, selected_model_name)
try:
    model = joblib.load(model_path)
    st.sidebar.success(f"Loaded: {selected_model_name}")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()

# Load scaler if it exists
scaler_path = os.path.join(model_dir, "scaler.pkl")
scaler = None
if os.path.exists(scaler_path):
    try:
        scaler = joblib.load(scaler_path)
        # st.sidebar.info("Scaler loaded successfully.")
    except:
        st.sidebar.warning("Could not load scaler.")

# 2. Dataset Upload
st.header("1. Upload Test Dataset (CSV)")
uploaded_file = st.file_uploader("Upload your CSV file (must have same features as training data)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:", df.head())
        
        # Check if target column exists for evaluation
        target_col = st.text_input("Enter the name of the Target Column (e.g., 'target')", "target")
        
        if target_col in df.columns:
            X_test = df.drop(target_col, axis=1)
            y_test = df[target_col]
            
            # Identify if preprocessing is needed
            # For simplicity, we assume the upload is already cleaner or compatible
            # Keep only numeric columns
            X_test = X_test.select_dtypes(include=[np.number])
            
            # Apply scaling if model needs it (heuristic based on model name or if scaler exists)
            # Generally, if scaler exists, we should try to use it. 
            # Ideally, we should know if the model was trained with scaled data.
            # Based on previous code, all except Trees were scaled OR separate logic was used.
            # To be safe/simple here: if scaler exists and model is KNN/Logistic, use it.
            
            model_name_lower = selected_model_name.lower()
            if scaler and ("logistic" in model_name_lower or "neighbor" in model_name_lower or "svm" in model_name_lower):
                try:
                    X_test_processed = scaler.transform(X_test)
                except Exception as e:
                    st.warning(f"Scaling failed (feature mismatch?): {e}. Using raw data.")
                    X_test_processed = X_test
            else:
                 X_test_processed = X_test

            # Predictions
            if st.button("Evaluate Model"):
                y_pred = model.predict(X_test_processed)
                
                try:
                    y_prob = model.predict_proba(X_test_processed)[:, 1]
                except:
                    y_prob = None
                
                # 3. Metrics
                st.header("2. Evaluation Metrics")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
                c2.metric("F1 Score", f"{f1_score(y_test, y_pred, zero_division=0):.4f}")
                
                if y_prob is not None:
                     c3.metric("AUC Score", f"{roc_auc_score(y_test, y_prob):.4f}")
                else:
                    c3.metric("AUC Score", "N/A")
                    
                c4, c5, c6 = st.columns(3)
                c4.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.4f}")
                c5.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.4f}")
                c6.metric("MCC Score", f"{matthews_corrcoef(y_test, y_pred):.4f}")

                # 4. Confusion Matrix & Report
                st.header("3. Detailed Analysis")
                
                tab1, tab2 = st.tabs(["Confusion Matrix", "Classification Report"])
                
                with tab1:
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_ylabel('Actual')
                    ax.set_xlabel('Predicted')
                    st.pyplot(fig)
                    
                with tab2:
                    report = classification_report(y_test, y_pred, output_dict=True)
                    st.dataframe(pd.DataFrame(report).transpose())
                    
        else:
            st.warning(f"Column '{target_col}' not found in dataset. Please enter the correct target column name to enable evaluation.")
            
    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("Please upload a CSV file to proceed.")
