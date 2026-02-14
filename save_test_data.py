import kagglehub
import glob
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def save_test_data():
    print("Downloading Heart Disease dataset to extract test data...")
    path = kagglehub.dataset_download("johnsmith88/heart-disease-dataset")
    
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    if not csv_files:
        print("Error: No CSV file found.")
        return

    csv_path = csv_files[0]
    print(f"Loading data from: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Ensure numeric only (same as training)
    df = df.select_dtypes(include=[np.number])
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split Data (Same random_state as training to get the actual test set)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Combine X_test and y_test to simulate a real test dataset file
    test_data = X_test.copy()
    test_data['target'] = y_test
    
    output_file = "test_data.csv"
    test_data.to_csv(output_file, index=False)
    print(f"Test data saved to: {os.path.abspath(output_file)}")
    print(f"Test data shape: {test_data.shape}")

if __name__ == "__main__":
    save_test_data()
