import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    print("--- Starting Multi-Class Preprocessing ---")
    
    df.columns = df.columns.str.strip()
    

    le = LabelEncoder()
    df['Label'] = le.fit_transform(df['Label'])
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(le, 'models/label_encoder.pkl')
    print(f"Encoded {len(le.classes_)} distinct classes.")
    
    X = df.drop('Label', axis=1)
    y = df['Label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    

    mean = X_train.mean()
    std = X_train.std()
    
    X_train_scaled = (X_train - mean) / (std + 1e-9)
    X_test_scaled = (X_test - mean) / (std + 1e-9)
    
    scaling_stats = {'mean': mean, 'std': std}
    joblib.dump(scaling_stats, 'models/scaling_stats.pkl')
    
    print(f"Preprocessing Complete. Training Shape: {X_train_scaled.shape}")
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    from data_loader import load_and_clean_data
    master_path = "data/processed/master_dataset.csv"
    
    if os.path.exists(master_path):
        raw_data = load_and_clean_data(master_path)
        X_train, X_test, y_train, y_test = preprocess_data(raw_data)
    else:
        print("Master dataset not found. Run create_master_data.py first.")