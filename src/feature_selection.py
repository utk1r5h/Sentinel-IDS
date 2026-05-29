import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

def get_top_features(X_train, y_train, top_n=10):
    print("--- Calculating Feature Importance ---")
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    importances = model.feature_importances_
    feature_names = X_train.columns
    
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    top_features = feature_importance_df.head(top_n)['Feature'].tolist()
    
    print(f"Top {top_n} Features Selected:")
    for i, feat in enumerate(top_features):
        print(f"{i+1}. {feat}")
        
    return top_features

if __name__ == "__main__":
    from data_loader import load_and_clean_data
    from preprocessing import preprocess_data
    
    master_path = "data/processed/master_dataset.csv"
    raw_data = load_and_clean_data(master_path)
    
    X_train, X_test, y_train, y_test = preprocess_data(raw_data)
    
    top_cols = get_top_features(X_train, y_train, top_n=15) # Increase to 15 for diversity