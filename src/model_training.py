import pandas as pd
import numpy as np
import joblib
import os
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scratch_models import RandomForestFromScratch




def train_scratch_model(X_train, X_test, y_train, y_test, features):
    print("\n--- Training FROM-SCRATCH Random Forest (Demo Dataset) ---")
    print(f"Training on {len(X_train)} samples, {len(features)} features")

    X_train_sub = X_train[features].values
    X_test_sub  = X_test[features].values
    y_train_arr = y_train.values
    y_test_arr  = y_test.values

    scratch_rf = RandomForestFromScratch(n_trees=20, max_depth=15, max_features='sqrt')

    start = time.time()
    scratch_rf.fit(X_train_sub, y_train_arr)
    elapsed = time.time() - start
    print(f"\nScratch RF trained in {elapsed:.1f} seconds")

    preds = scratch_rf.predict(X_test_sub)
    le = joblib.load('models/label_encoder.pkl')
    
    present_classes = np.unique(y_test_arr).astype(int)
    target_names = [le.classes_[i] for i in present_classes]

    print("\n--- Scratch RF Performance (Demo Dataset) ---")
    print(classification_report(y_test_arr, preds, labels=present_classes, target_names=target_names, zero_division=0))

    joblib.dump(scratch_rf, 'models/ids_random_forest.pkl')
    print("Scratch RF saved to models/ids_random_forest.pkl")
    return scratch_rf




def train_global_model(X_train, X_test, y_train, y_test, features):
    print(f"\n--- Training Global IDS Model (Full Dataset, sklearn) ---")
    print(f"Training on {len(X_train)} samples, {len(features)} features")

    X_train_sub = X_train[features]
    X_test_sub  = X_test[features]

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )

    start = time.time()
    print("Fitting model...")
    model.fit(X_train_sub, y_train)
    elapsed = time.time() - start
    print(f"Sklearn RF trained in {elapsed:.1f} seconds")

    y_pred = model.predict(X_test_sub)
    le = joblib.load('models/label_encoder.pkl')

    present_classes = np.unique(y_test).astype(int)
    target_names = [le.classes_[i] for i in present_classes]

    report_dict = classification_report(
        y_test, y_pred, labels=present_classes, target_names=target_names, output_dict=True, zero_division=0
    )
    
    print("\n--- Global Model Performance Summary ---")
    print(classification_report(y_test, y_pred, labels=present_classes, target_names=target_names, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)

    eval_results = {
        'report':           report_dict,
        'confusion_matrix': cm,
        'class_names':      le.classes_.tolist(),
        'train_time_sec':   round(elapsed, 1),
        'n_train':          len(X_train),
        'n_test':           len(X_test),
        'n_features':       len(features),
    }
    joblib.dump(eval_results, 'models/evaluation_results.pkl')
    print("Evaluation results saved to models/evaluation_results.pkl")

    joblib.dump(model, 'models/global_ids_model.pkl')
    joblib.dump(features, 'models/global_feature_list.pkl')
    print("Model saved to models/global_ids_model.pkl")
    return model


# ─────────────────────────────────────────────
# PART C: Save feature stats from UNSCALED data
# ─────────────────────────────────────────────

def save_feature_medians(X_train_raw, features):
    """
    Saves median, 5th percentile (min), and 95th percentile (max)
    for each feature to set slider ranges in the UI.
    """
    available = [f for f in features if f in X_train_raw.columns]
    medians = X_train_raw[available].median().to_dict()
    maxvals = X_train_raw[available].quantile(0.95).to_dict()
    minvals = X_train_raw[available].quantile(0.05).to_dict()

    joblib.dump({'medians': medians, 'maxvals': maxvals, 'minvals': minvals},
                'models/feature_medians.pkl')
    print("Feature stats saved to models/feature_medians.pkl")


if __name__ == "__main__":
    from data_loader import load_and_clean_data
    from preprocessing import preprocess_data
    from feature_selection import get_top_features

    print("=" * 60)
    print("STEP 1: Full dataset training (sklearn RF)")
    print("=" * 60)
    raw_data = load_and_clean_data("data/processed/master_dataset.csv")

    X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(raw_data)
    top_cols = get_top_features(X_train_scaled, y_train, top_n=15)

    raw_data_2 = load_and_clean_data("data/processed/master_dataset.csv")
    raw_data_2.columns = raw_data_2.columns.str.strip()
    raw_data_2 = raw_data_2.drop(columns=['Label'], errors='ignore')
    X_train_raw, _ = train_test_split(raw_data_2, test_size=0.2, random_state=42)
    save_feature_medians(X_train_raw, top_cols)

    train_global_model(X_train_scaled, X_test_scaled, y_train, y_test, top_cols)

    print("\n" + "=" * 60)
    print("STEP 2: Demo dataset training (Scratch RF)")
    print("=" * 60)
    demo_path = "data/processed/demo_traffic.csv"
    if os.path.exists(demo_path):
        demo_data = load_and_clean_data(demo_path)
        X_train_d, X_test_d, y_train_d, y_test_d = preprocess_data(demo_data)
        train_scratch_model(X_train_d, X_test_d, y_train_d, y_test_d, top_cols)
    else:
        print(f"Demo CSV not found at {demo_path}. Run create_demo_csv.py first.")