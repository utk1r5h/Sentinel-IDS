"""
create_demo_csv.py

Creates a small, demo-ready CSV for the Streamlit live simulation.

Why this exists:
  - The raw CICIDS 2017 files are hundreds of MB — cannot be uploaded in a browser
  - The master_dataset.csv is also large (~700k rows)
  - This script creates a ~1000 row file with every attack class represented
  - File is small enough to upload via Streamlit sidebar instantly

Output: data/processed/demo_traffic.csv
Run this AFTER model_training.py has been run at least once
(so global_feature_list.pkl exists).
"""

import pandas as pd
import numpy as np
import joblib
import os


def create_demo_csv(
    master_path="data/processed/master_dataset.csv",
    output_path="data/processed/demo_traffic.csv",
    rows_per_class=100,
    random_state=42
):
    print("--- Creating Demo CSV ---")

    # Load the feature list saved during training
    feature_list_path = "models/global_feature_list.pkl"
    if not os.path.exists(feature_list_path):
        print("ERROR: global_feature_list.pkl not found.")
        print("Run model_training.py first so features are selected and saved.")
        return

    top_features = joblib.load(feature_list_path)
    print(f"Loaded {len(top_features)} selected features.")

    # Load master dataset (only the columns we need + Label)
    cols_to_load = top_features + ['Label']
    print(f"Loading master dataset from {master_path}...")
    df = pd.read_csv(master_path, usecols=cols_to_load)
    df.columns = df.columns.str.strip()
    print(f"Master dataset loaded: {len(df)} rows, {len(df.columns)} columns")

    # Drop rows with any NaN or Inf in feature columns
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=top_features)

    # Stratified sampling: rows_per_class rows from each attack type
    class_counts = df['Label'].value_counts()
    print(f"\nClasses in master dataset: {len(class_counts)}")
    for label, count in class_counts.items():
        print(f"  {label}: {count} rows")

    sampled_frames = []
    for label in class_counts.index:
        class_df = df[df['Label'] == label]
        n = min(rows_per_class, len(class_df))
        sampled = class_df.sample(n=n, random_state=random_state)
        sampled_frames.append(sampled)
        print(f"  Sampled {n} rows from class: {label}")

    demo_df = pd.concat(sampled_frames, axis=0).sample(frac=1, random_state=random_state)
    demo_df = demo_df.reset_index(drop=True)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    demo_df.to_csv(output_path, index=False)

    print(f"\nSUCCESS: Demo CSV saved to {output_path}")
    print(f"Total rows: {len(demo_df)}")
    print(f"Columns: {list(demo_df.columns)}")
    print(f"\nClass distribution in demo file:")
    print(demo_df['Label'].value_counts().to_string())


if __name__ == "__main__":
    create_demo_csv(rows_per_class=100)
