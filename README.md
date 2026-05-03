# Sentinel-IDS: Advanced Machine Learning Intrusion Detection

This repository contains **Sentinel-IDS**, a modular, high-performance framework designed to detect and classify network intrusions[cite: 1]. By leveraging a Random Forest architecture, the system identifies anomalies in network traffic, distinguishing between legitimate operations and various attack vectors[cite: 1].

---

## 📂 Project Architecture & Directory Tree

The system is architected to separate data ingestion, preprocessing, and the final application layer[cite: 1].

```text
Sentinel-IDS/
├── app/                        # Application Layer
│   └── main.py                 # Core API / Entry point
├── data/                       # Data Repository
│   └── processed/              # Cleaned/Transformed data storage
├── models/                     # Serialized ML Binaries (.pkl)
│   ├── ids_random_forest.pkl   # The trained RF model
│   ├── label_encoder.pkl       # Target variable mapping
│   ├── scaling_stats.pkl       # Feature scaling parameters
│   └── evaluation_results.pkl  # Performance metadata
├── src/                        # Core Logic & Pipeline
│   ├── preprocessing.py        # Data cleaning & normalization
│   ├── feature_selection.py    # 15-feature extraction logic
│   ├── model_training.py       # Training & Serialization
│   └── evaluation.py           # Metrics calculation
├── requirements.txt            # System dependencies
└── .gitignore                  # Version control exclusions
```

---

## 🛠 Detailed Component Manifest

### 1. Source Logic (`src/`)
This directory contains the procedural scripts that define the ML pipeline's behavior[cite: 1].

*   **`create_master_data.py`**: Acts as the data aggregator. It compiles raw network logs from various sources into a unified "Master" dataset for analysis[cite: 1].
*   **`preprocessing.py`**: Performs rigorous data sanitization. It handles missing value imputation, removes outliers, and ensures all numerical inputs are normalized[cite: 1].
*   **`feature_selection.py`**: Uses statistical importance metrics to reduce dimensionality. It isolates the **15 most impactful features** from the raw data to minimize computational overhead and improve model accuracy[cite: 1].
*   **`data_loader.py`**: A specialized utility designed to stream and load large-scale CSV files into memory efficiently without causing system overflow[cite: 1].
*   **`model_training.py`**: The training harness. It implements the Random Forest algorithm, fits the model to the 15 selected features, and exports the final binary to the `models/` folder[cite: 1].
*   **`evaluation.py`**: Generates a comprehensive performance report, including Accuracy, Precision, Recall, and F1-Score metrics[cite: 1].
*   **`scratch_models.py`**: Contains custom implementations of underlying algorithms used for benchmarking against standard library versions[cite: 1].

### 2. Trained Assets (`models/`)
These files represent the "learned" state of the system[cite: 1].

*   **`ids_random_forest.pkl`**: The finalized Random Forest classifier[cite: 1].
*   **`feature_medians.pkl` & `scaling_stats.pkl`**: These files store the exact statistical distribution (medians/means) of the training data. This is critical for ensuring that real-time data is scaled identically to the data the model was trained on[cite: 1].
*   **`label_encoder.pkl`**: Translates numerical model outputs back into human-readable attack classifications (e.g., `0` becomes `DDoS`, `1` becomes `Normal`)[cite: 1].
*   **`global_feature_list.pkl`**: A reference file that ensures the order of features during prediction remains consistent with the training phase[cite: 1].

### 3. Application Interface (`app/`)
*   **`main.py`**: The production-facing script. It initializes the environment, loads the `.pkl` assets, and provides an interface for submitting live network traffic to the classifier for immediate threat detection[cite: 1].

---

## 🚀 Step-by-Step Technical Workflow

1.  **Ingestion**: Execute `src/create_master_data.py` to build the primary training set[cite: 1].
2.  **Transformation**: Run `src/preprocessing.py`. This step saves `scaling_stats.pkl` and `feature_medians.pkl` for future use[cite: 1].
3.  **Refinement**: Run `src/feature_selection.py` to identify the 15-feature subset[cite: 1].
4.  **Optimization**: Run `src/model_training.py` to train the classifier and generate the primary binary, `ids_random_forest.pkl`[cite: 1].
5.  **Operation**: Start `app/main.py`. This script will use the stored scaling parameters and the 15-feature list to analyze incoming traffic[cite: 1].

---

## 📊 Model Evaluation Summary
The model's performance is logged in `models/evaluation_results.pkl`[cite: 1]. Standard performance is calculated as follows:

$$ \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} $$

$$ \text{Precision} = \frac{TP}{TP + FP} $$

(Where $TP$ = True Positives, $TN$ = True Negatives, etc.)[cite: 1]
