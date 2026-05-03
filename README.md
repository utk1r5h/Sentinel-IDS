UT Sentinel-IDS: Advanced Network Intrusion Detection
This repository contains Sentinel-IDS, a machine learning-powered framework designed to identify and classify network security threats. By analyzing traffic patterns, it distinguishes between "Normal" operations and various "Intrusion" attempts using a refined Random Forest model


📂 Project Architecture
The system is built as a modular pipeline where data moves from raw logs to a processed state, then into training, and finally into a serving application


Sentinel-IDS/
├── app/                 # Serving Layer
│   └── main.py          # Application entry point
├── data/                # Data Storage
│   └── processed/       # Cleaned CSV files
├── models/              # Model Binaries & Metadata
│   ├── *.pkl            # Serialized ML components
├── src/                 # Logic & Processing
│   ├── preprocessing.py # Data cleaning
│   ├── training.py      # ML training logic
│   └── ...              # Helper scripts
└── requirements.txt     # Dependency list


UT Sentinel-IDS: Advanced Network Intrusion Detection
This repository contains Sentinel-IDS, a machine learning-powered framework designed to identify and classify network security threats. By analyzing traffic patterns, it distinguishes between "Normal" operations and various "Intrusion" attempts using a refined Random Forest model.  
+1

📂 Project Architecture
The system is built as a modular pipeline where data moves from raw logs to a processed state, then into training, and finally into a serving application.  

Directory Tree

Plaintext
Sentinel-IDS/
├── app/                 # Serving Layer
│   └── main.py          # Application entry point
├── data/                # Data Storage
│   └── processed/       # Cleaned CSV files
├── models/              # Model Binaries & Metadata
│   ├── *.pkl            # Serialized ML components
├── src/                 # Logic & Processing
│   ├── preprocessing.py # Data cleaning
│   ├── training.py      # ML training logic
│   └── ...              # Helper scripts
└── requirements.txt     # Dependency list
🛠 Detailed Component Breakdown
1. The Processing Engine (src/)

This folder contains the "blueprints" for how data is handled.  

create_master_data.py: The initial step. It aggregates scattered network logs into a single "Master" file, ensuring a unified training foundation.  

preprocessing.py: This script performs heavy lifting by handling missing data (Imputation) and scaling numerical values. It ensures that a feature like "Packet Size" (which could be in thousands) doesn't statistically overwhelm "Flag Count" (which might be 0 or 1).  
+1

feature_selection.py: Crucial for efficiency. It analyzes the original dataset (which may have dozens of features) and selects the top 15 most important features. This reduces the model's "noise" and significantly speeds up real-time detection.  
+1

scratch_models.py: Likely contains custom-coded implementations of algorithms (like Decision Trees) used for comparative testing against standard libraries.



2. The Model Brain (models/)

Files here are stored in .pkl (Pickle) format, which allows Python to save and load complex objects instantly.  

ids_random_forest.pkl: The core classifier. It has "learned" the difference between malicious and benign traffic during training.  

label_encoder.pkl: A translation key. It maps numerical predictions (e.g., 0, 1, 2) back to human-readable labels like Normal, DDoS, or PortScan.  

scaling_stats.pkl & feature_medians.pkl: These store the "mathematical state" of the training data. When new traffic arrives at the IDS, these files are used to scale that new data in the exact same way the training data was scaled.




3. The Serving Layer (app/)

main.py: This is the "face" of the project. It loads the trained model from the models/ folder and listens for network data. When data is passed to it, it applies the stored scaling stats, runs the 15-feature extraction, and outputs a security verdict.


🔄 Execution Workflow
To take this from a fresh repo to a working system, follow this sequence:

Environment Setup: Install dependencies via pip install -r requirements.txt.  

Data Preparation: Run create_master_data.py to compile your dataset, followed by create_demo_csv.py if you need a test file[cite: 1].

Pipeline Execution:

Run Preprocessing to clean the data[cite: 1].

Run Feature Selection to isolate the 15 critical columns[cite: 1].

Run Model Training to generate the .pkl files[cite: 1].

Deployment: Launch app/main.py to begin active monitoring or to test the demo_traffic.csv for intrusions[cite: 1].
