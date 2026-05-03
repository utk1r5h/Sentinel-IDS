import pandas as pd
import numpy as np

def load_and_clean_data(filepath):
    print(f"--- Loading: {filepath.split('/')[-1]} ---")
    
    # Load the dataset
    df = pd.read_csv(filepath)
    
    # 1. Strip whitespace from column names (common issue in this dataset)
    df.columns = df.columns.str.strip()
    
    # 2. Handle 'dirty' values
    # Replace 'Infinity' strings or actual np.inf with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Drop rows with any NaN values
    initial_count = len(df)
    df.dropna(inplace=True)
    print(f"Dropped {initial_count - len(df)} rows containing NaN/Inf values.")
    
    return df

if __name__ == "__main__":
    # Test the loader
    path = "data/raw/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
    data = load_and_clean_data(path)
    print(data['Label'].value_counts())