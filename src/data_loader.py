import pandas as pd
import numpy as np

def load_and_clean_data(filepath):
    print(f"--- Loading: {filepath.split('/')[-1]} ---")
    
    df = pd.read_csv(filepath)
    
    df.columns = df.columns.str.strip()
    

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    initial_count = len(df)
    df.dropna(inplace=True)
    print(f"Dropped {initial_count - len(df)} rows containing NaN/Inf values.")
    
    return df

if __name__ == "__main__":
    path = "data/raw/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
    data = load_and_clean_data(path)
    print(data['Label'].value_counts())