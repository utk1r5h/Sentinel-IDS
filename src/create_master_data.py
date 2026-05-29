import pandas as pd
import glob
import os

def merge_and_balance_data(input_folder, output_file):
    all_files = glob.glob(os.path.join(input_folder, "*.csv"))
    combined_data = []

    print(f"--- Starting Merge of {len(all_files)} files ---")

    for file_path in all_files:
        file_name = os.path.basename(file_path)
        print(f"Processing: {file_name}")
        
        df = pd.read_csv(file_path)
        
        df.columns = df.columns.str.strip()
        
        attacks = df[df['Label'] != 'BENIGN']
        benign = df[df['Label'] == 'BENIGN']
        
  
        benign_sampled = benign.sample(n=min(len(benign), 50000), random_state=42)
        
        day_balanced = pd.concat([attacks, benign_sampled])
        combined_data.append(day_balanced)
        print(f"   Added {len(attacks)} attacks and {len(benign_sampled)} benign rows.")

    # Final Merge
    master_df = pd.concat(combined_data, axis=0)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    master_df.to_csv(output_file, index=False)
    print(f"\nSUCCESS: Master dataset saved to {output_file}")
    print(f"Total Rows: {len(master_df)}")
    print("\nAttack Distribution:")
    print(master_df['Label'].value_counts())

if __name__ == "__main__":
    merge_and_balance_data("data/raw/", "data/processed/master_dataset.csv")