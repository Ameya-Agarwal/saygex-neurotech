import pandas as pd
import json
import os
import glob

MAPPING_FILE = "metadata.json" 


FILES_TO_PROCESS = glob.glob("csvs/top200_raw_expression*.csv") + glob.glob("csvs/top200_zscored_by_full_distribution*.csv")
# =================================================

def map_sample_barcodes(csv_path, mapping_dict):
    
    
    # Load the data into 'df'
    df = pd.read_csv(csv_path, index_col=0)
    
    # Check if Barcodes are in the Index (Rows)
    index_matches = df.index.isin(mapping_dict.keys()).sum()
    # Check if Barcodes are in the Columns (Headers)
    col_matches = df.columns.isin(mapping_dict.keys()).sum()

    target_axis = None
    
    if index_matches > col_matches and index_matches > 0:
        target_axis = 'index'
    elif col_matches > index_matches and col_matches > 0:
        target_axis = 'columns'
    else:
        print(f"  ! Warning: Low match rate (Index: {index_matches}, Cols: {col_matches}). Skipping.")
        return

    # --- KEY CHANGE: Apply Mapping WITHOUT averaging duplicates ---
    if target_axis == 'index':
        # Map the index. If a barcode isn't in the JSON, keep the original barcode.
        new_index = [mapping_dict.get(x, x) for x in df.index]
        df.index = new_index
    else:
        # Map the columns. If a barcode isn't in the JSON, keep the original barcode.
        new_cols = [mapping_dict.get(x, x) for x in df.columns]
        df.columns = new_cols

    # Save
    filename, ext = os.path.splitext(csv_path)
    new_filename = f"{filename}_TumorLabels{ext}"
    
    df.to_csv(new_filename)
    print(f"  -> Saved: {new_filename} (Kept all samples separate)\n")

def main():
    # 1. Load JSON
    if not os.path.exists(MAPPING_FILE):
        print(f"Error: Mapping file '{MAPPING_FILE}' not found.")
        return

    with open(MAPPING_FILE, 'r') as f:
        mapping_dict = json.load(f)
    print(f"Loaded {len(mapping_dict)} mappings.")

    if not FILES_TO_PROCESS:
        print("ERROR: No files found. Check your 'csvs' folder.")
        return

    # 2. Process Files
    for file_path in FILES_TO_PROCESS:
        map_sample_barcodes(file_path, mapping_dict)

if __name__ == "__main__":
    main()