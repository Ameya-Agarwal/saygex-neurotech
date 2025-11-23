## Importing libraries
import pandas as pd
import numpy as np
import gzip
import json
import os
import glob

barcodess = {}

def loader(type):
    with gzip.open(f"./gzips/TCGA.{type}.sampleMap_HiSeqV2.gz", "rt") as f:
        return pd.read_csv(f, sep="\t", index_col=0)

KICH_df = loader("KICH")
KIRC_df = loader("KIRC")
KIRP_df = loader("KIRP")


#cleaning up barcodes
def barc(df):
    barcodes = df.columns.tolist()
    return barcodes

#making the list
def addtolist(entries, type):
    for e in entries:
        barcodess[e] = type

#calling functions for each dataset
def allofit(df, type):
    barcodez = barc(df)
    addtolist(barcodez, type)

allofit(KIRP_df, "KIRP")
allofit(KIRC_df, "KIRC")
allofit(KICH_df, "KICH")

#saving json file
output_path = "./json/metadata.json"
with open(output_path, "w") as f:
    json.dump(barcodess, f, indent=4)

#combining dataframes
Exp_Levels = pd.concat([KICH_df, KIRC_df, KIRP_df], axis = 1)
Exp_Levels_KICH_KIRC = pd.concat([KICH_df, KIRC_df], axis = 1).transpose()
Exp_Levels_KICH_KIRP = pd.concat([KICH_df, KIRP_df], axis = 1).transpose()
Exp_Levels_KIRC_KIRP = pd.concat([KIRC_df, KIRP_df], axis = 1).transpose()

#saving data frames as csvs

Exp_Levels.to_csv('./csvs/ExpressionLevels.csv', index=True)
Exp_Levels_KICH_KIRC.to_csv('./csvs/ExpressionLevels_KICH_KIRC.csv', index=True)
Exp_Levels_KICH_KIRP.to_csv('./csvs/ExpressionLevels_KICH_KIRP.csv', index=True)
Exp_Levels_KIRC_KIRP.to_csv('./csvs/ExpressionLevels_KIRC_KIRP.csv', index=True)

#list of clock genes
Clock_Genes = ["CLOCK", "ARNTL", "ARNTL2", 
               "NPAS2", "NR1D1", "NR1D2", "CRY1", 
               "CRY2", "RORA", "RORB", 
               "RORC", "PER1", "PER2",
               "PER3"]

#comparing with ExpressionLevels.csv
exp = pd.read_csv("./csvs/ExpressionLevels.csv", index_col=0)
match = [g for g in Clock_Genes if g in exp.index]

#creating data frame with clock gene expression data
clock_exp_df = exp.loc[match]
clock_exp_df = clock_exp_df.copy()              
clock_exp_df["gene"] = clock_exp_df.index     
clock_exp_df = clock_exp_df.reset_index(drop=True)


#saving data frame as csv
clock_exp = clock_exp_df.set_index("gene").T 
clock_exp.to_csv("./csvs/ClockExpressionLevels.csv", index=True)


FILES_TO_PROCESS = glob.glob("./csvs/ExpressionLevels_KICH_KIRC.csv") + glob.glob("./csvs/ExpressionLevels_KICH_KIRP.csv") + glob.glob("./csvs/ExpressionLevels_KIRC_KIRP.csv") + glob.glob("./csvs/ClockExpressionLevels.csv")

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
        
    if target_axis == 'index':
        # Map the index. If a barcode isn't in the JSON, keep the original barcode.
        new_index = [mapping_dict.get(x, x) for x in df.index]
        df.index = new_index
    else:
        # Map the columns. If a barcode isn't in the JSON, keep the original barcode.
        new_cols = [mapping_dict.get(x, x) for x in df.columns]
        df.columns = new_cols

    filename, ext = os.path.splitext(csv_path)
    new_filename = f"{filename}_Labelled{ext}"
    
    df.to_csv(new_filename)

def remove_barcodes():

    with open(output_path, 'r') as f:
        mapping_dict = json.load(f)

    # 2. Process Files
    for file_path in FILES_TO_PROCESS:
        map_sample_barcodes(file_path, mapping_dict)

remove_barcodes()