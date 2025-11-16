
## Importing libraries
import pandas as pd
import numpy as np
import gzip
import json


barcodess = {}

def loader(type):
    with gzip.open(f"C:/Users/vinay/Desktop/Neurotech Project Proposals/SAYGEX/TCGA.{type}.sampleMap_HiSeqV2.gz", "rt") as f:
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
output_path = "C:/Users/vinay/Desktop/Neurotech Project Proposals/SAYGEX/metadata.json"
with open(output_path, "w") as f:
    json.dump(barcodess, f, indent=4)

#combining dataframes
Exp_Levels = pd.concat([KICH_df, KIRC_df, KIRP_df], axis = 1)

#saving data frame as csv
Exp_Levels.to_csv('C:/Users/vinay/Desktop/Neurotech Project Proposals/SAYGEX/ExpressionLevels.csv', index=True)





