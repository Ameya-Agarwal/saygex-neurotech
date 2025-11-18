## Importing libraries
import pandas as pd
import numpy as np
import gzip
import json

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
Exp_Levels_KICH_KIRC = pd.concat([KICH_df, KIRC_df], axis = 1)
Exp_Levels_KICH_KIRP = pd.concat([KICH_df, KIRP_df], axis = 1)
Exp_Levels_KIRC_KIRP = pd.concat([KIRC_df, KIRP_df], axis = 1)

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
clock_exp_df = clock_exp_df.reset_index().rename(columns={"index":"gene"})

#saving data frame as csv
clock_exp_df.to_csv("./csvs/ClockExpressionLevels.csv", index=False)
