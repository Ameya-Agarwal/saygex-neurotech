import pandas as pd
import numpy as np
import os

INPUT_CSV = "./csvs/200Genes_ClockGenes_results.csv"
OUTPUT_CSV = "./csvs/WeightedClockGenes.csv"

#the list of clock genes
CLOCK_GENES = ["CLOCK", "ARNTL", "ARNTL2",
               "NPAS2", "NR1D1", "NR1D2",
               "CRY1", "CRY2", "RORA",
               "RORB", "RORC", "PER1",
               "PER2", "PER3"]


#softmax weighing
def softmax(x):
    x = np.array(x, dtype=float)
    x = x - np.max(x)        
    e = np.exp(x)
    return e / e.sum()


df = pd.read_csv(INPUT_CSV)

#extracting clock genes
clock_df = df[df['Gene'].isin(CLOCK_GENES)].copy()

#ditching any duplicates
clock_df = clock_df.drop_duplicates(subset='Gene')


#computing the weights
scores = clock_df['Score'].values
weights = softmax(scores)

clock_df['Weight'] = weights

#arranging by weight
clock_df = clock_df.sort_values('Weight', ascending=False).reset_index(drop=True)

#saving
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
clock_df.to_csv(OUTPUT_CSV, index=False)