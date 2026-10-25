import pandas as pd
import numpy as np
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os

# remanipulating dataframe
df_raw = pd.read_csv("./csvs/ExpressionLevels_KICH_KIRC.csv", index_col=0)
        
with open('./json/metadata.json') as f:
    labels_dict = json.load(f)

df_T = df_raw.transpose()
df_T['Tumor_Type'] = df_T.index.map(labels_dict)

tumor_types = df_T['Tumor_Type'].unique()
t1, t2 = tumor_types
pair_name = f"{t1} vs {t2}"

df_T = df_T.dropna(axis=1, how='any')
X = df_T.drop('Tumor_Type', axis=1)
y = df_T['Tumor_Type']

# recalculating p-values (only for KICH vs KIRC)
group1 = X[y == t1]
group2 = X[y == t2]

log2_fc = group1.mean() - group2.mean()

t_stats, p_vals = stats.ttest_ind(group1, group2)
p_vals = np.where(p_vals == 0, 1e-300, p_vals)

neg_log_pval = -np.log10(p_vals)
volcano_score = np.abs(log2_fc) * neg_log_pval

results_df = pd.DataFrame({
    'Gene': group1.columns,
    'P_Value': p_vals,
    'Score': volcano_score
})

# screening out the clock genes specifically
clock_genes = ["CLOCK", "ARNTL", "ARNTL2", 
               "NPAS2", "NR1D1", "NR1D2", "CRY1", 
               "CRY2", "RORA", "RORB", 
               "RORC", "PER1", "PER2",
               "PER3"]
clock_df = results_df[results_df['Gene'].isin(clock_genes)]

# screening out other genes by score
top_genes_df = results_df.sort_values(by='Score', ascending=False).head(200)

# combining the two into one df
final_df = pd.concat([top_genes_df, clock_df]).drop_duplicates(subset='Gene')
final_df.to_csv('./csvs/200Genes_ClockGenes_results.csv', index=True)
