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


os.makedirs("plots", exist_ok=True)



df_KICH_KIRC = pd.read_csv(f"./csvs/ExpressionLevels_KICH_KIRC.csv", sep="\t", index_col=0)
df_KICH_KIRP = pd.read_csv(f"./csvs//ExpressionLevels_KICH_KIRP.csv", sep="\t", index_col=0)
df_KIRC_KIRP = pd.read_csv(f"./csvs/ExpressionLevels_KIRC_KIRP.csv", sep="\t", index_col=0)

#function to run all the tests on a pair of csv files
def analyze_tumor_pair(csv_file_path, json_labels_path):

    df_raw = pd.read_csv(csv_file_path, index_col=0)
        
    with open(json_labels_path) as f:
        labels_dict = json.load(f)

    df_T = df_raw.transpose()
    df_T['Tumor_Type'] = df_T.index.map(labels_dict)

    tumor_types = df_T['Tumor_Type'].unique()
    t1, t2 = tumor_types
    pair_name = f"{t1} vs {t2}"

    df_T = df_T.dropna(axis=1, how='any')
    X = df_T.drop('Tumor_Type', axis=1)
    y = df_T['Tumor_Type']
    
    # --- PCA ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    
    sil_score = silhouette_score(principal_components, y)
    
    # --- Volcano ---
    group1 = X[y == t1]
    group2 = X[y == t2]
    
    log2_fc = group1.mean() - group2.mean()
    
    t_stats, p_vals = stats.ttest_ind(group1, group2)
    p_vals = np.where(p_vals == 0, 1e-300, p_vals)
    neg_log_pval = -np.log10(p_vals)

    # Volcano coloring
    sig = (neg_log_pval > -np.log10(0.05)) & (abs(log2_fc) > 1)
    up = sig & (log2_fc > 1)
    down = sig & (log2_fc < -1)

    # --- PLOT PCA ---
    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=principal_components[:, 0],
                    y=principal_components[:, 1],
                    hue=y,
                    s=90,
                    alpha=0.9,
                    palette="Set1")
    
    plt.title(f"PCA: {pair_name}\nSilhouette: {sil_score:.3f}", fontweight="bold")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    
    plt.tight_layout()
    plt.savefig(f"plots/PCA_{t1}_vs_{t2}.png", dpi=300)
    plt.close()

    # --- PLOT VOLCANO ---
    plt.figure(figsize=(7, 6))

    # non-significant
    plt.scatter(log2_fc[~sig], neg_log_pval[~sig],
                color="lightgrey", s=12, alpha=0.6, label="Not significant")

    # upregulated (positive log2FC)
    plt.scatter(log2_fc[up], neg_log_pval[up],
                color="red", s=18, alpha=0.8, label="Upregulated")

    # downregulated (negative log2FC)
    plt.scatter(log2_fc[down], neg_log_pval[down],
                color="blue", s=18, alpha=0.8, label="Downregulated")

    plt.axhline(-np.log10(0.05), color='black', linestyle='--')
    plt.axvline(1, color='black', linestyle='--')
    plt.axvline(-1, color='black', linestyle='--')

    plt.title(f"Volcano Plot: {pair_name}", fontweight="bold")
    plt.xlabel(f"Log2 Fold Change ({t1}/{t2})")
    plt.ylabel("-Log10 P-Value")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"plots/Volcano_{t1}_vs_{t2}.png", dpi=300)
    plt.close()

    print(f"Analysis complete for {pair_name}. Score: {sil_score:.3f}\n")
    return pair_name, sil_score
    
# --- Run the Analysis for all 3 Pairs ---
all_results = []

# Pair 1: KIRC vs KIRP (Should be easy to separate)

pair1, score1 = analyze_tumor_pair('./csvs/ExpressionLevels_KIRC_KIRP.csv', './json/metadata.json')
all_results.append({"Pair": pair1, "Score": score1})

pair2, score2 = analyze_tumor_pair('./csvs/ExpressionLevels_KICH_KIRC.csv', './json/metadata.json')
all_results.append({"Pair": pair2, "Score": score2})

pair3, score3 = analyze_tumor_pair('./csvs/ExpressionLevels_KICH_KIRP.csv', './json/metadata.json')
all_results.append({"Pair": pair3, "Score": score3})

# saving results
results_df = pd.DataFrame(all_results).sort_values(by='Score', ascending=False)
results_df.to_csv('./csvs/math_results.csv', index=True)