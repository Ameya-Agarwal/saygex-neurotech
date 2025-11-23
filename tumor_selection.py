import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy import stats
import os

os.makedirs("plots", exist_ok=True)
os.makedirs("csvs", exist_ok=True)

def analyze_tumor_pair(csv_file_path):

    filename = os.path.basename(csv_file_path)

    df = pd.read_csv(csv_file_path)
    
    df.rename(columns={df.columns[0]: 'Tumor_Type'}, inplace=True)
    
    df = df.dropna(axis=0, how='any') 
        
    tumor_types = df['Tumor_Type'].unique()
    if len(tumor_types) != 2:
        return None, -1

    t1, t2 = tumor_types
    pair_name = f"{t1} vs {t2}"

    X = df.drop('Tumor_Type', axis=1)
    y = df['Tumor_Type']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    sil_score = silhouette_score(components, y)

    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=components[:,0], y=components[:,1], hue=y, s=80, alpha=0.8, palette="Set1")
    plt.title(f"PCA: {pair_name}\nSilhouette Score: {sil_score:.3f}", fontweight='bold')
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.tight_layout()
    plt.savefig(f"plots/PCA_{t1}_{t2}.png", dpi=300)
    plt.close()

    group1 = X[y == t1]
    group2 = X[y == t2]

    log2_fc = group1.mean() - group2.mean()
    t_stat, pvals = stats.ttest_ind(group1, group2)
    pvals = np.where(pvals == 0, 1e-300, pvals) # Safety fix
    neg_log_p = -np.log10(pvals)

    sig = (neg_log_p > -np.log10(0.05)) & (abs(log2_fc) > 1)
    colors = np.where(sig & (log2_fc > 1), 'indianred', 
             np.where(sig & (log2_fc < -1), 'steelblue', 'lightgrey'))

    plt.figure(figsize=(7, 6))
    plt.scatter(log2_fc, neg_log_p, c=colors, s=10, alpha=0.6)
    plt.axhline(-np.log10(0.05), color='k', linestyle='--', lw=1)
    plt.axvline(1, color='k', linestyle='--', lw=1)
    plt.axvline(-1, color='k', linestyle='--', lw=1)
    plt.title(f"Volcano: {pair_name}", fontweight='bold')
    plt.xlabel(f"Log2 Fold Change ({t1} - {t2})")
    plt.ylabel("-Log10 P-Value")
    plt.tight_layout()
    plt.savefig(f"plots/Volcano_{t1}_{t2}.png", dpi=300)
    plt.close()

    return pair_name, sil_score


files = [
    "ExpressionLevels_KICH_KIRC_Labelled.csv",
    "ExpressionLevels_KICH_KIRP_Labelled.csv",
    "ExpressionLevels_KIRC_KIRP_Labelled.csv"
]

results = []
for f in files:
    path = f"./csvs/{f}"
    if os.path.exists(path):
        name, score = analyze_tumor_pair(path)
        if name: results.append({"Pair": name, "Score": score})

pd.DataFrame(results).sort_values('Score', ascending=False).to_csv("./csvs/math_results.csv", index=False)

target_file = "./csvs/ExpressionLevels_KICH_KIRC_Labelled.csv"

df = pd.read_csv(target_file)
df.rename(columns={df.columns[0]: 'Tumor_Type'}, inplace=True)
X = df.drop('Tumor_Type', axis=1)
y = df['Tumor_Type']
t1, t2 = y.unique() # e.g., KICH, KIRC

grp1 = X[y == t1]
grp2 = X[y == t2]
_, pvals = stats.ttest_ind(grp1, grp2)
pvals = np.where(pvals == 0, 1e-300, pvals)

gene_stats = pd.Series(pvals, index=X.columns).sort_values()
top_200_genes = gene_stats.head(200)

top_200_df = pd.DataFrame({'Gene': top_200_genes.index, 'P_Value': top_200_genes.values})
top_200_df.to_csv("./csvs/200MostDiffGenes_KICH_KIRC.csv", index=False)
print("Saved Top 200 Genes.", flush=True)

if os.path.exists('./csvs/ClockExpressionLevels_Labelled.csv'):
    clock_header = pd.read_csv('./csvs/ClockExpressionLevels_Labelled.csv', nrows=0)
    clock_list = clock_header.columns[1:].tolist() 
    valid_clock = [g for g in clock_list if g in X.columns]
    
    if valid_clock:
        c_grp1 = grp1[valid_clock]
        c_grp2 = grp2[valid_clock]
        _, c_pvals = stats.ttest_ind(c_grp1, c_grp2)
        
        pd.DataFrame({
            'Gene': valid_clock,
            'P_Value': c_pvals,
            'Log2FC': c_grp1.mean() - c_grp2.mean()
        }).sort_values('P_Value').to_csv("./csvs/Clock_Genes_Analysis.csv", index=False)
        print("Saved Clock Stats.", flush=True)
        
        data_clock = X[valid_clock]
        data_top200 = X[top_200_genes.index]
        combined_data = pd.concat([data_clock, data_top200], axis=1)
        full_corr = combined_data.corr()
        final_corr = full_corr.loc[valid_clock, top_200_genes.index]

        plt.figure(figsize=(14, 8))
        sns.heatmap(final_corr, cmap='coolwarm', center=0)
        plt.title(f"Correlation: Clock Genes vs Top 200 Diff Genes ({t1} vs {t2})", fontweight='bold')
        plt.xlabel("Top 200 Differentially Expressed Tumor Genes")
        plt.ylabel("Clock Genes")
        plt.tight_layout()
        plt.savefig("plots/Correlation_Clock_vs_Tumor.png", dpi=300)
        plt.close()
else:
    print("Clock file not found, skipping heatmap.", flush=True)
