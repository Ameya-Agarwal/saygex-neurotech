import pandas as pd
import numpy as np
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


df_KICH_KIRC = pd.read_csv(f"./ExpressionLevels_KICH_KIRC.csv", sep="\t", index_col=0)
df_KICH_KIRP = pd.read_csv(f"./ExpressionLevels_KICH_KIRP.csv", sep="\t", index_col=0)
df_KIRC_KIRP = pd.read_csv(f"./ExpressionLevels_KIRC_KIRP.csv", sep="\t", index_col=0)

#function to run all the tests on a pair of csv files
def analyze_tumor_pair(csv_file_path, json_labels_path):

    df_raw = pd.read_csv(csv_file_path, index_col=0)
        
    with open(json_labels_path) as f:
        labels_dict = json.load(f)

    # Transpose the matrix
    df_T = df_raw.transpose()
    
    # Map labels from the json to the tumor
    df_T['Tumor_Type'] = df_T.index.map(labels_dict)


    # Check that we have exactly two tumor types
    tumor_types = df_T['Tumor_Type'].unique()
        
    t1, t2 = tumor_types
    pair_name = f"{t1} vs {t2}"

    # Separate features (X) and labels (y)
    # Ensure no NaN values, which can break PCA/T-tests
    df_T = df_T.dropna(axis=1, how='any') # Drop genes with any NaN
    X = df_T.drop('Tumor_Type', axis=1)
    y = df_T['Tumor_Type']
    
    # --- 2. PCA ANALYSIS (FOR CLUSTER SEPARATION) ---
    
    # Standardize data (z-score scaling) is CRUCIAL for PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform PCA to reduce to 2 components
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    
    # Calculate Silhouette Score:
    sil_score = silhouette_score(principal_components, y)
    
    # --- 3. VOLCANO PLOT ANALYSIS (FOR GENE DIFFERENCES) ---
    
    group1 = X[y == t1]
    group2 = X[y == t2]
    
    # 1. X-AXIS: Log2 Fold Change
    log2_fc = group1.mean() - group2.mean()
    
    # 2. Y-AXIS: -Log10 P-Value
    t_stats, p_vals = stats.ttest_ind(group1, group2)
    p_vals = np.where(p_vals == 0, 1e-300, p_vals) # Handle -log10(0)
    neg_log_pval = -np.log10(p_vals)
    
    # --- 4. PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: PCA
    sns.scatterplot(x=principal_components[:, 0], y=principal_components[:, 1], 
                    hue=y, ax=ax1, s=80, alpha=0.8)
    ax1.set_title(f'PCA: {pair_name}\nSeparation Score: {sil_score:.3f} (Higher is Better)', 
                  fontweight='bold')
    ax1.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax1.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    
    # Plot 2: Volcano Plot
    sns.scatterplot(x=log2_fc, y=neg_log_pval, ax=ax2, alpha=0.5, linewidth=0)
    ax2.axhline(-np.log10(0.05), color='red', linestyle='--', label='P-Value = 0.05')
    ax2.axvline(1, color='blue', linestyle='--', alpha=0.5, label='Log2 Fold Change = 1')
    ax2.axvline(-1, color='blue', linestyle='--', alpha=0.5)
    ax2.set_title(f'Volcano Plot: {pair_name}', fontweight='bold')
    ax2.set_xlabel(f'Log2 Fold Change ({t1} / {t2})')
    ax2.set_ylabel('-Log10 P-Value (Significance)')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"Analysis complete for {pair_name}. Score: {sil_score:.3f}\n")
    return pair_name, sil_score

    
# --- Run the Analysis for all 3 Pairs ---
all_results = []

# Pair 1: KIRC vs KIRP (Should be easy to separate)

pair1, score1 = analyze_tumor_pair('./ExpressionLevels_KIRC_KIRP.csv', './metadata.json')
all_results.append({"Pair": pair1, "Score": score1})

pair2, score2 = analyze_tumor_pair('./ExpressionLevels_KICH_KIRC.csv', './metadata.json')
all_results.append({"Pair": pair2, "Score": score2})

pair3, score3 = analyze_tumor_pair('./ExpressionLevels_KICH_KIRP.csv', './metadata.json')
all_results.append({"Pair": pair3, "Score": score3})

# saving results
results_df = pd.DataFrame(all_results).sort_values(by='Score', ascending=False)
results_df.to_csv('./math_results.csv', index=True)