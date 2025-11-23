import pandas as pd
import numpy as np
import os
import glob
from scipy import stats
from sklearn.preprocessing import StandardScaler

INPUT_FILE = "./csvs/ExpressionLevels_KICH_KIRC_Labelled.csv"

OUTPUT_DIR = "csvs"

CLOCK_GENES_ALL = [
    "CLOCK", "ARNTL", "ARNTL2", "NPAS2", "NR1D1", "NR1D2", 
    "CRY1", "CRY2", "RORA", "RORB", "RORC", "PER1", "PER2", "PER3"
]

def softmax(x):
    x = np.array(x, dtype=float)
    x = x - np.max(x) # Stability shift
    e = np.exp(x)
    return e / e.sum()

def run_full_pipeline():
    if not os.path.exists(INPUT_FILE):
        print(f"CRITICAL ERROR: Input file '{INPUT_FILE}' not found.")
        return
    
    df_raw = pd.read_csv(INPUT_FILE, index_col=0)
    y = df_raw.index
    X_raw = df_raw.values
    genes = df_raw.columns
    
    #z-score
    scaler = StandardScaler()
    X_z = scaler.fit_transform(X_raw)
    df_z = pd.DataFrame(X_z, index=y, columns=genes)

    #t-tests
    tumor_types = y.unique()
    if len(tumor_types) != 2:
        print(f"Error: Expected 2 tumor types, found {tumor_types}")
        return

    t1, t2 = tumor_types
    group1 = df_z.loc[t1]
    group2 = df_z.loc[t2]

    t_stats, p_vals = stats.ttest_ind(group1, group2)
    p_vals = np.where(p_vals == 0, 1e-300, p_vals) # Avoid log(0)
    
    #calculates the scores
    scores = -np.log10(p_vals)
    
    #creates the stats dataframe
    stats_df = pd.DataFrame({
        'Gene': genes, 
        'p_val': p_vals,
        'Score': scores 
    }).set_index('Gene')

   
    #identifying the top 200 genes
    top_200_genes = stats_df.sort_values('p_val', ascending=True).head(200).index.tolist()
    
    df_top200_raw = df_raw[top_200_genes]
    df_top200_z   = df_z[top_200_genes]
    
    valid_clock   = [g for g in CLOCK_GENES_ALL if g in genes]
    df_clock_raw  = df_raw[valid_clock]
    df_clock_z    = df_z[valid_clock]

    #saving the first 4 files
    path_top200_raw = os.path.join(OUTPUT_DIR, "top200_raw_expression_TumorLabels.csv")
    path_top200_z   = os.path.join(OUTPUT_DIR, "top200_zscored_by_full_distribution_TumorLabels.csv")
    path_clock_raw  = os.path.join(OUTPUT_DIR, "clock_expression_raw_TumorLabels.csv")
    path_clock_z    = os.path.join(OUTPUT_DIR, "clock_expression_zscored_TumorLabels.csv")
    
    df_top200_raw.to_csv(path_top200_raw)
    df_top200_z.to_csv(path_top200_z)
    df_clock_raw.to_csv(path_clock_raw)
    df_clock_z.to_csv(path_clock_z)

    clock_stats = stats_df.loc[valid_clock].copy()
    
    #calculate softmax weights based on p values
    weights = softmax(clock_stats['Score'].values)
    clock_stats['Weight'] = weights
    
    weight_map = clock_stats['Weight'].to_dict()
    
    #saving the weights
    clock_stats.reset_index().to_csv(os.path.join(OUTPUT_DIR, "WeightedClockGenes.csv"), index=False)
    


    #filtering the clock genes
    sorted_clock = sorted(weight_map.items(), key=lambda x: x[1])
    #drop the 3 lowest clock genes
    genes_to_keep = [x[0] for x in sorted_clock[3:]]
    
    print(f"   -> Dropped 3 weakest clock genes. Kept {len(genes_to_keep)}.")
    
    df_clock_raw_kept = df_clock_raw[genes_to_keep]
    df_clock_z_kept   = df_clock_z[genes_to_keep]

    #top 200 genes
    candidates = [g for g in top_200_genes if g not in genes_to_keep]
    df_cand_num = df_top200_raw[candidates]

    #correlation matrix calculation
    corr_matrix = pd.concat([df_cand_num, df_clock_raw_kept], axis=1).corr()
    #slicing
    target_matrix = corr_matrix.loc[candidates, genes_to_keep]
    
    
    w_vec = pd.Series({g: weight_map[g] for g in genes_to_keep})
    w_vec = w_vec.reindex(target_matrix.columns).fillna(0)
    
    final_scores = target_matrix.abs().dot(w_vec)
    
    #select top 100
    top_100_candidates = final_scores.sort_values(ascending=False).head(100).index.tolist()

    #saving the output csvs
    saved_ids = df_raw.index

    #merge the raw genes
    df_final_100_raw = df_top200_raw[top_100_candidates]
    df_merged_raw = pd.concat([df_final_100_raw, df_clock_raw_kept], axis=1)
    df_merged_raw.index = saved_ids 
    
    path_out_raw = os.path.join(OUTPUT_DIR, "top_100+11_clock_raw_expression.csv")
    df_merged_raw.to_csv(path_out_raw)

    #merge the z-scored genes
    df_final_100_z = df_top200_z[top_100_candidates]
    df_merged_z = pd.concat([df_final_100_z, df_clock_z_kept], axis=1)
    df_merged_z.index = saved_ids
    
    path_out_z = os.path.join(OUTPUT_DIR, "top_100+11_clock_z_scores.csv")
    df_merged_z.to_csv(path_out_z)

    print("\n--- PIPELINE COMPLETE ---")

if __name__ == "__main__":
    run_full_pipeline()