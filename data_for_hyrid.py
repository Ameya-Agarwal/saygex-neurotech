import pandas as pd
import numpy as np
import sys

# the files
FILE_WEIGHTS = 'WeightedClockGenes.csv'

FILE_CLOCK_RAW = 'clock_expression_raw_TumorLabels.csv'
FILE_TOP200_RAW = './csvs/top200_raw_expression_TumorLabels.csv'

FILE_CLOCK_Z = 'clock_expression_zscored_TumorLabels.csv'
FILE_TOP200_Z = './csvs/top200_zscored_by_full_distribution_TumorLabels.csv'

OUTPUT_RAW = 'top_100+14_clock_raw_expression.csv'
OUTPUT_Z = 'top_100+14_clock_z_scores.csv'


# since we have different index name but rows are in the exact same order
FORCE_MATCH_BY_POSITION = True 

def load_clean_orient(filename, name_tag):
    
    df = pd.read_csv(filename)
   

    # clean columns
    if 'Unnamed: 0' in df.columns:
        df.set_index('Unnamed: 0', inplace=True)
        df.index.name = None
    elif df.columns[0].lower() in ['sample', 'gene', 'id', 'sampleid']:
         df.set_index(df.columns[0], inplace=True)
    else:
        df.set_index(df.columns[0], inplace=True)

   
    return df

  
def main():
    # load raw expression data
    df_clock_raw = load_clean_orient(FILE_CLOCK_RAW, "Clock Raw")
    df_top200_raw = load_clean_orient(FILE_TOP200_RAW, "Top200 Raw")

    # keep labels
    saved_ids = df_top200_raw.index.tolist()

   
    # load weights and create dictionary
    df_weights = pd.read_csv(FILE_WEIGHTS)
    w_col = 'Weight' if 'Weight' in df_weights.columns else 'Score'
    weight_map = dict(zip(df_weights['Gene'], df_weights[w_col]))

    # ensure candidate genes are not clock genes
    clock_genes = df_clock_raw.columns.tolist()
    candidates = [g for g in df_top200_raw.columns if g not in clock_genes]


   # drop 3 least significant clock genes
    clock_gene_weights = pd.Series({gene: weight_map.get(gene, 0) for gene in clock_genes})
    genes_to_keep = clock_gene_weights.sort_values(ascending=True).iloc[3:].index.tolist()
    df_clock_raw = df_clock_raw[genes_to_keep]
    clock_genes = genes_to_keep 
    

    # calculate score
    df_cand_num = df_top200_raw[candidates].select_dtypes(include=np.number)
    df_clock_num = df_clock_raw.select_dtypes(include=np.number)

    # correlation matrix
    corr = pd.concat([df_cand_num, df_clock_num], axis=1).corr()
    target_matrix = corr.loc[candidates, df_clock_num.columns]
    
    # apply weights
    w_vec = pd.Series({g: weight_map.get(g, 0) for g in df_clock_num.columns})
    w_vec = w_vec.reindex(target_matrix.columns).fillna(0)
    final_scores = target_matrix.abs().dot(w_vec)
    top_100_genes = final_scores.sort_values(ascending=False).head(100).index.tolist()
    
    #raw expressions
      # extract top 100
    df_final_100 = df_top200_raw[top_100_genes]

      # merge
    df_merged_raw = pd.concat([df_final_100, df_clock_raw], axis=1)
    
      # keep labels
    df_merged_raw.index = saved_ids

      # save
    df_merged_raw.to_csv(OUTPUT_RAW)


    # z scores
    df_clock_z = load_clean_orient(FILE_CLOCK_Z, "Clock Z")
    df_top200_z = load_clean_orient(FILE_TOP200_Z, "Top200 Z")


      # keep only top 11
    df_clock_z = df_clock_z[clock_genes]

    if FORCE_MATCH_BY_POSITION:
        # Trim to match the Raw Data Length used above
        min_len = len(df_merged_raw)
        df_clock_z = df_clock_z.iloc[:min_len]
        df_top200_z = df_top200_z.iloc[:min_len]
        
        df_clock_z.index = range(min_len)
        df_top200_z.index = range(min_len)
    else:
        df_clock_z = df_clock_z.loc[df_merged_raw.index]
        df_top200_z = df_top200_z.loc[df_merged_raw.index]

    
      # merge
    valid_z_genes = [g for g in top_100_genes if g in df_top200_z.columns]
    df_final_100_z = df_top200_z[valid_z_genes]
    df_merged_z = pd.concat([df_final_100_z, df_clock_z], axis=1)

      # keep labels
    df_merged_z.index = saved_ids

      # save
    df_merged_z.to_csv(OUTPUT_Z)
    df_merged_z.to_csv(OUTPUT_Z)
   
