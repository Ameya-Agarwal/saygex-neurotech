import pandas as pd
import numpy as np

FILE_EXPRESSION = './csvs/top200+clock_combined.csv'
FILE_WEIGHTS = './csvs/WeightedClockGenes.csv'
OUTPUT_FILENAME = './csvs/top100_correlation_values.csv'

   # load and transpose
df_expr = pd.read_csv(FILE_EXPRESSION, index_col=0).T
df_clock_info = pd.read_csv(FILE_WEIGHTS)
    
   # create dictionary
clock_weight_map = dict(zip(df_clock_info['Gene'], df_clock_info['Score']))
   
clock_genes = [g for g in clock_weight_map.keys() if g in df_expr.columns]
candidate_genes = [g for g in df_expr.columns if g not in clock_genes]

df_candidates = df_expr[candidate_genes]
df_clocks = df_expr[clock_genes]

    # calculate correlation
corr_matrix = pd.concat([df_candidates, df_clocks], axis=1).corr()
    
    # keeping only rows as candidates, columns as clock
target_matrix = corr_matrix.loc[candidate_genes, clock_genes]


    # create series
weights_vector = pd.Series({g: clock_weight_map[g] for g in clock_genes})
    
    # align
weights_vector = weights_vector.reindex(target_matrix.columns)

weighted_scores = target_matrix.abs().dot(weights_vector)

    # get top 100
top_100 = weighted_scores.sort_values(ascending=False).head(100)
    
    # convert to dataframe 
output_df = top_100.to_frame(name='Weighted_Relevance_Score')

    # save
output_df.to_csv(OUTPUT_FILENAME)
    









