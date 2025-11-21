import pandas as pd
import numpy as np

FILE_EXPRESSION = './csvs/top200+clock_combined.csv'
FILE_WEIGHTS = './csvs/WeightedClockGenes.csv'
OUTPUT_FILENAME = './csvs/top_100_correlated_genes.csv'

    # load file
df_expr = pd.read_csv(FILE_EXPRESSION, index_col=0)
    
    # transpose 
df_expr = df_expr.T
    
    # load weights
df_weights = pd.read_csv(FILE_WEIGHTS)
    
  
    # create a dictionary for clock genes and respective weights
weight_map = dict(zip(df_weights['Gene'], df_weights['Weight']))
  
   
    # unnessecary step making sure the clock genes being used are there in both files
valid_clock_genes = [gene for gene in weight_map.keys() if gene in df_expr.columns]
    
   
    # create series for score with sample names
clock_scores = pd.Series(0.0, index=df_expr.index)
for gene in valid_clock_genes:
        weight = weight_map[gene]
  
        clock_scores += df_expr[gene] * weight
 
  
    # calculate correlation
correlations = df_expr.corrwith(clock_scores, axis=0)

    # convert to dataframe
results_df = correlations.to_frame(name='Correlation')

    # extract top 100
results_df['Abs_Correlation'] = results_df['Correlation'].abs()
    
    # sort
top_100 = results_df.sort_values(by='Abs_Correlation', ascending=False).head(100)
    
    # remove temporary column
final_output = top_100[['Correlation']]

    # save
final_output.to_csv(OUTPUT_FILENAME)
    
    







