import pandas as pd

#load files
top200 = pd.read_csv("./csvs/transposed_top200.csv")
clock = pd.read_csv("./csvs/ClockExpressionLevels.csv")

#renaming first column's name in top 200
top200.rename(columns={top200.columns[0]: "Gene"}, inplace=True)

#renaming the first column in clock
clock.rename(columns={"sample": "Gene"}, inplace=True)

#ensuring same column names 
common_samples = [c for c in top200.columns if c in clock.columns]

#reducing both to shared 
top200 = top200[["Gene"] + common_samples]
clock = clock[["Gene"] + common_samples]

#merging
combined = pd.concat([top200, clock], ignore_index=True)

#saving
combined.to_csv("top200_clock_combined.csv", index=False)


