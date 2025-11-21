import pandas as pd

# load files
top200 = pd.read_csv("./csvs/transposed_top200.csv")
clock = pd.read_csv("./csvs/ClockExpressionLevels.csv")

# rename columns
top200.rename(columns={top200.columns[0]: "Gene"}, inplace=True)
clock.rename(columns={"sample": "Gene"}, inplace=True)

# ensure same column names
common_samples = [c for c in top200.columns if c in clock.columns and c != "Gene"]

# reduce both to shared columns
top200 = top200[["Gene"] + common_samples]
clock = clock[["Gene"] + common_samples]

# merge
combined = pd.concat([top200, clock], ignore_index=True)

# remove duplicate column
combined.drop_duplicates(subset=['Gene'], keep='first', inplace=True)

# save
combined.to_csv("csvs/top200+clock_combined.csv", index=False)




