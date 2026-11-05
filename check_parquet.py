import pandas as pd
df = pd.read_parquet('data/processed/landmarks/full/landmarks_train.parquet')
print(df.columns)
print(df.head(2))
print("Total rows:", len(df))
