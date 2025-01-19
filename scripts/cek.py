import pandas as pd

# Load data yang sudah diproses
df = pd.read_csv("gold_usd_preprocessed.csv", index_col="time", parse_dates=True)

# Tampilkan 5 baris pertama
print(df.head())
