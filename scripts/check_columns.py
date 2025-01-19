import pandas as pd

# Baca file preprocessed
df = pd.read_csv("gold_usd_preprocessed.csv")

# Tampilkan semua kolom
print("Kolom yang tersedia:")
for col in df.columns:
    print(f"- {col}") 