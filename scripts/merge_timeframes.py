import pandas as pd

# Load data M1, M5, dan M15
df_m1 = pd.read_csv("gold_usd_M1.csv", index_col="time", parse_dates=True)
df_m5 = pd.read_csv("gold_usd_M5.csv", index_col="time", parse_dates=True)
df_m15 = pd.read_csv("gold_usd_M15.csv", index_col="time", parse_dates=True)

# Tambahkan suffix '_M15' ke kolom df_m15 (kecuali 'time' yang digunakan untuk merge)
df_m15.columns = [f"{col}_M15" if col != "time" else col for col in df_m15.columns]

# Merge berdasarkan timestamp (inner join)
df = df_m1.merge(df_m5, on="time", suffixes=("_M1", "_M5"))
df = df.merge(df_m15, on="time")

# Periksa kolom yang ada
print("\nKolom setelah merge:")
for col in df.columns:
    print(f"- {col}")

# Simpan dataset gabungan
df.to_csv("gold_usd_combined.csv")

print("\n✅ Data M1, M5, dan M15 berhasil digabungkan dan disimpan sebagai gold_usd_combined.csv")
print("\nContoh data:")
print(df.head())
