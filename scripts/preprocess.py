import pandas as pd
import ta
from sklearn.preprocessing import MinMaxScaler
import joblib  # Untuk menyimpan scaler

# Load dataset gabungan (M5 & M15)
df = pd.read_csv("gold_usd_combined.csv", index_col="time", parse_dates=True)

# Tambahkan indikator teknikal untuk timeframe M1
df["RSI_M1"] = ta.momentum.RSIIndicator(df["close_M1"], window=14).rsi()
df["EMA_M1"] = ta.trend.EMAIndicator(df["close_M1"], window=14).ema_indicator()
df["MACD_M1"] = ta.trend.MACD(df["close_M1"]).macd()
df["Bollinger_Upper_M1"] = ta.volatility.BollingerBands(df["close_M1"]).bollinger_hband()
df["Bollinger_Lower_M1"] = ta.volatility.BollingerBands(df["close_M1"]).bollinger_lband()

# Tambahkan indikator teknikal untuk timeframe M5
df["RSI_M5"] = ta.momentum.RSIIndicator(df["close_M5"], window=14).rsi()
df["EMA_M5"] = ta.trend.EMAIndicator(df["close_M5"], window=14).ema_indicator()
df["MACD_M5"] = ta.trend.MACD(df["close_M5"]).macd()
df["Bollinger_Upper_M5"] = ta.volatility.BollingerBands(df["close_M5"]).bollinger_hband()
df["Bollinger_Lower_M5"] = ta.volatility.BollingerBands(df["close_M5"]).bollinger_lband()

# Tambahkan indikator teknikal untuk timeframe M15
df["RSI_M15"] = ta.momentum.RSIIndicator(df["close_M15"], window=14).rsi()
df["EMA_M15"] = ta.trend.EMAIndicator(df["close_M15"], window=14).ema_indicator()
df["MACD_M15"] = ta.trend.MACD(df["close_M15"]).macd()
df["Bollinger_Upper_M15"] = ta.volatility.BollingerBands(df["close_M15"]).bollinger_hband()
df["Bollinger_Lower_M15"] = ta.volatility.BollingerBands(df["close_M15"]).bollinger_lband()

# Hapus nilai NaN akibat perhitungan indikator teknikal sebelum normalisasi
df.dropna(inplace=True)

# Daftar semua fitur yang akan dinormalisasi
features = ["open_M1", "high_M1", "low_M1", "close_M1", "RSI_M1", "EMA_M1", "MACD_M1", "Bollinger_Upper_M1", "Bollinger_Lower_M1",
            "open_M5", "high_M5", "low_M5", "close_M5", "RSI_M5", "EMA_M5", "MACD_M5", "Bollinger_Upper_M5", "Bollinger_Lower_M5",
            "open_M15", "high_M15", "low_M15", "close_M15", "RSI_M15", "EMA_M15", "MACD_M15", "Bollinger_Upper_M15", "Bollinger_Lower_M15"]

# Buat dan simpan scaler untuk semua fitur
scaler_features = MinMaxScaler(feature_range=(0, 1))
df[features] = scaler_features.fit_transform(df[features])
joblib.dump(scaler_features, "scaler_features.pkl")

# Buat dan simpan scaler khusus untuk close_M5 (untuk prediksi)
scaler_close = MinMaxScaler(feature_range=(0, 1))
close_m5_values = df[["close_M5"]].values
scaler_close.fit(close_m5_values)
joblib.dump(scaler_close, "scaler_close.pkl")

# Simpan hasil preprocessing
df.to_csv("gold_usd_preprocessed.csv")

print("✅ Preprocessing selesai! Data disimpan sebagai:")
print("- gold_usd_preprocessed.csv (data yang sudah dinormalisasi)")
print("- scaler_features.pkl (scaler untuk semua fitur)")
print("- scaler_close.pkl (scaler khusus untuk close_M5)")
print("\nContoh data yang sudah dinormalisasi:")
print(df.head())
