import pandas as pd
import ta
import joblib  # Untuk menyimpan scaler (tidak digunakan karena normalisasi dihapus)
import numpy as np

# Load dataset gabungan (Hanya M15 yang dipertahankan)
df = pd.read_csv("gold_usd_combined.csv", index_col="time", parse_dates=True)

# Tambahkan indikator teknikal untuk timeframe M15
df["RSI_M15"] = ta.momentum.RSIIndicator(df["close_M15"], window=14).rsi()
df["EMA_M15"] = ta.trend.EMAIndicator(df["close_M15"], window=14).ema_indicator()
df["MACD_M15"] = ta.trend.MACD(df["close_M15"]).macd()
bollinger_M15 = ta.volatility.BollingerBands(df["close_M15"])
df["Bollinger_Upper_M15"] = bollinger_M15.bollinger_hband()
df["Bollinger_Lower_M15"] = bollinger_M15.bollinger_lband()

# Tambahkan indikator tren tambahan
df["ADX_M15"] = ta.trend.ADXIndicator(df["high_M15"], df["low_M15"], df["close_M15"], window=14).adx()
df["SMA_M15"] = ta.trend.SMAIndicator(df["close_M15"], window=50).sma_indicator()

# Hapus nilai NaN akibat perhitungan indikator teknikal sebelum preprocessing
df.dropna(inplace=True)

# Daftar semua fitur yang akan digunakan sebagai input dan target (hanya M15 dan indikator tambahan)
features = [
    "open_M15", "high_M15", "low_M15", "close_M15",
    "RSI_M15", "EMA_M15", "MACD_M15", 
    "Bollinger_Upper_M15", "Bollinger_Lower_M15",
    "ADX_M15", "SMA_M15"
]

# **Hapus langkah normalisasi**
# scaler_features = MinMaxScaler(feature_range=(0, 1))
# df[features] = scaler_features.fit_transform(df[features])
# joblib.dump(scaler_features, "scaler_features.pkl")

# Simpan hasil preprocessing tanpa normalisasi
df.to_csv("gold_usd_preprocessed.csv")

print("✅ Preprocessing selesai! Data disimpan sebagai:")
print("- gold_usd_preprocessed.csv (data yang sudah diproses tanpa normalisasi)")
# print("- scaler_features.pkl (scaler untuk semua fitur)")  # Tidak disimpan karena normalisasi dihapus
print("\nContoh data yang sudah diproses:")
print(df.head())

# Membuat dataset dengan Sliding Window (250 candle sebelumnya sebagai input)
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])  # Semua fitur pada timestep berikutnya sebagai target
    return np.array(X), np.array(y)

n_steps = 250  # Gunakan 250 candle terakhir untuk prediksi
data = df[features].values
X, y = create_sequences(data, n_steps)

# Split Data (85% Training, 10% Testing, 5% Holdout)
total_samples = len(X)
train_size = int(total_samples * 0.85)
test_size = int(total_samples * 0.10)
holdout_size = total_samples - train_size - test_size

X_train, X_temp = X[:train_size], X[train_size:]
y_train, y_temp = y[:train_size], y[train_size:]

X_test, X_holdout = X_temp[:test_size], X_temp[test_size:]
y_test, y_holdout = y_temp[:test_size], y_temp[test_size:]

# Simpan dataset
np.savez("lstm_dataset.npz", 
         X_train=X_train, X_test=X_test, X_holdout=X_holdout, 
         y_train=y_train, y_test=y_test, y_holdout=y_holdout)

print("✅ Dataset LSTM berhasil dibuat dan disimpan sebagai lstm_dataset.npz")
print(f"\nUkuran dataset:")
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"X_holdout: {X_holdout.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test: {y_test.shape}")
print(f"y_holdout: {y_holdout.shape}")
