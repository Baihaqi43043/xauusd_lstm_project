import pandas as pd
import ta
from sklearn.preprocessing import MinMaxScaler
import joblib  # Untuk menyimpan scaler
import numpy as np

# 1. Memuat Libraries dan Dependencies
# Sudah dilakukan di atas

# 2. Memuat dan Mengolah Data
# Pastikan file 'gold_usd_combined.csv' berada di direktori yang sama dengan script ini
df = pd.read_csv("gold_usd_combined.csv", index_col="time", parse_dates=True)

# 3. Menambahkan Indikator Teknikal
# Fungsi untuk menambahkan indikator teknikal untuk berbagai timeframe
def add_technical_indicators(df, timeframe):
    # RSI
    df[f"RSI_{timeframe}"] = ta.momentum.RSIIndicator(df[f"close_{timeframe}"], window=14).rsi()
    
    # EMA
    df[f"EMA_{timeframe}"] = ta.trend.EMAIndicator(df[f"close_{timeframe}"], window=14).ema_indicator()
    
    # MACD
    macd = ta.trend.MACD(df[f"close_{timeframe}"])
    df[f"MACD_{timeframe}"] = macd.macd()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df[f"close_{timeframe}"])
    df[f"Bollinger_Upper_{timeframe}"] = bollinger.bollinger_hband()
    df[f"Bollinger_Lower_{timeframe}"] = bollinger.bollinger_lband()

# Tambahkan indikator teknikal untuk timeframe M1, M5, dan M15
for timeframe in ["M1", "M5", "M15"]:
    add_technical_indicators(df, timeframe)

# 4. Menghapus Nilai NaN
# Setelah menambahkan indikator teknikal, beberapa nilai mungkin NaN
df.dropna(inplace=True)

# 5. Normalisasi Data
# Daftar semua fitur yang akan digunakan sebagai input dan target
features = [
    "open_M1", "high_M1", "low_M1", "close_M1", "RSI_M1", "EMA_M1", "MACD_M1", 
    "Bollinger_Upper_M1", "Bollinger_Lower_M1",
    "open_M5", "high_M5", "low_M5", "close_M5", "RSI_M5", "EMA_M5", "MACD_M5", 
    "Bollinger_Upper_M5", "Bollinger_Lower_M5",
    "open_M15", "high_M15", "low_M15", "close_M15", "RSI_M15", "EMA_M15", "MACD_M15", 
    "Bollinger_Upper_M15", "Bollinger_Lower_M15"
]

# Inisialisasi scaler
scaler_features = MinMaxScaler(feature_range=(0, 1))

# Fit dan transformasi data
df[features] = scaler_features.fit_transform(df[features])

# Simpan scaler untuk digunakan nanti saat inverse transform
joblib.dump(scaler_features, "scaler_features.pkl")

# 6. Membuat Sequences dengan Sliding Window
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])  # Semua fitur pada timestep berikutnya sebagai target
    return np.array(X), np.array(y)

n_steps = 250  # Jumlah candle sebelumnya yang digunakan sebagai input

data_values = df[features].values
X, y = create_sequences(data_values, n_steps)

# 7. Membagi Dataset menjadi Train, Test, dan Holdout
total_samples = len(X)
train_size = int(total_samples * 0.85)
test_size = int(total_samples * 0.10)
holdout_size = total_samples - train_size - test_size

X_train, X_temp = X[:train_size], X[train_size:]
y_train, y_temp = y[:train_size], y[train_size:]

X_test, X_holdout = X_temp[:test_size], X_temp[test_size:]
y_test, y_holdout = y_temp[:test_size], y_temp[test_size:]

# Menampilkan ukuran dataset
print("✅ Dataset LSTM berhasil dibuat dan dibagi menjadi Train, Test, dan Holdout.")
print(f"\nUkuran dataset:")
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"X_holdout: {X_holdout.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test: {y_test.shape}")
print(f"y_holdout: {y_holdout.shape}")

# 8. Menyimpan Dataset dan Scaler
# Simpan dataset ke file NPZ
np.savez("lstm_dataset.npz", 
         X_train=X_train, X_test=X_test, X_holdout=X_holdout, 
         y_train=y_train, y_test=y_test, y_holdout=y_holdout)

# Simpan Data yang sudah dinormalisasi ke CSV (Opsional)
df.to_csv("gold_usd_preprocessed.csv")

print("\n✅ Preprocessing selesai! Data disimpan sebagai:")
print("- gold_usd_preprocessed.csv (data yang sudah dinormalisasi)")
print("- lstm_dataset.npz (dataset yang sudah dibagi)")
print("- scaler_features.pkl (scaler untuk semua fitur)")
print("\nContoh data yang sudah dinormalisasi:")
print(df.head())
