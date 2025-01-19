import numpy as np
import pandas as pd

# Load data yang sudah diproses
df = pd.read_csv("gold_usd_preprocessed.csv", index_col="time", parse_dates=True)

# Ambil fitur yang digunakan sebagai input model
features = ["open_M1", "high_M1", "low_M1", "close_M1", "RSI_M1", "EMA_M1", "MACD_M1", "Bollinger_Upper_M1", "Bollinger_Lower_M1",
            "open_M5", "high_M5", "low_M5", "close_M5", "RSI_M5", "EMA_M5", "MACD_M5", "Bollinger_Upper_M5", "Bollinger_Lower_M5",
            "open_M15", "high_M15", "low_M15", "close_M15", "RSI_M15", "EMA_M15", "MACD_M15", "Bollinger_Upper_M15", "Bollinger_Lower_M15"]

data = df[features].values

# Buat dataset dengan Sliding Window (120 candle sebelumnya sebagai input)
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps, 3])  # Harga Close M1 sebagai target
    return np.array(X), np.array(y)

n_steps = 180  # Gunakan 180 candle terakhir untuk prediksi
X, y = create_sequences(data, n_steps)

# Split Data (78% Training, 22% Testing)
split = int(len(X) * 0.78)
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# Simpan dataset
np.savez("lstm_dataset.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

print("✅ Dataset LSTM berhasil dibuat dan disimpan sebagai lstm_dataset.npz")
print(f"\nUkuran dataset:")
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test: {y_test.shape}")
