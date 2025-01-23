import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load model
model = tf.keras.models.load_model("lstm_goldusd_paper.h5")
# model = tf.keras.models.load_model("models/best_model.h5")

# Load dataset testing
data = np.load("lstm_dataset.npz")
X_test, y_test = data["X_test"], data["y_test"]

# Load data asli untuk mendapatkan range harga sebenarnya
df_original = pd.read_csv("gold_usd_combined.csv", parse_dates=['time'])
df_original.set_index('time', inplace=True)
close_m5_values = df_original[["close_M5"]].values

# Load scaler
scaler_close = MinMaxScaler(feature_range=(0, 1))
scaler_close.fit(close_m5_values)  # Fit scaler dengan data asli
scaler_features = joblib.load("scaler_features.pkl")

# Prediksi harga
y_pred = model.predict(X_test)

# De-normalisasi hasil prediksi dan harga aktual
y_test_denorm = scaler_close.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_denorm = scaler_close.inverse_transform(y_pred.reshape(-1, 1)).flatten()

# Hitung Error menggunakan harga yang sudah di-denormalisasi
mse = mean_squared_error(y_test_denorm, y_pred_denorm)
mae = mean_absolute_error(y_test_denorm, y_pred_denorm)

print(f"✅ Mean Squared Error (MSE): {mse:.2f}")
print(f"✅ Mean Absolute Error (MAE): {mae:.2f}")
print(f"✅ Root Mean Squared Error (RMSE): {np.sqrt(mse):.2f}")

# Ambil tanggal untuk test set (30% terakhir dari data)
test_dates = df_original.index[-len(y_test_denorm):]

# Plot hasil prediksi vs data asli dengan format harga yang lebih baik
plt.figure(figsize=(15, 7))
plt.plot(test_dates, y_test_denorm, label="Harga Aktual", color='blue', linewidth=2)
plt.plot(test_dates, y_pred_denorm, label="Harga Prediksi", color='red', linestyle="dashed", linewidth=2)

# Konfigurasi plot dengan format harga yang lebih baik
plt.xlabel("Tahun")
plt.ylabel("Harga XAU/USD (USD)")
plt.title("Prediksi vs Harga Aktual XAU/USD - Model LSTM")
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)

# Format sumbu Y untuk menampilkan harga dalam USD
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.2f}'))

# Format sumbu X untuk menampilkan tahun
plt.gcf().autofmt_xdate()  # Rotasi dan format otomatis untuk tanggal

# Mengatur margin plot agar label tidak terpotong
plt.tight_layout()

plt.show()

# Tampilkan beberapa sampel prediksi dengan format harga USD
print("\nSampel hasil prediksi:")
print("Tanggal            |     Aktual    |    Prediksi    |   Selisih")
print("-" * 65)
for i in range(5):  # Tampilkan 5 sampel pertama
    diff = abs(y_test_denorm[i] - y_pred_denorm[i])
    print(f"{test_dates[i].strftime('%Y-%m-%d %H:%M')} | ${y_test_denorm[i]:11,.2f} | ${y_pred_denorm[i]:11,.2f} | ${diff:10,.2f}")

# Tampilkan range harga untuk verifikasi
print("\nRange harga:")
print(f"Min Aktual: ${y_test_denorm.min():.2f}")
print(f"Max Aktual: ${y_test_denorm.max():.2f}")
print(f"Min Prediksi: ${y_pred_denorm.min():.2f}")
print(f"Max Prediksi: ${y_pred_denorm.max():.2f}")
