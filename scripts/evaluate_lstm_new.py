import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import matplotlib.pyplot as plt

# 1. Memuat Libraries dan Dependencies
# Sudah dilakukan di atas

# 2. Memuat Scaler dan Dataset Test
# Pastikan file 'lstm_dataset.npz', 'scaler_features.pkl', dan 'gold_usd_preprocessed.csv' berada di direktori yang sama dengan script ini

# Path ke file dataset, scaler, dan data preprocessed
dataset_path = "lstm_dataset.npz"
scaler_path = "scaler_features.pkl"
preprocessed_data_path = "gold_usd_preprocessed.csv"

# Memuat scaler
scaler_features = joblib.load(scaler_path)

# Memuat dataset
data = np.load(dataset_path)
X_test = data["X_test"]        # Shape: (samples, timesteps, features)
y_test = data["y_test"]        # Shape: (samples, features)

# Menampilkan bentuk data Test
print(f"✅ Bentuk data Test:")
print(f"X_test: {X_test.shape}")
print(f"y_test: {y_test.shape}")

# 3. Memuat Model yang Telah Dilatih
# Path ke model terbaik
model_path = "models/best_model.h5"  # Ganti jika menggunakan nama atau path yang berbeda

# Memuat model
model = load_model(model_path)
print(f"✅ Model '{model_path}' berhasil dimuat.")

# 4. Memuat Data Preprocessed untuk Mendapatkan Timestamps
df_preprocessed = pd.read_csv(preprocessed_data_path, index_col="time", parse_dates=True)
print(f"✅ Data preprocessed '{preprocessed_data_path}' berhasil dimuat.")

# 5. Mengaitkan y_test dengan Timestamps
# Asumsi selama pembuatan dataset:
# - Data dibagi menjadi 85% train, 10% test, 5% holdout
# - Menggunakan sliding window dengan n_steps = 250

n_steps = 250  # Jumlah langkah (candle) yang digunakan sebagai input

# Mendapatkan ukuran dataset
# Pastikan 'lstm_dataset.npz' berisi 'X_train', 'X_test', 'X_holdout', 'y_train', 'y_test', 'y_holdout'
if 'X_train' in data and 'X_test' in data and 'X_holdout' in data:
    train_size = len(data["X_train"])
    test_size = len(data["X_test"])
    holdout_size = len(data["X_holdout"])
    print(f"\nUkuran dataset:")
    print(f"Train samples: {train_size}")
    print(f"Test samples: {test_size}")
    print(f"Holdout samples: {holdout_size}")
else:
    # Jika 'X_train' tidak ada, asumsikan hanya 'X_test' dan 'y_test'
    train_size = 0
    test_size = len(X_test)
    holdout_size = 0
    print(f"\nUkuran dataset:")
    print(f"Test samples: {test_size}")

# Mendapatkan semua timestamps dari data preprocessed
all_timestamps = df_preprocessed.index

# Menghitung start index untuk test
# y_test[i] adalah data setelah X_test[i], jadi timestamp y_test[i] = timestamp X_test[i] + 1 step

# Compute the start index of y_test in the preprocessed data
# During dataset creation, sliding window was used, so y_test starts at (train_size) + n_steps
y_test_start_index = train_size + n_steps
y_test_end_index = y_test_start_index + test_size

# Memastikan indeks tidak melebihi panjang data
if y_test_end_index > len(all_timestamps):
    y_test_end_index = len(all_timestamps)
    y_test = y_test[:y_test_end_index - y_test_start_index]
    print(f"⚠️ Adjusted y_test_end_index to {y_test_end_index} to fit the data.")

# Mendapatkan timestamps untuk y_test
y_test_timestamps = all_timestamps[y_test_start_index:y_test_end_index]

# Memeriksa kesesuaian jumlah timestamps dan y_test
if len(y_test_timestamps) != len(y_test):
    print(f"⚠️ Jumlah timestamps ({len(y_test_timestamps)}) tidak sama dengan jumlah y_test ({len(y_test)}).")
    # Sesuaikan y_test dan timestamps jika diperlukan
    min_length = min(len(y_test_timestamps), len(y_test))
    y_test = y_test[:min_length]
    y_test_timestamps = y_test_timestamps[:min_length]

print(f"✅ Timestamps untuk y_test telah ditentukan.")

# 6. Inverse Transform Prediksi dan Aktual
# Karena semua fitur dinormalisasi menggunakan scaler yang sama, kita bisa menggunakan scaler_features untuk inverse transform
print("\n🔄 Melakukan inverse transform pada prediksi dan data aktual...")
y_pred_test = model.predict(X_test)
y_pred_test_inv = scaler_features.inverse_transform(y_pred_test)
y_test_inv = scaler_features.inverse_transform(y_test)
print("✅ Inverse transform selesai.")

# 7. Menghitung Metrik Evaluasi (MAE dan MSE)
# Mendefinisikan nama fitur sesuai dengan preprocessing
feature_names = [
    "open_M1", "high_M1", "low_M1", "close_M1", "RSI_M1", "EMA_M1", "MACD_M1", 
    "Bollinger_Upper_M1", "Bollinger_Lower_M1",
    "open_M5", "high_M5", "low_M5", "close_M5", "RSI_M5", "EMA_M5", "MACD_M5", 
    "Bollinger_Upper_M5", "Bollinger_Lower_M5",
    "open_M15", "high_M15", "low_M15", "close_M15", "RSI_M15", "EMA_M15", "MACD_M15", 
    "Bollinger_Upper_M15", "Bollinger_Lower_M15"
]

# Menghitung MAE dan MSE untuk setiap fitur
mae_test = mean_absolute_error(y_test_inv, y_pred_test_inv, multioutput='raw_values')
mse_test = mean_squared_error(y_test_inv, y_pred_test_inv, multioutput='raw_values')

# 8. Menampilkan Hasil Evaluasi
print("\n📊 Evaluasi Model pada Data Test:")
print(f"{'Fitur':<20} {'MAE':<15} {'MSE':<15}")
print("-" * 50)
for i, feature in enumerate(feature_names):
    print(f"{feature:<20} {mae_test[i]:<15.4f} {mse_test[i]:<15.4f}")

# 9. Visualisasi Hasil Prediksi vs Aktual dengan Timestamps
# Pilih beberapa fitur untuk divisualisasikan
features_to_plot = ["close_M1", "close_M5", "close_M15"]

for feature in features_to_plot:
    if feature in feature_names:
        idx = feature_names.index(feature)
        plt.figure(figsize=(12, 6))
        plt.plot(y_test_timestamps[:100], y_test_inv[:100, idx], label='Actual', marker='o')
        plt.plot(y_test_timestamps[:100], y_pred_test_inv[:100, idx], label='Predicted', marker='x')
        plt.title(f'Actual vs Predicted for {feature} (Test)')
        plt.xlabel('Waktu')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print(f"⚠️ Fitur '{feature}' tidak ditemukan dalam daftar fitur.")

# 10. Menyimpan Hasil Prediksi dan Aktual ke File CSV (Opsional)
# Ini berguna untuk analisis lebih lanjut atau dokumentasi
holdout_results_test = pd.DataFrame(y_test_inv, columns=feature_names, index=y_test_timestamps)
holdout_results_test_pred = pd.DataFrame(y_pred_test_inv, columns=[f"{name}_pred" for name in feature_names], index=y_test_timestamps)

# Gabungkan aktual dan prediksi
test_comparison = pd.concat([holdout_results_test, holdout_results_test_pred], axis=1)

# Simpan ke CSV
test_comparison.to_csv("test_predictions_with_timestamps.csv")
print("\n✅ Hasil prediksi dan aktual pada data Test telah disimpan sebagai 'test_predictions_with_timestamps.csv'.")
