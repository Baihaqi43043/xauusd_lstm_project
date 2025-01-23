import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import matplotlib.pyplot as plt

# 1. Memuat Libraries dan Dependencies
# Sudah dilakukan di atas

# 2. Memuat Scaler dan Dataset Holdout
# Pastikan file 'lstm_dataset.npz', 'scaler_features.pkl', dan 'gold_usd_preprocessed.csv' berada di direktori yang sama dengan script ini

# Path ke file dataset, scaler, dan data preprocessed
dataset_path = "lstm_dataset.npz"
scaler_path = "scaler_features.pkl"
preprocessed_data_path = "gold_usd_preprocessed.csv"

# Memuat scaler
scaler_features = joblib.load(scaler_path)

# Memuat dataset
data = np.load(dataset_path)
X_holdout = data["X_holdout"]  # Shape: (samples, timesteps, features)
y_holdout = data["y_holdout"]  # Shape: (samples, features)

# Menampilkan bentuk data Holdout
print(f"✅ Bentuk data Holdout:")
print(f"X_holdout: {X_holdout.shape}")
print(f"y_holdout: {y_holdout.shape}")

# 3. Memuat Model yang Telah Dilatih
# Path ke model terbaik
model_path = "models/best_model.h5"  # Ganti jika menggunakan nama atau path yang berbeda

# Memuat model
model = load_model(model_path)
print(f"✅ Model '{model_path}' berhasil dimuat.")

# 4. Memuat Data Preprocessed untuk Mendapatkan Timestamps
# Memuat data preprocessed untuk mendapatkan timestamp y_holdout
df_preprocessed = pd.read_csv(preprocessed_data_path, index_col="time", parse_dates=True)
print(f"✅ Data preprocessed '{preprocessed_data_path}' berhasil dimuat.")

# 5. Mengaitkan y_holdout dengan Timestamps
# Asumsi selama pembuatan dataset:
# - Data dibagi menjadi 85% train, 10% test, 5% holdout
# - Menggunakan sliding window dengan n_steps = 250

n_steps = 250  # Jumlah langkah (candle) yang digunakan sebagai input
total_samples = len(data["X_train"]) + len(data["X_test"]) + len(X_holdout)

train_size = len(data["X_train"])
test_size = len(data["X_test"])
holdout_size = len(X_holdout)

# Mendapatkan semua timestamps dari data preprocessed
all_timestamps = df_preprocessed.index

# Menghitung start index untuk holdout
# y_holdout[i] adalah data setelah X_holdout[i], jadi timestamp y_holdout[i] = timestamp X_holdout[i] + 1 step
holdout_start_index = train_size + test_size + n_steps
holdout_end_index = holdout_start_index + holdout_size

# Memastikan indeks tidak melebihi panjang data
if holdout_end_index > len(all_timestamps):
    holdout_end_index = len(all_timestamps)
    y_holdout = y_holdout[:holdout_end_index - holdout_start_index]
    print(f"⚠️ Adjusted holdout_end_index to {holdout_end_index} to fit the data.")

# Mendapatkan timestamps untuk y_holdout
y_holdout_timestamps = all_timestamps[holdout_start_index:holdout_end_index]

# Memeriksa kesesuaian jumlah timestamps dan y_holdout
if len(y_holdout_timestamps) != len(y_holdout):
    print(f"⚠️ Jumlah timestamps ({len(y_holdout_timestamps)}) tidak sama dengan jumlah y_holdout ({len(y_holdout)}).")
    # Sesuaikan y_holdout dan timestamps jika diperlukan
    min_length = min(len(y_holdout_timestamps), len(y_holdout))
    y_holdout = y_holdout[:min_length]
    y_holdout_timestamps = y_holdout_timestamps[:min_length]

print(f"✅ Timestamps untuk y_holdout telah ditentukan.")

# 6. Inverse Transform Prediksi dan Aktual
# Karena semua fitur dinormalisasi menggunakan scaler yang sama, kita bisa menggunakan scaler_features untuk inverse transform
print("\n🔄 Melakukan inverse transform pada prediksi dan data aktual...")
y_pred_holdout = model.predict(X_holdout)
y_pred_holdout_inv = scaler_features.inverse_transform(y_pred_holdout)
y_holdout_inv = scaler_features.inverse_transform(y_holdout)
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
mae_holdout = mean_absolute_error(y_holdout_inv, y_pred_holdout_inv, multioutput='raw_values')
mse_holdout = mean_squared_error(y_holdout_inv, y_pred_holdout_inv, multioutput='raw_values')

# 8. Menampilkan Hasil Evaluasi
print("\n📊 Evaluasi Model pada Data Holdout:")
print(f"{'Fitur':<20} {'MAE':<15} {'MSE':<15}")
print("-" * 50)
for i, feature in enumerate(feature_names):
    print(f"{feature:<20} {mae_holdout[i]:<15.4f} {mse_holdout[i]:<15.4f}")

# 9. Visualisasi Hasil Prediksi vs Aktual dengan Timestamps
# Pilih beberapa fitur untuk divisualisasikan
features_to_plot = ["close_M1", "close_M5", "close_M15"]

for feature in features_to_plot:
    if feature in feature_names:
        idx = feature_names.index(feature)
        plt.figure(figsize=(12, 6))
        plt.plot(y_holdout_timestamps[:100], y_holdout_inv[:100, idx], label='Actual', marker='o')
        plt.plot(y_holdout_timestamps[:100], y_pred_holdout_inv[:100, idx], label='Predicted', marker='x')
        plt.title(f'Actual vs Predicted for {feature} (Holdout)')
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
holdout_results = pd.DataFrame(y_holdout_inv, columns=feature_names, index=y_holdout_timestamps)
holdout_results_pred = pd.DataFrame(y_pred_holdout_inv, columns=[f"{name}_pred" for name in feature_names], index=y_holdout_timestamps)

# Gabungkan aktual dan prediksi
holdout_comparison = pd.concat([holdout_results, holdout_results_pred], axis=1)

# Simpan ke CSV
holdout_comparison.to_csv("holdout_predictions_with_timestamps.csv")
print("\n✅ Hasil prediksi dan aktual pada data Holdout telah disimpan sebagai 'holdout_predictions_with_timestamps.csv'.")
