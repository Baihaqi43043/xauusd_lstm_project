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
# Memuat data preprocessed untuk mendapatkan timestamp y_test
df_preprocessed = pd.read_csv(preprocessed_data_path, index_col="time", parse_dates=True)
print(f"✅ Data preprocessed '{preprocessed_data_path}' berhasil dimuat.")

# 5. Menentukan Jarak Waktu dalam Jam dan Menghitung Stride
# Definisikan jarak waktu dalam jam
time_distance_hours = 1  # Ganti sesuai kebutuhan, misalnya 1 untuk 1 jam

# Menghitung stride dalam langkah (steps)
stride_steps = time_distance_hours * 60  # Asumsi data pada M1 (1 menit per langkah)

print(f"\n📅 Menggunakan jarak waktu {time_distance_hours} jam ({stride_steps} langkah).")

# 6. Mengiterasi Data Test dengan Stride dan Mengaitkan dengan Timestamps
# Inisialisasi list untuk menyimpan prediksi, aktual, dan timestamps
y_pred_selected = []
y_test_selected = []
timestamps_selected = []

# Mendapatkan semua timestamps dari data preprocessed
all_timestamps = df_preprocessed.index

# Menghitung start index untuk test
# Asumsi pembagian dataset:
# - Train: 85%
# - Test: 10%
# - Holdout: 5%
# - Menggunakan sliding window dengan n_steps = 250
n_steps = 250  # Jumlah langkah (candle) yang digunakan sebagai input

# Mendapatkan ukuran dataset
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

# Menghitung indeks awal dan akhir untuk test
y_test_start_index = train_size + n_steps
y_test_end_index = y_test_start_index + test_size

# Memastikan indeks tidak melebihi panjang data
if y_test_end_index > len(all_timestamps):
    y_test_end_index = len(all_timestamps)
    y_test = y_test[:y_test_end_index - y_test_start_index]
    print(f"⚠️ Adjusted y_test_end_index to {y_test_end_index} to fit the data.")

# Mendapatkan timestamps untuk y_test
y_test_timestamps_full = all_timestamps[y_test_start_index:y_test_end_index]

# Mengiterasi data test dengan stride_steps
print("\n🔍 Melakukan prediksi dengan jarak waktu yang ditentukan...")
for i in range(0, len(X_test), stride_steps):
    if i >= len(y_test):
        break
    y_actual = y_test[i]       # Shape: (features,)
    y_pred = y_test[i]         # Temporary placeholder

    # Mendapatkan prediksi
    X_window = X_test[i:i+1]  # Shape: (1, timesteps, features)
    y_pred = model.predict(X_window)  # Shape: (1, features)
    y_pred = y_pred[0]                # Shape: (features,)

    # Append prediksi dan aktual
    y_pred_selected.append(y_pred)
    y_test_selected.append(y_actual)

    # Append timestamp
    if i < len(y_test_timestamps_full):
        timestamps_selected.append(y_test_timestamps_full[i])
    else:
        # Jika tidak ada timestamp yang sesuai, gunakan NaT
        timestamps_selected.append(pd.NaT)

# Konversi list ke numpy array
y_pred_selected = np.array(y_pred_selected)
y_test_selected = np.array(y_test_selected)
timestamps_selected = pd.to_datetime(timestamps_selected)

print(f"✅ Prediksi selesai. Jumlah prediksi: {y_pred_selected.shape[0]}")

# 7. Inverse Transform Prediksi dan Aktual
print("\n🔄 Melakukan inverse transform pada prediksi dan data aktual...")
y_pred_selected_inv = scaler_features.inverse_transform(y_pred_selected)
y_test_selected_inv = scaler_features.inverse_transform(y_test_selected)
print("✅ Inverse transform selesai.")

# 8. Menghitung Metrik Evaluasi (MAE dan MSE)
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
mae_test = mean_absolute_error(y_test_selected_inv, y_pred_selected_inv, multioutput='raw_values')
mse_test = mean_squared_error(y_test_selected_inv, y_pred_selected_inv, multioutput='raw_values')

# 9. Menampilkan Hasil Evaluasi
print("\n📊 Evaluasi Model pada Data Test dengan Jarak Waktu:")
print(f"{'Fitur':<20} {'MAE':<15} {'MSE':<15}")
print("-" * 50)
for i, feature in enumerate(feature_names):
    print(f"{feature:<20} {mae_test[i]:<15.4f} {mse_test[i]:<15.4f}")

# 10. Visualisasi Hasil Prediksi vs Aktual dengan Timestamps
# Pilih beberapa fitur untuk divisualisasikan
features_to_plot = ["close_M1", "close_M5", "close_M15"]

for feature in features_to_plot:
    if feature in feature_names:
        idx = feature_names.index(feature)
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps_selected[:100], y_test_selected_inv[:100, idx], label='Actual', marker='o')
        plt.plot(timestamps_selected[:100], y_pred_selected_inv[:100, idx], label='Predicted', marker='x')
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

# 11. (Opsional) Menyimpan Hasil Prediksi dan Aktual ke File CSV
# Ini berguna untuk analisis lebih lanjut atau dokumentasi
print("\n🗂️ Menyimpan hasil prediksi dan aktual ke file CSV...")
test_results = pd.DataFrame(y_test_selected_inv, columns=feature_names, index=timestamps_selected)
test_results_pred = pd.DataFrame(y_pred_selected_inv, columns=[f"{name}_pred" for name in feature_names], index=timestamps_selected)

# Gabungkan aktual dan prediksi
test_comparison = pd.concat([test_results, test_results_pred], axis=1)

# Simpan ke CSV
test_comparison.to_csv("test_predictions_with_timestamps.csv")
print("✅ Hasil prediksi dan aktual pada data Test telah disimpan sebagai 'test_predictions_with_timestamps.csv'.")
