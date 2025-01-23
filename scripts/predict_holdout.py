import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import matplotlib.pyplot as plt
import argparse
import os
import ta  # Import ta sebagai modul

def parse_arguments():
    parser = argparse.ArgumentParser(description="LSTM Multi-Feature Forecasting")
    parser.add_argument('--hours', type=int, default=1, help='Number of hours to predict (default: 1)')
    parser.add_argument('--input_file', type=str, default='gold_usd_preprocessed.csv', help='Path to the preprocessed input CSV file')
    parser.add_argument('--output_csv', type=str, default='external_predictions.csv', help='Path to save the prediction results')
    args = parser.parse_args()
    return args

def add_technical_indicators(df, timeframe):
    """
    Tambahkan indikator teknikal ke dataframe untuk timeframe tertentu.
    """
    # RSI
    df[f"RSI_{timeframe}"] = ta.momentum.RSIIndicator(close=df[f"close_{timeframe}"], window=14).rsi()
    
    # EMA
    df[f"EMA_{timeframe}"] = ta.trend.EMAIndicator(close=df[f"close_{timeframe}"], window=14).ema_indicator()
    
    # MACD
    macd = ta.trend.MACD(close=df[f"close_{timeframe}"])
    df[f"MACD_{timeframe}"] = macd.macd()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(close=df[f"close_{timeframe}"])
    df[f"Bollinger_Upper_{timeframe}"] = bollinger.bollinger_hband()
    df[f"Bollinger_Lower_{timeframe}"] = bollinger.bollinger_lband()

def preprocess_new_data(input_file, features, scaler):
    """
    Memuat dan memproses data baru untuk prediksi.
    """
    df_new = pd.read_csv(input_file, index_col="time", parse_dates=True)
    
    # Tambahkan indikator teknikal untuk timeframe M1, M5, M15
    for timeframe in ["M1", "M5", "M15"]:
        add_technical_indicators(df_new, timeframe)
    
    # Hapus nilai NaN akibat perhitungan indikator teknikal
    df_new.dropna(inplace=True)
    
    # Pastikan semua fitur ada
    missing_features = set(features) - set(df_new.columns)
    if missing_features:
        raise ValueError(f"Fitur berikut tidak ditemukan dalam data baru: {missing_features}")
    
    # Normalisasi data menggunakan scaler yang sama dengan training
    df_new_scaled = scaler.transform(df_new[features].values)
    
    return df_new_scaled, df_new.index

def create_sequence(data, n_steps):
    """
    Membuat sequences untuk prediksi.
    """
    X = []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
    return np.array(X)

def main():
    # 1. Mengambil Input dari Terminal untuk Jarak Waktu Prediksi
    args = parse_arguments()
    prediction_hours = args.hours
    input_file = args.input_file
    output_csv = args.output_csv
    
    # Asumsi data pada M1 (1 menit per langkah)
    steps_per_hour = 60  # 60 menit dalam 1 jam
    total_steps = prediction_hours * steps_per_hour
    
    print(f"\n📅 Anda telah memilih untuk memprediksi {prediction_hours} jam ke depan ({total_steps} langkah).")
    
    # 2. Memuat Model dan Scaler yang Telah Dilatih
    model_path = "models/best_model.h5"  # Ganti jika path berbeda
    scaler_path = "scaler_features.pkl"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file '{model_path}' tidak ditemukan.")
        return
    if not os.path.exists(scaler_path):
        print(f"❌ Scaler file '{scaler_path}' tidak ditemukan.")
        return
    
    # Memuat model
    model = load_model(model_path)
    print(f"✅ Model '{model_path}' berhasil dimuat.")
    
    # Memuat scaler
    scaler = joblib.load(scaler_path)
    print(f"✅ Scaler '{scaler_path}' berhasil dimuat.")
    
    # 3. Memuat dan Memproses Data Baru
    # Daftar semua fitur yang digunakan
    features = [
        "open_M1", "high_M1", "low_M1", "close_M1", "RSI_M1", "EMA_M1", "MACD_M1", 
        "Bollinger_Upper_M1", "Bollinger_Lower_M1",
        "open_M5", "high_M5", "low_M5", "close_M5", "RSI_M5", "EMA_M5", "MACD_M5", 
        "Bollinger_Upper_M5", "Bollinger_Lower_M5",
        "open_M15", "high_M15", "low_M15", "close_M15", "RSI_M15", "EMA_M15", "MACD_M15", 
        "Bollinger_Upper_M15", "Bollinger_Lower_M15"
    ]
    
    print("\n🔄 Memproses data baru...")
    try:
        data_new_scaled, data_new_dates = preprocess_new_data(input_file, features, scaler)
        print("✅ Data baru berhasil diproses dan dinormalisasi.")
    except Exception as e:
        print(f"❌ Error saat memproses data baru: {e}")
        return
    
    # 4. Membuat Prediksi dengan Jarak Waktu yang Ditentukan
    # Mengambil window terakhir untuk memulai prediksi
    n_steps = 250  # Jumlah candle sebelumnya yang digunakan sebagai input
    if len(data_new_scaled) < n_steps:
        print(f"❌ Data baru kurang dari {n_steps} langkah untuk membuat window.")
        return
    
    last_window = data_new_scaled[-n_steps:]  # Shape: (250, 27)
    
    # Inisialisasi list untuk menyimpan prediksi
    predictions = []
    
    # Melakukan prediksi secara rekursif
    current_window = last_window.copy()
    print("\n🔍 Mulai melakukan prediksi...")
    for step in range(total_steps):
        # Reshape untuk model input
        X_input = current_window.reshape((1, n_steps, len(features)))
        # Prediksi
        y_pred = model.predict(X_input, verbose=0)  # Shape: (1, features)
        # Simpan prediksi
        predictions.append(y_pred[0])
        # Update window dengan prediksi
        current_window = np.vstack((current_window[1:], y_pred[0]))
        if (step + 1) % 100 == 0 or (step + 1) == total_steps:
            print(f"✅ Prediksi {step + 1}/{total_steps} langkah selesai.")
    
    # Konversi prediksi ke numpy array
    predictions = np.array(predictions)  # Shape: (total_steps, 27)
    
    # 5. Inverse Transform Prediksi ke Skala Asli
    print("\n🔄 Melakukan inverse transform pada prediksi...")
    predictions_inv = scaler.inverse_transform(predictions)
    print("✅ Inverse transform selesai.")
    
    # 6. Menampilkan Hasil Prediksi
    # Membuat DataFrame untuk prediksi
    predicted_df = pd.DataFrame(predictions_inv, columns=features)
    
    # Membuat tanggal untuk prediksi
    last_date = data_new_dates[-1]
    date_range = pd.date_range(start=last_date + pd.Timedelta(minutes=1), periods=total_steps, freq='T')  # 'T' untuk menit
    
    predicted_df['time'] = date_range
    predicted_df.set_index('time', inplace=True)
    
    # 7. Visualisasi Hasil Prediksi dengan Axis Menampilkan Jam dan Tanggal
    features_to_plot = ["close_M1", "close_M5", "close_M15"]
    
    for feature in features_to_plot:
        if feature in features:
            plt.figure(figsize=(14, 7))
            plt.plot(predicted_df.index, predicted_df[feature], label='Predicted', marker='x')
            plt.title(f'Prediksi {prediction_hours} Jam ke Depan untuk {feature}')
            plt.xlabel('Waktu')
            plt.ylabel(feature)
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        else:
            print(f"⚠️ Fitur '{feature}' tidak ditemukan dalam daftar fitur.")
    
    # 8. (Opsional) Menyimpan Hasil Prediksi ke File CSV
    predicted_df.to_csv(output_csv)
    print(f"\n✅ Hasil prediksi telah disimpan sebagai '{output_csv}'.")
    
    # 9. (Opsional) Menampilkan Evaluasi jika Data Aktual Tersedia
    # Jika Anda memiliki data aktual untuk periode prediksi, Anda bisa menambahkan kode berikut:
    # pastikan untuk menyediakan data aktual yang sesuai dengan rentang waktu prediksi
    
    # Contoh (tidak akan dijalankan jika data aktual tidak tersedia):
    """
    actual_file = "actual_new_data.csv"
    if os.path.exists(actual_file):
        df_actual = pd.read_csv(actual_file, index_col="time", parse_dates=True)
        # Tambahkan indikator teknikal dan normalisasi seperti sebelumnya
        for timeframe in ["M1", "M5", "M15"]:
            add_technical_indicators(df_actual, timeframe)
        df_actual.dropna(inplace=True)
        actual_scaled = scaler.transform(df_actual[features].values)
        actual_pred_steps = len(predictions)
        actual_recent = actual_scaled[-actual_pred_steps:]
        actual_inv = scaler.inverse_transform(actual_recent)
        
        # Membuat DataFrame untuk aktual
        actual_df = pd.DataFrame(actual_inv, columns=features)
        actual_df['time'] = date_range
        actual_df.set_index('time', inplace=True)
        
        # Plot aktual vs prediksi
        for feature in features_to_plot:
            if feature in features:
                plt.figure(figsize=(14, 7))
                plt.plot(predicted_df.index, predicted_df[feature], label='Predicted', marker='x')
                plt.plot(actual_df.index, actual_df[feature], label='Actual', marker='o')
                plt.title(f'Actual vs Predicted {prediction_hours} Jam ke Depan untuk {feature}')
                plt.xlabel('Waktu')
                plt.ylabel(feature)
                plt.legend()
                plt.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()
            else:
                print(f"⚠️ Fitur '{feature}' tidak ditemukan dalam daftar fitur.")
    else:
        print("\n⚠️ Data aktual untuk periode prediksi tidak tersedia.")
    """

if __name__ == "__main__":
    main()
