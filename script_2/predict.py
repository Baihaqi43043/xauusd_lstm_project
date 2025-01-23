# predict.py

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt
import argparse
import os
import ta

def parse_arguments():
    parser = argparse.ArgumentParser(description="LSTM Multi-Step Multi-Feature Forecasting")
    parser.add_argument('--steps', type=int, default=(60 * 24), help='Number of steps to predict (default: 60)')
    parser.add_argument('--input_file', type=str, default='gold_usd_preprocessed_multi_step.csv', help='Path to the preprocessed input CSV file')
    parser.add_argument('--output_csv', type=str, default='multi_step_predictions.csv', help='Path to save the prediction results')
    args = parser.parse_args()
    return args

def add_technical_indicators(df, timeframe):
    """
    Menambahkan indikator teknikal ke dataframe untuk timeframe tertentu.
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
    Memuat dan memproses data baru untuk prediksi multi-step.
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

def create_input_sequence(data, n_steps):
    """
    Membuat sequences untuk input model.
    """
    return data[-n_steps:]  # Mengambil window terakhir

def main():
    # 1. Mengambil Input dari Terminal untuk Jarak Waktu Prediksi
    args = parse_arguments()
    forecast_horizon = args.steps
    input_file = args.input_file
    output_csv = args.output_csv
    
    # Asumsi data pada M1 (1 menit per langkah)
    steps_per_hour = 60  # 60 menit dalam 1 jam
    total_steps = forecast_horizon  # Langkah yang diprediksi
    
    print(f"\n📅 Anda telah memilih untuk memprediksi {forecast_horizon} langkah ke depan.")
    
    # 2. Memuat Model dan Scaler yang Telah Dilatih
    model_path = "models/best_model_multi_step.h5"  # Path model terbaik
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
    
    # 4. Membuat Input Sequence
    n_steps = 250  # Jumlah candle sebelumnya yang digunakan sebagai input
    input_sequence = create_input_sequence(data_new_scaled, n_steps)
    
    # 5. Reshape Input untuk Model
    X_input = input_sequence.reshape((1, n_steps, len(features)))
    
    # 6. Melakukan Prediksi Multi-Step
    print("\n🔍 Melakukan prediksi multi-step...")
    y_pred_scaled = model.predict(X_input, verbose=0)  # Shape: (1, forecast_horizon, n_features)
    
    # 7. Inverse Transform Prediksi ke Skala Asli
    # Flatten y_pred_scaled untuk inverse_transform
    y_pred_scaled_flat = y_pred_scaled.reshape(-1, len(features))
    y_pred_inv = scaler.inverse_transform(y_pred_scaled_flat)
    y_pred_inv = y_pred_inv.reshape(forecast_horizon, len(features))
    
    print("✅ Inverse transform selesai.")
    
    # 8. Menyimpan Hasil Prediksi
    # Membuat DataFrame untuk prediksi
    predicted_df = pd.DataFrame(y_pred_inv, columns=features)
    
    # Membuat tanggal untuk prediksi
    last_date = data_new_dates[-1]
    date_range = pd.date_range(start=last_date + pd.Timedelta(minutes=1), periods=forecast_horizon, freq='T')  # 'T' untuk menit
    
    predicted_df['time'] = date_range
    predicted_df.set_index('time', inplace=True)
    
    # 9. Menampilkan Hasil Prediksi
    features_to_plot = ["close_M1", "close_M5", "close_M15"]
    
    for feature in features_to_plot:
        if feature in features:
            plt.figure(figsize=(14, 7))
            plt.plot(predicted_df.index, predicted_df[feature], label='Predicted', marker='x')
            plt.title(f'Prediksi {forecast_horizon} Langkah ke Depan untuk {feature}')
            plt.xlabel('Waktu')
            plt.ylabel(feature)
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        else:
            print(f"⚠️ Fitur '{feature}' tidak ditemukan dalam daftar fitur.")
    
    # 10. Menyimpan Hasil Prediksi ke File CSV
    predicted_df.to_csv(output_csv)
    print(f"\n✅ Hasil prediksi telah disimpan sebagai '{output_csv}'.")

if __name__ == "__main__":
    main()
