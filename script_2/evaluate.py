# evaluate.py

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import ta

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

def evaluate_model(model, X, y, scaler, features, forecast_horizon):
    """
    Mengevaluasi model dengan menghitung MAE dan MSE.
    """
    y_pred_scaled = model.predict(X, verbose=0)  # Shape: (samples, forecast_horizon, n_features)
    
    # Reshape untuk inverse_transform
    samples = y_pred_scaled.shape[0]
    y_pred_scaled_flat = y_pred_scaled.reshape(-1, len(features))
    y_pred_inv = scaler.inverse_transform(y_pred_scaled_flat).reshape(samples, forecast_horizon, len(features))
    
    y_true_scaled_flat = y.reshape(-1, len(features))
    y_true_inv = scaler.inverse_transform(y_true_scaled_flat).reshape(samples, forecast_horizon, len(features))
    
    # Hitung MAE dan MSE untuk setiap fitur dan setiap langkah
    mae = mean_absolute_error(y_true_inv.flatten(), y_pred_inv.flatten())
    mse = mean_squared_error(y_true_inv.flatten(), y_pred_inv.flatten())
    
    print(f"✅ Mean Absolute Error (MAE): {mae}")
    print(f"✅ Mean Squared Error (MSE): {mse}")
    
    return y_true_inv, y_pred_inv

def main():
    # 1. Parameter
    model_path = "models/best_model_multi_step.h5"
    scaler_path = "scaler_features.pkl"
    dataset_path = "lstm_dataset_multi_step.npz"  # Bisa menggunakan dataset holdout atau test
    features = [
        "open_M1", "high_M1", "low_M1", "close_M1", "RSI_M1", "EMA_M1", "MACD_M1", 
        "Bollinger_Upper_M1", "Bollinger_Lower_M1",
        "open_M5", "high_M5", "low_M5", "close_M5", "RSI_M5", "EMA_M5", "MACD_M5", 
        "Bollinger_Upper_M5", "Bollinger_Lower_M5",
        "open_M15", "high_M15", "low_M15", "close_M15", "RSI_M15", "EMA_M15", "MACD_M15", 
        "Bollinger_Upper_M15", "Bollinger_Lower_M15"
    ]
    n_steps = 250
    forecast_horizon = 60  # Sesuaikan dengan yang digunakan saat preprocessing dan training
    
    # 2. Memuat Model dan Scaler
    if not os.path.exists(model_path):
        print(f"❌ Model file '{model_path}' tidak ditemukan.")
        return
    if not os.path.exists(scaler_path):
        print(f"❌ Scaler file '{scaler_path}' tidak ditemukan.")
        return
    
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    print(f"✅ Model '{model_path}' dan scaler '{scaler_path}' berhasil dimuat.")
    
    # 3. Memuat Dataset yang Sudah Diproses
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset file '{dataset_path}' tidak ditemukan.")
        return
    
    data = np.load(dataset_path)
    X_holdout = data["X_holdout"]
    y_holdout = data["y_holdout"]
    print(f"✅ Bentuk data holdout: X_holdout: {X_holdout.shape}, y_holdout: {y_holdout.shape}")
    
    # 4. Evaluasi Model
    y_true_inv, y_pred_inv = evaluate_model(model, X_holdout, y_holdout, scaler, features, forecast_horizon)
    
    # 5. Visualisasi Hasil Evaluasi
    # Pilih sampel untuk visualisasi (misalnya, sampel pertama)
    sample_idx = 0  # Ganti sesuai kebutuhan
    for feature in ["close_M1", "close_M5", "close_M15"]:
        if feature in features:
            feature_idx = features.index(feature)
            plt.figure(figsize=(14, 7))
            plt.plot(range(forecast_horizon), y_true_inv[sample_idx, :, feature_idx], label='Actual', marker='o')
            plt.plot(range(forecast_horizon), y_pred_inv[sample_idx, :, feature_idx], label='Predicted', marker='x')
            plt.title(f'Evaluasi Prediksi Multi-Step untuk {feature}')
            plt.xlabel('Langkah ke Depan')
            plt.ylabel(feature)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            print(f"⚠️ Fitur '{feature}' tidak ditemukan dalam daftar fitur.")

if __name__ == "__main__":
    main()
