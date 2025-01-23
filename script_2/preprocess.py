# preprocess.py

import pandas as pd
import ta
import numpy as np
import os

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
    
    # ADX (Average Directional Index)
    adx = ta.trend.ADXIndicator(high=df[f"high_{timeframe}"], low=df[f"low_{timeframe}"], close=df[f"close_{timeframe}"], window=14)
    df[f"ADX_{timeframe}"] = adx.adx()
    df[f"DI+_{timeframe}"] = adx.adx_pos()
    df[f"DI-_{timeframe}"] = adx.adx_neg()
    
    # SMA (Simple Moving Average) sebagai indikator tren tambahan
    sma = ta.trend.SMAIndicator(close=df[f"close_{timeframe}"], window=50)
    df[f"SMA_{timeframe}"] = sma.sma_indicator()

def create_sequences(data, n_steps, forecast_horizon):
    """
    Membuat sequences untuk multi-step forecasting.
    """
    X, y = [], []
    for i in range(len(data) - n_steps - forecast_horizon + 1):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps:i + n_steps + forecast_horizon])
    return np.array(X), np.array(y)

def main():
    # 1. Memuat Dataset Gabungan (M1, M5, M15)
    raw_data_path = "gold_usd_combined.csv"
    if not os.path.exists(raw_data_path):
        print(f"❌ File dataset mentah '{raw_data_path}' tidak ditemukan.")
        return
    
    df = pd.read_csv(raw_data_path, index_col="time", parse_dates=True)
    print("✅ Dataset mentah berhasil dimuat.")
    
    # 2. Menambahkan Indikator Teknikal untuk Timeframes M1, M5, M15
    for timeframe in ["M1", "M5", "M15"]:
        add_technical_indicators(df, timeframe)
        print(f"✅ Indikator teknikal untuk timeframe {timeframe} berhasil ditambahkan.")
    
    # 3. Menghapus Nilai NaN akibat Perhitungan Indikator Teknikal
    df.dropna(inplace=True)
    print("✅ Nilai NaN akibat perhitungan indikator teknikal telah dihapus.")
    
    # 4. Menentukan Daftar Fitur yang Akan Digunakan
    features = [
        "open_M1", "high_M1", "low_M1", "close_M1", "RSI_M1", "EMA_M1", "MACD_M1", 
        "Bollinger_Upper_M1", "Bollinger_Lower_M1",
        "ADX_M1", "DI+_M1", "DI-_M1", "SMA_M1",  # Indikator Tren untuk M1
        "open_M5", "high_M5", "low_M5", "close_M5", "RSI_M5", "EMA_M5", "MACD_M5", 
        "Bollinger_Upper_M5", "Bollinger_Lower_M5",
        "ADX_M5", "DI+_M5", "DI-_M5", "SMA_M5",  # Indikator Tren untuk M5
        "open_M15", "high_M15", "low_M15", "close_M15", "RSI_M15", "EMA_M15", "MACD_M15", 
        "Bollinger_Upper_M15", "Bollinger_Lower_M15",
        "ADX_M15", "DI+_M15", "DI-_M15", "SMA_M15"  # Indikator Tren untuk M15
    ]
    
    # 5. Memastikan Semua Fitur Tersedia
    for feature in features:
        if feature not in df.columns:
            raise ValueError(f"Fitur '{feature}' tidak ditemukan dalam data CSV.")
    
    # 6. Menyiapkan Data untuk Pembuatan Sequences
    data = df[features].values
    print("✅ Data siap untuk pembuatan sequences.")
    
    # 7. Membuat Sequences dengan Sliding Window untuk Multi-Step Forecasting
    n_steps = 250  # Jumlah candle sebelumnya yang digunakan sebagai input
    forecast_horizon = 60  # Jumlah langkah ke depan yang diprediksi (misalnya, 60 menit)
    
    X, y = create_sequences(data, n_steps, forecast_horizon)
    print("✅ Sequences untuk multi-step forecasting berhasil dibuat.")
    
    # 8. Membagi Dataset menjadi Train, Test, dan Holdout
    total_samples = len(X)
    train_size = int(total_samples * 0.85)
    test_size = int(total_samples * 0.10)
    holdout_size = total_samples - train_size - test_size
    
    X_train, X_temp = X[:train_size], X[train_size:]
    y_train, y_temp = y[:train_size], y[train_size:]
    
    X_test, X_holdout = X_temp[:test_size], X_temp[test_size:]
    y_test, y_holdout = y_temp[:test_size], y_temp[test_size:]
    
    # 9. Menampilkan Ukuran Dataset
    print("✅ Dataset berhasil dibagi menjadi Train, Test, dan Holdout.")
    print(f"\nUkuran dataset:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")
    print(f"X_holdout: {X_holdout.shape}")
    print(f"y_holdout: {y_holdout.shape}")
    
    # 10. Menyimpan Dataset
    dataset_path = "lstm_dataset_multi_step.npz"
    np.savez(dataset_path, 
             X_train=X_train, X_test=X_test, X_holdout=X_holdout, 
             y_train=y_train, y_test=y_test, y_holdout=y_holdout)
    print(f"✅ Dataset disimpan sebagai '{dataset_path}'.")
    
    # 11. Menyimpan Data yang Sudah Diproses ke CSV (Opsional)
    preprocessed_csv_path = "gold_usd_preprocessed_multi_step.csv"
    df.to_csv(preprocessed_csv_path)
    print(f"✅ Data yang sudah diproses disimpan sebagai '{preprocessed_csv_path}'.")
    
    # 12. Contoh Data yang Sudah Diproses
    print("\nContoh data yang sudah diproses:")
    print(df.head())

if __name__ == "__main__":
    main()
