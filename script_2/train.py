# train.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
import os
import math
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt

def get_one_cycle_lr(epochs, max_lr, steps_per_epoch, div_factor=25.):
    initial_learning_rate = max_lr / div_factor
    final_learning_rate = initial_learning_rate / 1000

    # Menghitung total steps
    total_steps = epochs * steps_per_epoch
    step_size = total_steps // 2  # Setengah siklus untuk naik, setengah untuk turun

    def one_cycle_lr(step):
        # Convert step ke dalam range [0,1]
        cycle = math.floor(1 + step / (2 * step_size))
        x = abs(step / step_size - 2 * cycle + 1)

        if step <= step_size:
            # Fase naik
            return initial_learning_rate + (max_lr - initial_learning_rate) * (1 - x)
        else:
            # Fase turun
            return final_learning_rate + (max_lr - final_learning_rate) * x

    return tf.keras.callbacks.LearningRateScheduler(one_cycle_lr)

def build_model(n_steps, n_features, forecast_horizon):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(n_steps, n_features)),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(100, return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),
        Dense(100, activation='relu'),
        Dense(forecast_horizon * n_features)  # Output: forecast_horizon * n_features
    ])
    
    model.add(tf.keras.layers.Reshape((forecast_horizon, n_features)))  # Mengubah output menjadi (forecast_horizon, n_features)
    
    return model

def main():
    # 1. Konfigurasi GPU
    print("Mengecek GPU yang tersedia...")
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        try:
            # Mengaktifkan memory growth untuk menghindari mengalokasikan semua memory GPU
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ Ditemukan {len(physical_devices)} GPU:")
            for gpu in physical_devices:
                print(f"- {gpu}")
        except RuntimeError as e:
            print(f"❌ Error saat mengkonfigurasi GPU: {e}")
    else:
        print("❌ Tidak ditemukan GPU, menggunakan CPU")
    
    # 2. Memuat Dataset yang Sudah Diproses
    dataset_path = "lstm_dataset_multi_step.npz"
    if not os.path.exists(dataset_path):
        print(f"❌ File dataset '{dataset_path}' tidak ditemukan. Pastikan Anda sudah menjalankan 'preprocess.py'.")
        return
    
    data = np.load(dataset_path)
    X_train, X_test, X_holdout = data["X_train"], data["X_test"], data["X_holdout"]
    y_train, y_test, y_holdout = data["y_train"], data["y_test"], data["y_holdout"]
    
    # Pastikan y_train dan y_test memiliki bentuk yang sesuai untuk multi-step
    print(f"\n✅ Bentuk data training: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"✅ Bentuk data testing: X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"✅ Bentuk data holdout: X_holdout: {X_holdout.shape}, y_holdout: {y_holdout.shape}")
    
    # 3. Memuat Scaler yang Digunakan Selama Preprocessing
    scaler_path = "scaler_features.pkl"
    if not os.path.exists(scaler_path):
        print(f"❌ Scaler file '{scaler_path}' tidak ditemukan. Pastikan Anda sudah menjalankan 'preprocess.py'.")
        return
    
    scaler_features = joblib.load(scaler_path)
    print(f"✅ Scaler '{scaler_path}' berhasil dimuat.")
    
    # 4. Menentukan Jumlah Fitur Output dan Forecast Horizon
    n_features_output = y_train.shape[2]  # Jumlah fitur yang diprediksi
    forecast_horizon = y_train.shape[1]  # Jumlah langkah ke depan
    
    # 5. Membangun Model
    n_steps = X_train.shape[1]
    n_features = X_train.shape[2]
    model = build_model(n_steps, n_features, forecast_horizon)
    model.summary()
    
    # 6. Hyperparameters untuk OneCycle LR
    BATCH_SIZE = 32
    EPOCHS = 50
    MAX_LR = 0.01
    steps_per_epoch = len(X_train) // BATCH_SIZE
    
    # 7. Compile Model dengan Optimizer yang Dioptimalkan
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=['mae'])
    
    # 8. Callback untuk Menyimpan Model Terbaik
    checkpoint_path = "models/best_model_multi_step.h5"
    os.makedirs("models", exist_ok=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    
    # 9. Early Stopping untuk Menghentikan Training jika Tidak Ada Improvement
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # 10. Buat OneCycle Learning Rate Scheduler
    one_cycle = get_one_cycle_lr(EPOCHS, MAX_LR, steps_per_epoch)
    
    # 11. Train Model dengan OneCycle LR
    print("\nMemulai training model dengan OneCycle Learning Rate...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint, early_stopping, one_cycle],
        verbose=1
    )
    
    # 12. Simpan Model Final
    final_model_path = "models/lstm_goldusd_paper_multi_step.h5"
    model.save(final_model_path)
    print(f"\n✅ Training selesai!")
    print(f"✅ Model terbaik disimpan di '{checkpoint_path}'")
    print(f"✅ Model final disimpan sebagai '{final_model_path}'")
    
    # 13. Plotting History
    # Plot Loss (MSE)
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss (MSE)')
    plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_over_epochs.png")
    plt.show()
    
    # Plot MAE
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('MAE Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("mae_over_epochs.png")
    plt.show()

if __name__ == "__main__":
    main()
