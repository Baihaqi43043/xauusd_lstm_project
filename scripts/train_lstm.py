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

# Konfigurasi GPU
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

# Load dataset yang sudah diproses
data = np.load("lstm_dataset.npz")
X_train, X_test, X_holdout = data["X_train"], data["X_test"], data["X_holdout"]
y_train, y_test, y_holdout = data["y_train"], data["y_test"], data["y_holdout"]

# Pastikan y_train dan y_test memiliki bentuk yang sesuai untuk multi-output
print(f"\n✅ Bentuk data training: X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"✅ Bentuk data testing: X_test: {X_test.shape}, y_test: {y_test.shape}")
print(f"✅ Bentuk data holdout: X_holdout: {X_holdout.shape}, y_holdout: {y_holdout.shape}")

# Memuat scaler yang digunakan selama preprocessing
scaler_features = joblib.load("scaler_features.pkl")

# Menentukan jumlah fitur output
n_features_output = y_train.shape[1]  # Harus sama dengan jumlah fitur yang diprediksi

# Model LSTM untuk forecasting multi-feature
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    BatchNormalization(),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    BatchNormalization(),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(n_features_output)  # Menyesuaikan jumlah unit dengan jumlah fitur output
])

# Hyperparameters untuk OneCycle LR
BATCH_SIZE = 32
EPOCHS = 25
MAX_LR = 0.01
steps_per_epoch = len(X_train) // BATCH_SIZE

# Compile model dengan optimizer yang dioptimalkan
optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=['mae'])

# Callback untuk menyimpan model terbaik
checkpoint_path = "models/best_model.h5"
os.makedirs("models", exist_ok=True)
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

# Early stopping untuk menghentikan training jika tidak ada improvement
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Buat OneCycle learning rate scheduler
one_cycle = get_one_cycle_lr(EPOCHS, MAX_LR, steps_per_epoch)

# Train model dengan OneCycle LR
print("\nMemulai training model dengan OneCycle Learning Rate...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint, early_stopping, one_cycle],
    verbose=1
)

# Simpan model final
model.save("lstm_goldusd_paper.h5")

print("\n✅ Training selesai!")
print("✅ Model terbaik disimpan di models/best_model.h5")
print("✅ Model final disimpan sebagai lstm_goldusd_paper.h5")

# (Opsional) Plotting History
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss (MSE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('MAE Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()
