import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

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
X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]

# Bentuk data
print(f"\n✅ Bentuk data training: {X_train.shape}")
print(f"✅ Bentuk data testing: {X_test.shape}")

# Model LSTM berdasarkan paper
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)
])

# Compile model dengan optimizer yang dioptimalkan
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
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
    patience=5,
    restore_best_weights=True
)

# Train model dengan batch size yang lebih besar untuk GPU
print("\nMemulai training model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=64,  # Batch size lebih besar untuk GPU
    callbacks=[checkpoint, early_stopping],
    verbose=1
)

# Simpan model final
model.save("lstm_goldusd_paper.h5")

print("\n✅ Training selesai!")
print("✅ Model terbaik disimpan di models/best_model.h5")
print("✅ Model final disimpan sebagai lstm_goldusd_paper.h5")
