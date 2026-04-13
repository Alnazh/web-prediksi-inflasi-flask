import numpy as np
import pandas as pd
import pickle
import json
import os

# ── TF 2.16+ / Keras 3: gunakan 'keras' langsung, bukan 'tensorflow.keras' ──
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

os.makedirs('model_data', exist_ok=True)

# ── Load data bersih ─────────────────────────────────────────────────────────
df = pd.read_csv('model_data/inflasi_clean.csv')
df_train = df[df['Tahun'] <= 2025].copy()

# ── Encoding ─────────────────────────────────────────────────────────────────
le = LabelEncoder()
df_train['Kota_Enc'] = le.fit_transform(df_train['Kota'])

# Fitur: Kota (encoded), Tahun, Bulan_Num | Target: Inflasi
X = df_train[['Kota_Enc', 'Tahun', 'Bulan_Num']].values
y = df_train['Inflasi'].values.reshape(-1, 1)

# ── Normalisasi ───────────────────────────────────────────────────────────────
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# ── Split: 70% Train / 15% Val / 15% Test ────────────────────────────────────
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y_scaled, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")


# ── Arsitektur ANN (Feedforward Neural Network) ───────────────────────────────
def build_model():
    """Definisi arsitektur — HARUS IDENTIK dengan yang di app.py"""
    m = Sequential([
        Dense(64,  activation='relu', input_shape=(3,)),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64,  activation='relu'),
        Dense(32,  activation='relu'),
        Dense(1,   activation='linear')
    ])
    m.compile(optimizer='adam', loss='mse')
    return m


model = build_model()
model.summary()

# ── Training ──────────────────────────────────────────────────────────────────
early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=300,
    batch_size=16,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=1
)

# ── Evaluasi ──────────────────────────────────────────────────────────────────
y_pred_sc   = model.predict(X_test, verbose=0)
y_pred_real = scaler_y.inverse_transform(y_pred_sc)
y_test_real = scaler_y.inverse_transform(y_test)

mae  = float(np.mean(np.abs(y_test_real - y_pred_real)))
rmse = float(np.sqrt(np.mean((y_test_real - y_pred_real) ** 2)))
mse  = float(np.mean((y_test_real - y_pred_real) ** 2))
print(f"\nTest MAE : {mae:.4f}%")
print(f"Test RMSE: {rmse:.4f}%")

# ── Simpan weights (universal, tidak terikat versi Keras) ────────────────────
model.save_weights('model_data/ann_weights.weights.h5')

# ── Simpan artefak lain ───────────────────────────────────────────────────────
with open('model_data/scaler_X.pkl', 'wb') as f: pickle.dump(scaler_X, f)
with open('model_data/scaler_y.pkl', 'wb') as f: pickle.dump(scaler_y, f)
with open('model_data/label_encoder.pkl', 'wb') as f: pickle.dump(le, f)

hist = history.history
pd.DataFrame({
    'epoch':    range(1, len(hist['loss']) + 1),
    'loss':     hist['loss'],
    'val_loss': hist['val_loss'],
    'mae':      hist['loss'],
    'val_mae':  hist['val_loss'],
}).to_csv('model_data/training_history.csv', index=False)

pd.DataFrame({
    'Aktual':   y_test_real.flatten(),
    'Prediksi': y_pred_real.flatten()
}).to_csv('model_data/test_results.csv', index=False)

with open('model_data/metrics.json', 'w') as f:
    json.dump({
        'mae':            round(mae, 4),
        'rmse':           round(rmse, 4),
        'mse':            round(mse, 4),
        'train_samples':  int(X_train.shape[0]),
        'val_samples':    int(X_val.shape[0]),
        'test_samples':   int(X_test.shape[0]),
        'epochs_trained': len(hist['loss'])
    }, f)

print("\nSemua artefak berhasil disimpan!")
print(f"Kota tersedia: {list(le.classes_)}")
