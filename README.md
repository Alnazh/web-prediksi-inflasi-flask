# Prediksi Inflasi Provinsi Jawa Barat – Aplikasi ANN dengan Flask

Aplikasi web berbasis Flask yang menggunakan **Artificial Neural Networks (ANN)**  
untuk memprediksi inflasi Y-on-Y (Year-on-Year) di 7 Kota &amp; Provinsi Jawa Barat.

---

## Struktur Proyek

```
inflasi_jabar/
├── app.py                  # Flask backend utama
├── train_model.py          # Script training model ANN
├── parse_data.py           # Script parsing CSV mentah
├── requirements.txt        # Dependensi Python
├── model_data/
│   ├── inflasi_clean.csv          # Dataset bersih (7 Kota + Provinsi Jawa Barat)
│   ├── ann_weights.weights.h5     # Bobot ANN terlatih (Keras)
│   ├── scaler_X.pkl               # MinMaxScaler fitur
│   ├── scaler_y.pkl               # MinMaxScaler target
│   ├── label_encoder.pkl          # LabelEncoder kota
│   ├── metrics.json        # Metrik evaluasi
│   ├── training_history.csv
│   └── test_results.csv
├── templates/
│   ├── base.html           # Layout dasar (sidebar + dark/light toggle)
│   ├── index.html          # Dashboard
│   ├── predict.html        # Halaman prediksi
│   ├── upload.html         # Upload dataset CSV
│   └── analisis.html       # Analisis model
└── static/
    └── css/
        └── theme.css       # Dasher Bootstrap theme
```

---

## Cara Menjalankan

### 1. Buat virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 2. Install dependensi
```bash
pip install -r requirements.txt
```

### 3. ⚠️ WAJIB — Latih ulang model sebelum pertama kali menjalankan
```bash
# Jalankan dari dalam folder inflasi_jabar/
python train_model.py
```
Script ini akan menghasilkan:
- `model_data/ann_weights.weights.h5` — bobot Keras ANN
- `model_data/scaler_X.pkl`, `scaler_y.pkl` — scaler yang disesuaikan data
- `model_data/label_encoder.pkl` — encoder kota
- `model_data/metrics.json`, `training_history.csv`, `test_results.csv`

### 4. Jalankan aplikasi
```bash
python app.py
```
Buka browser: **http://localhost:5000**

---

## Dataset
- **Sumber**: Inflasi Y-on-Y dari OpenData Jabar / BPS
- **Periode**: 2020 – 2026 (sebagian)
- **Cakupan**: 7 Kota + Agregat Provinsi Jawa Barat
- **Total rekaman**: ~1.056 data bulanan

## Arsitektur Model ANN
| Layer | Neuron | Aktivasi |
|-------|--------|----------|
| Input | 3 | – |
| Dense 1 | 64 | ReLU |
| Dropout | – | 0.2 |
| Dense 2 | 128 | ReLU |
| Dropout | – | 0.2 |
| Dense 3 | 64 | ReLU |
| Dense 4 | 32 | ReLU |
| Output | 1 | Linear |

- **Loss**: MSE | **Optimizer**: Adam
- **Split**: 70% Train / 15% Val / 15% Test
- **MAE Test**: ~0.47% | **RMSE Test**: ~0.63%

## Fitur Aplikasi
- **Dashboard** – statistik ringkasan, tren inflasi, prediksi cepat
- **Prediksi Inflasi** – input kota/tahun/bulan → prediksi ANN + grafik
- **Upload Dataset** – proses batch CSV, scatter plot aktual vs prediksi
- **Analisis Model** – training history, metrik, arsitektur ANN
- **Dark / Light Mode** – toggle tema Dasher Bootstrap

## Deployment (Heroku / Railway / Render)
```bash
# Procfile (buat file baru)
echo "web: gunicorn app:app" > Procfile
```

## Teknologi
- Python 3.10+, Flask 3, TensorFlow/Keras, Scikit-learn
- Bootstrap 5 + Dasher Theme (dark/light switch)
- Chart.js, Matplotlib

---
**Tugas 4 – Artificial Neural Networks**  
Mohammad Bayu Anggara, S.Kom., M.Kom. | Teknik Informatika – Universitas Bale Bandung
