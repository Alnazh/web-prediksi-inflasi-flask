import os, io, json, base64, pickle, sqlite3
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from flask import Flask, render_template, request, jsonify, g

app = Flask(__name__)
app.jinja_env.globals['enumerate'] = enumerate

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE, 'model_data')
DB_PATH   = os.path.join(BASE, 'model_data', 'history.db')

# ── Load model ────────────────────────────────────────────────────────────────
def build_model():
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
model.load_weights(os.path.join(MODEL_DIR, 'ann_weights.weights.h5'))

with open(os.path.join(MODEL_DIR, 'scaler_X.pkl'),      'rb') as f: scaler_X = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'scaler_y.pkl'),      'rb') as f: scaler_y = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'rb') as f: le       = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'metrics.json'))           as f: metrics  = json.load(f)

df_clean     = pd.read_csv(os.path.join(MODEL_DIR, 'inflasi_clean.csv'))
hist_df      = pd.read_csv(os.path.join(MODEL_DIR, 'training_history.csv'))
test_results = pd.read_csv(os.path.join(MODEL_DIR, 'test_results.csv'))

CITIES = sorted(le.classes_.tolist())
MONTHS = ['Januari','Februari','Maret','April','Mei','Juni',
          'Juli','Agustus','September','Oktober','November','Desember']

# ── SQLite Database ───────────────────────────────────────────────────────────
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop('db', None)
    if db: db.close()

def init_db():
    os.makedirs(MODEL_DIR, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    # ── Auto-migrate: jika skema lama tidak punya kolom 'prediksi', drop & recreate ──
    try:
        cols = [r[1] for r in con.execute("PRAGMA table_info(prediction_history)").fetchall()]
        if cols and 'prediksi' not in cols:
            con.execute('DROP TABLE IF EXISTS prediction_history')
            con.commit()
    except Exception:
        pass
    con.execute('''
        CREATE TABLE IF NOT EXISTS prediction_history (
            id         INTEGER  PRIMARY KEY AUTOINCREMENT,
            kota       TEXT     NOT NULL,
            tahun      INTEGER  NOT NULL,
            bulan      TEXT     NOT NULL,
            bulan_num  INTEGER  NOT NULL,
            prediksi   REAL     NOT NULL,
            label      TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    con.commit()
    con.close()

init_db()

def save_prediction(kota, tahun, bulan_name, bulan_num, prediksi, label):
    db = get_db()
    db.execute(
        'INSERT INTO prediction_history (kota,tahun,bulan,bulan_num,prediksi,label) VALUES (?,?,?,?,?,?)',
        (kota, tahun, bulan_name, bulan_num, prediksi, label)
    )
    db.commit()

def get_history(limit=50):
    db = get_db()
    rows = db.execute(
        'SELECT * FROM prediction_history ORDER BY created_at DESC LIMIT ?', (limit,)
    ).fetchall()
    return [dict(r) for r in rows]

def get_stats():
    db = get_db()
    total  = db.execute('SELECT COUNT(*) FROM prediction_history').fetchone()[0]
    avg    = db.execute('SELECT AVG(prediksi) FROM prediction_history').fetchone()[0]
    top    = db.execute('SELECT kota, COUNT(*) as c FROM prediction_history GROUP BY kota ORDER BY c DESC LIMIT 1').fetchone()
    return {'total': total, 'avg': round(avg, 2) if avg else 0, 'top_kota': dict(top) if top else None}

# ── Helpers ───────────────────────────────────────────────────────────────────
def inflasi_label(val):
    if val < 0:
        return ('Deflasi', 'danger',
                'Harga barang secara umum turun. Bisa jadi tanda ekonomi sedang lesu.')
    elif val <= 1:
        return ('Sangat Rendah', 'info',
                'Inflasi sangat terkendali. Harga hampir tidak berubah dari tahun lalu.')
    elif val <= 3:
        return ('Normal / Sehat', 'success',
                'Tingkat ideal. Harga naik sedikit, daya beli masyarakat masih terjaga.')
    elif val <= 5:
        return ('Sedang / Waspada', 'warning',
                'Harga mulai terasa naik. Perlu lebih bijak dalam pengeluaran.')
    else:
        return ('Tinggi / Berbahaya', 'danger',
                'Harga naik signifikan. Perlu kebijakan pemerintah untuk mengendalikan inflasi.')

def predict_inflasi(kota, tahun, bulan_num):
    kota_enc = le.transform([kota])[0]
    X_in = scaler_X.transform([[kota_enc, tahun, bulan_num]])
    y_sc = model.predict(X_in, verbose=0)
    return float(scaler_y.inverse_transform(y_sc)[0][0])

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=110,
                facecolor='none', transparent=True)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64

# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    latest_year = int(df_clean[df_clean['Tahun'] <= 2025]['Tahun'].max())
    d_yr = df_clean[df_clean['Tahun'] == latest_year]
    avg_inflasi = round(d_yr['Inflasi'].mean(), 2)
    max_row = d_yr.loc[d_yr['Inflasi'].idxmax()]
    min_row = d_yr.loc[d_yr['Inflasi'].idxmin()]
    yearly = (df_clean[df_clean['Tahun'] <= 2025]
              .groupby('Tahun')['Inflasi'].mean().reset_index())
    city_trend = (df_clean[df_clean['Tahun'].isin([2024, 2025])]
                  .groupby(['Kota','Tahun'])['Inflasi'].mean().reset_index()
                  .pivot(index='Kota', columns='Tahun', values='Inflasi').reset_index())
    label, badge, desc = inflasi_label(avg_inflasi)
    db_stats = get_stats()
    return render_template('index.html',
        cities=CITIES, months=MONTHS, metrics=metrics,
        avg_inflasi=avg_inflasi, latest_year=latest_year,
        max_kota=max_row['Kota'], max_val=round(max_row['Inflasi'], 2),
        min_kota=min_row['Kota'], min_val=round(min_row['Inflasi'], 2),
        yearly=yearly.to_dict('records'),
        city_trend=city_trend.to_dict('records'),
        total_data=len(df_clean),
        avg_label=label, avg_badge=badge, avg_desc=desc,
        db_stats=db_stats
    )

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    result = None
    charts = {}
    extra  = {}
    if request.method == 'POST':
        kota      = request.form['kota']
        tahun     = int(request.form['tahun'])
        bulan_num = int(request.form['bulan'])
        pred_val  = predict_inflasi(kota, tahun, bulan_num)
        bulan_name = MONTHS[bulan_num - 1]
        label, badge, desc = inflasi_label(pred_val)

        # Save to DB
        save_prediction(kota, tahun, bulan_name, bulan_num, round(pred_val, 2), label)

        city_data = df_clean[df_clean['Kota'] == kota].sort_values(['Tahun','Bulan_Num'])
        pred_vals = [predict_inflasi(kota, tahun, m) for m in range(1, 13)]

        # ── Chart 1: Trend historis + prediksi ──
        fig1, ax1 = plt.subplots(figsize=(14, 5))
        ax1.set_facecolor('none'); fig1.patch.set_alpha(0)
        for yr, grp in city_data.groupby('Tahun'):
            g = grp.sort_values('Bulan_Num')
            alpha = 0.35 if yr < 2020 else 0.55
            ax1.plot(g['Bulan_Num'], g['Inflasi'], marker='o', markersize=2.5,
                     linewidth=1.3, alpha=alpha, label=str(yr))
        ax1.plot(range(1,13), pred_vals, color='#FF6B6B', linewidth=2.5,
                 marker='s', markersize=5, label=f'Prediksi {tahun}', zorder=5)
        ax1.axvline(x=bulan_num, color='orange', linestyle='--', linewidth=1.4, alpha=0.7)
        ax1.scatter([bulan_num], [pred_val], s=130, color='#FF6B6B', zorder=10)
        ax1.set_xticks(range(1,13))
        ax1.set_xticklabels([m[:3] for m in MONTHS], fontsize=10, color='#aaa')
        ax1.tick_params(colors='#aaa'); [sp.set_edgecolor('#444') for sp in ax1.spines.values()]
        ax1.set_xlabel('Bulan', color='#aaa'); ax1.set_ylabel('Inflasi Y-on-Y (%)', color='#aaa')
        ax1.set_title(f'Tren Historis & Prediksi ANN — {kota}', color='#ddd', pad=8)
        ax1.legend(fontsize=6, framealpha=0.15, labelcolor='#ccc', ncol=4)
        ax1.grid(True, alpha=0.12, color='#888')
        charts['trend'] = fig_to_b64(fig1)

        # ── all_preds untuk tabel peringkat (tanpa chart) ──
        all_preds = {c: predict_inflasi(c, tahun, bulan_num) for c in CITIES}

        # ── Chart 2: Seasonal pattern (rata2 per bulan historis) ──
        city_hist = df_clean[(df_clean['Kota'] == kota) & (df_clean['Tahun'] <= 2025)]
        seasonal = city_hist.groupby('Bulan_Num')['Inflasi'].mean()
        fig3, ax3 = plt.subplots(figsize=(14, 4.5))
        ax3.set_facecolor('none'); fig3.patch.set_alpha(0)
        bar_c2 = ['#FF6B6B' if i+1 == bulan_num else '#4BC0C0' for i in range(12)]
        ax3.bar(range(1,13), seasonal.values, color=bar_c2, alpha=0.8)
        ax3.set_xticks(range(1,13))
        ax3.set_xticklabels([m[:3] for m in MONTHS], fontsize=10, color='#aaa')
        ax3.tick_params(colors='#aaa'); [sp.set_edgecolor('#444') for sp in ax3.spines.values()]
        ax3.set_ylabel('Rata-rata Inflasi (%)', color='#aaa')
        ax3.set_title(f'Pola Musiman Inflasi {kota} (2015–2025)\n(Merah = bulan yang diprediksi)', color='#ddd', pad=8)
        ax3.grid(True, alpha=0.12, color='#888', axis='y')
        charts['seasonal'] = fig_to_b64(fig3)

        # ── Chart 4: Inflasi tahun ini vs tahun lalu (line perbandingan) ──
        prev_year = tahun - 1
        curr_preds = [predict_inflasi(kota, tahun, m) for m in range(1,13)]
        hist_prev  = city_hist[city_hist['Tahun'] == prev_year].sort_values('Bulan_Num')
        fig4, ax4 = plt.subplots(figsize=(14, 4.5))
        ax4.set_facecolor('none'); fig4.patch.set_alpha(0)
        ax4.plot(range(1,13), curr_preds, color='#FF6B6B', linewidth=2.5, marker='o',
                 markersize=4, label=f'Prediksi {tahun}')
        if len(hist_prev) == 12:
            ax4.plot(hist_prev['Bulan_Num'], hist_prev['Inflasi'], color='#4BC0C0',
                     linewidth=2, marker='s', markersize=3.5,
                     label=f'Aktual {prev_year}', linestyle='--', alpha=0.8)
        ax4.axhline(y=3, color='#FFC107', linestyle=':', linewidth=1.2, alpha=0.6, label='Batas Normal 3%')
        ax4.set_xticks(range(1,13))
        ax4.set_xticklabels([m[:3] for m in MONTHS], fontsize=10, color='#aaa')
        ax4.tick_params(colors='#aaa'); [sp.set_edgecolor('#444') for sp in ax4.spines.values()]
        ax4.set_ylabel('Inflasi (%)', color='#aaa')
        ax4.set_title(f'Prediksi {tahun} vs Aktual {prev_year} — {kota}', color='#ddd', pad=8)
        ax4.legend(fontsize=8, framealpha=0.2, labelcolor='#ccc')
        ax4.grid(True, alpha=0.12, color='#888')
        charts['vs_prev'] = fig_to_b64(fig4)

        # ── Extra stats ──
        rank_sorted = sorted(all_preds.items(), key=lambda x: x[1], reverse=True)
        rank_pos = next(i+1 for i, (c, _) in enumerate(rank_sorted) if c == kota)
        hist_this_month = city_hist[city_hist['Bulan_Num'] == bulan_num]['Inflasi']
        hist_mean = round(hist_this_month.mean(), 2) if len(hist_this_month) else None
        hist_max  = round(hist_this_month.max(), 2)  if len(hist_this_month) else None
        hist_min  = round(hist_this_month.min(), 2)  if len(hist_this_month) else None
        yoy_change = None
        if prev_year >= 2015:
            prev_data = city_hist[(city_hist['Tahun'] == prev_year) & (city_hist['Bulan_Num'] == bulan_num)]
            if len(prev_data):
                yoy_change = round(pred_val - float(prev_data['Inflasi'].values[0]), 2)
        latest_year_val = int(df_clean[df_clean['Tahun'] <= 2025]['Tahun'].max())
        city_avg_year = round(city_hist[city_hist['Tahun'] == latest_year_val]['Inflasi'].mean(), 2) \
                        if latest_year_val in city_hist['Tahun'].values else None
        extra = {
            'rank': rank_pos, 'total_cities': len(CITIES),
            'hist_mean': hist_mean, 'hist_max': hist_max, 'hist_min': hist_min,
            'yoy_change': yoy_change, 'prev_year': prev_year,
            'city_avg_year': city_avg_year, 'latest_year': latest_year_val,
            'all_preds': dict(rank_sorted),
        }

        result = {
            'kota': kota, 'tahun': tahun, 'bulan': bulan_name,
            'bulan_num': bulan_num, 'nilai': round(pred_val, 2),
            'label': label, 'badge': badge, 'desc': desc
        }

    return render_template('predict.html',
        cities=CITIES, months=MONTHS, metrics=metrics,
        result=result, charts=charts, extra=extra)


@app.route('/history')
def history():
    page     = int(request.args.get('page', 1))
    per_page = 20
    db       = get_db()
    total    = db.execute('SELECT COUNT(*) FROM prediction_history').fetchone()[0]
    offset   = (page - 1) * per_page
    rows = db.execute(
        'SELECT * FROM prediction_history ORDER BY created_at DESC LIMIT ? OFFSET ?',
        (per_page, offset)
    ).fetchall()
    records = [dict(r) for r in rows]
    stats   = get_stats()

    # Chart: prediksi per kota (bar)
    kota_counts = db.execute(
        'SELECT kota, COUNT(*) as c, AVG(prediksi) as avg_pred FROM prediction_history GROUP BY kota ORDER BY c DESC'
    ).fetchall()
    kota_counts = [dict(r) for r in kota_counts]

    # Chart: timeline prediksi terakhir 30
    timeline = db.execute(
        'SELECT created_at, prediksi, kota FROM prediction_history ORDER BY created_at DESC LIMIT 30'
    ).fetchall()
    timeline = [dict(r) for r in reversed(timeline)]

    return render_template('history.html',
        records=records, stats=stats, kota_counts=kota_counts,
        timeline=timeline, page=page, per_page=per_page, total=total)


@app.route('/komparasi')
def komparasi():
    tahun = int(request.args.get('tahun', 2025))
    bulan = int(request.args.get('bulan', 6))
    years_available = sorted(df_clean['Tahun'].unique().tolist())
    return render_template('komparasi.html',
        months=MONTHS, tahun=tahun, bulan=bulan,
        cities=CITIES, years_available=years_available)


@app.route('/api/komparasi')
def api_komparasi():
    """Heavy computation called async from frontend"""
    tahun = int(request.args.get('tahun', 2025))
    bulan = int(request.args.get('bulan', 6))

    rows = []
    for kota in CITIES:
        try:
            pred  = predict_inflasi(kota, tahun, bulan)
            hist_row = df_clean[
                (df_clean['Kota'] == kota) &
                (df_clean['Tahun'] == tahun) &
                (df_clean['Bulan_Num'] == bulan)
            ]
            aktual = round(float(hist_row['Inflasi'].values[0]), 2) if len(hist_row) else None
            label, badge, _ = inflasi_label(pred)
            rows.append({'kota': kota, 'prediksi': round(pred, 2),
                         'aktual': aktual, 'label': label, 'badge': badge})
        except Exception:
            pass
    rows.sort(key=lambda x: x['prediksi'], reverse=True)

    # Bar chart
    abbr   = [r['kota'].replace('Kota ','').replace('Kab. ','Kab ') for r in rows]
    preds  = [r['prediksi'] for r in rows]
    colors = ['#FF6B6B' if v > 5 else '#FF9F40' if v > 3 else '#4BC0C0' for v in preds]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.set_facecolor('none'); fig.patch.set_alpha(0)
    bars = ax.barh(abbr, preds, color=colors, alpha=0.85)
    for bar, val in zip(bars, preds):
        ax.text(bar.get_width()+0.04, bar.get_y()+bar.get_height()/2,
                f'{val:.2f}%', va='center', fontsize=8, color='#ccc')
    ax.set_xlabel('Prediksi Inflasi (%)', color='#aaa')
    ax.set_title(f'Peringkat Prediksi Inflasi — {MONTHS[bulan-1]} {tahun}', color='#ddd')
    ax.tick_params(colors='#aaa')
    [sp.set_edgecolor('#444') for sp in ax.spines.values()]
    ax.grid(True, alpha=0.12, color='#888', axis='x')
    chart = fig_to_b64(fig)

    # Heatmap all months
    heat = {}
    for kota in CITIES:
        heat[kota] = [round(predict_inflasi(kota, tahun, m), 2) for m in range(1, 13)]

    return jsonify({'rows': rows, 'chart': chart, 'heat': heat,
                    'tahun': tahun, 'bulan': bulan, 'bulan_name': MONTHS[bulan-1]})


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    preview, summary, error, chart = None, None, None, None
    if request.method == 'POST':
        f = request.files.get('csvfile')
        if f and f.filename.endswith('.csv'):
            try:
                content = f.read().decode('utf-8-sig')
                df_up   = pd.read_csv(io.StringIO(content))
                df_up.columns = [c.strip() for c in df_up.columns]
                col_map   = {c.lower(): c for c in df_up.columns}
                kota_col  = col_map.get('kota', col_map.get('wilayah'))
                tahun_col = col_map.get('tahun')
                bulan_col = col_map.get('bulan_num', col_map.get('bulan'))
                inf_col   = col_map.get('inflasi')
                if not all([kota_col, tahun_col, bulan_col, inf_col]):
                    error = 'Kolom tidak ditemukan. Pastikan CSV memiliki: Kota, Tahun, Bulan_Num, Inflasi'
                else:
                    df_up = df_up[[kota_col, tahun_col, bulan_col, inf_col]].copy()
                    df_up.columns = ['Kota', 'Tahun', 'Bulan_Num', 'Inflasi']
                    df_up = df_up.dropna()
                    df_up['Tahun'] = df_up['Tahun'].astype(int)
                    df_up['Bulan_Num'] = df_up['Bulan_Num'].astype(int)
                    preds = []
                    for _, row in df_up.iterrows():
                        try: p = predict_inflasi(str(row['Kota']), int(row['Tahun']), int(row['Bulan_Num']))
                        except: p = None
                        preds.append(round(p, 2) if p is not None else None)
                    df_up['Prediksi_ANN'] = preds
                    df_up['Error (%)']    = (df_up['Inflasi'] - df_up['Prediksi_ANN']).abs().round(3)
                    preview = df_up.head(20).to_dict('records')
                    summary = {'total': len(df_up), 'mae': round(df_up['Error (%)'].mean(), 4),
                               'valid_kota': df_up['Kota'].nunique()}
                    fig, ax = plt.subplots(figsize=(9, 4))
                    ax.set_facecolor('none'); fig.patch.set_alpha(0)
                    ax.scatter(df_up['Inflasi'], df_up['Prediksi_ANN'],
                               alpha=0.6, color='#4BC0C0', edgecolors='#2a8a8a', s=40)
                    mn = min(df_up['Inflasi'].min(), df_up['Prediksi_ANN'].min())
                    mx = max(df_up['Inflasi'].max(), df_up['Prediksi_ANN'].max())
                    ax.plot([mn,mx],[mn,mx],'--', color='#FF6B6B', linewidth=1.5, label='Prediksi Sempurna')
                    ax.set_xlabel('Nilai Aktual (%)', color='#aaa')
                    ax.set_ylabel('Prediksi ANN (%)', color='#aaa')
                    ax.set_title('Aktual vs Prediksi ANN', color='#ddd')
                    ax.tick_params(colors='#aaa')
                    [sp.set_edgecolor('#444') for sp in ax.spines.values()]
                    ax.legend(labelcolor='#ccc', framealpha=0.2)
                    ax.grid(True, alpha=0.15, color='#888')
                    chart = fig_to_b64(fig)
            except Exception as e:
                error = f'Error membaca file: {str(e)}'
        else:
            error = 'Harap unggah file CSV yang valid (.csv)'
    return render_template('upload.html',
        preview=preview, summary=summary, error=error, chart=chart, months=MONTHS)


@app.route('/analisis')
def analisis():
    fig1, ax1 = plt.subplots(figsize=(9, 4))
    ax1.set_facecolor('none'); fig1.patch.set_alpha(0)
    ax1.plot(hist_df['epoch'], hist_df['loss'],     color='#4BC0C0', label='Train Loss', linewidth=2)
    ax1.plot(hist_df['epoch'], hist_df['val_loss'], color='#FF6B6B', label='Val Loss', linestyle='--', linewidth=2)
    ax1.set_xlabel('Epoch', color='#aaa'); ax1.set_ylabel('MSE Loss', color='#aaa')
    ax1.set_title('Grafik Training & Validation Loss', color='#ddd')
    ax1.tick_params(colors='#aaa'); [sp.set_edgecolor('#444') for sp in ax1.spines.values()]
    ax1.legend(labelcolor='#ccc', framealpha=0.2); ax1.grid(True, alpha=0.15, color='#888')
    loss_chart = fig_to_b64(fig1)

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.set_facecolor('none'); fig2.patch.set_alpha(0)
    ax2.scatter(test_results['Aktual'], test_results['Prediksi'],
                alpha=0.6, color='#4BC0C0', edgecolors='#2a8a8a', s=40)
    mn = min(test_results['Aktual'].min(), test_results['Prediksi'].min())
    mx = max(test_results['Aktual'].max(), test_results['Prediksi'].max())
    ax2.plot([mn,mx],[mn,mx],'--', color='#FF6B6B', linewidth=1.5, label='Ideal')
    ax2.set_xlabel('Aktual (%)', color='#aaa'); ax2.set_ylabel('Prediksi (%)', color='#aaa')
    ax2.set_title('Aktual vs Prediksi ANN (Data Uji)', color='#ddd')
    ax2.tick_params(colors='#aaa'); [sp.set_edgecolor('#444') for sp in ax2.spines.values()]
    ax2.legend(labelcolor='#ccc', framealpha=0.2); ax2.grid(True, alpha=0.15, color='#888')
    scatter_chart = fig_to_b64(fig2)

    kota_avg = (df_clean[df_clean['Tahun'] <= 2025]
                .groupby('Kota')['Inflasi'].mean().sort_values(ascending=False).reset_index())
    fig3, ax3 = plt.subplots(figsize=(9, 4))
    ax3.set_facecolor('none'); fig3.patch.set_alpha(0)
    colors3 = ['#4BC0C0' if 'Kab' not in k else '#FF9F40' for k in kota_avg['Kota']]
    ax3.barh(kota_avg['Kota'], kota_avg['Inflasi'], color=colors3, alpha=0.85)
    ax3.set_xlabel('Rata-rata Inflasi (%)', color='#aaa')
    ax3.set_title('Rata-rata Inflasi per Kota/Kab (2015–2025)', color='#ddd')
    ax3.tick_params(colors='#aaa'); [sp.set_edgecolor('#444') for sp in ax3.spines.values()]
    ax3.grid(True, alpha=0.15, color='#888', axis='x')
    bar_chart = fig_to_b64(fig3)

    return render_template('analisis.html',
        metrics=metrics, loss_chart=loss_chart,
        scatter_chart=scatter_chart, bar_chart=bar_chart,
        hist_df=hist_df.tail(10).to_dict('records'))


@app.route('/panduan')
def panduan():
    return render_template('panduan.html', months=MONTHS, cities=CITIES, metrics=metrics)


# ── API ───────────────────────────────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.json
    try:
        val = predict_inflasi(data['kota'], int(data['tahun']), int(data['bulan']))
        label, badge, desc = inflasi_label(val)
        return jsonify({'success': True, 'prediksi': round(val, 2),
                        'kota': data['kota'], 'tahun': data['tahun'],
                        'bulan': MONTHS[int(data['bulan'])-1],
                        'label': label, 'badge': badge, 'desc': desc})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/history/delete/<int:id>', methods=['DELETE'])
def delete_history(id):
    db = get_db()
    db.execute('DELETE FROM prediction_history WHERE id=?', (id,))
    db.commit()
    return jsonify({'success': True})

@app.route('/api/history/clear', methods=['DELETE'])
def clear_history():
    db = get_db()
    db.execute('DELETE FROM prediction_history')
    db.commit()
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
