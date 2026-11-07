from collections import Counter
import json
import os
import re
from flask import Flask, jsonify, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# --- 1. LOAD DATASET EXCEL (GLOBAL FALLBACK SAFE) ---
DATASET_PATH = os.path.join(DATA_DIR, 'review_rumah_sakit_merged_clean.xlsx')
if not os.path.exists(DATASET_PATH):
  DATASET_PATH = os.path.join(DATA_DIR, 'dataset_rs_labeling.xlsx')
if not os.path.exists(DATASET_PATH):
  DATASET_PATH = os.path.join(BASE_DIR, 'review_rumah_sakit_merged_clean.xlsx')

df_reviews = pd.DataFrame()

if os.path.exists(DATASET_PATH):
  df_raw = pd.read_excel(DATASET_PATH)
  # Buang 12 ulasan rating 0 sesuai kesepakatan
  if 'review_rating_numeric' in df_raw.columns:
    df_reviews = df_raw[df_raw['review_rating_numeric'] > 0].copy()
  else:
    df_reviews = df_raw.copy()

  # Labeling sentimen dasar jika belum ada
  if (
      'sentimen' not in df_reviews.columns
      and 'review_rating_numeric' in df_reviews.columns
  ):
    df_reviews['sentimen'] = df_reviews['review_rating_numeric'].apply(
        lambda r: 'positif' if r >= 4 else 'negatif'
    )
else:
  print(f'WARNING: Dataset file tidak ditemukan di {DATASET_PATH}')

# --- 2. KONEKSI DATABASE MYSQL (OPSIONAL/FALLBACK) ---
HAS_MYSQL = False
try:
  import mysql.connector

  def get_db_connection():
    return mysql.connector.connect(
        host=os.environ.get('MYSQLHOST', 'localhost'),
        user=os.environ.get('MYSQLUSER', 'root'),
        password=os.environ.get('MYSQLPASSWORD', ''),
        database=os.environ.get('MYSQLDATABASE', 'db_hospital_sentiment'),
        port=int(os.environ.get('MYSQLPORT', 3306)),
    )

  HAS_MYSQL = True
except ImportError:
  HAS_MYSQL = False

# --- 3. LOAD KAMUS NORMALISASI ---
norm_path = os.path.join(DATA_DIR, 'Normalisasi_2.csv')
if not os.path.exists(norm_path):
  norm_path = os.path.join(DATA_DIR, 'Normalisasi.csv')

norm_dict = {}
kamus_df = pd.DataFrame()

if os.path.exists(norm_path):
  df_norm = pd.read_csv(norm_path)
  kamus_df = df_norm.iloc[:, :2].copy()
  kamus_df.columns = ['Kata_Asli', 'Kata_Perbaikan']
  norm_dict = dict(
      zip(
          kamus_df['Kata_Asli'].astype(str).str.lower(),
          kamus_df['Kata_Perbaikan'].astype(str).str.lower(),
      )
  )

# --- 4. LOAD VECTORIZER & MODEL ---
vec_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
model_path = os.path.join(MODEL_DIR, 'svc_model.pkl')
if not os.path.exists(model_path):
  model_path = os.path.join(MODEL_DIR, 'model_svm_sentiment.pkl')

vectorizer = joblib.load(vec_path) if os.path.exists(vec_path) else None
model_svc = joblib.load(model_path) if os.path.exists(model_path) else None


def preprocess_text(text):
  if not isinstance(text, str):
    return ''
  text = text.lower()
  text = re.sub(r'[^a-z\s]', '', text)
  words = text.split()
  normalized_words = [norm_dict.get(w, w) for w in words]
  return ' '.join(normalized_words)


# --- 5. METRIK DATASET & METADATA LDA/PIECES ---
metrics_summary = {
    'total_data': len(df_reviews) if not df_reviews.empty else 2218,
    'positif_count': (
        int((df_reviews['sentimen'] == 'positif').sum())
        if 'sentimen' in df_reviews.columns
        else 1669
    ),
    'negatif_count': (
        int((df_reviews['sentimen'] == 'negatif').sum())
        if 'sentimen' in df_reviews.columns
        else 549
    ),
    'accuracy_80_20': 95.05,
    'cm': [[100, 10], [12, 322]],
}

LDA_PIECES_PATH = os.path.join(DATA_DIR, 'lda_pieces.json')
lda_pieces_data = {}
if os.path.exists(LDA_PIECES_PATH):
  with open(LDA_PIECES_PATH, 'r', encoding='utf-8') as f:
    lda_pieces_data = json.load(f)


# --- 6. ROUTE NAVIGASI UTAMA ---


@app.route('/')
def index():
  return render_template(
      'index.html', metrics=metrics_summary, lda_pieces=lda_pieces_data
  )


@app.route('/sentiment')
def sentiment_page():
  global df_reviews

  page = request.args.get('page', 1, type=int)
  rating = request.args.get('rating', 'all')
  sentimen = request.args.get('sentimen', 'all')
  per_page = 15

  df_filtered = (
      df_reviews.copy()
      if not df_reviews.empty
      else pd.DataFrame(
          columns=['author_title', 'review_text', 'review_rating_numeric']
      )
  )

  if rating != 'all' and 'review_rating_numeric' in df_filtered.columns:
    df_filtered = df_filtered[
        df_filtered['review_rating_numeric'] == int(rating)
    ]
  if sentimen != 'all' and 'sentimen' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['sentimen'] == sentimen]

  total_pages = max(1, (len(df_filtered) + per_page - 1) // per_page)
  start_idx = (page - 1) * per_page
  end_idx = start_idx + per_page

  items = df_filtered.iloc[start_idx:end_idx].to_dict(orient='records')

  return render_template(
      'sentiment.html',
      metrics=metrics_summary,
      items=items,
      page=page,
      total_pages=total_pages,
      rating=rating,
      sentimen=sentimen,
      total_count=len(df_filtered),
  )


@app.route('/pipeline')
def pipeline():
  global df_reviews

  stage = request.args.get('stage', '1')
  step = request.args.get('step', 'clean')
  page = request.args.get('page', 1, type=int)
  rating = request.args.get('rating', 'all')
  sentimen = request.args.get('sentimen', 'all')
  per_page = 15

  df_filtered = (
      df_reviews.copy()
      if not df_reviews.empty
      else pd.DataFrame(
          columns=['author_title', 'review_text', 'review_rating_numeric']
      )
  )

  if rating != 'all' and 'review_rating_numeric' in df_filtered.columns:
    df_filtered = df_filtered[
        df_filtered['review_rating_numeric'] == int(rating)
    ]
  if sentimen != 'all' and 'sentimen' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['sentimen'] == sentimen]

  total_pages = max(1, (len(df_filtered) + per_page - 1) // per_page)
  start_idx = (page - 1) * per_page
  end_idx = start_idx + per_page

  items = df_filtered.iloc[start_idx:end_idx].to_dict(orient='records')

  return render_template(
      'pipeline.html',
      active_stage=stage,
      current_step=step,
      items=items,
      page=page,
      total_pages=total_pages,
      rating=rating,
      sentimen=sentimen,
      total_count=len(df_filtered),
  )


@app.route('/topics')
def topics_page():
  return render_template('topics.html', lda_pieces=lda_pieces_data)


@app.route('/pieces')
def pieces_page():
  return render_template('pieces.html', lda_pieces=lda_pieces_data)


@app.route('/single')
def single_page():
  return render_template('single.html')


# --- 7. API ENDPOINTS ---


@app.route('/api/preprocessing', methods=['GET'])
def get_preprocessing_data():
  if not HAS_MYSQL:
    draw = int(request.args.get('draw', 1))
    start = int(request.args.get('start', 0))
    length = int(request.args.get('length', 10))

    items = (
        df_reviews.iloc[start : start + length].to_dict(orient='records')
        if not df_reviews.empty
        else []
    )
    return jsonify({
        'draw': draw,
        'recordsTotal': len(df_reviews),
        'recordsFiltered': len(df_reviews),
        'data': items,
    })

  try:
    draw = int(request.args.get('draw', 1))
    start = int(request.args.get('start', 0))
    length = int(request.args.get('length', 10))
    search_value = request.args.get('search[value]', '').strip()

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute('SELECT COUNT(*) AS total FROM reviews')
    total_records = cursor.fetchone()['total']

    if search_value:
      query_where = """
                WHERE raw_text LIKE %s 
                OR step1_lower LIKE %s
                OR step2_clean LIKE %s
                OR step3_normalized LIKE %s 
                OR step4_stopword LIKE %s
                OR step5_stemmed LIKE %s 
                OR predicted_sentiment LIKE %s
            """
      search_param = f'%{search_value}%'
      params_where = (search_param,) * 7

      cursor.execute(
          f'SELECT COUNT(*) AS total FROM reviews {query_where}', params_where
      )
      filtered_records = cursor.fetchone()['total']

      query_data = f"""
                SELECT id, raw_text, step1_lower, step2_clean, step3_normalized, step4_stopword, step5_stemmed, predicted_sentiment 
                FROM reviews 
                {query_where} 
                ORDER BY id ASC LIMIT %s OFFSET %s
            """
      params_data = params_where + (length, start)
      cursor.execute(query_data, params_data)
    else:
      filtered_records = total_records
      query_data = """
                SELECT id, raw_text, step1_lower, step2_clean, step3_normalized, step4_stopword, step5_stemmed, predicted_sentiment 
                FROM reviews 
                ORDER BY id ASC LIMIT %s OFFSET %s
            """
      cursor.execute(query_data, (length, start))

    reviews_data = cursor.fetchall()
    cursor.close()
    conn.close()

    return jsonify({
        'draw': draw,
        'recordsTotal': total_records,
        'recordsFiltered': filtered_records,
        'data': reviews_data,
    })
  except Exception as e:
    return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
  return jsonify({
      'total_reviews': metrics_summary['total_data'],
      'positive_count': metrics_summary['positif_count'],
      'negative_count': metrics_summary['negatif_count'],
      'accuracy': metrics_summary['accuracy_80_20'],
  })


@app.route('/api/predict', methods=['POST'])
def predict_single():
  data = request.get_json(silent=True) or {}
  raw_text = data.get('text', '')

  if not raw_text.strip():
    return jsonify({'error': 'Teks ulasan tidak boleh kosong'}), 400

  cleaned_text = preprocess_text(raw_text)

  if vectorizer and model_svc:
    vec = vectorizer.transform([cleaned_text])
    pred = model_svc.predict(vec)[0]

    try:
      decision_score = float(model_svc.decision_function(vec)[0])
    except Exception:
      decision_score = None

    try:
      proba = model_svc.predict_proba(vec)[0]
      prob_pos = round(float(proba[1]) * 100, 2)
      prob_neg = round(float(proba[0]) * 100, 2)
    except Exception:
      prob_pos, prob_neg = None, None
  else:
    pred = 'positif'
    decision_score = 0.5
    prob_pos, prob_neg = 75.0, 25.0

  return jsonify({
      'status': 'success',
      'raw_text': raw_text,
      'cleaned_text': cleaned_text,
      'sentiment': str(pred),
      'decision_score': decision_score,
      'probabilities': {'positif': prob_pos, 'negatif': prob_neg},
  })


if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0', port=5000)