import os
import re
import joblib
import mysql.connector
import pandas as pd
from collections import Counter
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# --- KONEKSI DATABASE MYSQL ---
def get_db_connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='db_hospital_sentiment'
    )

# 1. Load Kamus Normalisasi
norm_path = os.path.join(DATA_DIR, 'Normalisasi_2.csv')
if not os.path.exists(norm_path):
    norm_path = os.path.join(DATA_DIR, 'Normalisasi.csv')

df_norm = pd.read_csv(norm_path)
norm_dict = dict(zip(df_norm.iloc[:, 0].astype(str).str.lower(), df_norm.iloc[:, 1].astype(str).str.lower()))

# 2. Load Vectorizer & Model
vec_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
model_path = os.path.join(MODEL_DIR, 'svc_model.pkl')
if not os.path.exists(model_path):
    model_path = os.path.join(MODEL_DIR, 'model_svm_sentiment.pkl')

vectorizer = joblib.load(vec_path)
model_svc = joblib.load(model_path)

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    normalized_words = [norm_dict.get(w, w) for w in words]
    return " ".join(normalized_words)

# --- ROUTE HALAMAN ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/single')
def single_page():
    return render_template('single.html')

# ROUTE PREPROCESSING: Hanya render struktur HTML (Super Cepat)
@app.route('/preprocessing')
def preprocessing_page():
    return render_template('preprocessing.html')

# API DATATABLES SERVER-SIDE (Fetch Data Bertahap dari MySQL)
@app.route('/api/preprocessing', methods=['GET'])
def get_preprocessing_data():
    try:
        draw = int(request.args.get('draw', 1))
        start = int(request.args.get('start', 0))
        length = int(request.args.get('length', 10))
        search_value = request.args.get('search[value]', '').strip()

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Hitung total seluruh data
        cursor.execute("SELECT COUNT(*) AS total FROM reviews")
        total_records = cursor.fetchone()['total']

        # Filter jika ada pencarian via input DataTables
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
            search_param = f"%{search_value}%"
            params_where = (search_param,) * 7
            
            cursor.execute(f"SELECT COUNT(*) AS total FROM reviews {query_where}", params_where)
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
            "draw": draw,
            "recordsTotal": total_records,
            "recordsFiltered": filtered_records,
            "data": reviews_data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- API STATISTIK (DARI MYSQL) ---

@app.route('/api/stats', methods=['GET'])
def get_stats():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("SELECT step5_stemmed, predicted_sentiment FROM reviews")
        reviews = cursor.fetchall()
        cursor.close()
        conn.close()

        total_reviews = len(reviews)
        pos_count = sum(1 for r in reviews if str(r['predicted_sentiment']).strip().lower() in ['positif', '1'])
        neg_count = sum(1 for r in reviews if str(r['predicted_sentiment']).strip().lower() in ['negatif', '-1', '0'])

        stopwords = set(['dan', 'yang', 'di', 'ke', 'dari', 'ini', 'itu', 'dengan', 'untuk', 'pada', 'adalah', 'ada', 'saya', 'bisa', 'tidak', 'juga', 'akan', 'sudah', 'bgt', 'banget', 'rs', 'rumah', 'sakit', 'nya'])
        words_list = []

        for r in reviews:
            text = r['step5_stemmed'] or ""
            for w in text.split():
                if len(w) > 2 and w not in stopwords:
                    words_list.append(w)

        top_words_counts = Counter(words_list).most_common(7)
        top_keywords = [{'word': w, 'count': c} for w, c in top_words_counts]

        return jsonify({
            'total_reviews': total_reviews,
            'positive_count': pos_count,
            'negative_count': neg_count,
            'top_keywords': top_keywords
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict_single():
    data = request.get_json(silent=True) or {}
    raw_text = data.get('text', '')
    
    if not raw_text.strip():
        return jsonify({'error': 'Teks ulasan tidak boleh kosong'}), 400
        
    cleaned_text = preprocess_text(raw_text)
    
    vec = vectorizer.transform([cleaned_text])
    pred = model_svc.predict(vec)[0]
    
    try:
        decision_score = float(model_svc.decision_function(vec)[0])
    except:
        decision_score = None

    return jsonify({
        'raw_text': raw_text,
        'cleaned_text': cleaned_text,
        'sentiment': str(pred),
        'decision_score': decision_score
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)