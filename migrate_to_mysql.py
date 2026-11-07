import os
import re
import mysql.connector
import pandas as pd
import joblib
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# 1. Koneksi Database MySQL
def get_db_connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='db_hospital_sentiment'
    )

# 2. Inisialisasi Sastrawi
factory_stemmer = StemmerFactory()
stemmer = factory_stemmer.create_stemmer()

factory_stop = StopWordRemoverFactory()
stopword_remover = factory_stop.create_stop_word_remover()

# 3. Load Kamus Normalisasi (Membaca langsung dari data/Normalisasi_2.csv)
norm_dict = {}
norm_csv_path = os.path.join('data', 'Normalisasi_2.csv')

if os.path.exists(norm_csv_path):
    # Sesuaikan delimiter jika CSV menggunakan koma (,) atau titik koma (;)
    try:
        df_norm = pd.read_csv(norm_csv_path)
    except Exception:
        df_norm = pd.read_csv(norm_csv_path, sep=';')
    
    # Asumsi kolom pertama kata slang, kolom kedua kata baku
    col_slang = df_norm.columns[0]
    col_baku = df_norm.columns[1]
    norm_dict = dict(zip(df_norm[col_slang].astype(str).str.lower(), df_norm[col_baku].astype(str).str.lower()))

def normalize_text(text):
    words = text.split()
    normalized_words = [norm_dict.get(w, w) for w in words]
    return " ".join(normalized_words)

# 4. Load Model & Vectorizer dari folder 'models'
model_path = os.path.join('models', 'model_svm_sentiment.pkl')
vectorizer_path = os.path.join('models', 'tfidf_vectorizer.pkl')

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# 5. Load Dataset dari folder 'data'
# Menggunakan Excel review_rumah_sakit_merged_clean_2.xlsx atau dataset_rs_labeling.xlsx
dataset_path = os.path.join('data', 'review_rumah_sakit_merged_clean_2.xlsx')

if dataset_path.endswith('.xlsx'):
    df = pd.read_excel(dataset_path)
else:
    df = pd.read_csv(dataset_path)

conn = get_db_connection()
cursor = conn.cursor()

print("Memulai migrasi data ke MySQL...")

# Cari nama kolom ulasan secara dinamis (misal: 'ulasan', 'review', atau 'text')
text_column = None
for col in df.columns:
    if col.lower() in ['ulasan', 'review', 'text', 'content', 'review_text']:
        text_column = col
        break

if not text_column:
    text_column = df.columns[0] # Default ke kolom pertama jika tidak terdeteksi

for index, row in df.iterrows():
    raw_text = str(row[text_column]) if pd.notna(row[text_column]) else ""
    if not raw_text.strip():
        continue

    hospital_name = str(row.get('nama_rs', row.get('hospital_name', 'Rumah Sakit')))

    # --- Pipeline Preprocessing ---
    # Step 1: Case Folding
    step1_lower = raw_text.lower()

    # Step 2: Cleaning
    step2_clean = re.sub(r'[^a-zA-Z\s]', '', step1_lower)
    step2_clean = re.sub(r'\s+', ' ', step2_clean).strip()

    # Step 3: Normalization
    step3_normalized = normalize_text(step2_clean)

    # Step 4: Stopword Removal
    step4_stopword = stopword_remover.remove(step3_normalized)

    # Step 5: Stemming
    step5_stemmed = stemmer.stem(step4_stopword)

    # --- Prediksi Sentimen ---
    text_vectorized = vectorizer.transform([step5_stemmed])
    prediction = model.predict(text_vectorized)[0]

    # --- Simpan ke Database ---
    query = """
        INSERT INTO reviews 
        (hospital_name, raw_text, step1_lower, step2_clean, step3_normalized, step4_stopword, step5_stemmed, predicted_sentiment)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    values = (hospital_name, raw_text, step1_lower, step2_clean, step3_normalized, step4_stopword, step5_stemmed, str(prediction))
    
    cursor.execute(query, values)

conn.commit()
cursor.close()
conn.close()

print("Migrasi Selesai! Seluruh data dan tahapan preprocessing berhasil disimpan ke MySQL.")