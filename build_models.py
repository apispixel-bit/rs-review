import os
import re
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Cari File Kamus Normalisasi (di folder data/ atau root)
norm_path = None
for p in [
    os.path.join(BASE_DIR, 'data', 'Normalisasi_2.csv'),
    os.path.join(BASE_DIR, 'data', 'Normalisasi.csv'),
    os.path.join(BASE_DIR, 'Normalisasi_2.csv'),
    os.path.join(BASE_DIR, 'Normalisasi.csv')
]:
    if os.path.exists(p):
        norm_path = p
        break

norm_dict = {}
if norm_path:
    print(f"[+] Membaca kamus normalisasi dari: {norm_path}")
    df_norm = pd.read_csv(norm_path)
    norm_dict = dict(zip(df_norm.iloc[:, 0].astype(str).str.lower(), df_norm.iloc[:, 1].astype(str).str.lower()))
else:
    print("[-] Warning: File kamus normalisasi tidak ditemukan.")

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    normalized_words = [norm_dict.get(w, w) for w in words]
    return " ".join(normalized_words)

# 2. Cari File Dataset Berlabel `dataset_rs_labeling.xlsx` (di folder data/ atau root)
dataset_path = None
for p in [
    os.path.join(BASE_DIR, 'data', 'dataset_rs_labeling.xlsx'),
    os.path.join(BASE_DIR, 'dataset_rs_labeling.xlsx'),
    os.path.join(BASE_DIR, 'data', 'review_rumah_sakit_manual_2.xlsx'),
    os.path.join(BASE_DIR, 'review_rumah_sakit_manual_2.xlsx')
]:
    if os.path.exists(p):
        dataset_path = p
        break

if not dataset_path:
    print("[-] Error: File dataset_rs_labeling.xlsx tidak ditemukan!")
    exit()

print(f"[+] Membaca dataset dari: {dataset_path}")
df = pd.read_excel(dataset_path)

# Mengambil kolom ulasan dan label sentimen
df['text_clean'] = df['review_text'].astype(str).apply(preprocess_text)
X_text = df['text_clean']
y = df['sentiment'].astype(str).str.strip()

print(f"\n[+] Distribusi Label Dataset:\n{y.value_counts()}\n")

# 3. Fit TF-IDF Vectorizer dengan 1814 Fitur
vectorizer = TfidfVectorizer(max_features=1814)
X = vectorizer.fit_transform(X_text)

# 4. Fit Model SVC
model_svc = SVC(kernel='linear', C=1.0)
model_svc.fit(X, y)

# 5. Simpan Berkas Model & Vectorizer yang Sinkron
model_dir = os.path.join(BASE_DIR, 'models')
os.makedirs(model_dir, exist_ok=True)

joblib.dump(vectorizer, os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
joblib.dump(model_svc, os.path.join(model_dir, 'svc_model.pkl'))
joblib.dump(model_svc, os.path.join(model_dir, 'model_svm_sentiment.pkl'))

print(f"[✓] SUKSES! Model dan Vectorizer ({X.shape[1]} fitur) berhasil diperbarui di folder models/\n")