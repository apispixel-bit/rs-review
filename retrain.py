import os
import re
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Path Data & Output Model
data_path = os.path.join(BASE_DIR, 'data', 'Normalisasi_2.csv')

# Cari dataset review di folder data/
dataset_path = None
data_folder = os.path.join(BASE_DIR, 'data')

if os.path.exists(data_folder):
    for f in os.listdir(data_folder):
        if f.endswith('.xlsx') or f.endswith('.csv'):
            if any(k in f.lower() for k in ['review', 'label', 'clean', 'merged', 'dataset', 'rs']):
                dataset_path = os.path.join(data_folder, f)
                break

if not dataset_path:
    print("[-] File dataset tidak ditemukan di folder data/")
    exit()

print(f"[+] Menggunakan Dataset: {dataset_path}")

# Load Kamus Normalisasi jika ada
norm_dict = {}
if os.path.exists(data_path):
    df_norm = pd.read_csv(data_path)
    norm_dict = dict(zip(df_norm.iloc[:, 0].astype(str).str.lower(), df_norm.iloc[:, 1].astype(str).str.lower()))

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    normalized_words = [norm_dict.get(w, w) for w in words]
    return " ".join(normalized_words)

# Load Dataset
if dataset_path.endswith('.csv'):
    df = pd.read_csv(dataset_path)
else:
    df = pd.read_excel(dataset_path)

print("\n--- Daftar Kolom pada Dataset ---")
for i, col in enumerate(df.columns):
    print(f" {i+1}. {col}")
print("----------------------------------\n")

# Deteksi Kolom Teks Ulasan
text_keywords = ['review', 'teks', 'text', 'ulasan', 'comment', 'isi', 'content']
text_col = None
for c in df.columns:
    if any(k in str(c).lower() for k in text_keywords):
        text_col = c
        break
if not text_col:
    text_col = df.columns[0] # Fallback ke kolom pertama

# Deteksi Kolom Label Sentimen
label_keywords = ['label', 'sentimen', 'sentiment', 'rating', 'class', 'polarity', 'target', 'kategori', 'kat']
label_col = None
for c in df.columns:
    if any(k in str(c).lower() for k in label_keywords):
        label_col = c
        break
if not label_col:
    label_col = df.columns[-1] # Fallback ke kolom terakhir

print(f"[+] Kolom Teks yang terdeteksi : '{text_col}'")
print(f"[+] Kolom Label yang terdeteksi: '{label_col}'\n")

# Preprocessing Teks
df['cleaned'] = df[text_col].astype(str).apply(preprocess_text)

# Melatih TF-IDF & SVC Model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned'])
y = df[label_col].astype(str)

model_svc = SVC(kernel='linear')
model_svc.fit(X, y)

# Simpan File .pkl
model_dir = os.path.join(BASE_DIR, 'models')
os.makedirs(model_dir, exist_ok=True)

joblib.dump(vectorizer, os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
joblib.dump(model_svc, os.path.join(model_dir, 'svc_model.pkl'))

print(f"[✓] SUKSES! Model berhasil dilatih dengan {X.shape[1]} fitur.")
print(f"[✓] File tfidf_vectorizer.pkl & svc_model.pkl telah diperbarui di folder models/\n")