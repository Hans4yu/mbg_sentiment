import streamlit as st
import joblib
import re
import requests
import nltk
import os

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd

# Download NLTK resources dengan error handling yang lebih baik
@st.cache_resource
def download_nltk_data():
    nltk_data = ['punkt_tab', 'stopwords']
    for resource in nltk_data:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except:
                pass
    
    # Ensure punkt_tab exists for Indonesian
    try:
        nltk.data.find('tokenizers/punkt_tab/english')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)

download_nltk_data()

# =============================
# Load model & vectorizer
# =============================
@st.cache_resource
def load_model():
    model = joblib.load("models/logistic_regression_optimized.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# =============================
# Slangwords (ringkas, tapi sama logika)
# =============================
slangwords = {
    "mf": "maaf",
    "gk": "tidak",
    "gak": "tidak",
    "nggak": "tidak",
    "ga": "tidak",
    "kmi": "kami",
    "ntt": "nusa tenggara timur",
    "grtis": "gratis",
    "kenyangpembagi": "kenyang pembagi",
}

# =============================
# Preprocessing functions
# =============================
def cleaningText(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#[A-Za-z0-9_]+', '', text)
    text = re.sub(r'rt[\s]+', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def casefoldingText(text):
    return text.lower()

def fix_slangwords(text):
    words = text.split()
    fixed = [slangwords[w] if w in slangwords else w for w in words]
    return " ".join(fixed)

def tokenizingText(text):
    try:
        return word_tokenize(text)
    except LookupError:
        # Fallback simple tokenization jika NLTK data tidak tersedia
        return text.split()

def filteringText(tokens):
    try:
        stop_id = set(stopwords.words('indonesian'))
        stop_en = set(stopwords.words('english'))
    except LookupError:
        # Fallback: gunakan stopwords sederhana jika NLTK data tidak tersedia
        stop_id = {"dan", "atau", "yang", "di", "ke", "dari", "untuk"}
        stop_en = {"and", "or", "the", "a", "to", "from", "for"}
    
    stop_all = stop_id.union(stop_en)

    exceptions = {
        "tidak", "gak", "ga", "nggak",
        "mantap", "bagus", "jelek", "buruk",
        "tapi", "namun"
    }

    stop_all = stop_all - exceptions
    return [w for w in tokens if w not in stop_all]

def toSentence(tokens):
    return " ".join(tokens)

# =============================
# Prediction pipeline
# =============================
def predict_sentiment(text):
    clean = cleaningText(text)
    casefold = casefoldingText(clean)
    slang_fixed = fix_slangwords(casefold)
    tokens = tokenizingText(slang_fixed)
    filtered = filteringText(tokens)
    final_text = toSentence(filtered)

    tfidf = vectorizer.transform([final_text])
    pred = model.predict(tfidf)[0]
    proba = model.predict_proba(tfidf)[0]

    return pred, proba, final_text

# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="Analisis Sentimen MBG", layout="centered")

st.title("Analisis Sentimen Komentar YouTube")
st.write("Model: Logistic Regression + TF-IDF")

user_input = st.text_area(
    "Masukkan komentar YouTube:",
    height=150,
    placeholder="Contoh: kenapa portalnya tutup kmi dri daerah ntt"
)

if st.button("Prediksi Sentimen"):
    if user_input.strip() == "":
        st.warning("Teks tidak boleh kosong")
    else:
        label, proba, processed = predict_sentiment(user_input)

        st.subheader("Hasil Prediksi")
        st.write(f"**Sentimen:** {label}")
        st.write("**Probabilitas:**")
        for i, cls in enumerate(model.classes_):
            st.write(f"- {cls}: {proba[i]:.4f}")

        st.subheader("Teks setelah preprocessing")
        st.code(processed)

st.divider()
st.subheader("Uji Model Menggunakan File CSV")

uploaded_file = st.file_uploader(
    "Upload file CSV (harus ada kolom 'Isi Komentar' atau 'komentar')",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Deteksi kolom komentar otomatis
    possible_cols = ["Isi Komentar", "komentar", "text", "comment"]
    text_col = None
    for col in possible_cols:
        if col in df.columns:
            text_col = col
            break

    if text_col is None:
        st.error("Kolom komentar tidak ditemukan.")
    else:
        st.success(f"Menggunakan kolom: {text_col}")

        results = []
        for text in df[text_col].astype(str):
            label, proba, processed = predict_sentiment(text)
            results.append({
                "komentar_asli": text,
                "komentar_bersih": processed,
                "sentimen": label,
                "prob_positive": proba[list(model.classes_).index("positive")],
                "prob_neutral": proba[list(model.classes_).index("neutral")],
                "prob_negative": proba[list(model.classes_).index("negative")]
            })

        result_df = pd.DataFrame(results)

        st.dataframe(result_df)

        # Download hasil
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download hasil prediksi",
            csv,
            file_name="hasil_uji_model.csv",
            mime="text/csv"
        )