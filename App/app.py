import streamlit as st
import joblib
import re
import emoji
import gdown
import os

# Download your FIXED & WORKING models from Google Drive
@st.cache_resource(show_spinner="جاري تحميل الموديلات...")
def load_models():
    os.makedirs("models", exist_ok=True)
    
    urls = {
        "best_model_arabic.pkl":        "https://drive.google.com/uc?id=16cOlavKZ4AX3yCLCX21jj8BeGjfEWyWh",
        "best_model_english.pkl":       "https://drive.google.com/uc?id=1c4tR9y7ZKzGPi_gvRkzuxFmOg96ARBRG",
        "tfidf_vectorizer_arabic.pkl":  "https://drive.google.com/uc?id=19P4Cy1TkzqO9PRlsp56m9JwSTw_Ryyng",
        "tfidf_vectorizer_english.pkl": "https://drive.google.com/uc?id=16kpbK9b3BQ4ZN-D8rFnicC2r1JaWMuQi",
        "bilstm_arabic.pth":            "https://drive.google.com/uc?id=1--s_-1FqU79874fgSYmDGFfG94gWFzRc",
        "bilstm_english.pth":           "https://drive.google.com/uc?id=1nOWQtzGwR6ZMuctNQlgbfpXc-SXf89Jg",
        "vocab_arabic.pkl":             "https://drive.google.com/uc?id=1aWu75baJtNngA7Y6Hxk8NjUZKx5zzkWo",
        "vocab_english.pkl":            "https://drive.google.com/uc?id=1LNEgdeirtwQQUYmQU3f0UZeBTkOV1GKm"
    }
    
    for name, url in urls.items():
        path = f"models/{name}"
        if not os.path.exists(path):
            with st.spinner(f"تحميل {name}..."):
                gdown.download(url, path, quiet=True)
    
    # Load only the working ML models (fast & reliable)
    model_ar = joblib.load("models/best_model_arabic.pkl")
    model_en = joblib.load("models/best_model_english.pkl")
    vec_ar   = joblib.load("models/tfidf_vectorizer_arabic.pkl")
    vec_en   = joblib.load("models/tfidf_vectorizer_english.pkl")
    
    return model_ar, model_en, vec_ar, vec_en

model_ar, model_en, vec_ar, vec_en = load_models()

# Cleaning functions
def clean_arabic(text):
    text = re.sub(r'http\\S+|www\\S+|https\\S+', '', text)
    text = emoji.replace_emoji(text, '')
    text = re.sub(r'[@#]\\w+', '', text)
    text = re.sub(r'[^\\u0600-\\u06FF\\s]', '', text)
    return text.strip()

def clean_english(text):
    text = re.sub(r'http\\S+|www\\S+|https\\S+', '', text)
    text = emoji.replace_emoji(text, '')
    text = re.sub(r'[@#]\\w+', '', text)
    text = re.sub(r'[^a-zA-Z\\s]', '', text).lower()
    return text.strip()

# UI – Clean & Professional
st.set_page_config(page_title="HexSoftware Sentiment Analysis", page_icon="rocket", layout="centered")
st.title("HexSoftware – Arabic & English Sentiment Analysis")
st.markdown("### Machine Learning Track • Task 1 Completed")

language = st.selectbox("اختر اللغة / Select Language", ["العربية", "English"])

user_text = st.text_area("اكتب النص هنا / Enter your text:", height=150)

if st.button("تحليل المشاعر / Analyze Sentiment", type="primary"):
    if user_text.strip():
        try:
            if language == "العربية":
                X = vec_ar.transform([clean_arabic(user_text)])
                pred = "إيجابي" if model_ar.predict(X)[0] == 1 else "سلبي"
                conf = abs(model_ar.decision_function(X)[0])
            else:
                X = vec_en.transform([clean_english(user_text)])
                pred = "Positive" if model_en.predict(X)[0] == 1 else "Negative"
                conf = abs(model_en.decision_function(X)[0]) if hasattr(model_en, 'decision_function') else 0.9

            st.success(f"النتيجة: **{pred}**")
            st.progress(min(conf/3, 1.0))
            if conf > 1.2:
                st.balloons()
        except Exception as e:
            st.error("خطأ في التحليل – جرب تاني")
    else:
        st.warning("من فضلك اكتب نص")

st.caption("© HexSoftware ML Track 2025 – Built with passion")
