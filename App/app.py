
import streamlit as st
import joblib
import re
import emoji
import gdown
import os

# --- Download models from Google Drive (public links) ---
@st.cache_resource
def download_models():
    os.makedirs("models", exist_ok=True)
    
    # ضع هنا لينكات Google Drive الخاصة بيك (File → Share → Anyone with the link)
    # مثال (غيّرها بتاعتك):
    urls = {
        "best_model_arabic.pkl":        "https://drive.google.com/file/d/16cOlavKZ4AX3yCLCX21jj8BeGjfEWyWh/view?usp=drive_link",
        "best_model_english.pkl":       "https://drive.google.com/file/d/1c4tR9y7ZKzGPi_gvRkzuxFmOg96ARBRG/view?usp=drive_link", 
        "tfidf_vectorizer_arabic.pkl":  "https://drive.google.com/file/d/19P4Cy1TkzqO9PRlsp56m9JwSTw_Ryyng/view?usp=drive_link",
        "tfidf_vectorizer_english.pkl": "https://drive.google.com/file/d/16kpbK9b3BQ4ZN-D8rFnicC2r1JaWMuQi/view?usp=drive_link"
    }
    
    for name, url in urls.items():
        path = f"models/{name}"
        if not os.path.exists(path):
            with st.spinner(f"Downloading {name}..."):
                gdown.download(url, path, quiet=False)
    
    return (
        joblib.load("models/best_model_arabic.pkl"),
        joblib.load("models/best_model_english.pkl"),
        joblib.load("models/tfidf_vectorizer_arabic.pkl"),
        joblib.load("models/tfidf_vectorizer_english.pkl")
    )

model_ar, model_en, vec_ar, vec_en = download_models()

# --- Cleaning functions ---
def clean_arabic(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = emoji.replace_emoji(text, '')
    text = re.sub(r'[@#]\w+', '', text)
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    return text.strip()

def clean_english(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = emoji.replace_emoji(text, '')
    text = re.sub(r'[@#]\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    return text.strip()

# --- UI ---
st.set_page_config(page_title="HexSoftware Sentiment Analysis", page_icon="rocket", layout="centered")
st.title("HexSoftware – Arabic & English Sentiment Analysis")
st.markdown("### Machine Learning Internship Project")

language = st.selectbox("Select Language / اختر اللغة", ["العربية", "English"])

user_input = st.text_area("Enter text / اكتب النص:", height=150)

if st.button("Analyze Sentiment / تحليل المشاعر", type="primary"):
    if user_input.strip():
        try:
            if language == "العربية":
                cleaned = clean_arabic(user_input)
                X = vec_ar.transform([cleaned])
                pred = int(model_ar.predict(X)[0])
                result = "إيجابي" if pred == 1 else "سلبي"
                confidence = abs(model_ar.decision_function(X)[0])
            else:
                cleaned = clean_english(user_input)
                X = vec_en.transform([cleaned])
                pred = int(model_en.predict(X)[0])
                result = "Positive" if pred == 1 else "Negative"
                confidence = model_en.decision_function(X)[0] if hasattr(model_en, 'decision_function') else 0.9

            st.success(f"**النتيجة: {result}**")
            st.progress(min(abs(confidence)/3, 1.0))
            st.balloons()
        except Exception as e:
            st.error("Error during prediction. Check console.")
    else:
        st.warning("Please enter some text")

st.caption("Built for Hex Software ML Track • Powered by Streamlit")
