import streamlit as st
import joblib
import re
import emoji
import os

# --- Fix path issue (works whether folder is 'models' or 'Models') ---
def find_file(filename):
    possible_paths = [
        f"models/{filename}",
        f"Models/{filename}",
        f"/content/models/{filename}",
        filename  # if in same directory
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

# Load models with fallback
@st.cache_resource
def load_models():
    model_ar_path = find_file("best_model_arabic.pkl")
    model_en_path = find_file("best_model_english.pkl")
    vec_ar_path   = find_file("tfidf_vectorizer_arabic.pkl")
    vec_en_path   = find_file("tfidf_vectorizer_english.pkl")

    if not all([model_ar_path, model_en_path, vec_ar_path, vec_en_path]):
        st.error("Model files not found! Make sure 'models/' folder contains:")
        st.code("best_model_arabic.pkl\\nbest_model_english.pkl\\ntfidf_vectorizer_arabic.pkl\\ntfidf_vectorizer_english.pkl")
        st.stop()

    model_ar = joblib.load(model_ar_path)
    model_en = joblib.load(model_en_path)
    vec_ar   = joblib.load(vec_ar_path)
    vec_en   = joblib.load(vec_en_path)
    
    return model_ar, model_en, vec_ar, vec_en

model_ar, model_en, vec_ar, vec_en = load_models()

# --- Text cleaning ---
def clean_arabic(text):
    text = re.sub(r'http\\S+|www\\S+|https\\S+', '', text)
    text = emoji.replace_emoji(text, '')
    text = re.sub(r'[@#]\\w+', '', text)
    text = re.sub(r'[^\\u0600-\\u06FF\\s]', '', text)  # Arabic letters only
    return text.strip()

def clean_english(text):
    text = re.sub(r'http\\S+|www\\S+|https\\S+', '', text)
    text = emoji.replace_emoji(text, '')
    text = re.sub(r'[@#]\\w+', '', text)
    text = re.sub(r'[^a-zA-Z\\s]', '', text).lower()
    return text.strip()

# --- UI ---
st.set_page_config(page_title="Sentiment Analysis", page_icon="bar_chart", layout="centered")
st.title("Arabic & English Sentiment Analysis")
st.markdown("### Machine Learning Project – Hex Software Internship Track")

language = st.selectbox("Select Language", ["العربية", "English"])

user_input = st.text_area("Enter your text:", height=150, placeholder="اكتب النص هنا...")

if st.button("Analyze Sentiment", type="primary"):
    if user_input.strip():
        try:
            if language == "العربية":
                cleaned = clean_arabic(user_input)
                vectorized = vec_ar.transform([cleaned])
                prediction = int(model_ar.predict(vectorized)[0])
                result = "إيجابي" if prediction == 1 else "سلبي"
                confidence = abs(model_ar.decision_function(vectorized)[0])
            else:
                cleaned = clean_english(user_input)
                vectorized = vec_en.transform([cleaned])
                prediction = int(model_en.predict(vectorized)[0])
                result = "Positive" if prediction == 1 else "Negative"
                confidence = model_en.decision_function(vectorized)[0] if hasattr(model_en, 'decision_function') else 0.9

            st.success(f"**Prediction: {result}**")
            st.progress(min(abs(confidence)/3, 1.0))
            st.info(f"Confidence score: {abs(confidence):.3f}")
        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.warning("Please enter some text.")

st.caption("Built for Hex Software ML Track • Deployed with Streamlit")
