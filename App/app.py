import streamlit as st
import joblib
import re
import emoji
import numpy as np

# -------------------------------------------------
# Load Models and Vectorizers
# -------------------------------------------------

@st.cache_resource
def load_model(path):
    return joblib.load(path)

model_ar = load_model('models/best_model_arabic.pkl')
model_en = load_model('models/best_model_english.pkl')
vec_ar = load_model('models/tfidf_vectorizer_arabic.pkl')
vec_en = load_model('models/tfidf_vectorizer_english.pkl')


# -------------------------------------------------
# Text Cleaning Functions
# -------------------------------------------------

def clean_arabic(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = emoji.replace_emoji(text, '')
    text = re.sub(r'[@#]\w+', '', text)
    text = re.sub(r'[^ء-ي\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_english(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = emoji.replace_emoji(text, '')
    text = re.sub(r'[@#]\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------

st.set_page_config(page_title="Sentiment Analysis", layout="centered")

st.title("Arabic & English Sentiment Analysis")
st.subheader("Machine Learning Project – Hex Software Internship Track")

language = st.selectbox("Select Language", ["العربية", "English"])

user_input = st.text_area(
    "Enter your text:",
    height=150,
    placeholder="اكتب النص هنا..." if language == "العربية" else "Type your text here..."
)


# -------------------------------------------------
# Prediction Logic
# -------------------------------------------------

def get_confidence(model, vectorized):
    # If model has decision_function, use it
    if hasattr(model, "decision_function"):
        value = model.decision_function(vectorized)
        return float(abs(value[0]))
    # Fallback: use predict_proba
    if hasattr(model, "predict_proba"):
        return float(np.max(model.predict_proba(vectorized)))
    return 0.75  # Default fallback


if st.button("Analyze Sentiment"):

    if user_input.strip():

        if language == "العربية":
            cleaned = clean_arabic(user_input)
            vectorized = vec_ar.transform([cleaned])
            prediction = model_ar.predict(vectorized)[0]

            result = "إيجابي" if prediction == 1 else "سلبي"
            confidence = get_confidence(model_ar, vectorized)

        else:
            cleaned = clean_english(user_input)
            vectorized = vec_en.transform([cleaned])
            prediction = model_en.predict(vectorized)[0]

            result = "Positive" if prediction == 1 else "Negative"
            confidence = get_confidence(model_en, vectorized)

        st.success(f"Prediction: {result}")
        st.info(f"Confidence: {confidence:.3f}")

    else:
        st.warning("Please enter some text first.")


st.caption("Built with scikit-learn • Deployed with Streamlit • Hex Software ML Track")
