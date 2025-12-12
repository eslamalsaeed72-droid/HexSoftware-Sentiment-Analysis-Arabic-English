import streamlit as st
import joblib
import re
import emoji

# Load trained models and vectorizers
model_ar = joblib.load('models/best_model_arabic.pkl')
model_en = joblib.load('models/best_model_english.pkl')
vec_ar = joblib.load('models/tfidf_vectorizer_arabic.pkl')
vec_en = joblib.load('models/tfidf_vectorizer_english.pkl')

# Text cleaning functions
def clean_arabic(text):
    text = re.sub(r'http\\S+|www\\S+|https\\S+', '', text)
    text = emoji.replace_emoji(text, '')
    text = re.sub(r'[@#]\\w+', '', text)
    text = re.sub(r'[^ا-ي\\s]', '', text)
    return text.strip()

def clean_english(text):
    text = re.sub(r'http\\S+|www\\S+|https\\S+', '', text)
    text = emoji.replace_emoji(text, '')
    text = re.sub(r'[@#]\\w+', '', text)
    text = re.sub(r'[^a-zA-Z\\s]', '', text).lower()
    return text.strip()

# App UI
st.set_page_config(page_title="Sentiment Analysis", page_icon="sentiment", layout="centered")
st.title("Arabic & English Sentiment Analysis")
st.markdown("### Machine Learning Project – Hex Software Internship Track")

language = st.selectbox("Select Language", ["العربية", "English"])

user_input = st.text_area("Enter your text:", height=150, placeholder="اكتب النص هنا...")

if st.button("Analyze Sentiment", type="primary"):
    if user_input.strip():
        if language == "العربية":
            cleaned = clean_arabic(user_input)
            vectorized = vec_ar.transform([cleaned])
            prediction = model_ar.predict(vectorized)[0]
            result = "إيجابي" if prediction == 1 else "سلبي"
            confidence = abs(model_ar.decision_function(vectorized)[0])
        else:
            cleaned = clean_english(user_input)
            vectorized = vec_en.transform([cleaned])
            prediction = model_en.predict(vectorized)[0]
            result = "Positive" if prediction == 1 else "Negative"
            confidence = model_en.decision_function(vectorized)[0] if hasattr(model_en, 'decision_function') else 0.9
        
        st.success(f"**Prediction: {result}**")
        st.progress(min(confidence / 3 if confidence > 0 else 0.5, 1.0))
        st.info(f"Confidence: {confidence:.3f}")
    else:
        st.warning("Please enter some text to analyze.")

st.caption("Built with scikit-learn • Deployed with Streamlit • For Hex Software ML Track")
