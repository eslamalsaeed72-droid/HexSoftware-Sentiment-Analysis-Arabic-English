import streamlit as st
import joblib
import re
import emoji

# تحميل الموديلات
model_ar = joblib.load('models/best_model_arabic.pkl')
model_en = joblib.load('models/best_model_english.pkl')
vec_ar = joblib.load('models/tfidf_vectorizer_arabic.pkl')
vec_en = joblib.load('models/tfidf_vectorizer_english.pkl')

# دوال التنظيف (نفس اللي استخدمناها)
def clean_ar(text):
    text = re.sub(r'http\\S+|www\\S+|https\\S+', '', text)
    text = emoji.replace_emoji(text, '')
    text = re.sub(r'[@#]\\w+', '', text)
    text = re.sub(r'[^ا-ي\\s]', '', text)
    return text.strip()

def clean_en(text):
    text = re.sub(r'http\\S+|www\\S+|https\\S+', '', text)
    text = emoji.replace_emoji(text, '')
    text = re.sub(r'[@#]\\w+', '', text)
    text = re.sub(r'[^a-zA-Z\\s]', '', text)
    text = text.lower()
    return text.strip()

st.title("تحليل المشاعر - عربي وإنجليزي")
st.markdown("### مشروع Machine Learning - Hex Software Track")

lang = st.selectbox("اختر اللغة", ["العربية", "English"])

text = st.text_area("اكتب النص هنا:")

if st.button("تحليل المشاعر"):
    if text.strip():
        if lang == "العربية":
            cleaned = clean_ar(text)
            vec = vec_ar.transform([cleaned])
            pred = model_ar.predict(vec)[0]
            result = "إيجابي" if pred == 1 else "سلبي"
            confidence = model_ar.decision_function(vec)[0]
        else:
            cleaned = clean_en(text)
            vec = vec_en.transform([cleaned])
            pred = model_en.predict(vec)[0]
            result = "Positive" if pred == 1 else "Negative"
            confidence = model_en.decision_function(vec)[0] if hasattr(model_en, 'decision_function') else model_en.predict_proba(vec)[0].max()
        
        st.success(f"النتيجة: **{result}**")
        st.progress(abs(confidence) if confidence < 0 else confidence)
    else:
        st.error("اكتب نص أولاً!")
