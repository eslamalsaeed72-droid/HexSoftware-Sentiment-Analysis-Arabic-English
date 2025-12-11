Sentiment Analysis for Arabic and English

This repository provides a concise, production-oriented machine learning workflow for sentiment classification in Arabic and English text. The project includes preprocessing pipelines, feature engineering using TF-IDF, model training, evaluation, and prediction.

Project Summary

The solution applies traditional machine learning techniques to build sentiment analysis models capable of classifying text into positive, negative, or neutral sentiment. It is implemented in a single Jupyter Notebook for clarity and reproducibility.

Data

The datasets used in this project are stored on Google Drive:
https://drive.google.com/drive/folders/164q_gwG78BbWdWViFAdmCSFUnKngn3wN?usp=sharing

The notebook loads, inspects, and processes both Arabic and English datasets.

Key Capabilities

Unified workflow for Arabic and English text

Text normalization, cleaning, and linguistic preprocessing

Feature extraction with TF-IDF

Evaluation of multiple ML models (Logistic Regression, SVM, Naive Bayes, Random Forest)

Performance reporting using standard metrics and confusion matrices

Support for predicting sentiment on new text samples

Requirements

Install the required dependencies:

pip install numpy pandas scikit-learn matplotlib seaborn wordcloud nltk emoji
pip install tashaphyne arabic-reshaper python-bidi

Usage

Open the notebook:
Sentiment_Analysis_Arabic_&_English_(ML_Track_Hex_Software).ipynb

Run the notebook sequentially.

Update the data paths if needed.

Review model performance and execute the prediction section for custom inputs.

Future Enhancements

Integration of transformer-based language models (e.g., BERT, AraBERT)

Deployment as an API using FastAPI or Flask

Streamlit-based UI for real-time sentiment analysis

Improved model persistence and reproducible pipelines

License:
