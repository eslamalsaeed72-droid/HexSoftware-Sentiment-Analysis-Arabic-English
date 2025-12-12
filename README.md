# HexSoftware – Arabic & English Sentiment Analysis  
**Machine Learning Internship Project – ML Track**

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38-critical)
![License](https://img.shields.io/badge/License-MIT-green)

**Live Demo** → https://hexsoftware-sentiment-analysis.streamlit.app

## Project Overview
End-to-end sentiment analysis system for **Arabic** and **English** text, developed from scratch during the **Hex Software Machine Learning Track** internship.

The project demonstrates a complete ML pipeline: data ingestion, exploratory analysis, preprocessing, classical ML modeling, deep learning (BiLSTM), model comparison, and deployment via an interactive web application.

## Key Results
| Language | Best Classical Model | Accuracy | Deep Learning (BiLSTM) |
|----------|----------------------|----------|------------------------|
| Arabic   | Logistic Regression  | **87.69%** | ~85% (25 epochs)      |
| English  | LinearSVC            | **75.85%** | ~83% (15 epochs)      |

> Classical models outperformed deep learning on these datasets while being significantly faster and lighter – production-ready choice.

## Features
- Custom data loaders for folder-based Arabic dataset and Sentiment140 format
- Comprehensive EDA with interactive visualizations
- Language-specific text cleaning and stop-word removal
- TF-IDF vectorization (optimized per language)
- Training & hyperparameter tuning of 4 classical models using GridSearchCV
- BiLSTM implementation with PyTorch (bonus deep learning track)
- Automatic best-model selection and persistence
- **Interactive Streamlit web application** supporting real-time predictions in both languages
- All models hosted on Google Drive and downloaded on-the-fly (works perfectly on Streamlit Cloud)

## Tech Stack
- Python 3.10
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- PyTorch
- Streamlit
- gdown, joblib

## Project Structure
