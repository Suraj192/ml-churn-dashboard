# 📊 Customer Churn Prediction Dashboard

An end-to-end Machine Learning project that predicts telecom customer churn and presents results through an interactive Streamlit dashboard.

---

## 🚀 Live Demo

https://ml-churn-dashboard-dhvoymoqtmxymcwmweaulr.streamlit.app/

---

## 🧠 Project Overview

Customer churn prediction helps businesses identify customers who are likely to leave a service.

This project:

- Trains a Random Forest model
- Uses a proper preprocessing pipeline (ColumnTransformer + OneHotEncoder)
- Saves the trained pipeline
- Serves predictions via a Streamlit dashboard
- Displays churn probability and feature importance

---

## 🏗 Project Architecture
ml-churn-dashboard/
│
├── data/ # Dataset
├── models/ # Saved ML pipeline
├── train.py # Model training script
├── app.py # Streamlit dashboard
├── requirements.txt
└── README.md

## ⚙️ Tech Stack

- Python
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib

---

## 📈 Model Details

- Algorithm: Random Forest Classifier
- Preprocessing: ColumnTransformer
- Encoding: OneHotEncoder (handle_unknown='ignore')
- Accuracy: ~79%

---

## ▶️ Run Locally

1. Clone repository: https://github.com/Suraj192/ml-churn-dashboard.git

## 📊 Features

- Interactive customer input form
- Churn probability visualization
- Feature importance chart
- Production-safe ML pipeline

---

## 📌 Future Improvements

- FastAPI backend
- Docker containerization
- Cloud deployment (Render/AWS)
- Model monitoring

---

## 👤 Author

Suraj  
Data & Machine Learning Enthusiast