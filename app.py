import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# load trained model
model = joblib.load("models/churn_model.pkl")

# loading dataset
data = pd.read_csv("data/churn.csv")
data = data.drop("customerID", axis=1)

# fixing total charges column
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
data = data.dropna()

# Title
st.title("Customer Churn Prediction Dashboard")


# creating input fields 
input_data = {}

for column in data.drop("Churn", axis=1).columns:
    if data[column].dtype == "object":
        input_data[column] = st.selectbox(column, data[column].unique())
    else:
        input_data[column] = st.number_input(column, value=float(data[column].mean()))

#converting input to dataframe
input_df = pd.DataFrame([input_data])


# Predict button
if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"Customer is likely to churn (Probability: {probability:.2f})")
    else:
        st.success(f"Customer is likely to stay (Probability: {probability:.2f})")

    