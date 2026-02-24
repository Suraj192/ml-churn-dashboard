import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt


st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")

# Title
st.title("Customer Churn Prediction Dashboard")
st.write("Predict whether a telecom customer is likely to churn.")


# load trained model
model = joblib.load("models/churn_model.pkl")

# loading dataset
data = pd.read_csv("data/churn.csv")
data = data.drop("customerID", axis=1)

# fixing total charges column
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
data = data.dropna()

st.sidebar.header("Customer Information")


# creating input fields 
input_data = {}

for column in data.drop("Churn", axis=1).columns:
    if data[column].dtype == "object":
        input_data[column] = st.selectbox(column, data[column].unique())
    else:
        input_data[column] = st.number_input(column, value=float(data[column].mean()))

#converting input to dataframe
input_df = pd.DataFrame([input_data])


if st.sidebar.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == "Yes":
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is likely to stay.")

    st.write("Churn Probability:")
    st.progress(float(probability))

    st.write(f"Probability Score: {probability:.2f}")

    st.subheader("📈 Feature Importance")

    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()

    importances = classifier.feature_importances_

    feature_importance_df = pd.DataFrame({
        "Feature" : feature_names,
        "Importance" : importances
    }). sort_values(by="Importance", ascending=False).head(10)

    fig, ax = plt.subplots()
    ax.barh(feature_importance_df["Feature"], feature_importance_df["Importance"])
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score")

    st.pyplot(fig)