import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


def load_data(path):
    data=pd.read_csv(path)
    return data


def preprocess_data(data):
    # Dropping customer ID
    data= data.drop("customerID", axis=1)

    # converting TotalCharges column to numeric
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data = data.dropna()

    # encoding categorical columns
    for column in data.select_dtypes(include = ["object"]).columns:
        encoder = LabelEncoder()
        data[column] = encoder.fit_transform(data[column])

    X = data.drop("Churn", axis=1)
    y= data["Churn"]

    return X, y


def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators = 200,
        random_state = 42,
        n_jobs = -1
    )
    model.fit(X_train, y_train)
    return model

def main():
    data = load_data("data/churn.csv")
    X, y = preprocess_data(data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train)

    #Saving model
    joblib.dump(model, "models/churn_model.pk1")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Model Accuracy:", accuracy)
    print("Model saved as models/churn_model.pk1")

    





if __name__ == "__main__":
    main()
