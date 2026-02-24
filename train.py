import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib


def load_data(path):
    data=pd.read_csv(path)
    return data


def main():
    data = load_data("data/churn.csv")

    data = data.drop("customerID", axis=1)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data = data.dropna()

    X = data.drop("Churn", axis=1)
    y = data["Churn"]

    categorical_cols = X.select_dtypes(include=["object"]).columns
    numeric_cols = X.select_dtypes(exclude=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)

    accuracy = pipeline.score(X_test, y_test)
    print("Model Accuracy:", accuracy)

    joblib.dump(pipeline, "models/churn_model.pkl")
    print("Pipeline model saved.")




if __name__ == "__main__":
    main()
