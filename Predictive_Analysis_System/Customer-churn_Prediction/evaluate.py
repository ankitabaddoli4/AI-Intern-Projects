import joblib
import pandas as pd
from sklearn.metrics import classification_report

# Load model
model = joblib.load("../model/churn_model.pkl")

# Load data
df = pd.read_csv("../data/churn.csv")
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

X = df.drop("Churn", axis=1)
y = df["Churn"]

# Predict
y_pred = model.predict(X)

print("Classification Report:")
print(classification_report(y, y_pred))
