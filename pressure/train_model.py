import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("hypertension.csv")

# FINAL 8 FEATURES
X = df[
    [
        "gender",
        "age",
        "systolic",
        "diastolic",
        "family_history",
        "medication",
        "smoking",
        "exercise"
    ]
]

y = df["hypertension_stage"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))

# SAVE MODEL (AUTO-CREATES FILE)
with open("logreg_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained successfully")
print("📦 Saved as logreg_model.pkl")
print("🎯 Accuracy:", accuracy)
