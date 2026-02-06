import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

# sample dataset (same features as form)
data = {
    "gender": [0,1,0,1],
    "age": [1,2,3,4],
    "family_history": [0,1,1,1],
    "medication": [0,1,1,1],
    "severity": [0,1,2,2],
    "breath": [0,1,1,1],
    "vision": [0,1,0,1],
    "nose": [0,0,1,1],
    "systolic": [1,2,3,3],
    "diastolic": [1,2,2,2],
    "diet": [1,0,0,0],
    "target": [0,1,2,3]
}

df = pd.DataFrame(data)

X = df.drop("target", axis=1)
y = df["target"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

pickle.dump(model, open("logreg_model.pkl", "wb"))

print("Model saved successfully!")
