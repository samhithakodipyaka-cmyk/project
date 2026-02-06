from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# load trained model
model = pickle.load(open("logreg_model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""

    if request.method == "POST":
        gender = int(request.form["gender"])
        age = int(request.form["age"])
        family_history = int(request.form["family_history"])
        medication = int(request.form["medication"])
        severity = int(request.form["severity"])
        breath = int(request.form["breath"])
        vision = int(request.form["vision"])
        nose = int(request.form["nose"])
        systolic = int(request.form["systolic"])
        diastolic = int(request.form["diastolic"])
        diet = int(request.form["diet"])

        data = np.array([[gender, age, family_history, medication,
                          severity, breath, vision, nose,
                          systolic, diastolic, diet]])

        result = model.predict(data)[0]

        stages = {
            0: "Normal",
            1: "Stage-1 Hypertension",
            2: "Stage-2 Hypertension",
            3: "Hypertensive Crisis"
        }

        prediction = stages[result]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
