from flask import Flask, render_template, request, send_file
import pickle
import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

app = Flask(__name__)

# Load trained model
with open("logreg_model.pkl", "rb") as f:
    model = pickle.load(f)

last_result = {}

@app.route("/", methods=["GET", "POST"])
def index():
    global last_result

    result = confidence = advice = action = name = ""
    systolic = diastolic = 0
    risk_class = ""

    if request.method == "POST":
        name = request.form["name"]
        systolic = int(request.form["systolic"])
        diastolic = int(request.form["diastolic"])

        features = [
            int(request.form["gender"]),
            int(request.form["age"]),
            systolic,
            diastolic,
            int(request.form["family_history"]),
            int(request.form["medication"]),
            int(request.form["smoking"]),
            int(request.form["exercise"])
        ]

        pred = model.predict([features])[0]
        prob = model.predict_proba([features])[0]
        confidence = round(max(prob) * 100, 2)

        stages = [
            ("Normal Blood Pressure", "normal"),
            ("Stage-1 Hypertension", "stage1"),
            ("Stage-2 Hypertension", "stage2"),
            ("Hypertensive Crisis", "crisis")
        ]

        recommendations = [
            "Maintain healthy lifestyle and routine BP monitoring.",
            "Reduce salt intake and increase physical activity.",
            "Consult physician and strictly follow medication.",
            "Emergency condition – immediate hospital care required."
        ]

        actions = [
            "Annual BP check-up recommended.",
            "Weekly BP monitoring advised.",
            "Doctor visit within 1–2 weeks.",
            "Go to nearest emergency department immediately."
        ]

        result, risk_class = stages[pred]
        advice = recommendations[pred]
        action = actions[pred]

        last_result = {
            "name": name,
            "result": result,
            "confidence": confidence,
            "advice": advice,
            "action": action
        }

    return render_template(
        "index.html",
        name=name,
        result=result,
        confidence=confidence,
        advice=advice,
        action=action,
        systolic=systolic,
        diastolic=diastolic,
        risk_class=risk_class
    )

@app.route("/download")
def download():
    # Create folder ONLY when needed
    if not os.path.exists("reports"):
        os.mkdir("reports")

    path = "reports/health_report.pdf"
    c = canvas.Canvas(path, pagesize=A4)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 800, "Predictive Pulse – Blood Pressure Report")

    c.setFont("Helvetica", 12)
    c.drawString(100, 760, f"Patient Name: {last_result['name']}")
    c.drawString(100, 740, f"Risk Level: {last_result['result']}")
    c.drawString(100, 720, f"Confidence: {last_result['confidence']}%")

    c.drawString(100, 690, "Clinical Recommendation:")
    c.drawString(120, 670, last_result["advice"])

    c.drawString(100, 640, "Suggested Action:")
    c.drawString(120, 620, last_result["action"])

    c.drawString(100, 580, "Note: This system assists clinical decision-making.")
    c.save()

    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
