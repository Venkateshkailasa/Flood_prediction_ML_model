from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

FEATURES = [
    {"name": "Temp", "label": "Temperature (°C)", "placeholder": "28"},
    {"name": "Humidity", "label": "Humidity (%)", "placeholder": "75"},
    {"name": "Cloud Cover", "label": "Cloud Cover (%)", "placeholder": "40"},
    {"name": "ANNUAL", "label": "Annual Rainfall (mm)", "placeholder": "3326.6"},
    {"name": "Jan-Feb", "label": "Jan-Feb Rainfall (mm)", "placeholder": "9.3"},
    {"name": "Mar-May", "label": "Mar-May Rainfall (mm)", "placeholder": "275.7"},
    {"name": "Jun-Sep", "label": "Jun-Sep Rainfall (mm)", "placeholder": "2403.4"},
    {"name": "Oct-Dec", "label": "Oct-Dec Rainfall (mm)", "placeholder": "638.2"},
    {"name": "avgjune", "label": "Avg June Rainfall (mm)", "placeholder": "130.3"},
    {"name": "sub", "label": "Seasonal Flood Index", "placeholder": "256.4"}
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "floods.save")
scaler_path = os.path.join(BASE_DIR, "transform.save")
load_error = None

try:
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    else:
        model = None
        scaler = None
        load_error = "Saved model or scaler file not found. Run train_model.py first."
except Exception as exc:
    model = None
    scaler = None
    load_error = str(exc)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if model is None or scaler is None:
        return jsonify({"error": load_error or "Prediction assets are missing."}), 500

    data = request.get_json(silent=True)
    if data is None:
        data = request.form.to_dict()

    try:
        values = [float(data.get(feature['name'], 0)) for feature in FEATURES]
        X = pd.DataFrame([values], columns=[feature['name'] for feature in FEATURES])
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]
        probability = float(model.predict_proba(X_scaled)[0][1]) * 100
        risk_score = round(probability, 2)
        safe_score = round(100 - probability, 2)

        if risk_score >= 70:
            status = "High risk"
        elif risk_score >= 40:
            status = "Moderate risk"
        else:
            status = "Safe"

        return jsonify({
            "risk": risk_score,
            "safe": safe_score,
            "status": status,
            "model": type(model).__name__
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

if __name__ == '__main__':
    app.run(debug=True)
