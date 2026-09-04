"""
Disease Prediction from Medical Data -- Flask backend
Loads the actual trained models (no mocking) and serves live predictions.
"""
import json
import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")

app = Flask(__name__, static_folder="static")

DISEASES = ["diabetes", "heart", "breast_cancer"]

_cache = {}


def load_disease(key):
    if key not in _cache:
        model = joblib.load(f"{MODEL_DIR}/{key}_model.pkl")
        scaler = joblib.load(f"{MODEL_DIR}/{key}_scaler.pkl")
        with open(f"{MODEL_DIR}/{key}_meta.json") as f:
            meta = json.load(f)
        _cache[key] = (model, scaler, meta)
    return _cache[key]


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/diseases", methods=["GET"])
def list_diseases():
    out = {}
    for key in DISEASES:
        _, _, meta = load_disease(key)
        out[key] = {
            "disease": meta["disease"],
            "best_model": meta["best_model"],
            "feature_names": meta["feature_names"],
        }
    return jsonify(out)


@app.route("/api/predict/<disease_key>", methods=["POST"])
def predict(disease_key):
    if disease_key not in DISEASES:
        return jsonify({"error": f"unknown disease '{disease_key}'"}), 400

    model, scaler, meta = load_disease(disease_key)
    payload = request.get_json(force=True)

    try:
        features = [float(payload[f]) for f in meta["feature_names"]]
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({
            "error": f"Invalid input. Expected numeric fields: {meta['feature_names']}",
            "detail": str(e),
        }), 400

    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)

    pred = int(model.predict(X_scaled)[0])
    proba = float(model.predict_proba(X_scaled)[0, 1])

    return jsonify({
        "disease": meta["disease"],
        "model_used": meta["best_model"],
        "prediction": pred,
        "prediction_label": "Disease likely present" if pred == 1 else "No disease indicated",
        "probability": round(proba, 4),
    })


if __name__ == "__main__":
    for key in DISEASES:
        load_disease(key)  # warm cache / fail fast if a model is missing
    print("Models loaded:", DISEASES)
    app.run(host="0.0.0.0", port=5050, debug=False)
