"""
Disease Prediction from Medical Data -- Flask backend
Loads the actual trained models (no mocking) and serves live predictions.
"""
import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
import joblib
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history.db")

app = Flask(__name__, static_folder="static")

DISEASES = ["diabetes", "heart", "breast_cancer"]

_cache = {}


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with closing(get_db()) as conn, conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                disease TEXT,
                model_used TEXT,
                inputs_json TEXT,
                prediction INTEGER,
                prediction_label TEXT,
                probability REAL
            )
        """)


init_db()


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

    response = {
        "disease": meta["disease"],
        "model_used": meta["best_model"],
        "prediction": pred,
        "prediction_label": "Disease likely present" if pred == 1 else "No disease indicated",
        "probability": round(proba, 4),
    }

    with closing(get_db()) as conn, conn:
        conn.execute(
            "INSERT INTO predictions "
            "(timestamp, disease, model_used, inputs_json, prediction, prediction_label, probability) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                response["disease"],
                response["model_used"],
                json.dumps(payload),
                response["prediction"],
                response["prediction_label"],
                response["probability"],
            ),
        )

    return jsonify(response)


@app.route("/api/history", methods=["GET"])
def get_history():
    with closing(get_db()) as conn:
        rows = conn.execute(
            "SELECT id, timestamp, disease, model_used, inputs_json, prediction_label, probability "
            "FROM predictions ORDER BY id DESC LIMIT 50"
        ).fetchall()
    return jsonify([
        {
            "id": row["id"],
            "timestamp": row["timestamp"],
            "disease": row["disease"],
            "model_used": row["model_used"],
            "inputs": json.loads(row["inputs_json"]),
            "prediction_label": row["prediction_label"],
            "probability": row["probability"],
        }
        for row in rows
    ])


@app.route("/api/history/<int:record_id>", methods=["DELETE"])
def delete_history_record(record_id):
    with closing(get_db()) as conn, conn:
        cur = conn.execute("DELETE FROM predictions WHERE id = ?", (record_id,))
    if cur.rowcount == 0:
        return jsonify({"error": f"no record with id {record_id}"}), 404
    return jsonify({"deleted": record_id})


@app.route("/api/history", methods=["DELETE"])
def clear_history():
    with closing(get_db()) as conn, conn:
        conn.execute("DELETE FROM predictions")
    return jsonify({"cleared": True})


if __name__ == "__main__":
    for key in DISEASES:
        load_disease(key)  # warm cache / fail fast if a model is missing
    print("Models loaded:", DISEASES)
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
