"""Flask web UI, JSON prediction API, admin filters, and analytics dashboard."""
from __future__ import annotations

import os

from flask import Flask, jsonify, redirect, render_template, request, url_for

import charts
import config
import database as db
import ml_model

app = Flask(__name__)


def ensure_app_ready():
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
    if not os.path.isfile(config.DATABASE_PATH):
        db.init_db()


@app.route("/")
def index():
    ensure_app_ready()
    metrics = None
    try:
        mae, rmse, n = ml_model.evaluate_on_db()
        metrics = {"mae": mae, "rmse": rmse, "rows": n}
    except FileNotFoundError:
        metrics = None
    return render_template("index.html", metrics=metrics, error=None)


@app.route("/predict", methods=["POST"])
def predict_form():
    ensure_app_ready()
    try:
        distance = float(request.form["distance"])
        order_time = int(request.form["order_time"])
        traffic = request.form["traffic_level"]
        weather = request.form["weather"]
    except (KeyError, TypeError, ValueError):
        return "Invalid form data", 400
    save_order = request.form.get("save_order") == "on"
    actual_raw = (request.form.get("actual_delivery") or "").strip()

    try:
        pipe = ml_model.load_pipeline()
    except FileNotFoundError as e:
        return render_template("index.html", metrics=None, error=str(e)), 503

    predicted = ml_model.predict_delivery(
        distance, order_time, traffic, weather, pipeline=pipe
    )
    order_id = None
    if save_order:
        if not actual_raw:
            return (
                "When saving an order, provide actual delivery time (minutes).",
                400,
            )
        try:
            actual = float(actual_raw)
        except ValueError:
            return "Actual delivery time must be a number.", 400
        order_id = db.insert_order(distance, order_time, traffic, weather, actual)
    db.insert_prediction(order_id, predicted)
    return render_template(
        "result.html",
        predicted=predicted,
        distance=distance,
        order_time=order_time,
        traffic=traffic,
        weather=weather,
        order_id=order_id,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    ensure_app_ready()
    payload = request.get_json(force=True, silent=True) or {}
    try:
        distance = float(payload["distance"])
        order_time = int(payload["order_time"])
        traffic = str(payload["traffic_level"])
        weather = str(payload["weather"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "distance, order_time, traffic_level, weather required"}), 400
    try:
        pipe = ml_model.load_pipeline()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    predicted = ml_model.predict_delivery(
        distance, order_time, traffic, weather, pipeline=pipe
    )
    order_id = payload.get("order_id")
    if order_id is not None:
        try:
            order_id = int(order_id)
        except (TypeError, ValueError):
            order_id = None
    prediction_id = db.insert_prediction(order_id, predicted)
    return jsonify(
        {
            "predicted_time_minutes": round(predicted, 3),
            "prediction_id": prediction_id,
            "order_id": order_id,
        }
    )


@app.route("/admin")
def admin():
    ensure_app_ready()
    traffic = request.args.get("traffic") or None
    weather = request.args.get("weather") or None
    min_d = request.args.get("min_distance")
    max_d = request.args.get("max_distance")
    min_distance = float(min_d) if min_d not in (None, "") else None
    max_distance = float(max_d) if max_d not in (None, "") else None
    rows = db.fetch_orders_filtered(
        traffic=traffic,
        weather=weather,
        min_distance=min_distance,
        max_distance=max_distance,
        limit=250,
    )
    return render_template(
        "admin.html",
        rows=rows,
        traffic=traffic or "",
        weather=weather or "",
        min_distance=min_d or "",
        max_distance=max_d or "",
    )


@app.route("/dashboard")
def dashboard():
    ensure_app_ready()
    try:
        ml_model.load_pipeline()
    except FileNotFoundError:
        return render_template(
            "dashboard.html",
            error="Train the model first: python train_model.py",
            plots=None,
        ), 503
    p1 = charts.plot_delivery_vs_distance()
    p2 = charts.plot_traffic_impact()
    p3 = charts.plot_pred_vs_actual()
    plots = {
        "distance": os.path.basename(p1),
        "traffic": os.path.basename(p2),
        "compare": os.path.basename(p3),
    }
    return render_template("dashboard.html", plots=plots, error=None)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/metrics")
def metrics_json():
    ensure_app_ready()
    try:
        mae, rmse, n = ml_model.evaluate_on_db()
        return jsonify({"mae": mae, "rmse": rmse, "order_rows": n})
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503


@app.route("/train", methods=["POST"])
def train_trigger():
    """Dev convenience: retrain from UI (optional)."""
    ensure_app_ready()
    try:
        r = ml_model.train_and_save()
        return redirect(
            url_for("index", trained=1, mae=f"{r.mae:.4f}", rmse=f"{r.rmse:.4f}")
        )
    except ValueError as e:
        return str(e), 400


if __name__ == "__main__":
    ensure_app_ready()
    app.run(debug=True, port=5000)
