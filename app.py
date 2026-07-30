"""Flask Application Server & Endpoint Controllers."""
from __future__ import annotations

import os

from flask import Flask, jsonify, redirect, render_template, request, url_for

import charts
import config
import database as db
import ml_model
import seed_data

app = Flask(__name__)


def ensure_app_ready():
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
    if not os.path.isfile(config.DATABASE_PATH):
        db.init_db()
    _bootstrap_if_needed()


def _bootstrap_if_needed() -> None:
    """Startup bootstrap for fresh deployments."""
    bootstrap_enabled = os.environ.get("BOOTSTRAP_ON_START", "1") == "1"
    if not bootstrap_enabled or os.path.isfile(config.MODEL_PATH):
        return

    orders_count = db.count_orders()
    if orders_count < 10:
        seed_count = int(os.environ.get("BOOTSTRAP_SEED_COUNT", "300"))
        seed_data.seed(count=seed_count, seed=42)

    try:
        ml_model.train_and_save()
    except ValueError:
        return


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


@app.route("/scenario", methods=["POST"])
def scenario():
    return "Scenario endpoint placeholder", 200


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
    # Load model to ensure pipeline exists
    try:
        ml_model.load_pipeline()
    except FileNotFoundError:
        return render_template(
            "dashboard.html",
            error="Model pipeline not trained. Run training via home page or CLI script.",
            plots=None,
            kpis=None,
        ), 503

    # Fetch data for KPI calculations
    rows = db.fetch_orders_for_training()
    df = ml_model.rows_to_dataframe(rows)
    if df.empty:
        avg_delivery = 0.0
    else:
        avg_delivery = float(df[ml_model.TARGET].mean())
    # Confidence based on recent MAE
    mae, _, _ = ml_model.evaluate_on_db()
    confidence = max(0.0, min(1.0, 1 - mae / avg_delivery)) if avg_delivery != 0 else 0.0
    kpis = {
        "avg_delivery_time": round(avg_delivery, 2),
        "delay_risk": ml_model.assess_delay_risk(avg_delivery),
        "confidence": round(confidence * 100, 1),
    }

    # Generate plots
    p1 = charts.plot_delivery_vs_distance()
    p2 = charts.plot_traffic_impact()
    p3 = charts.plot_pred_vs_actual()
    plots = {
        "distance": os.path.basename(p1),
        "traffic": os.path.basename(p2),
        "compare": os.path.basename(p3),
    }
    return render_template("dashboard.html", plots=plots, error=None, kpis=kpis)


@app.route("/health")
def health():
    return jsonify({"status": "healthy"})


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
    ensure_app_ready()
    try:
        r = ml_model.train_and_save()
        return redirect(
            url_for("index", trained=1, mae=f"{r.mae:.2f}", rmse=f"{r.rmse:.2f}")
        )
    except ValueError as e:
        return str(e), 400


if __name__ == "__main__":
    ensure_app_ready()
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, host="0.0.0.0", port=port)
