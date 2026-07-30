"""Flask Application Server & Endpoint Controllers."""
from __future__ import annotations

import os

from flask import Flask, jsonify, redirect, render_template, request, url_for

import charts
import config
import database as db
import ml_model
import seed_data
from dtdc_model import DTDCPredictor, MODEL_ALGORITHM, MODEL_VERSION

app = Flask(__name__)

# ---------------------------------------------------------------------------
# DTDC model - initialised once at module load (singleton guarantees one load)
# ---------------------------------------------------------------------------
_dtdc_predictor = DTDCPredictor()


def ensure_app_ready():
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
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
        meta = _dtdc_predictor.meta
        m = meta.get("metrics", {})
        metrics = {
            "mae": m.get("mae_days", 0),
            "rmse": m.get("rmse_days", 0),
            "r2": m.get("r2", 0),
            "algorithm": meta.get("algorithm", MODEL_ALGORITHM),
            "model_version": meta.get("model_version", MODEL_VERSION),
        }
    except (FileNotFoundError, RuntimeError):
        metrics = None
    return render_template("index.html", metrics=metrics, error=None)


def _parse_dtdc_input(values):
    """Extract and validate DTDC prediction inputs from a form/JSON dict.

    Returns keyword arguments suitable for DTDCPredictor.predict().
    Raises ValueError on invalid input.
    """
    origin = str(values.get("origin", "")).strip()
    destination = str(values.get("destination", "")).strip()
    booking_weekday = str(values.get("booking_weekday", "")).strip()
    mode = str(values.get("mode", "")).strip()
    nature_of_consignment = str(values.get("nature_of_consignment", "")).strip()

    if not origin:
        raise ValueError("origin is required.")
    if not destination:
        raise ValueError("destination is required.")
    if not booking_weekday:
        raise ValueError("booking_weekday is required.")
    if not mode:
        raise ValueError("mode is required.")
    if not nature_of_consignment:
        raise ValueError("nature_of_consignment is required.")

    try:
        total_pieces = int(values.get("total_pieces", ""))
    except (TypeError, ValueError):
        raise ValueError("total_pieces must be a valid integer.")
    if total_pieces <= 0:
        raise ValueError("total_pieces must be positive.")

    for name in ("actual_weight", "volumetric_weight", "chargeable_weight"):
        try:
            val = float(values.get(name, ""))
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a valid number.")
        if val <= 0:
            raise ValueError(f"{name} must be positive.")

    return {
        "origin": origin,
        "destination": destination,
        "booking_weekday": booking_weekday,
        "mode": mode,
        "nature_of_consignment": nature_of_consignment,
        "total_pieces": total_pieces,
        "actual_weight": float(values.get("actual_weight", 0)),
        "volumetric_weight": float(values.get("volumetric_weight", 0)),
        "chargeable_weight": float(values.get("chargeable_weight", 0)),
    }


@app.route("/predict", methods=["POST"])
def predict_form():
    ensure_app_ready()
    try:
        kwargs = _parse_dtdc_input(request.form)
        result = _dtdc_predictor.predict(**kwargs)

        # Log prediction to audit table
        db.insert_dtdc_prediction(
            origin=kwargs["origin"],
            destination=kwargs["destination"],
            booking_weekday=kwargs["booking_weekday"],
            mode=kwargs["mode"],
            nature_of_consignment=kwargs["nature_of_consignment"],
            total_pieces=kwargs["total_pieces"],
            actual_weight=kwargs["actual_weight"],
            volumetric_weight=kwargs["volumetric_weight"],
            chargeable_weight=kwargs["chargeable_weight"],
            predicted_days=result.predicted_days,
            model_version=result.model_version,
        )

        # Load model metadata for result template context
        meta = _dtdc_predictor.meta
        m = meta.get("metrics", {})
        dataset_rows = meta.get("dataset_rows", 0)

        return render_template(
            "result.html",
            predicted_days=result.predicted_days,
            origin=kwargs["origin"],
            destination=kwargs["destination"],
            booking_weekday=kwargs["booking_weekday"],
            mode=kwargs["mode"],
            nature_of_consignment=kwargs["nature_of_consignment"],
            total_pieces=kwargs["total_pieces"],
            actual_weight=kwargs["actual_weight"],
            volumetric_weight=kwargs["volumetric_weight"],
            chargeable_weight=kwargs["chargeable_weight"],
            algorithm=result.algorithm,
            model_version=result.model_version,
            model_mae=m.get("mae_days", 0),
            dataset_rows=dataset_rows,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        return render_template("index.html", metrics=None, error=str(exc)), 400


@app.route("/api/predict", methods=["POST"])
def predict_api():
    ensure_app_ready()
    values = request.get_json(silent=True)
    if not isinstance(values, dict):
        return jsonify({"error": "A JSON object is required."}), 400

    try:
        kwargs = _parse_dtdc_input(values)
        result = _dtdc_predictor.predict(**kwargs)

        # Log prediction to audit table
        db.insert_dtdc_prediction(
            origin=kwargs["origin"],
            destination=kwargs["destination"],
            booking_weekday=kwargs["booking_weekday"],
            mode=kwargs["mode"],
            nature_of_consignment=kwargs["nature_of_consignment"],
            total_pieces=kwargs["total_pieces"],
            actual_weight=kwargs["actual_weight"],
            volumetric_weight=kwargs["volumetric_weight"],
            chargeable_weight=kwargs["chargeable_weight"],
            predicted_days=result.predicted_days,
            model_version=result.model_version,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(
        {
            "predicted_time_days": result.predicted_days,
            "model_version": result.model_version,
            "algorithm": result.algorithm,
        }
    )


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
    ensure_app_ready()

    # Load DTDC model metadata for KPI calculations
    try:
        meta = _dtdc_predictor.meta
        m = meta.get("metrics", {})
    except (FileNotFoundError, RuntimeError):
        return render_template(
            "dashboard.html",
            error="DTDC model not found. Train with ``python -m dtdc_model train``",
            plots=None,
            kpis=None,
        ), 503

    # Prediction audit stats
    pred_count = db.count_dtdc_predictions()
    pred_rows = db.fetch_dtdc_predictions(limit=10000)
    if pred_rows:
        import pandas as pd
        pdf = pd.DataFrame([{k: r[k] for k in r.keys()} for r in pred_rows])
        avg_pred = float(pdf["predicted_days"].mean())
    else:
        avg_pred = 0.0

    kpis = {
        "avg_predicted_days": round(avg_pred, 2),
        "model_mae": m.get("mae_days", 0),
        "model_r2": m.get("r2", 0),
        "prediction_count": pred_count,
        "model_version": meta.get("model_version", ""),
        "algorithm": meta.get("algorithm", ""),
    }

    # Generate DTDC-based charts
    p1 = charts.plot_mode_impact()
    p2 = charts.plot_prediction_distribution()
    p3 = charts.plot_top_routes()
    plots = {
        "mode_impact": os.path.basename(p1),
        "distribution": os.path.basename(p2),
        "routes": os.path.basename(p3),
    }
    return render_template("dashboard.html", plots=plots, error=None, kpis=kpis)

@app.route("/explain")
def explain():
    """Generate and serve model explanation plot."""
    try:
        path = ml_model.generate_explanation_plot()
        if not path:
            return "Explanation not available", 404
        # Serve image file
        from flask import send_file
        return send_file(path, mimetype='image/png')
    except Exception as e:
        return f"Error generating explanation: {e}", 500



@app.route("/health")
def health():
    return jsonify({"status": "healthy"})


@app.route("/metrics")
def metrics_json():
    ensure_app_ready()
    try:
        meta = _dtdc_predictor.meta
        m = meta.get("metrics", {})
        return jsonify({
            "mae_days": m.get("mae_days", 0),
            "rmse_days": m.get("rmse_days", 0),
            "r2": m.get("r2", 0),
            "model_version": meta.get("model_version", MODEL_VERSION),
            "algorithm": meta.get("algorithm", MODEL_ALGORITHM),
            "training_date": meta.get("training_date", ""),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 503
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
