# routes/dashboard.py
"""Dashboard route blueprint."""

from flask import Blueprint, render_template, current_app
import os
import ml_model
import database as db

bp = Blueprint('dashboard', __name__, url_prefix='')

@bp.route('/dashboard')
def dashboard():
    # Ensure model is loaded
    try:
        ml_model.load_pipeline()
    except FileNotFoundError:
        return render_template('dashboard.html', error='Model not trained.', plots=None, kpis=None), 503

    rows = db.fetch_orders_for_training()
    df = ml_model.rows_to_dataframe(rows)
    if df.empty:
        avg_delivery = 0.0
    else:
        avg_delivery = float(df[ml_model.TARGET].mean())
    mae, _, _ = ml_model.evaluate_on_db()
    confidence = max(0.0, min(1.0, 1 - mae / avg_delivery)) if avg_delivery != 0 else 0.0
    kpis = {
        'avg_delivery_time': round(avg_delivery, 2),
        'delay_risk': ml_model.assess_delay_risk(avg_delivery),
        'confidence': round(confidence * 100, 1),
    }
    # Generate plots
    p1 = ml_model.plot_delivery_vs_distance()
    p2 = ml_model.plot_traffic_impact()
    p3 = ml_model.plot_pred_vs_actual()
    plots = {
        'distance': os.path.basename(p1),
        'traffic': os.path.basename(p2),
        'compare': os.path.basename(p3),
    }
    return render_template('dashboard.html', plots=plots, error=None, kpis=kpis)
