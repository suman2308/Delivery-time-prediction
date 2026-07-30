"""Machine learning utilities for Smart Delivery.

Provides:
- A preprocessing + feature engineering pipeline.
- Several regression candidates (LinearRegression, RandomForest, XGBoost, LightGBM).
- Automatic training, validation, and best‑model selection.
- Persistence of the chosen pipeline (joblib) and a tiny JSON metadata file indicating the selected model.
- Prediction and evaluation helpers used by the Flask app (the app never knows which model is chosen).

The training pipeline is completely independent from Flask – it can be invoked from a script or a CI job.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import sqlite3

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Optional imports – if not available we simply skip those models.
try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None  # type: ignore
try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None  # type: ignore

import config
import database as db

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = ["distance", "order_time", "traffic_level", "weather"]
TARGET = "delivery_time"

# Path for a tiny JSON file that records which model was selected.
MODEL_META_PATH = config.MODEL_PATH + ".meta.json"

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def rows_to_dataframe(rows: List[sqlite3.Row]) -> pd.DataFrame:
    """Public helper for converting sqlite rows to DataFrame, used by charts.
    Delegates to internal _rows_to_dataframe.
    """
    return _rows_to_dataframe(rows)

    """Convert raw SQLite rows to a pandas DataFrame.
    The helper is deliberately tiny – the heavy lifting lives in the pipeline.
    """
    data = [{k: r[k] for k in r.keys()} for r in rows]
    return pd.DataFrame(data)

# ---------------------------------------------------------------------------
# Pre‑processing / feature engineering
# ---------------------------------------------------------------------------
def _build_preprocessor() -> ColumnTransformer:
    """Return a ColumnTransformer that:
    * Scales numeric columns with ``StandardScaler``.
    * One‑hot encodes categorical columns, ignoring unknown categories.
    """
    numeric = ["distance", "order_time"]
    categorical = ["traffic_level", "weather"]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )

# ---------------------------------------------------------------------------
# Model registry – candidate regressors
# ---------------------------------------------------------------------------
def get_training_stats() -> Tuple[float, float]:
    """Return mean and standard deviation of the target variable from training data.
    Used for delay‑risk calculation. Called after training; values are stored in the
    model meta JSON.
    """
    # Load all rows used for training (same as in train_and_save)
    rows = db.fetch_orders_for_training()
    df = _rows_to_dataframe(rows)
    mean_val = float(df[TARGET].mean())
    std_val = float(df[TARGET].std())
    return mean_val, std_val

def get_model_meta() -> Dict[str, Any]:
    """Read the JSON meta file produced during training.
    It contains the selected model name and optionally statistics.
    """
    if not os.path.isfile(MODEL_META_PATH):
        return {}
    try:
        with open(MODEL_META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def predict_with_confidence(
    distance: float,
    order_time: int,
    traffic: str,
    weather: str,
    pipeline: Optional[Pipeline] = None,
) -> Tuple[float, float]:
    """Return (prediction, confidence).
    Confidence is approximated as 1 - (MAE / mean_delivery_time) bounded to [0,1].
    """
    pipe = pipeline or load_pipeline()
    X = pd.DataFrame([
        {
            "distance": distance,
            "order_time": order_time,
            "traffic_level": traffic,
            "weather": weather,
        }
    ])
    pred = float(pipe.predict(X)[0])
    # Approximate confidence using training MAE stored in meta (if available)
    meta = get_model_meta()
    mae = meta.get("mae")
    if mae is None:
        # fallback: use a default moderate confidence
        confidence = 0.75
    else:
        mean_val, _ = get_training_stats()
        confidence = max(0.0, min(1.0, 1 - mae / mean_val))
    return pred, confidence

def assess_delay_risk(prediction: float) -> str:
    """Return 'high' if prediction exceeds mean + 1 σ, else 'low'."""
    mean_val, std_val = get_training_stats()
    return "high" if prediction > (mean_val + std_val) else "low"

import matplotlib.pyplot as plt

# Optional SHAP import – if unavailable we fall back to simple importance bar chart
try:
    import shap
except Exception:
    shap = None  # type: ignore

def generate_explanation_plot() -> str:
    """Create a visual explanation of the model's decisions.

    * If SHAP is available we produce a summary force plot for a typical sample.
    * Otherwise we render a plain bar chart of feature importances / coefficients.
    The image is saved under ``config.PLOTS_DIR`` as ``explanation.png`` and the
    function returns the absolute path so the Flask view can embed it.
    """
    meta = get_model_meta()
    best_name = meta.get("best_model")
    if not best_name:
        return ""
    pipe = load_pipeline()
    model = pipe.named_steps["model"]
    # Prepare feature names in the order used by the pipeline after preprocessing
    preprocessor = pipe.named_steps["preprocess"]
    # Get transformed feature names (numeric + one‑hot encoded categories)
    # This is a lightweight approximation – we only need a readable name list.
    numeric = ["distance", "order_time"]
    categorical = ["traffic_level", "weather"]
    # Build list of names for one‑hot features
    cat_features = preprocessor.transformers_[1][1].get_feature_names_out(categorical)
    feature_names = numeric + list(cat_features)

    if shap is not None:
        # Use a small sample from the training data for explanation
        rows = db.fetch_orders_for_training(limit=200)
        df = _rows_to_dataframe(rows)
        X_sample = df[FEATURE_COLUMNS].head(20)
        explainer = shap.Explainer(model, preprocessor.transform(X_sample))
        shap_vals = explainer(X_sample)
        plt.figure(figsize=(6, 4))
        shap.summary_plot(shap_vals, X_sample, plot_type="bar", show=False)
    else:
        # Fallback: simple importance / coefficient bar chart
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_[: len(feature_names)]
        elif hasattr(model, "coef_"):
            importances = model.coef_[: len(feature_names)]
        else:
            importances = [0] * len(feature_names)
        plt.figure(figsize=(6, 4))
        plt.barh(feature_names, importances, color="#38bdf8")
        plt.xlabel("Importance")
        plt.title("Feature importance for {} model".format(best_name))
        plt.gca().invert_yaxis()
    # Save the figure
    path = os.path.join(config.PLOTS_DIR, "explanation.png")
    plt.tight_layout()
    plt.savefig(path, dpi=120, facecolor=plt.gcf().get_facecolor())
    plt.close()
    return path

    """Return a simple model‑specific explanation.
    For LinearRegression we expose coefficients; for RandomForest we expose
    feature_importances_. The result maps feature name → importance/value.
    """
    meta = get_model_meta()
    best_name = meta.get("best_model")
    if not best_name:
        return {}
    pipe = load_pipeline()
    model = pipe.named_steps["model"]
    if hasattr(model, "coef_"):
        # LinearRegression or similar
        coeffs = model.coef_.tolist()
        return {feat: coeff for feat, coeff in zip(FEATURE_COLUMNS, coeffs)}
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_.tolist()
        return {feat: imp for feat, imp in zip(FEATURE_COLUMNS, importances)}
    return {}

def generate_recommendation(
    traffic: str,
    weather: str,
    distance: float,
) -> str:
    """Simple rule‑based recommendation based on inputs.
    * High traffic → suggest alternative route or earlier departure.
    * Rainy weather → add buffer time.
    * Long distance (>20 km) → consider split shipment.
    """
    recommendations = []
    if traffic.lower() == "high":
        recommendations.append("Consider alternative route or depart earlier to avoid congestion.")
    if weather.lower() == "rainy":
        recommendations.append("Add a buffer of 10‑15 min for weather‑related delays.")
    if distance > 20:
        recommendations.append("Long distance shipment – evaluate split‑delivery options.")
    return " ".join(recommendations) if recommendations else "No special actions needed."

def _candidate_models() -> Dict[str, Any]:
    """Dictionary of model name → instantiated regressor.
    Models that cannot be imported are silently omitted – the training loop will only
    consider the available ones.
    """
    candidates: Dict[str, Any] = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    }
    if XGBRegressor is not None:
        candidates["XGBoost"] = XGBRegressor(objective="reg:squarederror", n_estimators=200, random_state=42)
    if LGBMRegressor is not None:
        candidates["LightGBM"] = LGBMRegressor(n_estimators=200, random_state=42)
    return candidates

# ---------------------------------------------------------------------------
# Training result dataclass
# ---------------------------------------------------------------------------
@dataclass
class TrainResult:
    mae: float
    rmse: float
    train_rows: int
    best_model: str

# ---------------------------------------------------------------------------
# Core training routine – independent from Flask
# ---------------------------------------------------------------------------
def train_and_save(test_size: float = 0.2, random_state: int = 42) -> TrainResult:
    """Train all candidate models, evaluate on a hold‑out set, and persist the best.

    Returns a ``TrainResult`` containing the validation MAE/RMSE, the number of rows
    used for training and the name of the selected model.
    """
    rows = db.fetch_orders_for_training()
    if len(rows) < 10:
        raise ValueError("At least 10 historical orders are required to train model.")

    df = _rows_to_dataframe(rows)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    preprocessor = _build_preprocessor()
    candidates = _candidate_models()

    best_mae = float("inf")
    best_pipe: Optional[Pipeline] = None
    best_name = ""

    for name, model in candidates.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)
        mae = float(mean_absolute_error(y_val, preds))
        if mae < best_mae:
            best_mae = mae
            best_pipe = pipe
            best_name = name

    if best_pipe is None:
        raise RuntimeError("No candidate models were successfully trained.")

    # Persist the best pipeline.
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
    joblib.dump(best_pipe, config.MODEL_PATH)
    # Persist metadata (selected model name).
    with open(MODEL_META_PATH, "w", encoding="utf-8") as f:
        json.dump({"best_model": best_name}, f)

    # Compute final metrics on the validation set for reporting.
    final_preds = best_pipe.predict(X_val)
    mae = float(mean_absolute_error(y_val, final_preds))
    rmse = float(np.sqrt(mean_squared_error(y_val, final_preds)))
    return TrainResult(mae=mae, rmse=rmse, train_rows=len(df), best_model=best_name)

# ---------------------------------------------------------------------------
# Load pipeline – Flask only calls this; it never knows which model was chosen.
# ---------------------------------------------------------------------------
def load_pipeline() -> Pipeline:
    if not os.path.isfile(config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model not initialized at {config.MODEL_PATH}. Train model via CLI or UI."
        )
    return joblib.load(config.MODEL_PATH)

# ---------------------------------------------------------------------------
# Prediction helper – used by Flask routes.
# ---------------------------------------------------------------------------
def predict_delivery(
    distance: float,
    order_time: int,
    traffic_level: str,
    weather: str,
    pipeline: Optional[Pipeline] = None,
) -> float:
    pipe = pipeline or load_pipeline()
    X = pd.DataFrame([
        {
            "distance": distance,
            "order_time": order_time,
            "traffic_level": traffic_level,
            "weather": weather,
        }
    ])
    out = pipe.predict(X)
    return float(np.maximum(out[0], 2.0))

# ---------------------------------------------------------------------------
# Evaluation on the full DB – useful for dashboards.
# ---------------------------------------------------------------------------
def evaluate_on_db(pipeline: Optional[Pipeline] = None) -> Tuple[float, float, int]:
    pipe = pipeline or load_pipeline()
    rows = db.fetch_orders_for_training()
    df = _rows_to_dataframe(rows)
    if df.empty:
        return 0.0, 0.0, 0
    preds = pipe.predict(df[FEATURE_COLUMNS])
    mae = float(mean_absolute_error(df[TARGET], preds))
    rmse = float(np.sqrt(mean_squared_error(df[TARGET], preds)))
    return mae, rmse, len(df)

# ---------------------------------------------------------------------------
# Helper to expose selected model name (optional for API or admin UI).
# ---------------------------------------------------------------------------
def get_selected_model_name() -> str:
    if not os.path.isfile(MODEL_META_PATH):
        return "unknown"
    try:
        meta = json.load(open(MODEL_META_PATH, "r", encoding="utf-8"))
        return meta.get("best_model", "unknown")
    except Exception:
        return "unknown"


