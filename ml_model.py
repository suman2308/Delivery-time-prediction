"""Train linear regression on SQL data, persist with joblib, expose predict + metrics."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import database as db
import config


FEATURE_COLUMNS = ["distance", "order_time", "traffic_level", "weather"]
TARGET = "delivery_time"


@dataclass
class TrainResult:
    mae: float
    rmse: float
    train_rows: int


def _build_pipeline() -> Pipeline:
    categorical = ["traffic_level", "weather"]
    numeric = ["distance", "order_time"]
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    return Pipeline(
        steps=[
            ("prep", pre),
            ("model", LinearRegression()),
        ]
    )


def rows_to_dataframe(rows) -> pd.DataFrame:
    data = [{k: r[k] for k in r.keys()} for r in rows]
    return pd.DataFrame(data)


def train_and_save(test_size: float = 0.2, random_state: int = 42) -> TrainResult:
    rows = db.fetch_orders_for_training()
    if len(rows) < 10:
        raise ValueError("Need at least 10 orders in the database to train.")
    df = rows_to_dataframe(rows)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    pipe = _build_pipeline()
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    mae = float(mean_absolute_error(y_test, pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))

    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
    joblib.dump(pipe, config.MODEL_PATH)
    return TrainResult(mae=mae, rmse=rmse, train_rows=len(rows))


def load_pipeline() -> Pipeline:
    if not os.path.isfile(config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {config.MODEL_PATH}. Run: python train_model.py"
        )
    return joblib.load(config.MODEL_PATH)


def predict_delivery(
    distance: float,
    order_time: int,
    traffic_level: str,
    weather: str,
    pipeline: Optional[Pipeline] = None,
) -> float:
    pipe = pipeline or load_pipeline()
    X = pd.DataFrame(
        [
            {
                "distance": distance,
                "order_time": order_time,
                "traffic_level": traffic_level,
                "weather": weather,
            }
        ]
    )
    out = pipe.predict(X)
    return float(np.maximum(out[0], 1.0))


def evaluate_on_db(pipeline: Optional[Pipeline] = None) -> tuple[float, float, int]:
    """MAE / RMSE on all stored orders (in-sample diagnostic)."""
    pipe = pipeline or load_pipeline()
    rows = db.fetch_orders_for_training()
    df = rows_to_dataframe(rows)
    if df.empty:
        return 0.0, 0.0, 0
    pred = pipe.predict(df[FEATURE_COLUMNS])
    mae = float(mean_absolute_error(df[TARGET], pred))
    rmse = float(np.sqrt(mean_squared_error(df[TARGET], pred)))
    return mae, rmse, len(df)
