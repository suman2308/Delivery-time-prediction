"""Evaluate candidate regressors for the DTDC delivery-duration contract.

Evaluation is intentionally in-memory only: this module never saves or
replaces a model artifact.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data.dtdc_preprocessing import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    PreprocessingReport,
    preprocess_dtdc_csv,
)


RANDOM_STATE = 42
CATEGORICAL_FEATURES = (
    "origin",
    "destination",
    "booking_weekday",
    "mode",
    "nature_of_consignment",
)
NUMERIC_FEATURES = (
    "total_pieces",
    "actual_weight",
    "volumetric_weight",
    "chargeable_weight",
)


@dataclass(frozen=True)
class ModelEvaluation:
    """Metrics and feature importance for one candidate model."""

    name: str
    mae: float
    rmse: float
    r2: float
    train_mae: float
    training_seconds: float
    prediction_seconds: float
    cv_mae_mean: float
    cv_mae_std: float
    cv_rmse_mean: float
    cv_r2_mean: float
    feature_importance: pd.Series


@dataclass(frozen=True)
class DTDCModelComparison:
    """Complete in-memory result for a DTDC candidate-model evaluation."""

    preprocessing_report: PreprocessingReport
    metrics: pd.DataFrame
    best_model: str
    evaluations: dict[str, ModelEvaluation]


def evaluate_dtdc_csv(csv_path: str | Path) -> DTDCModelComparison:
    """Preprocess a DTDC CSV and evaluate all available candidate models."""
    frame, preprocessing_report = preprocess_dtdc_csv(csv_path)
    return evaluate_transformed_data(frame, preprocessing_report)


def evaluate_transformed_data(
    frame: pd.DataFrame,
    preprocessing_report: PreprocessingReport,
) -> DTDCModelComparison:
    """Evaluate candidates using only Phase 1 transformed DTDC data."""
    expected_columns = set(FEATURE_COLUMNS) | {TARGET_COLUMN}
    missing_columns = expected_columns - set(frame.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"Transformed DTDC data is missing required columns: {names}")

    features = frame.loc[:, list(FEATURE_COLUMNS)]
    target = frame[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    evaluations: dict[str, ModelEvaluation] = {}

    for name, (model, sparse_categories) in _candidate_models().items():
        pipeline = Pipeline(
            [
                ("preprocessor", _build_preprocessor(sparse_categories)),
                ("model", model),
            ]
        )
        cv_scores = cross_validate(
            pipeline,
            x_train,
            y_train,
            cv=cv,
            n_jobs=1,
            scoring={
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
                "r2": "r2",
            },
        )

        started = perf_counter()
        pipeline.fit(x_train, y_train)
        training_seconds = perf_counter() - started

        started = perf_counter()
        predictions = pipeline.predict(x_test)
        prediction_seconds = perf_counter() - started
        train_predictions = pipeline.predict(x_train)

        evaluations[name] = ModelEvaluation(
            name=name,
            mae=float(mean_absolute_error(y_test, predictions)),
            rmse=float(np.sqrt(mean_squared_error(y_test, predictions))),
            r2=float(r2_score(y_test, predictions)),
            train_mae=float(mean_absolute_error(y_train, train_predictions)),
            training_seconds=training_seconds,
            prediction_seconds=prediction_seconds,
            cv_mae_mean=float(-cv_scores["test_mae"].mean()),
            cv_mae_std=float(cv_scores["test_mae"].std()),
            cv_rmse_mean=float(-cv_scores["test_rmse"].mean()),
            cv_r2_mean=float(cv_scores["test_r2"].mean()),
            feature_importance=_feature_importance(pipeline, x_test, y_test),
        )

    metrics = pd.DataFrame(
        {
            name: {
                "mae_days": result.mae,
                "rmse_days": result.rmse,
                "r2": result.r2,
                "train_mae_days": result.train_mae,
                "training_seconds": result.training_seconds,
                "prediction_seconds": result.prediction_seconds,
                "cv_mae_days_mean": result.cv_mae_mean,
                "cv_mae_days_std": result.cv_mae_std,
                "cv_rmse_days_mean": result.cv_rmse_mean,
                "cv_r2_mean": result.cv_r2_mean,
            }
            for name, result in evaluations.items()
        }
    ).T.sort_values(["mae_days", "rmse_days", "r2"], ascending=[True, True, False])
    return DTDCModelComparison(
        preprocessing_report=preprocessing_report,
        metrics=metrics,
        best_model=metrics.index[0],
        evaluations=evaluations,
    )


def _build_preprocessor(sparse_categories: bool) -> ColumnTransformer:
    """Encode categorical inputs and preserve numeric values unchanged."""
    return ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore", sparse_output=sparse_categories
                ),
                list(CATEGORICAL_FEATURES),
            ),
            ("numeric", "passthrough", list(NUMERIC_FEATURES)),
        ]
    )


def _candidate_models() -> dict[str, tuple[Any, bool]]:
    """Return the requested regressors, omitting XGBoost when unavailable."""
    candidates: dict[str, tuple[Any, bool]] = {
        "RandomForestRegressor": (
            RandomForestRegressor(
                n_estimators=10,
                min_samples_leaf=10,
                n_jobs=1,
                random_state=RANDOM_STATE,
            ),
            True,
        ),
        "HistGradientBoostingRegressor": (
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=50,
                l2_regularization=0.1,
                random_state=RANDOM_STATE,
            ),
            False,
        ),
    }
    try:
        from xgboost import XGBRegressor
    except ImportError:
        return candidates

    candidates["XGBoostRegressor"] = (
        XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        True,
    )
    return candidates


def _feature_importance(
    pipeline: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.Series:
    """Return transformed-feature importance or permutation importance."""
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = _clean_feature_names(preprocessor.get_feature_names_out())

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:
        encoded_test = preprocessor.transform(x_test)
        values = permutation_importance(
            model,
            encoded_test,
            y_test,
            scoring="neg_mean_absolute_error",
            n_repeats=5,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ).importances_mean
    return pd.Series(values, index=feature_names).sort_values(ascending=False)


def _clean_feature_names(feature_names: np.ndarray) -> list[str]:
    """Make ColumnTransformer names readable in the evaluation report."""
    return [
        name.replace("categorical__", "").replace("numeric__", "")
        for name in feature_names
    ]
