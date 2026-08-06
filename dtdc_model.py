"""Production DTDC delivery-duration prediction model.

Provides:
- Training the tuned HistGradientBoostingRegressor on the full DTDC dataset.
- Versioned model persistence (joblib) with companion metadata (JSON).
- A reusable ``DTDCPredictor`` class with input validation, category
  standardisation, and safe handling of unseen categories.
- CLI entry points for training and interactive prediction.

The module is completely independent of the Flask application — the app may
import it, but no Flask-specific imports exist here.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import joblib

# Reuse the canonical dataset-preprocessing pipeline from Phase 1.
from data.dtdc_preprocessing import preprocess_dtdc_csv

# ---------------------------------------------------------------------------
# Version / path constants
# ---------------------------------------------------------------------------
MODEL_VERSION = "1.0.0"
MODEL_ALGORITHM = "HistGradientBoostingRegressor"
DATASET_VERSION = "DTDC_Improved_v1"
PREPROCESSING_VERSION = "1.0"
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_STEM = f"dtdc_hgb_v{MODEL_VERSION.replace('.', '_')}"
MODEL_PATH = MODEL_DIR / f"{MODEL_STEM}.joblib"
META_PATH = MODEL_DIR / f"{MODEL_STEM}.meta.json"
CSV_PATH = Path(__file__).resolve().parent / "DTDC_Improved_Dataset.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Feature contract — matches ``dtdc_preprocessing.FEATURE_COLUMNS``
FEATURE_COLUMNS = [
    "origin",
    "destination",
    "booking_weekday",
    "mode",
    "nature_of_consignment",
    "total_pieces",
    "actual_weight",
    "volumetric_weight",
    "chargeable_weight",
]
TARGET_COLUMN = "delivery_duration_days"

CATEGORICAL_FEATURES = [
    "origin",
    "destination",
    "booking_weekday",
    "mode",
    "nature_of_consignment",
]
NUMERIC_FEATURES = [
    "total_pieces",
    "actual_weight",
    "volumetric_weight",
    "chargeable_weight",
]

TUNED_HYPERPARAMETERS = {
    "learning_rate": 0.03,
    "max_iter": 150,
    "max_depth": 6,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
    "random_state": RANDOM_STATE,
}


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------
def _build_preprocessor() -> ColumnTransformer:
    """Return a ColumnTransformer with known-category-safe OneHotEncoder."""
    return ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )


def _build_pipeline() -> Pipeline:
    """Return a scikit-learn Pipeline wrapping the tuned HGB model."""
    return Pipeline(
        [
            ("preprocessor", _build_preprocessor()),
            (
                "model",
                HistGradientBoostingRegressor(**TUNED_HYPERPARAMETERS),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Category standardisation (mirrors dtdc_preprocessing._standardize_category)
# ---------------------------------------------------------------------------
def _standardize(text: str) -> str:
    """Normalise a single category string the same way training data was
    normalised by ``dtdc_preprocessing._standardize_category``.

    The logic mirrors the canonical source verbatim so that production
    inference applies identical normalisation to the Phase 1 training
    pipeline.  Keep these two implementations in sync.
    """
    import unicodedata
    import re
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.casefold()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
@dataclass
class PredictionInput:
    """Validated, standardised input to the DTDC model.

    All fields are coerced to the correct type; categories are casefolded
    and whitespace-normalised.
    """
    origin: str
    destination: str
    booking_weekday: str
    mode: str
    nature_of_consignment: str
    total_pieces: int
    actual_weight: float
    volumetric_weight: float
    chargeable_weight: float

    def to_dataframe(self) -> pd.DataFrame:
        """Return a single-row DataFrame matching the training feature columns."""
        return pd.DataFrame([{
            "origin": self.origin,
            "destination": self.destination,
            "booking_weekday": self.booking_weekday,
            "mode": self.mode,
            "nature_of_consignment": self.nature_of_consignment,
            "total_pieces": self.total_pieces,
            "actual_weight": self.actual_weight,
            "volumetric_weight": self.volumetric_weight,
            "chargeable_weight": self.chargeable_weight,
        }])


def parse_prediction_input(
    *,
    origin: str,
    destination: str,
    booking_weekday: str,
    mode: str,
    nature_of_consignment: str,
    total_pieces: Any,
    actual_weight: Any,
    volumetric_weight: Any,
    chargeable_weight: Any,
) -> PredictionInput:
    """Validate and standardise raw prediction inputs.

    Raises ``ValueError`` on invalid values.
    """
    errors: list[str] = []

    origin_s = _standardize(str(origin))
    destination_s = _standardize(str(destination))
    booking_weekday_s = _standardize(str(booking_weekday))
    mode_s = _standardize(str(mode))
    nature_s = _standardize(str(nature_of_consignment))

    if not origin_s:
        errors.append("origin must be a non-empty string")
    if not destination_s:
        errors.append("destination must be a non-empty string")
    if not booking_weekday_s:
        errors.append("booking_weekday must be a non-empty string")
    if not mode_s:
        errors.append("mode must be a non-empty string")
    if not nature_s:
        errors.append("nature_of_consignment must be a non-empty string")

    valid_weekdays = {
        "monday", "tuesday", "wednesday", "thursday",
        "friday", "saturday", "sunday",
    }
    if booking_weekday_s not in valid_weekdays:
        errors.append(
            f"booking_weekday must be a weekday name, got '{booking_weekday_s}'"
        )

    try:
        pieces = int(total_pieces)
        if pieces <= 0:
            errors.append("total_pieces must be a positive integer")
    except (TypeError, ValueError):
        errors.append("total_pieces must be a valid integer")

    for name, val in [
        ("actual_weight", actual_weight),
        ("volumetric_weight", volumetric_weight),
        ("chargeable_weight", chargeable_weight),
    ]:
        try:
            v = float(val)
            if v <= 0:
                errors.append(f"{name} must be positive")
        except (TypeError, ValueError):
            errors.append(f"{name} must be a valid number")

    if errors:
        raise ValueError("; ".join(errors))

    return PredictionInput(
        origin=origin_s,
        destination=destination_s,
        booking_weekday=booking_weekday_s,
        mode=mode_s,
        nature_of_consignment=nature_s,
        total_pieces=pieces,
        actual_weight=float(actual_weight),
        volumetric_weight=float(volumetric_weight),
        chargeable_weight=float(chargeable_weight),
    )


# ---------------------------------------------------------------------------
# Predictor singleton
# ---------------------------------------------------------------------------
class DTDCPredictor:
    """Lazy-loaded predictor that wraps the saved pipeline.

    Usage::

        predictor = DTDCPredictor()
        result = predictor.predict(
            origin="Mumbai",
            destination="Pune",
            booking_weekday="Monday",
            mode="Surface",
            nature_of_consignment="Dox",
            total_pieces=1,
            actual_weight=0.5,
            volumetric_weight=0.8,
            chargeable_weight=0.5,
        )
        # result.predicted_days == 2.34
    """

    _instance: Optional[DTDCPredictor] = None
    _pipeline: Optional[Pipeline] = None
    _meta: Optional[dict[str, Any]] = None

    def __new__(cls) -> DTDCPredictor:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load(self, path: str | Path = MODEL_PATH) -> None:
        """Load the pipeline from disk.  Raises FileNotFoundError if missing."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"DTDC model not found at {path}. "
                "Train first with ``python -m dtdc_model train``"
            )
        self._pipeline = joblib.load(path)
        meta_path = path.with_suffix(".meta.json")
        if meta_path.is_file():
            with open(meta_path, "r", encoding="utf-8") as f:
                self._meta = json.load(f)

    @property
    def pipeline(self) -> Pipeline:
        if self._pipeline is None:
            self.load()
        if self._pipeline is None:
            raise RuntimeError(
                "DTDC model not loaded. Ensure the model artifact exists "
                f"at {MODEL_PATH}."
            )
        return self._pipeline

    @property
    def meta(self) -> dict[str, Any]:
        if self._meta is None:
            self.load()
            if self._meta is None:
                raise RuntimeError(
                    "DTDC model metadata not loaded. Ensure the meta file "
                    f"exists at {META_PATH}."
                )
        return self._meta

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    @dataclass
    class PredictionResult:
        predicted_days: float
        model_version: str
        algorithm: str

    def predict(
        self,
        *,
        origin: str,
        destination: str,
        booking_weekday: str,
        mode: str,
        nature_of_consignment: str,
        total_pieces: Any,
        actual_weight: Any,
        volumetric_weight: Any,
        chargeable_weight: Any,
    ) -> PredictionResult:
        """Validate inputs and return a prediction.

        Unknown categories (cities, weekday spellings, modes not seen during
        training) are handled safely by the OneHotEncoder's
        ``handle_unknown="ignore"`` setting.
        """
        parsed = parse_prediction_input(
            origin=origin,
            destination=destination,
            booking_weekday=booking_weekday,
            mode=mode,
            nature_of_consignment=nature_of_consignment,
            total_pieces=total_pieces,
            actual_weight=actual_weight,
            volumetric_weight=volumetric_weight,
            chargeable_weight=chargeable_weight,
        )
        df = parsed.to_dataframe()
        out = self.pipeline.predict(df)
        days = float(np.maximum(out[0], 0.5))  # clamp to sane minimum
        meta = self.meta
        return self.PredictionResult(
            predicted_days=round(days, 4),
            model_version=meta.get("model_version", MODEL_VERSION),
            algorithm=meta.get("algorithm", MODEL_ALGORITHM),
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@dataclass
class TrainResult:
    """Metrics from a completed training run."""
    mae: float
    rmse: float
    r2: float
    train_rows: int
    test_rows: int
    model_version: str
    artifact_path: str


def train(csv_path: str | Path = CSV_PATH) -> TrainResult:
    """Train the tuned HGB pipeline on the full DTDC dataset and persist it.

    Steps
    -----
    1. Load and pre-process the CSV using the same logic as Phase 1.
    2. Split train / test (80/20, random_state=42).
    3. Build and fit the pipeline.
    4. Evaluate on the hold-out set.
    5. Persist the pipeline as a versioned ``.joblib`` file.
    6. Write a companion ``.meta.json`` with version info + metrics.

    Returns a ``TrainResult`` dataclass.
    """
    print(f"Reading and preprocessing DTDC dataset ... ", end="", flush=True)
    frame, report = preprocess_dtdc_csv(csv_path)
    print(f"{len(frame):,} rows ready.")

    features = frame[FEATURE_COLUMNS]
    target = frame[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )

    print(f"Train set: {len(x_train):,} rows | Test set: {len(x_test):,} rows")

    pipeline = _build_pipeline()

    print("Fitting tuned HGB pipeline ... ", end="", flush=True)
    pipeline.fit(x_train, y_train)
    print("done.")

    preds = pipeline.predict(x_test)
    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))

    print(f"Hold-out MAE : {mae:.4f} days")
    print(f"Hold-out RMSE: {rmse:.4f} days")
    print(f"Hold-out R²  : {r2:.4f}")

    # ---- Persist ----
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved  -> {MODEL_PATH}")

    meta: dict[str, Any] = {
        "model_version": MODEL_VERSION,
        "algorithm": MODEL_ALGORITHM,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "dataset_version": DATASET_VERSION,
        "dataset_rows": report.valid_rows,
        "features": list(FEATURE_COLUMNS),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "numeric_features": list(NUMERIC_FEATURES),
        "target": TARGET_COLUMN,
        "preprocessing_version": PREPROCESSING_VERSION,
        "hyperparameters": dict(TUNED_HYPERPARAMETERS),
        "metrics": {
            "mae_days": round(mae, 4),
            "rmse_days": round(rmse, 4),
            "r2": round(r2, 4),
        },
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
    }

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Metadata     -> {META_PATH}")

    return TrainResult(
        mae=mae,
        rmse=rmse,
        r2=r2,
        train_rows=len(x_train),
        test_rows=len(x_test),
        model_version=MODEL_VERSION,
        artifact_path=str(MODEL_PATH),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli_train() -> None:
    result = train()
    print()
    print("=" * 60)
    print(f"  DTDC model v{result.model_version} trained successfully.")
    print(f"  MAE: {result.mae:.4f} days | RMSE: {result.rmse:.4f} | R²: {result.r2:.4f}")
    print(f"  Artifact: {result.artifact_path}")
    print("=" * 60)


def _cli_predict() -> None:
    predictor = DTDCPredictor()
    try:
        predictor.load()
    except FileNotFoundError:
        print("Model not found. Run ``python -m dtdc_model train`` first.",
              file=sys.stderr)
        sys.exit(1)

    # Read from stdin as JSON lines or prompt interactively
    if not sys.stdin.isatty():
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Invalid JSON: {exc}", file=sys.stderr)
                continue
            try:
                result = predictor.predict(**data)
                print(json.dumps({
                    "predicted_days": result.predicted_days,
                    "model_version": result.model_version,
                    "algorithm": result.algorithm,
                }))
            except ValueError as exc:
                print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return

    # Interactive mode
    print("Enter prediction inputs (blank line to exit):")
    while True:
        try:
            origin = input("  origin: ").strip()
            if not origin:
                break
            dest = input("  destination: ").strip()
            weekday = input("  booking_weekday (e.g. Monday): ").strip()
            mode = input("  mode (e.g. Surface): ").strip()
            noc = input("  nature_of_consignment (Dox/Non-Dox): ").strip()
            pieces = input("  total_pieces: ").strip()
            act_wt = input("  actual_weight: ").strip()
            vol_wt = input("  volumetric_weight: ").strip()
            chg_wt = input("  chargeable_weight: ").strip()
        except EOFError:
            break

        try:
            result = predictor.predict(
                origin=origin,
                destination=dest,
                booking_weekday=weekday,
                mode=mode,
                nature_of_consignment=noc,
                total_pieces=pieces,
                actual_weight=act_wt,
                volumetric_weight=vol_wt,
                chargeable_weight=chg_wt,
            )
            print(f"  >> Predicted delivery: {result.predicted_days:.2f} days "
                  f"(model v{result.model_version})")
        except ValueError as exc:
            print(f"  >> Error: {exc}")
        print()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        _cli_train()
    elif len(sys.argv) > 1 and sys.argv[1] == "predict":
        _cli_predict()
    else:
        print("Usage: python -m dtdc_model <train|predict>")
        print()
        print("  train    — train and persist the tuned HGB model")
        print("  predict  — interactive or JSON-lines prediction")
        sys.exit(1)
