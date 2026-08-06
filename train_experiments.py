"""DTDC Model Experiment Harness — base, hybrid and stacking comparisons.

This module adapts the Kaggle notebook "DTDC Advanced Delivery Prediction" for
the CourierAI platform. It benchmarks five base models (Random Forest,
XGBoost, CatBoost, SVR, MLP) plus voting/stacking hybrids and stacking
combinations, for BOTH regression (delivery days) and classification
(delayed / not delayed), on the same held-out test split.

Consumers:
  - ``python train_experiments.py --scope smoke``   — CLI (smoke/quick/reduced/full)
  - ``run_experiment(scope, progress=...)``          — used by the admin panel
  - ``load_experiment_results()``                    — read the JSON for /model-comparison

Artifacts:
  - ``models/experiment_results.json`` — final results consumed by the web app
  - ``dtdc_results/``                  — per-scope progress CSVs (crash-resume) +
    ``status.json`` for the admin UI
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR

warnings.filterwarnings("ignore")

# Optional libraries — degrade gracefully when unavailable.
try:
    from xgboost import XGBClassifier, XGBRegressor

    HAS_XGB = True
except Exception:  # pragma: no cover - import guard
    HAS_XGB = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor

    HAS_CAT = True
except Exception:  # pragma: no cover - import guard
    HAS_CAT = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "DTDC_Improved_Dataset.csv")
RESULTS_PATH = os.path.join(BASE_DIR, "models", "experiment_results.json")
# Artifacts always land in the project tree, regardless of the launch cwd.
PROGRESS_DIR = os.path.join(BASE_DIR, "dtdc_results")
STATUS_PATH = os.path.join(PROGRESS_DIR, "status.json")

MODEL_NAMES = {
    "rf": "RandomForest",
    "xgb": "XGBoost",
    "cat": "CatBoost",
    "svr": "SVR",
    "svc": "SVC",
    "mlp": "MLP",
}

# Stacking combinations from the notebook's SECTION 3.
COMBINATIONS = [
    ["rf", "xgb"], ["rf", "svm"], ["rf", "mlp"],
    ["xgb", "svm"], ["xgb", "mlp"], ["cat", "svm"], ["cat", "mlp"],
    ["rf", "xgb", "svm"], ["rf", "xgb", "mlp"],
    ["rf", "cat", "svm"], ["rf", "cat", "mlp"],
    ["xgb", "cat", "svm"], ["xgb", "cat", "mlp"],
    ["rf", "xgb", "cat", "svm"], ["rf", "xgb", "cat", "mlp"],
    ["rf", "xgb", "svm", "mlp"],
    ["rf", "xgb", "cat", "svm", "mlp"],
]

REDUCED_COMBINATIONS = [
    ["rf", "xgb"],                      # bagging + boosting
    ["rf", "svm"],                      # bagging + kernel-based
    ["rf", "mlp"],                      # bagging + neural network
    ["rf", "xgb", "svm"],               # bagging + boosting + kernel
    ["rf", "xgb", "mlp"],               # bagging + boosting + neural
    ["rf", "xgb", "cat", "svm", "mlp"],  # all five (flagship)
]

# Experiment scopes. `rows` subsamples the dataset (None = full);
# `stacking` selects which combination set to run.
SCOPES = {
    "smoke": {"rows": 600, "stacking": None},            # seconds — CI/admin sanity
    "quick": {"rows": 5000, "stacking": None},           # ~1-2 min
    "reduced": {"rows": None, "stacking": "reduced"},    # full data, 6 combos
    "full": {"rows": None, "stacking": "full"},          # full data, 17 combos
}

METRIC_COLS = {
    "regression": ["MAE", "RMSE", "R2"],
    "classification": ["Accuracy", "Precision", "Recall", "F1"],
}

FEATURES = [
    "Origin", "Destination",
    "Sender_State", "Receiver_State",
    "Mode", "Mode_of_Payment",
    "Nature_of_Consignment",
    "Risk_Surcharge",
    "Total_Pieces",
    "Actual_Wt",
    "Volumetric_Wt",
    "Chargeable_Wt",
    "Tariff",
    "VAS_Charges",
    "Total_Amount",
    "Weight_Diff",
    "Charge_Ratio",
    "VAS_Ratio",
    "Amount_Per_Piece",
    "Route",
]

CAT_FEATURES = [
    "Origin", "Destination",
    "Sender_State", "Receiver_State",
    "Mode", "Mode_of_Payment",
    "Nature_of_Consignment",
    "Risk_Surcharge",
    "Route",
]

NUM_FEATURES = [
    "Total_Pieces",
    "Actual_Wt",
    "Volumetric_Wt",
    "Chargeable_Wt",
    "Tariff",
    "VAS_Charges",
    "Total_Amount",
    "Weight_Diff",
    "Charge_Ratio",
    "VAS_Ratio",
    "Amount_Per_Piece",
    "Route_Frequency",
]


# ---------------------------------------------------------------------------
# Status tracking (admin UI polls status.json)
# ---------------------------------------------------------------------------
def write_status(state: str, message: str = "", **extra) -> None:
    """Persist experiment status for the admin panel to poll (atomic replace)."""
    os.makedirs(os.path.dirname(STATUS_PATH) or ".", exist_ok=True)
    payload = {
        "state": state,  # idle | running | done | error
        "message": message,
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }
    payload.update(extra)
    tmp = STATUS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, STATUS_PATH)


def read_status() -> dict:
    """Read the last experiment status (defaults to idle)."""
    try:
        with open(STATUS_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {"state": "idle", "message": "No experiment run yet."}


# ---------------------------------------------------------------------------
# Results (consumed by /model-comparison)
# ---------------------------------------------------------------------------
def load_experiment_results() -> dict | None:
    """Return the latest experiment results JSON, or None if unavailable."""
    try:
        with open(RESULTS_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def save_experiment_results(results: dict) -> None:
    """Write results to models/experiment_results.json (atomic replace)."""
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    tmp = RESULTS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    os.replace(tmp, RESULTS_PATH)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def evaluate_regression(y_true, y_pred):
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
    }


def evaluate_classification(y_true, y_pred):
    return {
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "F1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }


def train_and_evaluate(model, X_tr, X_te, y_tr, y_true, task):
    """Fit a model; return (metrics, prediction_time_s)."""
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    elapsed = time.perf_counter() - t0
    if task == "regression":
        return evaluate_regression(y_true, y_pred), round(elapsed, 3)
    return evaluate_classification(y_true, y_pred), round(elapsed, 3)


# ---------------------------------------------------------------------------
# Model factories — FRESH, unfitted instances every call
# ---------------------------------------------------------------------------
def _regressor_set():
    models = {
        "rf": RandomForestRegressor(
            n_estimators=150, max_depth=15, random_state=42, n_jobs=-1
        ),
        "svr": SVR(kernel="rbf", gamma="scale"),
        "mlp": MLPRegressor(
            hidden_layer_sizes=(100, 50), max_iter=1000,
            early_stopping=True, n_iter_no_change=20,
            validation_fraction=0.1, random_state=42,
        ),
    }
    if HAS_XGB:
        models["xgb"] = XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=8,
            tree_method="hist", random_state=42,
        )
    if HAS_CAT:
        models["cat"] = CatBoostRegressor(
            iterations=200, learning_rate=0.05, depth=8,
            verbose=0, allow_writing_files=False, random_state=42,
        )
    return models


def _classifier_set():
    models = {
        "rf": RandomForestClassifier(
            n_estimators=150, max_depth=15, random_state=42, n_jobs=-1
        ),
        "svc": SVC(kernel="rbf", gamma="scale", probability=True, random_state=42),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(100, 50), max_iter=1000,
            early_stopping=True, n_iter_no_change=20,
            validation_fraction=0.1, random_state=42,
        ),
    }
    if HAS_XGB:
        models["xgb"] = XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=8,
            eval_metric="logloss", tree_method="hist", random_state=42,
        )
    if HAS_CAT:
        models["cat"] = CatBoostClassifier(
            iterations=200, learning_rate=0.05, depth=8,
            verbose=0, allow_writing_files=False, random_state=42,
        )
    return models


def _resolve_keys(comb, task):
    """Resolve the 'svm' placeholder to 'svr' or 'svc'."""
    key = "svr" if task == "regression" else "svc"
    return [key if k == "svm" else k for k in comb]


def _available_keys(task):
    """Keys actually usable for this task (respects missing libraries)."""
    pool = _regressor_set() if task == "regression" else _classifier_set()
    return set(pool.keys())


# ---------------------------------------------------------------------------
# Data loading + feature engineering (identical to the notebook)
# ---------------------------------------------------------------------------
def load_data(data_path: str | None = None, max_rows: int | None = None):
    """Read + engineer features. Returns (X, y_reg, y_cls) plus row count."""
    path = data_path or os.environ.get("DTDC_DATA_PATH") or DEFAULT_DATA_PATH
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"DTDC dataset not found at {path}. Set DTDC_DATA_PATH or place the "
            "CSV in the project root."
        )
    data = pd.read_csv(path)
    if max_rows and len(data) > max_rows:
        data = data.sample(n=max_rows, random_state=42)
    data.columns = data.columns.str.replace(" ", "_")

    y_reg = data["Improved_Delivery_Days"]
    y_cls = data["Improved_Delayed"]

    # Booking-time monetary features are legitimate predictors (no leakage).
    data["Weight_Diff"] = data["Volumetric_Wt"] - data["Actual_Wt"]
    data["Charge_Ratio"] = data["Total_Amount"] / (data["Chargeable_Wt"] + 1)
    data["VAS_Ratio"] = data["VAS_Charges"] / (data["Total_Amount"] + 1)
    data["Amount_Per_Piece"] = data["Total_Pieces"] / (data["Actual_Wt"] + 1)
    data["Route"] = (
        data["Origin"].astype(str) + "_TO_" + data["Destination"].astype(str)
    )

    X = data[FEATURES]
    return X, y_reg, y_cls, len(data)


def build_preprocessor():
    """ColumnTransformer: median-impute+scale numerics, one-hot categories."""
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]),
                NUM_FEATURES,
            ),
            (
                "cat",
                Pipeline([
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore")),
                ]),
                CAT_FEATURES,
            ),
        ]
    )


def prepare_data(data_path: str | None = None, max_rows: int | None = None):
    """Full pipeline: load, engineer, split, frequency-encode, preprocess.

    Route_Frequency is fitted on the TRAINING split only (anti data-leakage,
    exactly as in the notebook).
    """
    X, y_reg, y_cls, n_rows = load_data(data_path, max_rows)

    X_train, X_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls = (
        train_test_split(
            X, y_reg, y_cls,
            test_size=0.20, random_state=42, stratify=y_cls,
        )
    )

    route_freq = X_train["Route"].value_counts()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train["Route_Frequency"] = X_train["Route"].map(route_freq)
    X_test["Route_Frequency"] = X_test["Route"].map(route_freq)

    preprocessor = build_preprocessor()
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    return (
        X_train_p, X_test_p,
        y_train_reg, y_test_reg,
        y_train_cls, y_test_cls,
        n_rows,
    )


# ---------------------------------------------------------------------------
# Experiment sections
# ---------------------------------------------------------------------------
def _run_base(task, X_tr, X_te, y_tr, y_true, progress=None):
    """SECTION 1: evaluate every base model individually."""
    factories = _regressor_set if task == "regression" else _classifier_set
    models = factories()
    results = []
    total = len(models)
    for i, (key, model) in enumerate(models.items(), start=1):
        if progress:
            progress(f"base-{task}", MODEL_NAMES[key], i, total)
        metrics, train_s = train_and_evaluate(
            model, X_tr, X_te, y_tr, y_true, task
        )
        results.append({"Model": MODEL_NAMES[key], **metrics, "train_s": train_s})
    results.sort(key=lambda r: r["R2"] if task == "regression" else r["F1"],
                 reverse=True)
    return results


def _run_hybrid(task, X_tr, X_te, y_tr, y_true, progress=None):
    """SECTION 2: voting + stacking hybrids over ALL available base models."""
    if progress:
        progress(f"hybrid-{task}", "Voting", 1, 2)
    fresh = _regressor_set() if task == "regression" else _classifier_set()
    estimators = [(k, fresh[k]) for k in fresh]

    if task == "regression":
        voting = VotingRegressor(estimators=estimators)
        stacking = StackingRegressor(
            estimators=estimators,
            final_estimator=CatBoostRegressor(
                iterations=200, verbose=0,
                allow_writing_files=False, random_state=42,
            ) if HAS_CAT else RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
            cv=3,
        )
    else:
        voting = VotingClassifier(estimators=estimators, voting="soft")
        stacking = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=3000),
            cv=3,
        )

    results = []
    metrics, train_s = train_and_evaluate(
        voting, X_tr, X_te, y_tr, y_true, task
    )
    results.append({"Hybrid": "Voting", **metrics, "train_s": train_s})

    if progress:
        progress(f"hybrid-{task}", "Stacking", 2, 2)
    metrics, train_s = train_and_evaluate(
        stacking, X_tr, X_te, y_tr, y_true, task
    )
    results.append({"Hybrid": "Stacking", **metrics, "train_s": train_s})
    return results


def _run_stacking_combinations(task, X_tr, X_te, y_tr, y_true, scope_stack,
                               progress=None):
    """SECTION 3: stacking over base-model subsets (crash-resumable)."""
    combo_list = COMBINATIONS if scope_stack == "full" else REDUCED_COMBINATIONS
    results = []
    available = _available_keys(task)
    progress_csv = os.path.join(
        PROGRESS_DIR, f"stacking_{task}_{scope_stack}.csv"
    )

    completed = {}
    if os.path.exists(progress_csv):
        try:
            done = pd.read_csv(progress_csv)
            for _, row in done.iterrows():
                completed[row["Combination"]] = row
        except Exception:
            completed = {}

    for i, comb in enumerate(combo_list, start=1):
        keys = _resolve_keys(comb, task)
        # Skip combinations that reference a library we don't have.
        if not all(k in available for k in keys):
            continue
        comb_name = "+".join(MODEL_NAMES[k].upper() for k in keys)
        if progress:
            progress(f"stacking-{task}", comb_name, i, len(combo_list))

        if comb_name in completed:
            row = completed[comb_name]
            metrics = {c: float(row[c]) for c in METRIC_COLS[task]}
            results.append({"Combination": comb_name, **metrics,
                            "train_s": float(row.get("train_s", 0))})
            continue

        fresh = _regressor_set() if task == "regression" else _classifier_set()
        base = [(k, fresh[k]) for k in keys]

        if task == "regression":
            stack = StackingRegressor(
                estimators=base,
                final_estimator=CatBoostRegressor(
                    iterations=200, verbose=0,
                    allow_writing_files=False, random_state=42,
                ) if HAS_CAT else RandomForestRegressor(
                    n_estimators=100, random_state=42, n_jobs=-1
                ),
                cv=3,
            )
        else:
            stack = StackingClassifier(
                estimators=base,
                final_estimator=LogisticRegression(max_iter=3000),
                cv=3,
            )

        metrics, train_s = train_and_evaluate(
            stack, X_tr, X_te, y_tr, y_true, task
        )
        row = {"Combination": comb_name, **metrics, "train_s": train_s}
        results.append(row)

        # Persist immediately (crash-resume support)
        os.makedirs(PROGRESS_DIR, exist_ok=True)
        pd.DataFrame([row]).to_csv(
            progress_csv, mode="a",
            header=not os.path.exists(progress_csv), index=False,
        )

    sort_key = "R2" if task == "regression" else "F1"
    results.sort(key=lambda r: r[sort_key], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_experiment(scope: str = "smoke", data_path: str | None = None,
                   progress=None) -> dict:
    """Run every section for the given scope and persist the results JSON.

    ``progress`` is an optional callable(section, label, index, total) that
    receives progress updates (the admin worker wires this to status.json).
    """
    if scope not in SCOPES:
        raise ValueError(f"Unknown scope {scope!r}; choose from {sorted(SCOPES)}")

    spec = SCOPES[scope]
    if progress:
        progress("prepare", "Loading + preprocessing", 0, 1)
    X_tr, X_te, ytr_reg, yte_reg, ytr_cls, yte_cls, n_rows = prepare_data(
        data_path, spec["rows"]
    )

    results = {
        "scope": scope,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_rows": n_rows,
        "libs": {"xgboost": HAS_XGB, "catboost": HAS_CAT},
        "regression": {},
        "classification": {},
    }

    results["regression"]["base"] = _run_base(
        "regression", X_tr, X_te, ytr_reg, yte_reg, progress
    )
    results["regression"]["hybrid"] = _run_hybrid(
        "regression", X_tr, X_te, ytr_reg, yte_reg, progress
    )
    if spec["stacking"]:
        results["regression"]["stacking"] = _run_stacking_combinations(
            "regression", X_tr, X_te, ytr_reg, yte_reg, spec["stacking"], progress
        )

    results["classification"]["base"] = _run_base(
        "classification", X_tr, X_te, ytr_cls, yte_cls, progress
    )
    results["classification"]["hybrid"] = _run_hybrid(
        "classification", X_tr, X_te, ytr_cls, yte_cls, progress
    )
    if spec["stacking"]:
        results["classification"]["stacking"] = _run_stacking_combinations(
            "classification", X_tr, X_te, ytr_cls, yte_cls, spec["stacking"], progress
        )

    save_experiment_results(results)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the CourierAI DTDC model experiments."
    )
    parser.add_argument(
        "--scope", choices=sorted(SCOPES), default="smoke",
        help="Experiment scope (smoke = seconds, full = hours).",
    )
    parser.add_argument("--data-path", default=None,
                        help="Path to the DTDC CSV (default: project root).")
    args = parser.parse_args()

    write_status("running", f"Starting {args.scope} experiment…", scope=args.scope)

    def _progress(section, label, index, total):
        msg = f"[{section}] {label} ({index}/{total})"
        write_status("running", msg, scope=args.scope,
                     progress={"section": section, "label": label,
                               "index": index, "total": total})

    t0 = time.time()
    try:
        results = run_experiment(args.scope, args.data_path, progress=_progress)
        write_status("done", f"Finished in {time.time() - t0:.0f}s.",
                     scope=args.scope)
        print(f"\nResults written to {RESULTS_PATH}")
        for task in ("regression", "classification"):
            sort_key = "R2" if task == "regression" else "F1"
            best = None
            for group in results[task].values():
                for row in group:
                    if best is None or row[sort_key] > best[sort_key]:
                        best = row
            label = best.get("Model") or best.get("Hybrid") or best.get("Combination")
            print(f"Best {task}: {label} ({sort_key}={best[sort_key]})")
    except Exception as exc:  # pragma: no cover - CLI error path
        write_status("error", f"{type(exc).__name__}: {exc}", scope=args.scope)
        raise


if __name__ == "__main__":
    main()
