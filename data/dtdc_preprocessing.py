"""Preprocess DTDC booking records into model-ready training data.

This module deliberately uses only booking-time, non-PII fields from the
DTDC dataset. It does not train or persist a model.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


CATEGORICAL_SOURCE_COLUMNS = (
    "Origin",
    "Destination",
    "Mode",
    "Nature of Consignment",
)
NUMERIC_SOURCE_COLUMNS = (
    "Total Pieces",
    "Actual Wt",
    "Volumetric Wt",
    "Chargeable Wt",
)
DATE_SOURCE_COLUMN = "Date"
TARGET_SOURCE_COLUMN = "Improved_Delivery_Days"
REQUIRED_SOURCE_COLUMNS = (
    *CATEGORICAL_SOURCE_COLUMNS,
    *NUMERIC_SOURCE_COLUMNS,
    DATE_SOURCE_COLUMN,
    TARGET_SOURCE_COLUMN,
)

FEATURE_COLUMNS = (
    "origin",
    "destination",
    "booking_weekday",
    "mode",
    "nature_of_consignment",
    "total_pieces",
    "actual_weight",
    "volumetric_weight",
    "chargeable_weight",
)
TARGET_COLUMN = "delivery_duration_days"


@dataclass(frozen=True)
class PreprocessingReport:
    """Audit details for one preprocessing run."""

    total_rows: int
    valid_rows: int
    rejected_rows: int
    duplicate_rows_removed: int
    missing_values: dict[str, int]
    invalid_values: dict[str, int]
    category_statistics: dict[str, pd.Series]
    target_distribution: pd.Series


def preprocess_dtdc_csv(csv_path: str | Path) -> tuple[pd.DataFrame, PreprocessingReport]:
    """Read a DTDC CSV and return validated model features plus the target.

    Only columns listed in ``REQUIRED_SOURCE_COLUMNS`` are read. This avoids
    bringing PII, identifiers, and post-delivery fields into the pipeline.
    """
    path = Path(csv_path)
    raw = pd.read_csv(path, usecols=list(REQUIRED_SOURCE_COLUMNS))
    return preprocess_dtdc_dataframe(raw)


def preprocess_dtdc_dataframe(raw: pd.DataFrame) -> tuple[pd.DataFrame, PreprocessingReport]:
    """Validate and transform a DTDC dataframe into the approved contract."""
    missing_columns = set(REQUIRED_SOURCE_COLUMNS) - set(raw.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"DTDC dataset is missing required columns: {names}")

    source = raw.loc[:, list(REQUIRED_SOURCE_COLUMNS)].copy()
    total_rows = len(source)

    categorical = pd.DataFrame(
        {
            "origin": _standardize_category(source["Origin"]),
            "destination": _standardize_category(source["Destination"]),
            "mode": _standardize_category(source["Mode"]),
            "nature_of_consignment": _standardize_category(
                source["Nature of Consignment"]
            ),
        }
    )
    booking_date = pd.to_datetime(source[DATE_SOURCE_COLUMN], format="%Y-%m-%d", errors="coerce")
    numeric = pd.DataFrame(
        {
            "total_pieces": pd.to_numeric(source["Total Pieces"], errors="coerce"),
            "actual_weight": pd.to_numeric(source["Actual Wt"], errors="coerce"),
            "volumetric_weight": pd.to_numeric(
                source["Volumetric Wt"], errors="coerce"
            ),
            "chargeable_weight": pd.to_numeric(
                source["Chargeable Wt"], errors="coerce"
            ),
            TARGET_COLUMN: pd.to_numeric(source[TARGET_SOURCE_COLUMN], errors="coerce"),
        }
    )

    missing_values = {
        "Origin": int(categorical["origin"].isna().sum()),
        "Destination": int(categorical["destination"].isna().sum()),
        "Date": int(booking_date.isna().sum()),
        "Mode": int(categorical["mode"].isna().sum()),
        "Nature of Consignment": int(
            categorical["nature_of_consignment"].isna().sum()
        ),
        "Total Pieces": int(numeric["total_pieces"].isna().sum()),
        "Actual Wt": int(numeric["actual_weight"].isna().sum()),
        "Volumetric Wt": int(numeric["volumetric_weight"].isna().sum()),
        "Chargeable Wt": int(numeric["chargeable_weight"].isna().sum()),
        "Improved_Delivery_Days": int(numeric[TARGET_COLUMN].isna().sum()),
    }

    total_pieces_is_integer = numeric["total_pieces"].mod(1).eq(0)
    invalid_values = {
        "Date": int(booking_date.isna().sum()),
        "Total Pieces": int(
            (~np.isfinite(numeric["total_pieces"]) | (numeric["total_pieces"] <= 0) | ~total_pieces_is_integer).sum()
        ),
        "Actual Wt": int(
            (~np.isfinite(numeric["actual_weight"]) | (numeric["actual_weight"] <= 0)).sum()
        ),
        "Volumetric Wt": int(
            (~np.isfinite(numeric["volumetric_weight"]) | (numeric["volumetric_weight"] <= 0)).sum()
        ),
        "Chargeable Wt": int(
            (~np.isfinite(numeric["chargeable_weight"]) | (numeric["chargeable_weight"] <= 0)).sum()
        ),
        "Improved_Delivery_Days": int(
            (~np.isfinite(numeric[TARGET_COLUMN]) | (numeric[TARGET_COLUMN] <= 0)).sum()
        ),
    }

    valid = categorical.notna().all(axis=1) & booking_date.notna()
    valid &= np.isfinite(numeric).all(axis=1)
    valid &= numeric["total_pieces"].gt(0) & total_pieces_is_integer
    valid &= numeric[["actual_weight", "volumetric_weight", "chargeable_weight"]].gt(0).all(axis=1)
    valid &= numeric[TARGET_COLUMN].gt(0)

    transformed = pd.concat(
        [categorical, numeric], axis=1
    ).loc[valid].copy()
    transformed.insert(
        2,
        "booking_weekday",
        booking_date.loc[valid].dt.day_name().str.casefold(),
    )
    transformed["total_pieces"] = transformed["total_pieces"].astype("int64")
    transformed = transformed.loc[:, [*FEATURE_COLUMNS, TARGET_COLUMN]]

    duplicate_mask = transformed.duplicated(
        subset=[*FEATURE_COLUMNS, TARGET_COLUMN], keep="first"
    )
    duplicate_rows_removed = int(duplicate_mask.sum())
    transformed = transformed.loc[~duplicate_mask].reset_index(drop=True)

    category_statistics = {
        column: transformed[column].value_counts().sort_index()
        for column in (
            "origin",
            "destination",
            "booking_weekday",
            "mode",
            "nature_of_consignment",
        )
    }
    target_distribution = transformed[TARGET_COLUMN].value_counts().sort_index()
    report = PreprocessingReport(
        total_rows=total_rows,
        valid_rows=len(transformed),
        rejected_rows=total_rows - int(valid.sum()),
        duplicate_rows_removed=duplicate_rows_removed,
        missing_values=missing_values,
        invalid_values=invalid_values,
        category_statistics=category_statistics,
        target_distribution=target_distribution,
    )
    return transformed, report


def _standardize_category(values: pd.Series) -> pd.Series:
    """Normalize category text without introducing semantic mappings."""
    normalized = values.astype("string").str.normalize("NFKC")
    normalized = normalized.str.strip().str.replace(r"\s+", " ", regex=True)
    normalized = normalized.mask(normalized.eq(""))
    return normalized.str.casefold()
