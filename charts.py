"""Generate dark-theme tailored matplotlib analytics plots for dashboard.

Phase 5: All chart functions now draw from DTDC prediction audit data.
The legacy synthetic-model chart functions are removed.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import database as db

# Apply modern dark theme aesthetics to Matplotlib
plt.style.use("dark_background")
PLT_BG = "#111827"
PLT_CARD = "#1f2937"
PLT_TEXT = "#f3f4f6"
PLT_ACCENT = "#38bdf8"
PLT_GRID = "#374151"


def ensure_plots_dir() -> None:
    os.makedirs(config.PLOTS_DIR, exist_ok=True)


def _rows_to_df(rows: list) -> pd.DataFrame:
    """Convert SQLite Row list to DataFrame for charting."""
    if not rows:
        return pd.DataFrame()
    data = [{k: r[k] for k in r.keys()} for r in rows]
    return pd.DataFrame(data)


def plot_mode_impact() -> str:
    """Bar chart: average predicted delivery days by shipment mode."""
    ensure_plots_dir()
    path = os.path.join(config.PLOTS_DIR, "mode_impact.png")
    rows = db.fetch_dtdc_predictions(limit=10000)
    if not rows:
        _empty_chart("No DTDC predictions yet — submit a prediction first", path)
        return path
    df = _rows_to_df(rows)
    grouped = df.groupby("mode")["predicted_days"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(6, 4.2), facecolor=PLT_BG)
    ax.set_facecolor(PLT_CARD)
    colors_list = ["#38bdf8", "#818cf8", "#34d399"][:len(grouped)]
    bars = ax.bar(
        grouped.index.astype(str), grouped.values,
        color=colors_list, width=0.55, edgecolor="none",
    )
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}d",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom",
            color=PLT_TEXT, fontsize=9, fontweight="bold",
        )
    ax.set_ylabel("Avg Predicted Days", color=PLT_TEXT, fontsize=10)
    ax.set_xlabel("Shipment Mode", color=PLT_TEXT, fontsize=10)
    ax.set_title("Predicted Duration by Mode", color=PLT_TEXT, fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3, color=PLT_GRID)
    ax.tick_params(colors=PLT_TEXT)
    for spine in ax.spines.values():
        spine.set_color(PLT_GRID)
    plt.tight_layout()
    plt.savefig(path, dpi=140, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    return path


def plot_prediction_distribution() -> str:
    """Histogram: distribution of predicted delivery days."""
    ensure_plots_dir()
    path = os.path.join(config.PLOTS_DIR, "pred_distribution.png")
    rows = db.fetch_dtdc_predictions(limit=10000)
    if not rows:
        _empty_chart("No DTDC predictions yet — submit a prediction first", path)
        return path
    df = _rows_to_df(rows)
    values = df["predicted_days"].values

    fig, ax = plt.subplots(figsize=(6, 4.2), facecolor=PLT_BG)
    ax.set_facecolor(PLT_CARD)
    ax.hist(values, bins=30, color=PLT_ACCENT, edgecolor="none", alpha=0.8)
    ax.axvline(
        values.mean(), color="#fbbf24", linestyle="--", linewidth=1.5,
        label=f"Mean: {values.mean():.2f}d",
    )
    ax.set_xlabel("Predicted Days", color=PLT_TEXT, fontsize=10)
    ax.set_ylabel("Frequency", color=PLT_TEXT, fontsize=10)
    ax.set_title("Distribution of Predicted Durations", color=PLT_TEXT, fontsize=12, fontweight="bold", pad=12)
    ax.legend(facecolor=PLT_CARD, edgecolor=PLT_GRID, labelcolor=PLT_TEXT, fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3, color=PLT_GRID)
    ax.tick_params(colors=PLT_TEXT)
    for spine in ax.spines.values():
        spine.set_color(PLT_GRID)
    plt.tight_layout()
    plt.savefig(path, dpi=140, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    return path


def plot_top_routes() -> str:
    """Horizontal bar chart: top origin-destination pairs by prediction count."""
    ensure_plots_dir()
    path = os.path.join(config.PLOTS_DIR, "top_routes.png")
    rows = db.fetch_dtdc_predictions(limit=10000)
    if not rows:
        _empty_chart("No DTDC predictions yet — submit a prediction first", path)
        return path
    df = _rows_to_df(rows)
    route_counts = (
        df.groupby(["origin", "destination"])
        .size()
        .sort_values(ascending=False)
        .head(10)
    )
    labels = [f"{o} \u2192 {d}" for o, d in route_counts.index]

    fig, ax = plt.subplots(figsize=(6.5, 4.2), facecolor=PLT_BG)
    ax.set_facecolor(PLT_CARD)
    y_pos = range(len(labels))
    ax.barh(y_pos, route_counts.values, color="#818cf8", edgecolor="none", height=0.6)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=8, color=PLT_TEXT)
    ax.set_xlabel("Prediction Count", color=PLT_TEXT, fontsize=10)
    ax.set_title("Top Routes by Prediction Volume", color=PLT_TEXT, fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3, color=PLT_GRID)
    ax.tick_params(colors=PLT_TEXT)
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_color(PLT_GRID)
    plt.tight_layout()
    plt.savefig(path, dpi=140, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    return path


def _empty_chart(message: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 3), facecolor=PLT_BG)
    ax.set_facecolor(PLT_CARD)
    ax.text(0.5, 0.5, message, ha="center", va="center", color=PLT_TEXT, fontsize=10)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=100, facecolor=fig.get_facecolor())
    plt.close()


def pd_isna(val) -> bool:
    return val != val
