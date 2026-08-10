"""Generate dark-theme tailored matplotlib analytics plots for dashboard.

Phase 5: All chart functions now draw from DTDC prediction audit data.
The legacy synthetic-model chart functions are removed.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import database as db

# Apply modern dark theme aesthetics to Matplotlib — matches the design system
# tokens (bg #070b14, surfaces rgba white 0.04, brand gradient #6d5ef9 → #38bdf8).
plt.style.use("dark_background")
PLT_BG = "#0b1120"
PLT_CARD = "#131c33"
PLT_TEXT = "#eef2f9"
PLT_ACCENT = "#6d5ef9"
PLT_GRID = "#1f2b44"


def ensure_plots_dir() -> None:
    os.makedirs(config.PLOTS_DIR, exist_ok=True)


def _cache_valid(path: str) -> bool:
    """Return True if chart PNG is newer than the database file (cached)."""
    if not os.path.isfile(path):
        return False
    db_path = config.DATABASE_PATH
    if not os.path.isfile(db_path):
        return False
    return os.path.getmtime(path) > os.path.getmtime(db_path)


def plot_mode_impact(user_id=None) -> str:
    """Bar chart: average predicted delivery days by shipment mode.

    user_id scopes the chart to one account (personal dashboard); None is the
    platform-wide view (admin analytics). A per-user chart file keeps each
    user's picture private.
    """
    ensure_plots_dir()
    suffix = f"_u{user_id}" if user_id is not None else ""
    path = os.path.join(config.PLOTS_DIR, f"mode_impact{suffix}.png")
    if _cache_valid(path):
        return path
    rows = db.fetch_dtdc_predictions(limit=10000, user_id=user_id)
    if not rows:
        _empty_chart("No predictions yet — submit a prediction to see charts", path)
        return path
    df = db.rows_to_dataframe(rows)
    grouped = df.groupby("mode")["predicted_days"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(6, 4.2), facecolor=PLT_BG)
    ax.set_facecolor(PLT_CARD)
    colors_list = ["#6d5ef9", "#38bdf8", "#34d399"][:len(grouped)]
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


def plot_prediction_distribution(user_id=None) -> str:
    """Histogram: distribution of predicted delivery days."""
    ensure_plots_dir()
    suffix = f"_u{user_id}" if user_id is not None else ""
    path = os.path.join(config.PLOTS_DIR, f"pred_distribution{suffix}.png")
    if _cache_valid(path):
        return path
    rows = db.fetch_dtdc_predictions(limit=10000, user_id=user_id)
    if not rows:
        _empty_chart("No predictions yet — submit a prediction to see charts", path)
        return path
    df = db.rows_to_dataframe(rows)
    values = df["predicted_days"].values

    fig, ax = plt.subplots(figsize=(6, 4.2), facecolor=PLT_BG)
    ax.set_facecolor(PLT_CARD)
    ax.hist(values, bins=30, color=PLT_ACCENT, edgecolor="none", alpha=0.8)
    ax.axvline(
        values.mean(), color="#38bdf8", linestyle="--", linewidth=1.5,
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


def plot_top_routes(user_id=None) -> str:
    """Horizontal bar chart: top origin-destination pairs by prediction count."""
    ensure_plots_dir()
    suffix = f"_u{user_id}" if user_id is not None else ""
    path = os.path.join(config.PLOTS_DIR, f"top_routes{suffix}.png")
    if _cache_valid(path):
        return path
    rows = db.fetch_dtdc_predictions(limit=10000, user_id=user_id)
    if not rows:
        _empty_chart("No predictions yet — submit a prediction to see charts", path)
        return path
    df = db.rows_to_dataframe(rows)
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
    ax.barh(y_pos, route_counts.values, color="#38bdf8", edgecolor="none", height=0.6)
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
    """Create a placeholder chart for empty-data state."""
    fig, ax = plt.subplots(figsize=(5, 3), facecolor=PLT_BG)
    ax.set_facecolor(PLT_CARD)
    ax.text(0.5, 0.5, message, ha="center", va="center", color=PLT_TEXT, fontsize=10)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=100, facecolor=fig.get_facecolor())
    plt.close()
