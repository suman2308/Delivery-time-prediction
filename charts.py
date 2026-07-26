"""Generate dark-theme tailored matplotlib analytics plots for dashboard."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import database as db
import ml_model

# Apply modern dark theme aesthetics to Matplotlib
plt.style.use("dark_background")
PLT_BG = "#111827"
PLT_CARD = "#1f2937"
PLT_TEXT = "#f3f4f6"
PLT_ACCENT = "#38bdf8"
PLT_GRID = "#374151"


def ensure_plots_dir() -> None:
    os.makedirs(config.PLOTS_DIR, exist_ok=True)


def plot_delivery_vs_distance() -> str:
    ensure_plots_dir()
    rows = db.fetch_orders_for_training()
    path = os.path.join(config.PLOTS_DIR, "distance_vs_time.png")
    if not rows:
        _empty_chart("No order data available", path)
        return path
    df = ml_model.rows_to_dataframe(rows)

    fig, ax = plt.subplots(figsize=(6.5, 4.2), facecolor=PLT_BG)
    ax.set_facecolor(PLT_CARD)
    
    ax.scatter(df["distance"], df["delivery_time"], alpha=0.6, c=PLT_ACCENT, edgecolors="none", s=35)
    ax.set_xlabel("Distance (km)", color=PLT_TEXT, fontsize=10)
    ax.set_ylabel("Delivery Time (mins)", color=PLT_TEXT, fontsize=10)
    ax.set_title("Delivery Time vs. Distance", color=PLT_TEXT, fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.3, color=PLT_GRID)
    ax.tick_params(colors=PLT_TEXT)
    for spine in ax.spines.values():
        spine.set_color(PLT_GRID)

    plt.tight_layout()
    plt.savefig(path, dpi=140, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    return path


def plot_traffic_impact() -> str:
    ensure_plots_dir()
    rows = db.fetch_orders_for_training()
    path = os.path.join(config.PLOTS_DIR, "traffic_impact.png")
    if not rows:
        _empty_chart("No order data available", path)
        return path
    df = ml_model.rows_to_dataframe(rows)
    order = ["Low", "Medium", "High"]
    grouped = df.groupby("traffic_level")["delivery_time"].mean().reindex(order)

    fig, ax = plt.subplots(figsize=(6, 4.2), facecolor=PLT_BG)
    ax.set_facecolor(PLT_CARD)
    
    colors = ["#34d399", "#fbbf24", "#f87171"]
    bars = ax.bar(grouped.index.astype(str), grouped.values, color=colors, width=0.55, edgecolor="none")
    
    # Value annotations on top of bars
    for bar in bars:
        height = bar.get_height()
        if not pd_isna(height):
            ax.annotate(
                f"{height:.1f} m",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color=PLT_TEXT,
                fontsize=9,
                fontweight="bold"
            )

    ax.set_ylabel("Avg Delivery Time (mins)", color=PLT_TEXT, fontsize=10)
    ax.set_xlabel("Traffic Condition", color=PLT_TEXT, fontsize=10)
    ax.set_title("Impact of Traffic Level", color=PLT_TEXT, fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3, color=PLT_GRID)
    ax.tick_params(colors=PLT_TEXT)
    for spine in ax.spines.values():
        spine.set_color(PLT_GRID)

    plt.tight_layout()
    plt.savefig(path, dpi=140, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    return path


def plot_pred_vs_actual() -> str:
    ensure_plots_dir()
    rows = db.fetch_predictions_with_orders(limit=300)
    pairs = [(r["predicted_time"], r["delivery_time"]) for r in rows if r["delivery_time"] is not None]
    path = os.path.join(config.PLOTS_DIR, "pred_vs_actual.png")
    if len(pairs) < 3:
        _empty_chart("Requires predictions linked to completed orders", path)
        return path
    pred, actual = zip(*pairs)

    fig, ax = plt.subplots(figsize=(6, 4.2), facecolor=PLT_BG)
    ax.set_facecolor(PLT_CARD)
    
    ax.scatter(actual, pred, alpha=0.6, c="#818cf8", edgecolors="none", s=35)
    lim = max(max(actual), max(pred))
    ax.plot([0, lim], [0, lim], linestyle="--", color="#9ca3af", alpha=0.6, label="Ideal (Exact Match)")
    
    ax.set_xlabel("Actual Time (mins)", color=PLT_TEXT, fontsize=10)
    ax.set_ylabel("Predicted Time (mins)", color=PLT_TEXT, fontsize=10)
    ax.set_title("Predicted vs. Actual Accuracy", color=PLT_TEXT, fontsize=12, fontweight="bold", pad=12)
    ax.legend(facecolor=PLT_CARD, edgecolor=PLT_GRID, labelcolor=PLT_TEXT)
    ax.grid(True, linestyle="--", alpha=0.3, color=PLT_GRID)
    ax.tick_params(colors=PLT_TEXT)
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
