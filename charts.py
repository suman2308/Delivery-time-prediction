"""Generate matplotlib analytics PNGs for the web dashboard."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import database as db
import ml_model


def ensure_plots_dir() -> None:
    os.makedirs(config.PLOTS_DIR, exist_ok=True)


def plot_delivery_vs_distance() -> str:
    ensure_plots_dir()
    rows = db.fetch_orders_for_training()
    if not rows:
        path = os.path.join(config.PLOTS_DIR, "distance_vs_time.png")
        _empty_chart("No order data", path)
        return path
    df = ml_model.rows_to_dataframe(rows)
    plt.figure(figsize=(7, 4.5))
    plt.scatter(df["distance"], df["delivery_time"], alpha=0.45, c="#2563eb", edgecolors="none")
    plt.xlabel("Distance (km)")
    plt.ylabel("Delivery time (minutes)")
    plt.title("Delivery time vs distance")
    plt.grid(True, alpha=0.3)
    path = os.path.join(config.PLOTS_DIR, "distance_vs_time.png")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def plot_traffic_impact() -> str:
    ensure_plots_dir()
    rows = db.fetch_orders_for_training()
    if not rows:
        path = os.path.join(config.PLOTS_DIR, "traffic_impact.png")
        _empty_chart("No order data", path)
        return path
    df = ml_model.rows_to_dataframe(rows)
    order = ["Low", "Medium", "High"]
    grouped = df.groupby("traffic_level")["delivery_time"].mean().reindex(order)
    plt.figure(figsize=(6, 4.5))
    colors = ["#22c55e", "#eab308", "#ef4444"]
    plt.bar(grouped.index.astype(str), grouped.values, color=colors, edgecolor="#334155")
    plt.ylabel("Avg delivery time (minutes)")
    plt.xlabel("Traffic")
    plt.title("Traffic impact on delivery time")
    plt.grid(True, axis="y", alpha=0.3)
    path = os.path.join(config.PLOTS_DIR, "traffic_impact.png")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def plot_pred_vs_actual() -> str:
    ensure_plots_dir()
    rows = db.fetch_predictions_with_orders(limit=300)
    pairs = [(r["predicted_time"], r["delivery_time"]) for r in rows if r["delivery_time"] is not None]
    path = os.path.join(config.PLOTS_DIR, "pred_vs_actual.png")
    if len(pairs) < 3:
        _empty_chart("Need predictions linked to orders with actual times", path)
        return path
    pred, actual = zip(*pairs)
    plt.figure(figsize=(6, 6))
    plt.scatter(actual, pred, alpha=0.5, c="#7c3aed", edgecolors="none")
    lim = max(max(actual), max(pred))
    plt.plot([0, lim], [0, lim], "k--", alpha=0.35, label="Perfect match")
    plt.xlabel("Actual delivery time (minutes)")
    plt.ylabel("Predicted time (minutes)")
    plt.title("Predicted vs actual")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def _empty_chart(message: str, path: str) -> None:
    plt.figure(figsize=(5, 3))
    plt.text(0.5, 0.5, message, ha="center", va="center", fontsize=11)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
