"""Interactive CLI for delivery time prediction."""
import sys

import database as db
import ml_model


def prompt_float(label: str) -> float:
    while True:
        raw = input(f"{label}: ").strip()
        try:
            return float(raw)
        except ValueError:
            print("Please enter a number.")


def prompt_int(label: str, lo: int = 0, hi: int = 23) -> int:
    while True:
        raw = input(f"{label} (0-23 hour): ").strip()
        try:
            v = int(raw)
            if lo <= v <= hi:
                return v
        except ValueError:
            pass
        print(f"Enter an integer between {lo} and {hi}.")


def prompt_choice(label: str, options: list[str]) -> str:
    opts = "/".join(options)
    while True:
        raw = input(f"{label} ({opts}): ").strip()
        for o in options:
            if raw.lower() == o.lower():
                return o
        print(f"Choose one of: {', '.join(options)}")


def main():
    db.init_db()
    try:
        pipe = ml_model.load_pipeline()
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        print("Run: python seed_data.py && python train_model.py", file=sys.stderr)
        sys.exit(1)

    print("Smart Delivery Time Prediction (CLI)")
    distance = prompt_float("Distance (km)")
    hour = prompt_int("Time of day")
    traffic = prompt_choice("Traffic", ["Low", "Medium", "High"])
    weather = prompt_choice("Weather", ["Clear", "Rainy"])
    pred = ml_model.predict_delivery(distance, hour, traffic, weather, pipeline=pipe)
    print(f"\nPredicted delivery time: {pred:.2f} minutes")
    save = input("Log this prediction to DB without linking to an order? (y/N): ").strip().lower()
    if save == "y":
        pid = db.insert_prediction(None, pred)
        print(f"Saved prediction_id={pid}")


if __name__ == "__main__":
    main()
