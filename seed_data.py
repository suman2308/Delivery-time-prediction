"""Generate synthetic orders and persist to SQLite."""
import argparse
import random

import database as db
import config


TRAFFIC = ["Low", "Medium", "High"]
WEATHER = ["Clear", "Rainy"]


def synthetic_delivery_minutes(
    distance_km: float,
    hour: int,
    traffic: str,
    weather: str,
    rng: random.Random,
) -> float:
    """Ground truth: farther, peak hours, traffic, rain increase minutes."""
    base = 12.0
    dist_effect = distance_km * 4.2
    hour_effect = 0.35 * abs(hour - 14)  # rush around 8–10 and 17–19 approximated
    if hour in (8, 9, 17, 18, 19):
        hour_effect += 6.0
    traffic_bonus = {"Low": 0.0, "Medium": 5.5, "High": 12.0}[traffic]
    weather_bonus = 0.0 if weather == "Clear" else 7.0
    noise = rng.gauss(0, 2.8)
    return max(5.0, base + dist_effect + hour_effect + traffic_bonus + weather_bonus + noise)


def seed(count: int = 400, seed: int = 42) -> int:
    rng = random.Random(seed)
    inserted = 0
    for _ in range(count):
        distance = round(rng.uniform(0.5, 25.0), 2)
        hour = rng.randint(6, 22)
        traffic = rng.choice(TRAFFIC)
        weather = rng.choice(WEATHER)
        delivery = round(
            synthetic_delivery_minutes(distance, hour, traffic, weather, rng), 2
        )
        db.insert_order(distance, hour, traffic, weather, delivery)
        inserted += 1
    return inserted


def main():
    parser = argparse.ArgumentParser(description="Seed orders table with synthetic data.")
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    db.init_db()
    n = seed(count=args.count, seed=args.seed)
    print(f"Inserted {n} orders.")
    print(f"Database file: {config.DATABASE_PATH}")


if __name__ == "__main__":
    main()
