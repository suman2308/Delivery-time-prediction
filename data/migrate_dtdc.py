# data/migrate_dtdc.py
"""Migrate DTDC dataset CSV into the SQLite database used by the application.

The CSV file ``DTDC_Improved_Dataset.csv`` resides at the project root.
Each row contains the columns:
- distance (float)
- order_time (int) – hour of day (0‑23)
- traffic_level (str)
- weather (str)
- delivery_time (float)

The function ``migrate_dtdc`` reads the file, inserts each record into the
``orders`` table via the ``database`` module and returns the number of rows
imported.  It is idempotent – if the table already contains rows it will not
re‑insert them.
"""

import sys
import os
import csv
from typing import Tuple
# Ensure project root is in sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
import database as db


def _row_to_types(row: dict) -> Tuple[float, int, str, str, float]:
    """Convert CSV row strings to typed values expected by ``insert_order``.

    The CSV uses strings for all fields; we coerce to the appropriate Python
    types.  Any conversion error raises ``ValueError`` so that failures are
    visible during the bootstrap process.
    """
    distance = float(row["distance"].strip())
    order_time = int(row["order_time"].strip())
    traffic_level = row["traffic_level"].strip()
    weather = row["weather"].strip()
    delivery_time = float(row["delivery_time"].strip())
    return distance, order_time, traffic_level, weather, delivery_time


def migrate_dtdc() -> int:
    """Import the DTDC dataset into the ``orders`` table.

    Returns
    -------
    int
        Number of rows successfully inserted.
    """
    csv_path = os.path.join(config.BASE_DIR, "DTDC_Improved_Dataset.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"DTDC dataset not found at {csv_path}")

    inserted = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            distance, order_time, traffic_level, weather, delivery_time = _row_to_types(row)
            db.insert_order(distance, order_time, traffic_level, weather, delivery_time)
            inserted += 1
    return inserted

if __name__ == "__main__":
    count = migrate_dtdc()
    print(f"Inserted {count} rows from DTDC dataset.")
