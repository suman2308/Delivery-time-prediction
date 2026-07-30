"""Database schema & SQLite access layer for Smart Delivery platform."""
import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Optional

import config


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def connection():
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    schema_path = os.path.join(config.BASE_DIR, "schema.sql")
    with open(schema_path, "r", encoding="utf-8") as f:
        sql = f.read()
    with connection() as conn:
        conn.executescript(sql)
        _migrate_created_at_columns(conn)


def _migrate_created_at_columns(conn: sqlite3.Connection) -> None:
    """Add timestamps to databases created before the current schema."""
    for table_name in ("orders", "predictions"):
        columns = {
            row["name"]
            for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        if "created_at" not in columns:
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN created_at TIMESTAMP")
            conn.execute(
                f"UPDATE {table_name} SET created_at = CURRENT_TIMESTAMP "
                "WHERE created_at IS NULL"
            )


def insert_order(
    distance: float,
    order_time: int,
    traffic_level: str,
    weather: str,
    delivery_time: float,
) -> int:
    with connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO orders
                (distance, order_time, traffic_level, weather, delivery_time, created_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (distance, order_time, traffic_level, weather, delivery_time),
        )
        return int(cur.lastrowid)


def insert_prediction(order_id: Optional[int], predicted_time: float) -> int:
    with connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO predictions (order_id, predicted_time, created_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            """,
            (order_id, predicted_time),
        )
        return int(cur.lastrowid)


def fetch_orders_for_training() -> list[sqlite3.Row]:
    with connection() as conn:
        cur = conn.execute(
            "SELECT order_id, distance, order_time, traffic_level, weather, delivery_time FROM orders"
        )
        return cur.fetchall()


def count_orders() -> int:
    with connection() as conn:
        cur = conn.execute("SELECT COUNT(*) AS c FROM orders")
        row = cur.fetchone()
        return int(row["c"]) if row else 0


def fetch_orders_filtered(
    traffic: Optional[str] = None,
    weather: Optional[str] = None,
    min_distance: Optional[float] = None,
    max_distance: Optional[float] = None,
    limit: int = 250,
) -> list[sqlite3.Row]:
    clauses: list[str] = []
    params: list[Any] = []
    if traffic:
        clauses.append("traffic_level = ?")
        params.append(traffic)
    if weather:
        clauses.append("weather = ?")
        params.append(weather)
    if min_distance is not None:
        clauses.append("distance >= ?")
        params.append(min_distance)
    if max_distance is not None:
        clauses.append("distance <= ?")
        params.append(max_distance)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = f"SELECT * FROM orders{where} ORDER BY order_id DESC LIMIT ?"
    params.append(limit)
    with connection() as conn:
        cur = conn.execute(sql, params)
        return cur.fetchall()


def fetch_predictions_with_orders(limit: int = 500) -> list[sqlite3.Row]:
    with connection() as conn:
        cur = conn.execute(
            """
            SELECT p.prediction_id, p.order_id, p.predicted_time, p.created_at,
                   o.distance, o.order_time, o.traffic_level, o.weather, o.delivery_time
            FROM predictions p
            LEFT JOIN orders o ON o.order_id = p.order_id
            ORDER BY p.prediction_id DESC
            LIMIT ?
            """,
            (limit,),
        )
        return cur.fetchall()


def fetch_recent_predictions(limit: int = 10) -> list[sqlite3.Row]:
    with connection() as conn:
        cur = conn.execute(
            """
            SELECT p.prediction_id, p.order_id, p.predicted_time, p.created_at,
                   o.distance, o.order_time, o.traffic_level, o.weather
            FROM predictions p
            LEFT JOIN orders o ON o.order_id = p.order_id
            ORDER BY p.prediction_id DESC
            LIMIT ?
            """,
            (limit,),
        )
        return cur.fetchall()


def insert_dtdc_prediction(
    origin: str,
    destination: str,
    booking_weekday: str,
    mode: str,
    nature_of_consignment: str,
    total_pieces: int,
    actual_weight: float,
    volumetric_weight: float,
    chargeable_weight: float,
    predicted_days: float,
    model_version: str,
) -> int:
    """Log a DTDC model prediction to the audit table."""
    with connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO dtdc_predictions (
                origin, destination, booking_weekday, mode,
                nature_of_consignment, total_pieces, actual_weight,
                volumetric_weight, chargeable_weight, predicted_days,
                model_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                origin, destination, booking_weekday, mode,
                nature_of_consignment, total_pieces, actual_weight,
                volumetric_weight, chargeable_weight, predicted_days,
                model_version,
            ),
        )
        return int(cur.lastrowid)


def fetch_dtdc_predictions(limit: int = 5000) -> list[sqlite3.Row]:
    """Fetch recent DTDC prediction audit records."""
    with connection() as conn:
        cur = conn.execute(
            """
            SELECT * FROM dtdc_predictions
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return cur.fetchall()


def count_dtdc_predictions() -> int:
    """Return total number of DTDC predictions logged."""
    with connection() as conn:
        cur = conn.execute("SELECT COUNT(*) AS c FROM dtdc_predictions")
        row = cur.fetchone()
        return int(row["c"]) if row else 0
