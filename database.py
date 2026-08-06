"""Database schema & SQLite access layer for CourierAI platform."""
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

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
        _migrate_api_key_columns(conn)
        _migrate_plan_columns(conn)
        _migrate_demo_column(conn)


def _migrate_created_at_columns(conn: sqlite3.Connection) -> None:
    """Add timestamps to databases created before the current schema."""
    for table_name in ("orders",):
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


def _migrate_api_key_columns(conn: sqlite3.Connection) -> None:
    """Add API-key columns to databases created before the current schema."""
    columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(users)").fetchall()
    }
    if "api_key_hash" not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN api_key_hash TEXT")
    if "api_key_hint" not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN api_key_hint TEXT")
    if "api_key_enc" not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN api_key_enc TEXT")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_api_key_hash "
        "ON users(api_key_hash)"
    )


def _migrate_demo_column(conn: sqlite3.Connection) -> None:
    """Add an is_demo flag so admin demo-mode records never mix with real
    prediction audit data (real analytics filter them out by default)."""
    columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(dtdc_predictions)").fetchall()
    }
    if "is_demo" not in columns:
        conn.execute(
            "ALTER TABLE dtdc_predictions ADD COLUMN is_demo INTEGER NOT NULL DEFAULT 0"
        )


def _migrate_plan_columns(conn: sqlite3.Connection) -> None:
    """Add subscription-plan columns to databases created before the current schema."""
    columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(users)").fetchall()
    }
    if "plan" not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN plan TEXT NOT NULL DEFAULT 'basic'")
    if "plan_expires_at" not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN plan_expires_at TIMESTAMP")
    if "usage_month" not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN usage_month TEXT")
    if "predictions_used" not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN predictions_used INTEGER NOT NULL DEFAULT 0")


def rows_to_dataframe(rows: list[sqlite3.Row]) -> pd.DataFrame:
    """Convert SQLite Row list to a DataFrame (shared by charts & dashboard)."""
    if not rows:
        return pd.DataFrame()
    data = [{k: r[k] for k in r.keys()} for r in rows]
    return pd.DataFrame(data)


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
    is_demo: int = 0,
) -> int:
    """Log a DTDC model prediction to the audit table.

    is_demo=1 marks admin demo-mode records so they are excluded from real
    analytics by default.
    """
    with connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO dtdc_predictions (
                origin, destination, booking_weekday, mode,
                nature_of_consignment, total_pieces, actual_weight,
                volumetric_weight, chargeable_weight, predicted_days,
                model_version, is_demo, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                origin, destination, booking_weekday, mode,
                nature_of_consignment, total_pieces, actual_weight,
                volumetric_weight, chargeable_weight, predicted_days,
                model_version, is_demo,
            ),
        )
        return int(cur.lastrowid)


def fetch_dtdc_predictions(limit: int = 5000, include_demo: bool = False) -> list[sqlite3.Row]:
    """Fetch recent DTDC prediction audit records.

    Demo-mode records are excluded unless include_demo=True, so admin sample
    data never skews real analytics.
    """
    where = "" if include_demo else "WHERE is_demo = 0"
    with connection() as conn:
        cur = conn.execute(
            f"""
            SELECT * FROM dtdc_predictions
            {where}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return cur.fetchall()


def count_dtdc_predictions(include_demo: bool = False) -> int:
    """Return total number of DTDC predictions logged.

    Demo-mode records are excluded unless include_demo=True.
    """
    where = "" if include_demo else "WHERE is_demo = 0"
    with connection() as conn:
        cur = conn.execute(f"SELECT COUNT(*) AS c FROM dtdc_predictions {where}")
        row = cur.fetchone()
        return int(row["c"]) if row else 0


# ---------------------------------------------------------------------------
# User accounts
# ---------------------------------------------------------------------------
def create_user(
    full_name: str,
    email: str,
    password_hash: str,
    company: str = "",
    api_key_hash: Optional[str] = None,
    api_key_hint: Optional[str] = None,
    api_key_enc: Optional[str] = None,
    plan: str = "basic",
) -> int:
    """Insert a new user (starts on the free basic plan).

    Raises ValueError if the email is already registered.
    """
    with connection() as conn:
        existing = conn.execute(
            "SELECT id FROM users WHERE email = ? COLLATE NOCASE", (email,)
        ).fetchone()
        if existing:
            raise ValueError("An account with that email already exists.")
        try:
            cur = conn.execute(
                """
                INSERT INTO users (
                    full_name, company, email, password_hash,
                    api_key_hash, api_key_hint, api_key_enc, plan, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    full_name, company, email, password_hash,
                    api_key_hash, api_key_hint, api_key_enc, plan,
                ),
            )
        except sqlite3.IntegrityError:
            # Concurrent registration of the same email — keep the API contract.
            raise ValueError("An account with that email already exists.")
        return int(cur.lastrowid)


def get_user_by_email(email: str) -> Optional[sqlite3.Row]:
    """Return the user row for an email (case-insensitive) or None."""
    with connection() as conn:
        cur = conn.execute(
            "SELECT * FROM users WHERE email = ? COLLATE NOCASE", (email,)
        )
        return cur.fetchone()


def get_user_by_id(user_id: int) -> Optional[sqlite3.Row]:
    """Return the user row for an id or None."""
    with connection() as conn:
        cur = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        return cur.fetchone()


def count_users() -> int:
    """Return the total number of registered user accounts."""
    with connection() as conn:
        cur = conn.execute("SELECT COUNT(*) AS c FROM users")
        row = cur.fetchone()
        return int(row["c"]) if row else 0


def set_api_key(
    user_id: int,
    api_key_hash: str,
    api_key_hint: str,
    api_key_enc: Optional[str] = None,
) -> None:
    """Store a new API key (hash + encrypted copy) for a user, revoking the
    previous key immediately. The encrypted copy lets the owner reveal/copy
    their key on the account page without storing it in plaintext."""
    with connection() as conn:
        conn.execute(
            "UPDATE users SET api_key_hash = ?, api_key_hint = ?, api_key_enc = ? "
            "WHERE id = ?",
            (api_key_hash, api_key_hint, api_key_enc, user_id),
        )


def get_user_by_api_key(api_key_hash: str) -> Optional[sqlite3.Row]:
    """Return the user row owning the given API-key hash, or None."""
    with connection() as conn:
        cur = conn.execute(
            "SELECT * FROM users WHERE api_key_hash = ?", (api_key_hash,)
        )
        return cur.fetchone()


# ---------------------------------------------------------------------------
# Subscription plans & prediction quotas
# ---------------------------------------------------------------------------
def _now_month() -> str:
    """Current billing month as 'YYYY-MM'."""
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _now_text() -> str:
    """Current UTC timestamp in SQLite 'YYYY-MM-DD HH:MM:SS' format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def set_user_plan(user_id: int, plan: str, expires_at: Optional[str] = None) -> None:
    """Set a user's plan and optional expiry timestamp.

    Changing plans also resets the monthly usage counter so the new plan
    starts with a clean slate (fresh 50 free predictions on downgrade, a
    clear unlimited slate on upgrade).
    """
    with connection() as conn:
        conn.execute(
            "UPDATE users SET plan = ?, plan_expires_at = ?, "
            "usage_month = ?, predictions_used = 0 WHERE id = ?",
            (plan, expires_at, _now_month(), user_id),
        )


def get_user_plan_status(user_id: int) -> dict:
    """Return a user's plan + usage info for UI display and quota checks.

    Handles two bookkeeping concerns:
    - an expired paid plan falls back to the free basic plan;
    - the monthly counter resets when the calendar month rolls over.
    """
    user = get_user_by_id(user_id)
    if user is None:
        limit = config.plan_limit("basic")
        return {
            "plan": "basic", "plan_name": "Basic", "price": 0,
            "period": None, "limit": limit, "used": 0,
            "remaining": limit, "unlimited": False, "expires_at": None,
        }

    plan = user["plan"]
    if plan != "basic" and user["plan_expires_at"] and user["plan_expires_at"] < _now_text():
        # Paid plan expired → fall back to the free plan.
        set_user_plan(user_id, "basic", None)
        plan = "basic"

    month = _now_month()
    if user["usage_month"] != month:
        with connection() as conn:
            conn.execute(
                "UPDATE users SET usage_month = ?, predictions_used = 0 WHERE id = ?",
                (month, user_id),
            )
        user = get_user_by_id(user_id)

    limit = config.plan_limit(plan)
    used = int(user["predictions_used"] or 0)
    return {
        "plan": plan,
        "plan_name": config.PLANS[plan]["name"],
        "price": config.PLANS[plan]["price"],
        "period": config.PLANS[plan]["period"],
        "limit": limit,
        "used": used,
        "remaining": None if limit is None else max(limit - used, 0),
        "unlimited": limit is None,
        "expires_at": user["plan_expires_at"],
    }


def consume_prediction(user_id: int) -> tuple[bool, Optional[str]]:
    """Record one prediction against the user's quota.

    Returns (allowed, error_message). Unlimited plans always pass; limited
    plans pass while the counter is below the monthly cap. The increment is a
    single conditional UPDATE so concurrent requests cannot overshoot.
    """
    status = get_user_plan_status(user_id)
    if status["unlimited"]:
        with connection() as conn:
            conn.execute(
                "UPDATE users SET usage_month = ?, predictions_used = predictions_used + 1 "
                "WHERE id = ?",
                (_now_month(), user_id),
            )
        return True, None

    with connection() as conn:
        cur = conn.execute(
            "UPDATE users SET predictions_used = predictions_used + 1 "
            "WHERE id = ? AND usage_month = ? AND predictions_used < ?",
            (user_id, _now_month(), status["limit"]),
        )
        if cur.rowcount == 1:
            return True, None
    return False, (
        f"You have used all {status['limit']} free predictions this month. "
        "Upgrade to Pro for unlimited predictions."
    )
