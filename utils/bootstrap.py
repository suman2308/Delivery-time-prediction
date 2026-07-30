# utils/bootstrap.py
"""Application bootstrap utilities.
Ensures required directories exist, database is initialized, and optional data
migration is performed.
"""
import os
import importlib
from . import config
import database as db

def ensure_app_ready():
    """Prepare filesystem and database, then run optional bootstrap steps."""
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
    if not os.path.isfile(config.DATABASE_PATH):
        db.init_db()
    _bootstrap_if_needed()

def _bootstrap_if_needed():
    """Bootstrap on first run.
    - If synthetic data usage is enabled (`USE_SYNTHETIC_DATA=1`) and the DB has
      very few rows, seed synthetic data.
    - If the DB is empty, attempt to import the real DTDC dataset.
    """
    use_synthetic = os.getenv("USE_SYNTHETIC_DATA", "0") == "1"
    if db.count_orders() == 0:
        # Try real data import first
        try:
            import sys
            # Ensure project root is in sys.path for module imports
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
            from data.migrate_dtdc import migrate_dtdc
            migrated = migrate_dtdc()
            if migrated:
                return
        except Exception as e:
            # If import fails, fall back to synthetic if allowed
            print(f"DTDC migration failed: {e}")
    if use_synthetic:
        # Seed synthetic data (default 300 rows)
        seed_count = int(os.getenv("BOOTSTRAP_SEED_COUNT", "300"))
        import seed_data
        seed_data.seed(count=seed_count, seed=42)
