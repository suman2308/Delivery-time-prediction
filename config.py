"""Configuration settings for CourierAI platform."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.environ.get("DELIVERY_DB_PATH", os.path.join(BASE_DIR, "delivery.db"))
# Legacy path — kept for _bootstrap_if_needed which trains the old synthetic model.
# The DTDC model manages its own path internally in dtdc_model.py.
MODEL_PATH = os.environ.get("DELIVERY_MODEL_PATH", os.path.join(BASE_DIR, "models", "delivery_regressor.joblib"))
PLOTS_DIR = os.path.join(BASE_DIR, "static", "plots")
# Session signing key. Override in production via the SECRET_KEY env var.
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-me-in-production")
# REST API rate limiting (per API key). Overridable for tests / tuning.
API_RATE_LIMIT = os.environ.get("API_RATE_LIMIT", "30 per minute")
API_RATE_LIMIT_DAILY = os.environ.get("API_RATE_LIMIT_DAILY", "1000 per day")
LOGIN_RATE_LIMIT = os.environ.get("LOGIN_RATE_LIMIT", "10 per minute")
REGISTER_RATE_LIMIT = os.environ.get("REGISTER_RATE_LIMIT", "5 per minute")
REGENERATE_KEY_RATE_LIMIT = os.environ.get("REGENERATE_KEY_RATE_LIMIT", "3 per hour")
# Rate-limit storage: Redis in production (REDIS_URL), in-memory otherwise.
# NOTE: in-memory storage is per-process — with multiple gunicorn workers the
# effective budget multiplies by the worker count, so use Redis when scaling.
RATE_LIMIT_STORAGE_URI = (
    os.environ.get("RATE_LIMIT_STORAGE_URI") or os.environ.get("REDIS_URL") or "memory://"
)

# ---------------------------------------------------------------------------
# Subscription plans & prediction quotas
# ---------------------------------------------------------------------------
# Every new account starts on the free "basic" plan with a monthly prediction
# quota. Pro plans are unlimited (demo billing — no real payment processed).
FREE_PLAN_LIMIT = int(os.environ.get("FREE_PLAN_LIMIT", "50"))

PLANS = {
    # limit is resolved dynamically via plan_limit() so FREE_PLAN_LIMIT can be
    # tuned per-environment (and per-test) without rebuilding the dict.
    "basic": {"name": "Free", "price": 0, "period": None, "limit": None},
    "pro_monthly": {"name": "Pro", "price": 299, "period": "month", "limit": None},
    "pro_yearly": {"name": "Pro+", "price": 1299, "period": "year", "limit": None},
}

# Comma-separated emails granted admin access (gates the Admin navigation link
# and the admin demo-mode tooling). Leave empty to hide admin features.
ADMIN_EMAILS = {
    e.strip().lower()
    for e in os.environ.get("ADMIN_EMAILS", "").split(",")
    if e.strip()
}

# How long each paid plan lasts before it expires back to the free plan.
PLAN_DURATION_DAYS = {"pro_monthly": 30, "pro_yearly": 365}


def plan_limit(plan: str) -> int | None:
    """Monthly prediction limit for a plan. None means unlimited."""
    if plan == "basic":
        return FREE_PLAN_LIMIT
    return PLANS.get(plan, {}).get("limit")
