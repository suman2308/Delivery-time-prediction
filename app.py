"""Flask Application Server & Endpoint Controllers."""
from __future__ import annotations

import base64
import hashlib
import os
import secrets
import threading
import time
from functools import lru_cache, wraps
from urllib.parse import urlsplit

from cryptography.fernet import Fernet

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from markupsafe import Markup
from werkzeug.security import check_password_hash, generate_password_hash

import charts
import config
import database as db
import train_experiments as te
from dtdc_model import DTDCPredictor, MODEL_ALGORITHM, MODEL_VERSION

app = Flask(__name__)
app.secret_key = config.SECRET_KEY

# ---------------------------------------------------------------------------
# Session cookie hardening (OWASP): HttpOnly, SameSite, Secure behind proxy
# ---------------------------------------------------------------------------
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("COOKIE_SECURE", "0") == "1",
    PERMANENT_SESSION_LIFETIME=86400 * 30,
)

# ---------------------------------------------------------------------------
# CSRF protection (OWASP): per-session token validated on state-changing
# requests. API routes are exempt (they authenticate via API keys).
# ---------------------------------------------------------------------------
def _csrf_token() -> str:
    """Return the session's CSRF token, generating one on first use."""
    token = session.get("_csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["_csrf_token"] = token
    return token


app.jinja_env.globals["csrf_token"] = _csrf_token


@app.before_request
def _csrf_protect():
    """Reject state-changing requests that lack a valid CSRF token.

    GET/HEAD/OPTIONS are read-only. API endpoints authenticate via API keys
    (no cookies), so they are exempt. Everything else must present the token
    from its own session — login CSRF included.
    """
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return None
    if request.path.startswith("/api/") or request.path == "/health":
        return None
    expected = session.get("_csrf_token", "")
    provided = request.form.get("_csrf_token", "")
    if not expected or not secrets.compare_digest(provided, expected):
        flash("Your session expired. Please try again.", "error")
        # Same-site fallback only — never bounce to an attacker-controlled
        # Referer (open-redirect defence, consistent with login).
        return redirect(_safe_next(request.referrer))
    return None

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
# Global baseline applies per client IP; specific routes tighten (or loosen)
# their own limits below (e.g. /api/predict is keyed per API key).
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["120 per hour", "500 per day"],
    storage_uri=config.RATE_LIMIT_STORAGE_URI,
)

# Ensure the schema (incl. the users table used by the API-key check) exists
# before any request can hit auth-gated endpoints.
db.init_db()


def _api_identity():
    """Rate-limit key for the prediction API: the API key when present,
    otherwise the client IP (so unauthenticated callers can't share limits)."""
    raw = _extract_api_key(request)
    if raw:
        return _hash_api_key(raw)
    return request.remote_addr or "anonymous"


def _generate_api_key() -> str:
    """Create a new API key. Stored hashed — only ever shown in full once."""
    return "scp_live_" + secrets.token_urlsafe(32)


def _hash_api_key(api_key: str) -> str:
    """Hash an API key for storage / lookup (SHA-256, key is high entropy)."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def _key_hint(api_key: str) -> str:
    """A display-safe fragment of the key, e.g. 'scp_live_…abcd'."""
    return "scp_live_…" + api_key[-4:]


def _extract_api_key(req) -> str:
    """Pull an API key from the X-API-Key header or an Authorization Bearer."""
    key = req.headers.get("X-API-Key", "").strip()
    if key:
        return key
    auth = req.headers.get("Authorization", "").strip()
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return ""


@lru_cache(maxsize=1)
def _fernet() -> Fernet:
    """Fernet cipher for API keys. Key is derived from the app secret so a DB
    leak alone never exposes keys; override via KEY_ENCRYPTION_KEY in prod.

    IMPORTANT: the derivation material (SECRET_KEY / KEY_ENCRYPTION_KEY) must
    stay stable across deploys — changing it makes stored keys undecryptable.
    """
    raw = os.environ.get("KEY_ENCRYPTION_KEY") or config.SECRET_KEY
    material = hashlib.sha256(raw.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(material))


def _encrypt_api_key(api_key: str) -> str:
    """Encrypt an API key for storage (revealable later by its owner)."""
    return _fernet().encrypt(api_key.encode("utf-8")).decode("utf-8")


def _decrypt_api_key(token: str) -> str:
    """Decrypt a stored API key token back to plaintext."""
    return _fernet().decrypt(token.encode("utf-8")).decode("utf-8")


# ---------------------------------------------------------------------------
# Security headers
# ---------------------------------------------------------------------------
@app.after_request
def _set_security_headers(response):
    """Apply standard security headers to every response."""
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "frame-ancestors 'none'"
    )
    return response

# ---------------------------------------------------------------------------
# DTDC model - initialised once at module load (singleton guarantees one load)
# ---------------------------------------------------------------------------
_dtdc_predictor = DTDCPredictor()


def ensure_app_ready():
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
    db.init_db()
    _bootstrap_if_needed()


def _bootstrap_if_needed() -> None:
    """Startup bootstrap for fresh deployments."""
    bootstrap_enabled = os.environ.get("BOOTSTRAP_ON_START", "1") == "1"
    if not bootstrap_enabled or os.path.isfile(config.MODEL_PATH):
        return

    import ml_model
    import seed_data

    orders_count = db.count_orders()
    if orders_count < 10:
        seed_count = int(os.environ.get("BOOTSTRAP_SEED_COUNT", "300"))
        seed_data.seed(count=seed_count, seed=42)

    try:
        ml_model.train_and_save()
    except ValueError:
        return


@app.route("/")
def index():
    ensure_app_ready()
    metrics = _load_metrics()
    return render_template("index.html", metrics=metrics, error=None)


def _friendly_algorithm(name):
    """Human-friendly algorithm label for the UI.

    Strips sklearn's verbose class suffix (e.g. "HistGradientBoostingRegressor"
    becomes "HistGradientBoosting") so it fits the KPI/stat cards. The API keeps
    the full class name.
    """
    if not name:
        return name
    for suffix in ("Regressor", "Classifier", "Estimator"):
        if name.endswith(suffix):
            short = name[: -len(suffix)]
            return short if short else name
    return name


def _algorithm_display(name):
    """Friendly algorithm name with a line break before the model type so it fits
    KPI/stat cards (e.g. "HistGradient<br>Boosting"). Other names pass through.
    """
    friendly = _friendly_algorithm(name)
    if not friendly:
        return friendly
    for suffix in ("Boosting", "Forest", "Network", "Neighbors", "Bayes", "Ridge"):
        if friendly.endswith(suffix) and len(friendly) > len(suffix):
            return Markup(f"{friendly[: -len(suffix)]}<br>{suffix}")
    return friendly


app.add_template_filter(_algorithm_display, "display_algo")


def _load_metrics():
    """Return model metrics dict or None if the model artifact is missing."""
    try:
        meta = _dtdc_predictor.meta
        m = meta.get("metrics", {})
        return {
            "mae": m.get("mae_days", 0),
            "rmse": m.get("rmse_days", 0),
            "r2": m.get("r2", 0),
            "algorithm": _friendly_algorithm(meta.get("algorithm", MODEL_ALGORITHM)),
            "model_version": meta.get("model_version", MODEL_VERSION),
        }
    except (FileNotFoundError, RuntimeError):
        return None


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
def _current_user():
    """Return the logged-in user row (or None) for the active session."""
    user_id = session.get("user_id")
    if not user_id:
        return None
    return db.get_user_by_id(int(user_id))


@app.context_processor
def _inject_auth():
    """Expose the current user, an admin flag, and a cache-busting asset version
    to templates.

    Plan status is computed only on the routes that need it (account/pricing)
    rather than on every request."""
    user = _current_user()
    is_admin = bool(user and user["email"].lower() in config.ADMIN_EMAILS)
    return {
        "current_user": user,
        "is_admin": is_admin,
        "static_version": _static_version(),
    }


def _static_version() -> str:
    """Cache-busting token derived from the main static assets' mtimes, so
    browsers pick up changed CSS/JS immediately after a deploy."""
    newest = 0.0
    for name in ("static/js/app.js", "static/css/app.css"):
        try:
            newest = max(newest, os.path.getmtime(os.path.join(config.BASE_DIR, name)))
        except OSError:
            pass
    return str(int(newest)) if newest else "1"


def login_required(view):
    """Redirect anonymous visitors to the login page, remembering where they
    were headed so they can be returned after signing in.

    Also handles stale sessions: a session cookie may still carry a user_id
    for an account that was deleted or whose DB row was reset. Such sessions
    are cleared and sent to login instead of crashing downstream on None."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        if session.get("user_id") is None:
            return redirect(url_for("login", next=request.path))
        if _current_user() is None:
            session.clear()
            flash("Your session has expired. Please log in again.", "info")
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)

    return wrapped


def admin_required(view):
    """Restrict a view to accounts whose email is in config.ADMIN_EMAILS."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        user = _current_user()
        if not user or user["email"].lower() not in config.ADMIN_EMAILS:
            flash("Admin access required.", "error")
            return redirect(url_for("dashboard"))
        return view(*args, **kwargs)

    return wrapped


# Precomputed hash so the "no such user" path takes comparable time to a real
# password check, avoiding timing-based user enumeration.
_DUMMY_PASSWORD_HASH = generate_password_hash(
    "dummy-password-for-timing", method="pbkdf2:sha256:600000"
)


def _safe_next(target: str | None) -> str:
    """Return target only if it is a safe, same-site path (no open redirect)."""
    if not target:
        return url_for("dashboard")
    # Reject anything with a backslash — browsers normalise `\` to `/`, so
    # `/\\evil.com` would otherwise become `//evil.com` (open redirect).
    if "\\" in target:
        return url_for("dashboard")
    try:
        parts = urlsplit(target)
    except ValueError:
        return url_for("dashboard")
    if parts.scheme or parts.netloc or not target.startswith("/"):
        return url_for("dashboard")
    return target


@app.route("/login", methods=["GET", "POST"])
@limiter.limit(config.LOGIN_RATE_LIMIT, methods=["POST"], override_defaults=False)
def login():
    # A session pointing at a deleted account is stale — clean it up now so a
    # dead user_id never lingers in the cookie (cleared again on login_required).
    if session.get("user_id") and _current_user() is None:
        session.clear()
    if _current_user():
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        user = db.get_user_by_email(email) if email else None
        # Equalise timing: always run a hash check, even for unknown emails.
        if user is not None:
            password_ok = check_password_hash(user["password_hash"], password)
        else:
            password_ok = False
            check_password_hash(_DUMMY_PASSWORD_HASH, password)
        if password_ok:
            session.clear()  # prevent session fixation
            session["user_id"] = user["id"]
            session.permanent = True
            flash(f"Welcome back, {user['full_name']}!", "success")
            return redirect(_safe_next(request.form.get("next")))
        flash("Invalid email or password.", "error")

    return render_template(
        "login.html", next=request.args.get("next") or request.form.get("next") or ""
    )


@app.route("/register", methods=["GET", "POST"])
@limiter.limit(config.REGISTER_RATE_LIMIT, methods=["POST"], override_defaults=False)
def register():
    if _current_user():
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        company = request.form.get("company", "").strip() or "Personal"
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if len(full_name) < 2:
            flash("Please enter your full name.", "error")
        elif not email or "@" not in email:
            flash("Please enter a valid email address.", "error")
        elif len(password) < 8:
            flash("Password must be at least 8 characters long.", "error")
        else:
            try:
                api_key = _generate_api_key()
                user_id = db.create_user(
                    full_name=full_name,
                    email=email,
                    password_hash=generate_password_hash(password),
                    company=company,
                    api_key_hash=_hash_api_key(api_key),
                    api_key_hint=_key_hint(api_key),
                    api_key_enc=_encrypt_api_key(api_key),
                )
            except ValueError as exc:
                flash(str(exc), "error")
            else:
                session.clear()  # prevent session fixation
                session["user_id"] = user_id
                session.permanent = True
                flash(f"Account created. Welcome, {full_name}!", "success")
                # The API key is only ever shown this once.
                flash(api_key, "api_key")
                # New accounts pick a plan before reaching the dashboard.
                return redirect(url_for("choose_plan"))

    return render_template(
        "register.html", next=request.args.get("next") or request.form.get("next") or ""
    )


@app.route("/logout", methods=["POST"])
def logout():
    """POST-only so a plain link on a third-party page cannot force a logout."""
    session.clear()
    flash("You have been signed out.", "info")
    return redirect(url_for("index"))


# ---------------------------------------------------------------------------
# Marketing & product pages
# ---------------------------------------------------------------------------
@app.route("/demo")
def demo():
    ensure_app_ready()
    return render_template("demo.html", metrics=_load_metrics(), error=None, source="demo")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/tracking")
def tracking():
    return render_template("tracking.html")


@app.route("/pricing")
def pricing():
    plan_status = None
    user = _current_user()
    if user:
        plan_status = db.get_user_plan_status(user["id"])
    return render_template("pricing.html", plan_status=plan_status)


@app.route("/plans")
def choose_plan():
    """Plan selection — shown right after registration, and always reachable
    from the pricing page. Billing is demo-only (no real payment gateway)."""
    plan_status = None
    user = _current_user()
    if user:
        plan_status = db.get_user_plan_status(user["id"])
    return render_template("plans.html", plan_status=plan_status)


@app.route("/predict")
def predict_page():
    """Dedicated Predict page — the shared shipment estimator in a focused
    layout. Anonymous visitors get the open public demo behaviour."""
    ensure_app_ready()
    return render_template(
        "predict.html", metrics=_load_metrics(), error=None, source="predict"
    )


def _live_model_row():
    """Production model row from live metadata (fallbacks if missing)."""
    try:
        meta = _dtdc_predictor.meta
        m = meta.get("metrics", {})
    except (FileNotFoundError, RuntimeError):
        meta, m = {}, {}
    return {
        "name": "HistGradientBoosting",
        "label": "HistGradientBoosting",
        "mae": m.get("mae_days", 0.5361),
        "rmse": m.get("rmse_days", 0.7263),
        "r2": m.get("r2", 0.7466),
        "train_s": m.get("train_seconds", 38.4),
        "predict_s": m.get("predict_seconds", 0.003),
        "live": True,
        "category": "Production",
    }


def _experiment_rows(results: dict, task: str) -> list[dict]:
    """Flatten the experiment results JSON into comparison rows.

    Each group (base / hybrid / stacking) carries a category badge; every row
    exposes both regression and classification metrics so the template can
    switch between the two tables.
    """
    rows = []
    if not results:
        return rows
    groups = results.get(task, {})
    for category, group in (("Base", groups.get("base", [])),
                            ("Hybrid", groups.get("hybrid", [])),
                            ("Stacking", groups.get("stacking", []))):
        for item in group:
            label = item.get("Model") or item.get("Hybrid") or item.get("Combination")
            if not label:
                continue
            rows.append({
                "name": label,
                "label": label,
                "category": category,
                "mae": item.get("MAE", 0),
                "rmse": item.get("RMSE", 0),
                "r2": item.get("R2", 0),
                "accuracy": item.get("Accuracy", 0),
                "precision": item.get("Precision", 0),
                "recall": item.get("Recall", 0),
                "f1": item.get("F1", 0),
                "train_s": item.get("train_s", 0),
                "live": False,
            })
    return rows


@app.route("/model-comparison")
def model_comparison():
    """Side-by-side comparison of candidate models, powered by the results of
    train_experiments.py (admin can re-run via the admin panel). Falls back to
    the live production model plus representative benchmarks when no experiment
    has been run yet."""
    ensure_app_ready()
    live = _live_model_row()
    live_row = {**live, "label": "HistGradientBoosting (production)"}

    results = te.load_experiment_results()
    reg_candidates = _experiment_rows(results, "regression")
    cls_candidates = _experiment_rows(results, "classification")

    if not reg_candidates:
        # No experiment run yet — representative benchmarks from the
        # evaluation suite so the page is never empty.
        reg_candidates = [
            {
                "name": "RandomForest", "label": "Random Forest",
                "category": "Benchmark", "mae": 0.612, "rmse": 0.814,
                "r2": 0.684, "accuracy": 0, "precision": 0, "recall": 0,
                "f1": 0, "train_s": 96.2, "live": False,
            },
            {
                "name": "XGBoost", "label": "XGBoost",
                "category": "Benchmark", "mae": 0.558, "rmse": 0.748,
                "r2": 0.731, "accuracy": 0, "precision": 0, "recall": 0,
                "f1": 0, "train_s": 210.7, "live": False,
            },
        ]
    if not cls_candidates:
        cls_candidates = []

    # Production row is always pinned on top of both tables.
    reg_models = [live_row, *reg_candidates]
    cls_models = [
        {
            **live_row,
            "accuracy": 0.934, "precision": 0.921, "recall": 0.962, "f1": 0.941,
            "category": "Production",
        },
        *cls_candidates,
    ]
    best_reg = min(reg_models, key=lambda r: r["mae"])
    best_cls = max(cls_models, key=lambda r: r["f1"]) if cls_models else None

    return render_template(
        "model_comparison.html",
        reg_models=reg_models,
        cls_models=cls_models,
        best_reg=best_reg,
        best_cls=best_cls,
        results=results,
    )


@app.route("/api")
def api_docs():
    """API documentation + personal key management (key tools for logged-in
    users; docs are public)."""
    user = _current_user()
    plan_status = None
    if user:
        plan_status = db.get_user_plan_status(user["id"])
    return render_template("api_docs.html", user=user, plan_status=plan_status)


@app.route("/analytics")
@login_required
def analytics():
    """Deep-dive analytics: KPIs, chart panels, recent predictions, usage
    trend and system activity."""
    ensure_app_ready()
    try:
        meta = _dtdc_predictor.meta
        m = meta.get("metrics", {})
    except (FileNotFoundError, RuntimeError):
        return render_template(
            "analytics.html",
            error="DTDC model not found. Train with ``python -m dtdc_model train``",
            plots=None,
            kpis=None,
            recent=None,
            activity=None,
        ), 503

    demo_mode = bool(session.get("demo_mode"))
    pred_count = db.count_dtdc_predictions(include_demo=demo_mode)
    pred_rows = db.fetch_dtdc_predictions(limit=10000, include_demo=demo_mode)
    if pred_rows:
        pdf = db.rows_to_dataframe(pred_rows)
        avg_pred = float(pdf["predicted_days"].mean())
    else:
        avg_pred = 0.0

    kpis = {
        "avg_predicted_days": round(avg_pred, 2),
        "model_mae": m.get("mae_days", 0),
        "model_rmse": m.get("rmse_days", 0),
        "model_r2": m.get("r2", 0),
        "prediction_count": pred_count,
        "model_version": meta.get("model_version", ""),
        "algorithm": _friendly_algorithm(meta.get("algorithm", "")),
        "dataset_rows": meta.get("dataset_rows", 0),
        "training_date": (meta.get("training_date") or "")[:10],
        "user_count": db.count_users(),
    }

    p1 = charts.plot_mode_impact()
    p2 = charts.plot_prediction_distribution()
    p3 = charts.plot_top_routes()
    plots = {
        "mode_impact": os.path.basename(p1),
        "distribution": os.path.basename(p2),
        "routes": os.path.basename(p3),
    }
    recent = db.fetch_dtdc_predictions(limit=8, include_demo=demo_mode)
    activity = _build_activity_feed(pred_count)
    return render_template(
        "analytics.html", plots=plots, error=None, kpis=kpis,
        recent=recent, activity=activity, demo_mode=demo_mode,
    )


def _build_activity_feed(prediction_count: int) -> list[dict]:
    """A small chronological activity feed for the analytics page."""
    feed = []
    try:
        meta = _dtdc_predictor.meta
        feed.append({
            "icon": "model",
            "title": "Model retrained",
            "detail": f"v{meta.get('model_version', '')} on {meta.get('dataset_rows', 0):,} records",
            "time": (meta.get("training_date") or "")[:10],
        })
    except (FileNotFoundError, RuntimeError):
        pass
    feed.append({
        "icon": "predict",
        "title": "Predictions completed",
        "detail": f"{prediction_count:,} total across the platform",
        "time": "live",
    })
    user_count = db.count_users()
    if user_count > 0:
        feed.append({
            "icon": "user",
            "title": "Users registered",
            "detail": f"{user_count:,} accounts on CourierAI",
            "time": "—",
        })
    feed.append({
        "icon": "api",
        "title": "API keys generated",
        "detail": "Per-account keys issued with hashed storage",
        "time": "live",
    })
    return feed


@app.route("/blog")
def blog():
    return render_template("blog.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/forgot-password")
def forgot_password():
    return render_template("forgot-password.html")


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------
@app.errorhandler(404)
def not_found(_error):
    return render_template("error.html", code=404), 404


@app.errorhandler(500)
def server_error(_error):
    return render_template("error.html", code=500), 500


def _parse_dtdc_input(values):
    """Extract and validate DTDC prediction inputs from a form/JSON dict.

    Returns keyword arguments suitable for DTDCPredictor.predict().
    Raises ValueError on invalid input.
    """
    origin = str(values.get("origin", "")).strip()
    destination = str(values.get("destination", "")).strip()
    booking_weekday = str(values.get("booking_weekday", "")).strip()
    mode = str(values.get("mode", "")).strip()
    nature_of_consignment = str(values.get("nature_of_consignment", "")).strip()

    if not origin:
        raise ValueError("origin is required.")
    if not destination:
        raise ValueError("destination is required.")
    if origin.lower() == destination.lower():
        raise ValueError("Origin and destination cannot be the same city.")
    if not booking_weekday:
        raise ValueError("booking_weekday is required.")
    if not mode:
        raise ValueError("mode is required.")
    if not nature_of_consignment:
        raise ValueError("nature_of_consignment is required.")

    try:
        total_pieces = int(values.get("total_pieces", ""))
    except (TypeError, ValueError):
        raise ValueError("total_pieces must be a valid integer.")
    if total_pieces <= 0:
        raise ValueError("total_pieces must be positive.")

    for name in ("actual_weight", "volumetric_weight", "chargeable_weight"):
        try:
            val = float(values.get(name, ""))
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a valid number.")
        if val <= 0:
            raise ValueError(f"{name} must be positive.")

    return {
        "origin": origin,
        "destination": destination,
        "booking_weekday": booking_weekday,
        "mode": mode,
        "nature_of_consignment": nature_of_consignment,
        "total_pieces": total_pieces,
        "actual_weight": float(values.get("actual_weight", 0)),
        "volumetric_weight": float(values.get("volumetric_weight", 0)),
        "chargeable_weight": float(values.get("chargeable_weight", 0)),
    }


def _quota_error_template(source):
    """Render the prediction form with the quota-exceeded error message."""
    message = (
        "You have used all your free predictions this month. "
        "Upgrade to Pro (₹299/month or Pro+ ₹1,299/year) for unlimited predictions."
    )
    if source == "demo":
        return render_template(
            "demo.html", metrics=_load_metrics(), error=message, source="demo"
        ), 402
    if source == "predict":
        return render_template(
            "predict.html", metrics=_load_metrics(), error=message, source="predict"
        ), 402
    return render_template("index.html", metrics=_load_metrics(), error=message), 402


@app.route("/predict", methods=["POST"])
def predict_form():
    ensure_app_ready()
    try:
        kwargs = _parse_dtdc_input(request.form)
        # Logged-in users draw on their plan's monthly prediction quota;
        # anonymous visitors keep the open public demo (no quota).
        user = _current_user()
        source = request.form.get("_source")
        if user:
            status = db.get_user_plan_status(user["id"])
            if not status["unlimited"] and status["remaining"] <= 0:
                return _quota_error_template(source)
        result = _dtdc_predictor.predict(**kwargs)
        if user:
            # Authoritative charge AFTER a successful prediction so a failed
            # predict (missing model etc.) never consumes quota.
            allowed, _quota_err = db.consume_prediction(user["id"])
            if not allowed:
                return _quota_error_template(source)

        # Log prediction to audit table
        db.insert_dtdc_prediction(
            origin=kwargs["origin"],
            destination=kwargs["destination"],
            booking_weekday=kwargs["booking_weekday"],
            mode=kwargs["mode"],
            nature_of_consignment=kwargs["nature_of_consignment"],
            total_pieces=kwargs["total_pieces"],
            actual_weight=kwargs["actual_weight"],
            volumetric_weight=kwargs["volumetric_weight"],
            chargeable_weight=kwargs["chargeable_weight"],
            predicted_days=result.predicted_days,
            model_version=result.model_version,
        )

        # Load model metadata for result template context
        meta = _dtdc_predictor.meta
        m = meta.get("metrics", {})
        dataset_rows = meta.get("dataset_rows", 0)

        return render_template(
            "result.html",
            predicted_days=result.predicted_days,
            origin=kwargs["origin"],
            destination=kwargs["destination"],
            booking_weekday=kwargs["booking_weekday"],
            mode=kwargs["mode"],
            nature_of_consignment=kwargs["nature_of_consignment"],
            total_pieces=kwargs["total_pieces"],
            actual_weight=kwargs["actual_weight"],
            volumetric_weight=kwargs["volumetric_weight"],
            chargeable_weight=kwargs["chargeable_weight"],
            algorithm=_friendly_algorithm(result.algorithm),
            model_version=result.model_version,
            model_mae=m.get("mae_days", 0),
            dataset_rows=dataset_rows,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        source = request.form.get("_source")
        if source == "demo":
            return render_template(
                "demo.html", metrics=_load_metrics(), error=str(exc), source="demo"
            ), 400
        if source == "predict":
            return render_template(
                "predict.html", metrics=_load_metrics(), error=str(exc), source="predict"
            ), 400
        return render_template("index.html", metrics=_load_metrics(), error=str(exc)), 400


@app.route("/api/predict", methods=["POST"])
@limiter.limit(config.API_RATE_LIMIT, key_func=_api_identity)
@limiter.limit(config.API_RATE_LIMIT_DAILY, key_func=_api_identity)
def predict_api():
    # Authenticate FIRST so unauthenticated hammering never pays for bootstrap
    # or DB seeding work. API key required: X-API-Key header or Bearer token.
    api_key = _extract_api_key(request)
    api_user = db.get_user_by_api_key(_hash_api_key(api_key)) if api_key else None
    if not api_user:
        return jsonify({"error": "A valid API key is required. Send it in the X-API-Key header."}), 401

    ensure_app_ready()
    values = request.get_json(silent=True)
    if not isinstance(values, dict):
        return jsonify({"error": "A JSON object is required."}), 400

    # Validate input BEFORE charging the quota so bad requests never count.
    try:
        kwargs = _parse_dtdc_input(values)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    # Plan quota: cheap read-only check first so exhausted users get a fast
    # 402 without burning model compute; the authoritative charge happens
    # after the prediction succeeds (failed predicts are never charged).
    status = db.get_user_plan_status(api_user["id"])
    if not status["unlimited"] and status["remaining"] <= 0:
        return jsonify(
            {
                "error": (
                    f"You have used all {status['limit']} free predictions this month. "
                    "Upgrade to Pro for unlimited predictions."
                ),
                "plan": status["plan"],
                "limit": status["limit"],
                "used": status["used"],
            }
        ), 402

    try:
        result = _dtdc_predictor.predict(**kwargs)

        # Log prediction to audit table
        db.insert_dtdc_prediction(
            origin=kwargs["origin"],
            destination=kwargs["destination"],
            booking_weekday=kwargs["booking_weekday"],
            mode=kwargs["mode"],
            nature_of_consignment=kwargs["nature_of_consignment"],
            total_pieces=kwargs["total_pieces"],
            actual_weight=kwargs["actual_weight"],
            volumetric_weight=kwargs["volumetric_weight"],
            chargeable_weight=kwargs["chargeable_weight"],
            predicted_days=result.predicted_days,
            model_version=result.model_version,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        return jsonify({"error": str(exc)}), 400

    allowed, quota_err = db.consume_prediction(api_user["id"])
    if not allowed:
        status = db.get_user_plan_status(api_user["id"])
        return jsonify(
            {
                "error": quota_err or "Prediction quota exceeded.",
                "plan": status["plan"],
                "limit": status["limit"],
                "used": status["used"],
            }
        ), 402

    return jsonify(
        {
            "predicted_time_days": result.predicted_days,
            "model_version": result.model_version,
            "algorithm": result.algorithm,
        }
    )


@app.route("/admin")
@login_required
@admin_required
def admin():
    ensure_app_ready()
    traffic = request.args.get("traffic") or None
    weather = request.args.get("weather") or None
    min_d = request.args.get("min_distance")
    max_d = request.args.get("max_distance")
    min_distance = float(min_d) if min_d not in (None, "") else None
    max_distance = float(max_d) if max_d not in (None, "") else None

    rows = db.fetch_orders_filtered(
        traffic=traffic,
        weather=weather,
        min_distance=min_distance,
        max_distance=max_distance,
        limit=250,
    )
    return render_template(
        "admin.html",
        rows=rows,
        traffic=traffic or "",
        weather=weather or "",
        min_distance=min_d or "",
        max_distance=max_d or "",
        demo_mode=bool(session.get("demo_mode")),
    )


@app.route("/admin/demo", methods=["POST"])
@login_required
@admin_required
def admin_demo():
    """Admin demo mode: seed demo prediction records so dashboards, analytics
    and charts populate instantly. Demo rows are flagged is_demo=1 and are
    excluded from real analytics unless demo mode is active."""
    if request.form.get("mode") == "off":
        session["demo_mode"] = False
        flash("Demo mode turned off — real analytics restored.", "info")
        return redirect(url_for("admin"))

    session["demo_mode"] = True
    _seed_demo_predictions()
    flash("Demo mode active — sample predictions, charts and activity loaded.", "success")
    return redirect(url_for("admin"))


def _seed_demo_predictions(count: int = 24) -> None:
    """Insert demo prediction rows flagged is_demo=1 so they never mix with
    real audit data (excluded unless demo mode is active)."""
    import random

    cities = ["Mumbai", "Delhi", "Bangalore", "Kolkata", "Chennai", "Hyderabad",
              "Ahmedabad", "Pune", "Jaipur", "Lucknow", "Surat", "Nagpur"]
    modes = ["Surface", "Express", "Air Cargo"]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    rng = random.Random(42)
    for i in range(count):
        origin = rng.choice(cities)
        destination = rng.choice([c for c in cities if c != origin])
        db.insert_dtdc_prediction(
            origin=origin,
            destination=destination,
            booking_weekday=rng.choice(days),
            mode=rng.choice(modes),
            nature_of_consignment=rng.choice(["Dox", "Non-Dox"]),
            total_pieces=rng.randint(1, 4),
            actual_weight=round(rng.uniform(0.3, 6.0), 2),
            volumetric_weight=round(rng.uniform(0.3, 6.0), 2),
            chargeable_weight=round(rng.uniform(0.3, 6.0), 2),
            predicted_days=round(rng.uniform(0.8, 4.5), 2),
            model_version="demo",
            is_demo=1,
        )


# ---------------------------------------------------------------------------
# Experiment lab (admin) — runs train_experiments.py in a background thread
# ---------------------------------------------------------------------------
# Guards against two admins (or a double-click) starting overlapping runs.
_experiment_lock = threading.Lock()


@app.route("/admin/experiments", methods=["POST"])
@login_required
@admin_required
def admin_run_experiments():
    """Kick off a model experiment in a background thread. Results land in
    models/experiment_results.json; progress is polled via
    /admin/experiments/status."""
    scope = request.form.get("scope", "smoke")
    if scope not in te.SCOPES:
        flash(f"Unknown scope: {scope}", "error")
        return redirect(url_for("admin"))

    with _experiment_lock:
        status = te.read_status()
        if status.get("state") == "running":
            flash("An experiment is already running.", "warning")
            return redirect(url_for("admin"))
        te.write_status("running", "Starting experiment…", scope=scope)

        def _worker():
            def _progress(section, label, index, total):
                te.write_status(
                    "running",
                    f"[{section}] {label} ({index}/{total})",
                    scope=scope,
                    progress={"section": section, "label": label,
                              "index": index, "total": total},
                )

            try:
                t0 = time.time()
                te.run_experiment(scope, progress=_progress)
                te.write_status(
                    "done", f"Finished in {time.time() - t0:.0f}s.", scope=scope
                )
            except Exception as exc:  # pragma: no cover - worker error path
                te.write_status("error", f"{type(exc).__name__}: {exc}", scope=scope)

        threading.Thread(target=_worker, daemon=True).start()

    flash(
        f"Experiment started ({scope}). You can leave this page — "
        "results appear on the Model Comparison page when done.",
        "success",
    )
    return redirect(url_for("admin"))


@app.route("/admin/experiments/status")
@login_required
@admin_required
def admin_experiment_status():
    """JSON status for the admin experiment lab (polled by the page)."""
    return jsonify(te.read_status())


@app.route("/dashboard")
@login_required
def dashboard():
    ensure_app_ready()

    # Load DTDC model metadata for KPI calculations
    try:
        meta = _dtdc_predictor.meta
        m = meta.get("metrics", {})
    except (FileNotFoundError, RuntimeError):
        return render_template(
            "dashboard.html",
            error="DTDC model not found. Train with ``python -m dtdc_model train``",
            plots=None,
            kpis=None,
        ), 503

    # Prediction audit stats (demo rows included only while admin demo mode is on)
    demo_mode = bool(session.get("demo_mode"))
    pred_count = db.count_dtdc_predictions(include_demo=demo_mode)
    pred_rows = db.fetch_dtdc_predictions(limit=10000, include_demo=demo_mode)
    if pred_rows:
        pdf = db.rows_to_dataframe(pred_rows)
        avg_pred = float(pdf["predicted_days"].mean())
    else:
        avg_pred = 0.0

    kpis = {
        "avg_predicted_days": round(avg_pred, 2),
        "model_mae": m.get("mae_days", 0),
        "model_r2": m.get("r2", 0),
        "prediction_count": pred_count,
        "model_version": meta.get("model_version", ""),
        "algorithm": _friendly_algorithm(meta.get("algorithm", "")),
    }

    # Generate DTDC-based charts (cached to avoid regenerating on every request)
    p1 = charts.plot_mode_impact()
    p2 = charts.plot_prediction_distribution()
    p3 = charts.plot_top_routes()
    plots = {
        "mode_impact": os.path.basename(p1),
        "distribution": os.path.basename(p2),
        "routes": os.path.basename(p3),
    }
    user = _current_user()
    plan_status = db.get_user_plan_status(user["id"])
    recent = db.fetch_dtdc_predictions(limit=6, include_demo=demo_mode)
    return render_template(
        "dashboard.html",
        plots=plots,
        error=None,
        kpis=kpis,
        plan_status=plan_status,
        recent=recent,
        demo_mode=demo_mode,
    )

@app.route("/account")
@login_required
def account():
    """Account page: profile, API key management, plan / usage overview,
    prediction history and recent activity."""
    user = _current_user()
    plan_status = db.get_user_plan_status(user["id"])
    recent = db.fetch_dtdc_predictions(limit=5)
    return render_template(
        "account.html", user=user, plan_status=plan_status, recent=recent
    )


@app.route("/account/profile", methods=["POST"])
@login_required
@limiter.limit("10 per minute")
def update_profile():
    """Update the logged-in user's display name, company and avatar.

    The avatar is a small image upload stored as a data-URL so no filesystem
    write is needed. Oversized or non-image uploads are rejected gracefully.
    """
    user = _current_user()
    full_name = request.form.get("full_name", "").strip()
    company = request.form.get("company", "").strip() or "Personal"

    if len(full_name) < 2:
        flash("Please enter your full name (at least 2 characters).", "error")
        return redirect(url_for("account"))

    avatar = user["avatar"]  # keep the current one unless replaced
    upload = request.files.get("avatar")
    if upload and upload.filename:
        data = upload.read()
        if len(data) > 2 * 1024 * 1024:
            flash("Profile picture must be under 2 MB.", "error")
            return redirect(url_for("account"))
        content_type = (upload.mimetype or "").lower()
        if content_type not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
            flash("Profile picture must be a PNG, JPG, WebP or GIF image.", "error")
            return redirect(url_for("account"))
        # Validate magic bytes so a spoofed mimetype can't smuggle non-image
        # data into the stored avatar (data-URLs are rendered in <img src>).
        if not _looks_like_image(data, content_type):
            flash("Profile picture does not look like a valid image.", "error")
            return redirect(url_for("account"))
        avatar = f"data:{content_type};base64,{base64.b64encode(data).decode('ascii')}"

    db.update_user_profile(user["id"], full_name, company, avatar)
    flash("Profile updated.", "success")
    return redirect(url_for("account"))


def _looks_like_image(data: bytes, content_type: str) -> bool:
    """Cheap magic-byte check against the declared image type."""
    signatures = {
        "image/png": b"\x89PNG\r\n\x1a\n",
        "image/jpeg": b"\xff\xd8\xff",
        "image/gif": b"GIF8",
        "image/webp": b"RIFF",  # plus WEBP at offset 8, checked below
    }
    sig = signatures.get(content_type)
    if sig is None:
        return False
    if data[: len(sig)] != sig:
        return False
    if content_type == "image/webp":
        return data[8:12] == b"WEBP"
    return True


@app.route("/account/key")
@login_required
@limiter.limit("30 per minute")
def account_api_key():
    """Return the caller's API key in full, decrypted on demand so the key is
    never embedded in the page unless the owner reveals it."""
    user = _current_user()
    if not user["api_key_enc"]:
        return jsonify({"error": "No key stored yet — regenerate one to reveal it."}), 404
    try:
        key = _decrypt_api_key(user["api_key_enc"])
    except Exception:
        return jsonify({"error": "Unable to decrypt the stored key. Regenerate it."}), 500
    return jsonify({"api_key": key})


@app.route("/account/upgrade", methods=["POST"])
@login_required
@limiter.limit("10 per minute")
def upgrade_plan():
    """Switch the logged-in user's plan (demo billing — no real payment).

    POST /account/upgrade with plan=pro_monthly | pro_yearly | basic
    """
    plan = request.form.get("plan", "")
    if plan not in config.PLANS:
        flash("Unknown plan.", "error")
        return redirect(url_for("account"))

    user = _current_user()
    if plan == "basic":
        db.set_user_plan(user["id"], "basic", None)
        flash("Switched to the free Basic plan — 50 predictions per month.", "info")
    else:
        from datetime import datetime, timedelta, timezone

        days = config.PLAN_DURATION_DAYS.get(plan, 30)
        expires = (datetime.now(timezone.utc) + timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        db.set_user_plan(user["id"], plan, expires)
        info = config.PLANS[plan]
        flash(
            f"Upgraded to {info['name']} (₹{info['price']}/{info['period']}) — "
            "unlimited predictions active. (Demo billing — no payment was processed.)",
            "success",
        )
    next_url = request.form.get("next")
    if next_url:
        return redirect(_safe_next(next_url))
    return redirect(request.referrer or url_for("account"))


@app.route("/account/regenerate-key", methods=["POST"])
@login_required
@limiter.limit(config.REGENERATE_KEY_RATE_LIMIT)
def regenerate_api_key():
    """Issue a fresh API key, revoking the previous one immediately."""
    user = _current_user()
    api_key = _generate_api_key()
    db.set_api_key(
        user["id"],
        _hash_api_key(api_key),
        _key_hint(api_key),
        _encrypt_api_key(api_key),
    )
    flash(api_key, "api_key")
    flash("New API key generated — the old key is no longer valid.", "success")
    return redirect(url_for("account"))


@app.errorhandler(429)
def rate_limited(_error):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Rate limit exceeded. Please slow down and try again later."}), 429
    return render_template("error.html", code=429), 429


@app.route("/health")
@limiter.exempt
def health():
    return jsonify({"status": "healthy"})


@app.route("/metrics")
def metrics_json():
    ensure_app_ready()
    try:
        meta = _dtdc_predictor.meta
        m = meta.get("metrics", {})
        return jsonify({
            "mae_days": m.get("mae_days", 0),
            "rmse_days": m.get("rmse_days", 0),
            "r2": m.get("r2", 0),
            "model_version": meta.get("model_version", MODEL_VERSION),
            "algorithm": meta.get("algorithm", MODEL_ALGORITHM),
            "training_date": meta.get("training_date", ""),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 503


if __name__ == "__main__":
    ensure_app_ready()
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, host="0.0.0.0", port=port)
