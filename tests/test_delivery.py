import importlib
from pathlib import Path

import pytest


def _reload_with_paths(monkeypatch, tmp_path):
    plots = tmp_path / "plots"
    plots.mkdir()
    import config

    monkeypatch.setattr(config, "DATABASE_PATH", str(tmp_path / "test.db"))
    monkeypatch.setattr(config, "MODEL_PATH", str(tmp_path / "model.joblib"))
    monkeypatch.setattr(config, "PLOTS_DIR", str(plots))

    import app as flask_app
    import charts
    import database as db
    import ml_model

    importlib.reload(db)
    importlib.reload(ml_model)
    importlib.reload(charts)
    importlib.reload(flask_app)
    return db, ml_model, charts, flask_app


def test_train_predict_api(monkeypatch, tmp_path):
    db, ml_model, _, flask_app = _reload_with_paths(monkeypatch, tmp_path)
    db.init_db()
    # Seed synthetic-data orders (still used by dashboard/legacy routes)
    for i in range(30):
        d = 1.0 + i * 0.4
        h = (i % 12) + 6
        tr = ["Low", "Medium", "High"][i % 3]
        w = ["Clear", "Rainy"][i % 2]
        y = 10.0 + d * 3.5 + (5 if tr == "Medium" else 12 if tr == "High" else 0)
        y += 6 if w == "Rainy" else 0
        db.insert_order(d, h, tr, w, round(y, 2))

    # Train legacy model (still used by dashboard)
    r = ml_model.train_and_save()
    assert r.train_rows == 30
    assert r.mae >= 0

    # Verify DTDC model is loaded (used by /api/predict)
    from dtdc_model import DTDCPredictor
    predictor = DTDCPredictor()
    predictor.load()
    assert predictor.is_loaded

    client = flask_app.app.test_client()
    api_key = _register_get_key(client)
    import json
    resp = client.post(
        "/api/predict",
        data=json.dumps({
            "origin": "Mumbai",
            "destination": "Pune",
            "booking_weekday": "Monday",
            "mode": "Surface",
            "nature_of_consignment": "Dox",
            "total_pieces": 1,
            "actual_weight": 0.5,
            "volumetric_weight": 0.8,
            "chargeable_weight": 0.5,
        }).encode("utf-8"),
        content_type="application/json",
        headers={"X-API-Key": api_key},
    )
    assert resp.status_code == 200, f"Got {resp.status_code}: {resp.data.decode()}"
    body = resp.get_json()
    assert "predicted_time_days" in body, f"Missing predicted_time_days in {body}"
    assert body["predicted_time_days"] > 0
    assert body["algorithm"] == "HistGradientBoostingRegressor"
    assert body["model_version"] == "1.0.0"


def test_charts_generate(monkeypatch, tmp_path):
    db, ml_model, charts, _ = _reload_with_paths(monkeypatch, tmp_path)
    db.init_db()
    # Seed synthetic orders (legacy, still used by some routes)
    for i in range(15):
        db.insert_order(
            1.0 + i * 0.5,
            8 + (i % 10),
            ["Low", "Medium", "High"][i % 3],
            ["Clear", "Rainy"][i % 2],
            20.0 + i * 1.2,
        )
    ml_model.train_and_save()

    # Seed a DTDC prediction so charts have data
    db.insert_dtdc_prediction(
        origin="Mumbai", destination="Pune",
        booking_weekday="Monday", mode="Surface",
        nature_of_consignment="Dox",
        total_pieces=1, actual_weight=0.5,
        volumetric_weight=0.8, chargeable_weight=0.5,
        predicted_days=3.84, model_version="1.0.0",
    )

    # Test new DTDC chart functions
    p1 = charts.plot_mode_impact()
    assert Path(p1).is_file(), f"mode_impact chart not created: {p1}"

    p2 = charts.plot_prediction_distribution()
    assert Path(p2).is_file(), f"distribution chart not created: {p2}"

    p3 = charts.plot_top_routes()
    assert Path(p3).is_file(), f"top_routes chart not created: {p3}"


def test_marketing_pages_render(monkeypatch, tmp_path):
    """All public marketing/product pages render successfully (HTTP 200)."""
    monkeypatch.setenv("BOOTSTRAP_ON_START", "0")
    db, ml_model, _, flask_app = _reload_with_paths(monkeypatch, tmp_path)
    db.init_db()
    client = flask_app.app.test_client()

    pages = [
        "/",
        "/about",
        "/demo",
        "/tracking",
        "/pricing",
        "/blog",
        "/contact",
        "/login",
        "/register",
        "/forgot-password",
        "/predict",
        "/plans",
        "/model-comparison",
        "/api",
    ]
    for path in pages:
        resp = client.get(path)
        assert resp.status_code == 200, f"{path} returned {resp.status_code}"

    # Unknown route must render the styled 404 page
    resp = client.get("/does-not-exist")
    assert resp.status_code == 404
    assert b"Page Not Found" in resp.data or b"404" in resp.data


def _csrf(client, token="test-csrf-token"):
    """Seed a CSRF token into the client's session and return it.

    Mirrors the real browser flow where the first GET of any page renders
    a form whose {{ csrf_token() }} call stores the token in the session.
    """
    with client.session_transaction() as sess:
        sess["_csrf_token"] = token
    return token


def _post(client, url, data=None, **kwargs):
    """POST with a valid CSRF token attached, like a real browser would."""
    data = dict(data or {})
    data.setdefault("_csrf_token", _csrf(client))
    return client.post(url, data=data, **kwargs)


def _register_and_login(client, email="demo@example.com", password="SuperSecret1!"):
    """Register a fresh user via the form and return the resulting session client."""
    resp = _post(
        client,
        "/register",
        {
            "full_name": "Demo User",
            "company": "RapidCargo",
            "email": email,
            "password": password,
        },
        follow_redirects=False,
    )
    assert resp.status_code == 302, f"register returned {resp.status_code}"
    return resp


def _register_get_key(client, email="api@example.com", password="SuperSecret1!"):
    """Register a user and return the API key flashed once at registration."""
    _register_and_login(client, email=email, password=password)
    with client.session_transaction() as sess:
        flashes = sess.get("_flashes", [])
    key = next((m for c, m in flashes if c == "api_key"), None)
    assert key and key.startswith("scp_live_"), f"no API key flashed at registration ({key!r})"
    return key


def test_auth_register_login_logout(monkeypatch, tmp_path):
    """Register creates an account + session; logout clears it; login works."""
    monkeypatch.setenv("BOOTSTRAP_ON_START", "0")
    db, ml_model, _, flask_app = _reload_with_paths(monkeypatch, tmp_path)
    db.init_db()
    client = flask_app.app.test_client()

    # Register → redirected to plan selection and logged in
    resp = _register_and_login(client)
    assert resp.headers["Location"].endswith("/plans")
    with client.session_transaction() as sess:
        assert sess.get("user_id") is not None

    # Logout clears the session (POST-only)
    resp = _post(client, "/logout")
    assert resp.status_code == 302
    assert client.get("/logout").status_code == 405, "GET logout must be rejected"
    with client.session_transaction() as sess:
        assert sess.get("user_id") is None

    # Duplicate email (different case) is rejected with a validation flash
    resp = _post(
        client,
        "/register",
        {
            "full_name": "Other User",
            "email": "DEMO@example.com",
            "password": "SuperSecret1!",
        },
    )
    assert resp.status_code == 200
    assert b"already exists" in resp.data

    # Wrong password → back to the login page with an error
    resp = _post(client, "/login", {"email": "demo@example.com", "password": "wrong"})
    assert resp.status_code == 200
    assert b"Invalid email or password" in resp.data

    # Correct login → dashboard
    resp = _post(
        client, "/login", {"email": "demo@example.com", "password": "SuperSecret1!"}
    )
    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/dashboard")

    # Password is stored hashed, never in plaintext
    row = db.get_user_by_email("demo@example.com")
    assert row is not None
    assert row["password_hash"] != "SuperSecret1!"
    assert row["password_hash"].startswith("scrypt:") or row["password_hash"].startswith("pbkdf2:")


def test_protected_routes_require_login(monkeypatch, tmp_path):
    """/dashboard and /admin redirect anonymous visitors to the login page."""
    monkeypatch.setenv("BOOTSTRAP_ON_START", "0")
    db, ml_model, _, flask_app = _reload_with_paths(monkeypatch, tmp_path)
    db.init_db()
    client = flask_app.app.test_client()

    resp = client.get("/dashboard")
    assert resp.status_code == 302
    assert resp.headers["Location"] == "/login?next=/dashboard"

    resp = client.get("/admin")
    assert resp.status_code == 302
    assert resp.headers["Location"] == "/login?next=/admin"

    # Login, then both pages are reachable and render
    _register_and_login(client)
    assert client.get("/dashboard").status_code in (200, 503)
    # Non-admins are redirected away from /admin
    assert client.get("/admin").status_code == 302
    # Granting admin access (via ADMIN_EMAILS) unlocks the data explorer
    import config
    monkeypatch.setattr(config, "ADMIN_EMAILS", {"demo@example.com"})
    assert client.get("/admin").status_code == 200


def test_api_key_required_and_valid(monkeypatch, tmp_path):
    """The prediction API requires a valid API key; regeneration revokes it."""
    monkeypatch.setenv("BOOTSTRAP_ON_START", "0")
    db, ml_model, _, flask_app = _reload_with_paths(monkeypatch, tmp_path)
    db.init_db()
    client = flask_app.app.test_client()
    api_key = _register_get_key(client)
    payload = {
        "origin": "Mumbai", "destination": "Pune",
        "booking_weekday": "Monday", "mode": "Surface",
        "nature_of_consignment": "Dox",
        "total_pieces": 1, "actual_weight": 0.5,
        "volumetric_weight": 0.8, "chargeable_weight": 0.5,
    }

    # No key -> 401
    resp = client.post("/api/predict", json=payload)
    assert resp.status_code == 401

    # Wrong key -> 401
    resp = client.post("/api/predict", json=payload, headers={"X-API-Key": "scp_live_bogus"})
    assert resp.status_code == 401

    # Bearer auth header also works
    resp = client.post("/api/predict", json=payload, headers={"Authorization": f"Bearer {api_key}"})
    assert resp.status_code == 200

    # Regenerate via the account page revokes the old key
    resp = _post(client, "/account/regenerate-key")
    assert resp.status_code == 302
    with client.session_transaction() as sess:
        flashes = sess.get("_flashes", [])
    keys = [m for c, m in flashes if c == "api_key"]
    assert len(keys) >= 2, "regeneration should add a second api_key flash"
    new_key = keys[-1]
    assert new_key and new_key != api_key

    resp = client.post("/api/predict", json=payload, headers={"X-API-Key": api_key})
    assert resp.status_code == 401, "old key must be revoked after regeneration"
    resp = client.post("/api/predict", json=payload, headers={"X-API-Key": new_key})
    assert resp.status_code == 200


def test_account_key_reveal(monkeypatch, tmp_path):
    """The /account/key endpoint returns the owner's key on demand."""
    monkeypatch.setenv("BOOTSTRAP_ON_START", "0")
    db, ml_model, _, flask_app = _reload_with_paths(monkeypatch, tmp_path)
    db.init_db()
    client = flask_app.app.test_client()

    # Anonymous access is blocked
    assert client.get("/account/key").status_code == 302

    # Owner can reveal the exact key flashed at registration
    api_key = _register_get_key(client, email="reveal@example.com")
    resp = client.get("/account/key")
    assert resp.status_code == 200
    assert resp.get_json()["api_key"] == api_key

    # After regeneration the endpoint returns the new key
    _post(client, "/account/regenerate-key")
    with client.session_transaction() as sess:
        flashes = sess.get("_flashes", [])
    new_key = [m for c, m in flashes if c == "api_key"][-1]
    resp = client.get("/account/key")
    assert resp.get_json()["api_key"] == new_key

    # A user without a stored (encrypted) key gets a clean 404
    uid = db.create_user("No Key", "nokey@example.com", "not-a-real-hash")
    client2 = flask_app.app.test_client()
    with client2.session_transaction() as sess:
        sess["user_id"] = uid
    assert client2.get("/account/key").status_code == 404


def test_api_rate_limited(monkeypatch, tmp_path):
    """The API enforces its per-key rate limit with HTTP 429."""
    import config
    monkeypatch.setattr(config, "API_RATE_LIMIT", "2 per minute")
    monkeypatch.setenv("BOOTSTRAP_ON_START", "0")
    db, ml_model, _, flask_app = _reload_with_paths(monkeypatch, tmp_path)
    db.init_db()
    client = flask_app.app.test_client()
    api_key = _register_get_key(client, email="rate@example.com")
    payload = {
        "origin": "Mumbai", "destination": "Pune",
        "booking_weekday": "Monday", "mode": "Surface",
        "nature_of_consignment": "Dox",
        "total_pieces": 1, "actual_weight": 0.5,
        "volumetric_weight": 0.8, "chargeable_weight": 0.5,
    }

    assert client.post("/api/predict", json=payload, headers={"X-API-Key": api_key}).status_code == 200
    assert client.post("/api/predict", json=payload, headers={"X-API-Key": api_key}).status_code == 200
    resp = client.post("/api/predict", json=payload, headers={"X-API-Key": api_key})
    assert resp.status_code == 429, f"expected 429, got {resp.status_code}"
    assert resp.get_json()["error"]


def test_login_next_redirect_is_safe(monkeypatch, tmp_path):
    """The next parameter cannot be abused for open redirects."""
    monkeypatch.setenv("BOOTSTRAP_ON_START", "0")
    db, ml_model, _, flask_app = _reload_with_paths(monkeypatch, tmp_path)
    db.init_db()
    client = flask_app.app.test_client()
    _register_and_login(client)
    _post(client, "/logout")

    for evil in ("//evil.example.com", "/\\evil.example.com", "https://evil.example.com"):
        resp = _post(
            client,
            "/login",
            {"email": "demo@example.com", "password": "SuperSecret1!", "next": evil},
        )
        assert resp.status_code == 302, f"next={evil!r} returned {resp.status_code}"
        assert resp.headers["Location"].endswith("/dashboard"), f"open redirect via {evil!r}"


def test_csrf_protection_blocks_unsigned_post(monkeypatch, tmp_path):
    """A state-changing POST without a valid CSRF token is rejected (OWASP)."""
    monkeypatch.setenv("BOOTSTRAP_ON_START", "0")
    db, ml_model, _, flask_app = _reload_with_paths(monkeypatch, tmp_path)
    db.init_db()
    client = flask_app.app.test_client()
    _register_and_login(client)

    # No token → logout is blocked; the session must survive.
    resp = client.post("/logout")
    assert resp.status_code == 302, "unsigned POST should be redirected away"
    with client.session_transaction() as sess:
        assert sess.get("user_id") is not None, "unsigned POST must not log the user out"

    # A wrong/forged token is equally rejected.
    resp = client.post("/logout", data={"_csrf_token": "forged-token"})
    assert resp.status_code == 302
    with client.session_transaction() as sess:
        assert sess.get("user_id") is not None

    # The correct token (as a real browser sends) is accepted.
    resp = _post(client, "/logout")
    assert resp.status_code == 302
    with client.session_transaction() as sess:
        assert sess.get("user_id") is None


def test_dashboard_renders(monkeypatch, tmp_path):
    """Analytics dashboard renders with KPIs when the model artifact exists."""
    monkeypatch.setenv("BOOTSTRAP_ON_START", "0")
    db, ml_model, _, flask_app = _reload_with_paths(monkeypatch, tmp_path)
    db.init_db()
    client = flask_app.app.test_client()
    _register_and_login(client)
    resp = client.get("/dashboard")
    assert resp.status_code in (200, 503), f"/dashboard returned {resp.status_code}"


_PAYLOAD = {
    "origin": "Mumbai", "destination": "Pune",
    "booking_weekday": "Monday", "mode": "Surface",
    "nature_of_consignment": "Dox",
    "total_pieces": 1, "actual_weight": 0.5,
    "volumetric_weight": 0.8, "chargeable_weight": 0.5,
}


def test_plan_quota_enforced(monkeypatch, tmp_path):
    """Free plan caps predictions; upgrading unlocks unlimited predictions."""
    import config

    monkeypatch.setattr(config, "FREE_PLAN_LIMIT", 3)
    monkeypatch.setenv("BOOTSTRAP_ON_START", "0")
    db, ml_model, _, flask_app = _reload_with_paths(monkeypatch, tmp_path)
    db.init_db()
    client = flask_app.app.test_client()
    api_key = _register_get_key(client, email="quota@example.com")

    # A new user always starts on the free basic plan
    row = db.get_user_by_email("quota@example.com")
    assert row["plan"] == "basic"

    # First 3 predictions pass
    for _ in range(3):
        resp = client.post("/api/predict", json=_PAYLOAD, headers={"X-API-Key": api_key})
        assert resp.status_code == 200, f"free prediction failed: {resp.data.decode()}"

    # 4th is blocked with 402 + usage details
    resp = client.post("/api/predict", json=_PAYLOAD, headers={"X-API-Key": api_key})
    assert resp.status_code == 402
    body = resp.get_json()
    assert body["plan"] == "basic"
    assert body["limit"] == 3
    assert body["used"] == 3
    assert "Upgrade" in body["error"]

    # Invalid payloads never consume quota: with 0 left, a 400 (not 402)
    bad = dict(_PAYLOAD, total_pieces=-5)
    resp = client.post("/api/predict", json=bad, headers={"X-API-Key": api_key})
    assert resp.status_code == 400, "invalid input should 400, not charge quota"

    # Upgrade to Pro → unlimited again
    resp = _post(client, "/account/upgrade", {"plan": "pro_monthly"})
    assert resp.status_code == 302
    row = db.get_user_by_email("quota@example.com")
    assert row["plan"] == "pro_monthly"
    assert row["plan_expires_at"] is not None
    resp = client.post("/api/predict", json=_PAYLOAD, headers={"X-API-Key": api_key})
    assert resp.status_code == 200, "pro plan should allow predictions again"

    # Downgrade back to free restores the (fresh) quota
    _post(client, "/account/upgrade", {"plan": "basic"})
    row = db.get_user_by_email("quota@example.com")
    assert row["plan"] == "basic"
    resp = client.post("/api/predict", json=_PAYLOAD, headers={"X-API-Key": api_key})
    assert resp.status_code == 200


def test_web_form_quota_for_logged_in(monkeypatch, tmp_path):
    """Logged-in web-form predictions count against the plan quota."""
    import config

    monkeypatch.setattr(config, "FREE_PLAN_LIMIT", 1)
    monkeypatch.setenv("BOOTSTRAP_ON_START", "0")
    db, ml_model, _, flask_app = _reload_with_paths(monkeypatch, tmp_path)
    db.init_db()
    client = flask_app.app.test_client()
    _register_and_login(client, email="webquota@example.com")

    data = {
        "origin": "Mumbai", "destination": "Pune",
        "booking_weekday": "Monday", "mode": "Surface",
        "nature_of_consignment": "Dox",
        "total_pieces": "1", "actual_weight": "0.5",
        "volumetric_weight": "0.8", "chargeable_weight": "0.5",
    }

    resp = _post(client, "/predict", data)
    assert resp.status_code == 200, f"first form prediction failed: {resp.status_code}"
    resp = _post(client, "/predict", data)
    assert resp.status_code == 402, f"quota should block the 2nd form prediction, got {resp.status_code}"
    assert b"Upgrade" in resp.data, "error message should point to Pro upgrade"

    # Anonymous visitors are NOT quota-limited (public demo)
    anon = flask_app.app.test_client()
    resp = _post(anon, "/predict", data)
    assert resp.status_code == 200


def test_plan_upgrade_page_and_account(monkeypatch, tmp_path):
    """Account + pricing pages expose the plan and usage."""
    monkeypatch.setenv("BOOTSTRAP_ON_START", "0")
    db, ml_model, _, flask_app = _reload_with_paths(monkeypatch, tmp_path)
    db.init_db()
    client = flask_app.app.test_client()

    # Anonymous pricing renders
    assert client.get("/pricing").status_code == 200

    # Logged-in account shows the free plan + usage meter
    _register_and_login(client, email="planpage@example.com")
    resp = client.get("/account")
    assert resp.status_code == 200
    assert b"Free" in resp.data
    assert b"50" in resp.data and b"predictions used" in resp.data

    # Yearly upgrade works and sets expiry
    resp = _post(client, "/account/upgrade", {"plan": "pro_yearly"})
    assert resp.status_code == 302
    row = db.get_user_by_email("planpage@example.com")
    assert row["plan"] == "pro_yearly"
    assert row["plan_expires_at"] is not None

    # Unknown plan is rejected
    resp = _post(client, "/account/upgrade", {"plan": "bogus"})
    assert resp.status_code == 302
    row = db.get_user_by_email("planpage@example.com")
    assert row["plan"] == "pro_yearly", "unknown plan must not change the plan"


def test_model_comparison_page_renders(monkeypatch, tmp_path):
    """The Model Comparison page renders with regression + classification tables,
    with and without experiment results present."""
    import train_experiments as te

    # Isolate from any real results file: start from a missing results path,
    # then drop in a results file and confirm real rows appear.
    monkeypatch.setattr(te, "RESULTS_PATH", str(tmp_path / "experiment_results.json"))
    monkeypatch.setenv("BOOTSTRAP_ON_START", "0")
    db, ml_model, _, flask_app = _reload_with_paths(monkeypatch, tmp_path)
    db.init_db()
    client = flask_app.app.test_client()

    resp = client.get("/model-comparison")
    assert resp.status_code == 200, f"/model-comparison returned {resp.status_code}"
    # Both tab panels exist; regression table always has the production row.
    assert b"panel-regression" in resp.data
    assert b"panel-classification" in resp.data
    assert b"HistGradientBoosting" in resp.data

    # With results present, real candidate rows (e.g. CatBoost) show up.
    te.save_experiment_results({
        "scope": "smoke", "dataset_rows": 600,
        "regression": {"base": [{"Model": "CatBoost", "MAE": 0.52,
                                   "RMSE": 0.58, "R2": 0.858, "train_s": 3.2}],
                        "hybrid": []},
        "classification": {"base": [], "hybrid": []},
    })
    resp = client.get("/model-comparison")
    assert resp.status_code == 200
    assert b"CatBoost" in resp.data


def test_experiment_results_roundtrip(monkeypatch, tmp_path):
    """Experiment results JSON round-trips through save/load; status defaults idle."""
    import train_experiments as te

    monkeypatch.setattr(te, "RESULTS_PATH", str(tmp_path / "experiment_results.json"))
    monkeypatch.setattr(te, "STATUS_PATH", str(tmp_path / "dtdc_status" / "status.json"))

    payload = {
        "scope": "smoke",
        "dataset_rows": 600,
        "regression": {
            "base": [{"Model": "CatBoost", "MAE": 0.41, "RMSE": 0.55,
                       "R2": 0.858, "train_s": 3.2}],
            "hybrid": [],
        },
        "classification": {
            "base": [{"Model": "MLP", "Accuracy": 0.93, "Precision": 0.91,
                       "Recall": 0.95, "F1": 0.93, "train_s": 4.1}],
            "hybrid": [],
        },
    }
    te.save_experiment_results(payload)
    assert te.load_experiment_results() == payload
    assert te.load_experiment_results()["regression"]["base"][0]["Model"] == "CatBoost"

    # Status file absent -> idle default; write + read round-trip
    assert te.read_status()["state"] == "idle"
    te.write_status("done", "Finished in 30s.", scope="smoke")
    status = te.read_status()
    assert status["state"] == "done"
    assert status["scope"] == "smoke"

    # Missing results file -> None
    Path(te.RESULTS_PATH).unlink()
    assert te.load_experiment_results() is None


def test_experiment_scopes_and_data_prep(monkeypatch, tmp_path):
    """SCOPES table is sane and prepare_data builds compatible arrays."""
    import train_experiments as te

    assert set(te.SCOPES) == {"smoke", "quick", "reduced", "full"}
    assert te.SCOPES["smoke"]["rows"] < te.SCOPES["quick"]["rows"]
    # reduced/full run on the full dataset
    assert te.SCOPES["reduced"]["rows"] is None
    assert te.SCOPES["full"]["stacking"] == "full"
    # The stacking combo list matches the notebook's shape
    assert te.COMBINATIONS and len(te.COMBINATIONS) == 17
    assert len(te.REDUCED_COMBINATIONS) == 6
    # The flagship 5-model stack is in both sets
    flagship = ["rf", "xgb", "cat", "svm", "mlp"]
    assert flagship in te.COMBINATIONS
    assert flagship in te.REDUCED_COMBINATIONS


def test_plan_month_rollover_and_expiry(monkeypatch, tmp_path):
    """The monthly counter resets on rollover; expired paid plans fall back."""
    import config

    monkeypatch.setattr(config, "FREE_PLAN_LIMIT", 2)
    monkeypatch.setenv("BOOTSTRAP_ON_START", "0")
    db, ml_model, _, flask_app = _reload_with_paths(monkeypatch, tmp_path)
    db.init_db()
    client = flask_app.app.test_client()
    api_key = _register_get_key(client, email="rollover@example.com")
    uid = db.get_user_by_email("rollover@example.com")["id"]

    # Backdate a full month of usage → next prediction still works (rollover reset)
    with db.connection() as conn:
        conn.execute(
            "UPDATE users SET usage_month = '2000-01', predictions_used = 2 WHERE id = ?",
            (uid,),
        )
    resp = client.post("/api/predict", json=_PAYLOAD, headers={"X-API-Key": api_key})
    assert resp.status_code == 200, "stale usage month must reset the counter"
    assert db.get_user_by_id(uid)["predictions_used"] == 1

    # Exhaust the current month's quota
    resp = client.post("/api/predict", json=_PAYLOAD, headers={"X-API-Key": api_key})
    assert resp.status_code == 200
    resp = client.post("/api/predict", json=_PAYLOAD, headers={"X-API-Key": api_key})
    assert resp.status_code == 402

    # Upgrade to a paid plan (resets usage), then backdate its expiry
    _post(client, "/account/upgrade", {"plan": "pro_yearly"})
    assert db.get_user_by_id(uid)["plan"] == "pro_yearly"
    resp = client.post("/api/predict", json=_PAYLOAD, headers={"X-API-Key": api_key})
    assert resp.status_code == 200

    with db.connection() as conn:
        conn.execute(
            "UPDATE users SET plan_expires_at = '2000-01-01 00:00:00' WHERE id = ?",
            (uid,),
        )
    # An expired paid plan falls back to Basic (fresh monthly quota)
    resp = client.post("/api/predict", json=_PAYLOAD, headers={"X-API-Key": api_key})
    assert resp.status_code == 200
    assert db.get_user_by_id(uid)["plan"] == "basic", "expired plan must downgrade"
