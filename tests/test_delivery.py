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
    for i in range(30):
        d = 1.0 + i * 0.4
        h = (i % 12) + 6
        tr = ["Low", "Medium", "High"][i % 3]
        w = ["Clear", "Rainy"][i % 2]
        y = 10.0 + d * 3.5 + (5 if tr == "Medium" else 12 if tr == "High" else 0)
        y += 6 if w == "Rainy" else 0
        db.insert_order(d, h, tr, w, round(y, 2))

    r = ml_model.train_and_save()
    assert r.train_rows == 30
    assert r.mae >= 0

    client = flask_app.app.test_client()
    resp = client.post(
        "/api/predict",
        json={"distance": 5.0, "order_time": 10, "traffic_level": "High", "weather": "Rainy"},
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert "predicted_time_minutes" in body
    assert body["predicted_time_minutes"] > 0


def test_charts_generate(monkeypatch, tmp_path):
    db, ml_model, charts, _ = _reload_with_paths(monkeypatch, tmp_path)
    db.init_db()
    for i in range(15):
        db.insert_order(
            1.0 + i * 0.5,
            8 + (i % 10),
            ["Low", "Medium", "High"][i % 3],
            ["Clear", "Rainy"][i % 2],
            20.0 + i * 1.2,
        )
    ml_model.train_and_save()
    p = charts.plot_delivery_vs_distance()
    assert Path(p).is_file()
