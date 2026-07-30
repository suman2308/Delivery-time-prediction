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
