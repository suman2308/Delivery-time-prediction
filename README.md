# Smart Delivery Time Prediction

End-to-end system that predicts delivery time from historical order data: **SQLite** storage, **scikit-learn** regression, **Flask** web UI + JSON API, **matplotlib** analytics, and **pytest** tests.

## Features

- **Database**: `orders` (features + actual delivery time) and `predictions` (model output, optional link to an order).
- **ML**: Linear regression with one-hot encoded traffic/weather; hold-out **MAE** and **RMSE**; model saved with **joblib**.
- **Web**: Predict form, analytics dashboard, admin panel with SQL filters, theme toggle + sidebar UI.
- **API**: `POST /api/predict` for real-time predictions.
- **CLI**: Interactive prompts (`python cli.py`).

## Tech stack

Python · Flask · SQLite · pandas · scikit-learn · matplotlib · joblib · pytest

## Prerequisites

- Python **3.10+** recommended  
- `pip`

## Quick start (local)

From the project root:

```bash
python -m venv .venv
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create tables and seed synthetic orders (default 400 rows; change with `--count`):

```bash
python seed_data.py
```

Train the model and save it under `models/delivery_regressor.joblib`:

```bash
python train_model.py
```

Run the web app:

```bash
python app.py
```

Open **http://127.0.0.1:5000** — use **Predict**, **Dashboard**, and **Admin**.

## CLI

```bash
python cli.py
```

Follow prompts for distance, hour, traffic, and weather. Optionally log a prediction row without a new order.

## API example

```bash
curl -X POST http://127.0.0.1:5000/api/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"distance\": 8, \"order_time\": 18, \"traffic_level\": \"High\", \"weather\": \"Rainy\"}"
```

On macOS/Linux use `\` instead of `^` for line continuation. Optional field: `"order_id": <existing id>` to link the prediction.

Other useful endpoints:

- `GET /health` — health check  
- `GET /metrics` — MAE / RMSE JSON (requires trained model)

## Tests

```bash
pytest
```

(`pytest.ini` sets `pythonpath = .` so imports resolve.)

## Environment variables (optional)

| Variable | Purpose |
|----------|---------|
| `DELIVERY_DB_PATH` | SQLite file path (default: `delivery.db` in project root) |
| `DELIVERY_MODEL_PATH` | Saved model path (default: `models/delivery_regressor.joblib`) |

## Project layout (main files)

| Path | Role |
|------|------|
| `schema.sql` | Table definitions |
| `database.py` | SQLite access, filters, inserts |
| `seed_data.py` | Synthetic data generator |
| `ml_model.py` | Train / load / predict |
| `train_model.py` | CLI training entry |
| `app.py` | Flask application |
| `charts.py` | Matplotlib figures for dashboard |
| `cli.py` | Interactive CLI |
| `templates/` | HTML UI |
| `tests/` | Pytest suite |

## Notes

- `*.db`, `models/*.joblib`, and generated plot PNGs under `static/plots/` are **gitignored** — run seed + train locally after clone.
- The dev server (`python app.py`) is for **local development** only; use a production WSGI server (e.g. gunicorn) for deployment.

## License

Use and modify freely for learning and portfolio projects.
