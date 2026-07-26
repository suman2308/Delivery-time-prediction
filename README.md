# Smart Delivery Time Prediction Platform

A machine learning application for predicting order delivery times based on route distance, departure hour, traffic conditions, and weather. Built with **Python**, **Flask**, **scikit-learn**, **SQLite**, and modern web design principles.

---

## Key Features

- **Predictive Engine**: Linear regression pipeline using scikit-learn with categorical feature encoding for traffic density and weather conditions.
- **RESTful API**: Clean JSON endpoints (`/api/predict`, `/metrics`, `/health`) for easy system integration.
- **Interactive Analytics Dashboard**: Visualizations of feature correlations, traffic impacts, and prediction error margins powered by Matplotlib.
- **Order Database Explorer**: Filter and inspect historical delivery records with SQL parameters.
- **Dynamic Glassmorphic UI**: Ambient UI with light/dark theme switching and responsive controls.

---

## Tech Stack

- **Backend**: Python 3.10+, Flask, SQLite3, Joblib
- **Machine Learning**: scikit-learn, pandas, NumPy
- **Analytics & Visuals**: Matplotlib
- **Frontend**: Modern Vanilla CSS (Glassmorphism, Design Tokens), JavaScript (ES6)
- **Testing**: pytest

---

## Quick Start (Local Setup)

### 1. Environment Setup

```bash
python -m venv .venv
```

**Activate Environment:**
- **Windows (PowerShell):** `.\.venv\Scripts\Activate.ps1`
- **macOS / Linux:** `source .venv/bin/activate`

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Initialize Database & Model

Generate initial dataset (400 synthetic orders):
```bash
python seed_data.py
```

Train regression model:
```bash
python train_model.py
```

### 4. Run Application

```bash
python app.py
```

Access the application in your browser at `http://127.0.0.1:5000`.

---

## API Usage

### Predict Delivery Time (`POST /api/predict`)

**Request:**
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"distance": 6.5, "order_time": 14, "traffic_level": "Medium", "weather": "Clear"}'
```

**Response:**
```json
{
  "order_id": null,
  "prediction_id": 1,
  "predicted_time_minutes": 31.45
}
```

---

## Testing

Run unit & integration tests:
```bash
pytest
```

---

## Project Architecture

```
├── app.py           # Flask server & route handlers
├── ml_model.py      # Regression pipeline & model inference
├── database.py      # SQLite connection & query handlers
├── charts.py        # Analytics plot generator
├── seed_data.py     # Data generation utilities
├── train_model.py   # Model training script
├── cli.py           # CLI interactive predictor
├── schema.sql       # Database schema setup
├── static/          # CSS design tokens & client JavaScript
└── templates/       # Glassmorphic HTML templates
```

---

## Deploy on Render

This repo includes `render.yaml` for deployment:
1. Push repository to GitHub.
2. In Render, select **New +** -> **Blueprint**.
3. Select this repository to deploy automatically.

---

## License

Distributed under the MIT License.
