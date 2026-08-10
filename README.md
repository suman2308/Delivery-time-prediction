<div align="center">
  <h1>📦 CourierAI — AI-Powered Delivery Time Prediction</h1>
  <p>
    <strong>End-to-end machine learning platform</strong> for predicting shipment delivery times across Indian cities.<br>
    Trained on 49,639 real courier records. Achieves <strong>MAE 0.54 days</strong> (≈13 hours) with a tuned HistGradientBoosting model.
  </p>
  <p>
    <a href="#-features">Features</a> •
    <a href="#-tech-stack">Tech Stack</a> •
    <a href="#-architecture">Architecture</a> •
    <a href="#-ml-pipeline">ML Pipeline</a> •
    <a href="#-installation">Installation</a> •
    <a href="#-api-docs">API Docs</a> •
    <a href="#-deployment">Deployment</a>
  </p>

  <p>
    <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python">
    <img alt="Flask" src="https://img.shields.io/badge/Flask-3.0%2B-black?logo=flask">
    <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-1.8%2B-orange?logo=scikit-learn">
    <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
    <img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-brightgreen">
  </p>
</div>

---

## 🧠 The Problem

Indian logistics companies handle millions of shipments daily across a complex network of cities, each with different transit modes (Surface, Express, Air Cargo), varying parcel characteristics, and unpredictable delivery timelines. **Customers and businesses need accurate delivery estimates** to plan inventory, manage expectations, and optimize logistics operations.

This project builds a production-ready ML system that predicts delivery duration in **days** given origin, destination, shipment mode, parcel weight, and other booking-time features — trained on **49,639 real DTDC courier records**.

---

## ✨ Features

- **🔮 ML-Powered Predictions** — Tuned HistGradientBoostingRegressor achieving MAE of 0.54 days (≈13 hours)
- **🌐 REST API** — Clean JSON API (`POST /api/predict`) for easy integration into any logistics workflow
- **📊 Live Dashboard** — Real-time analytics with model KPIs (MAE, R²), prediction volume, mode impact charts, and top-route analytics
- **📝 Prediction Audit Log** — Every prediction is logged with full input parameters, predicted value, and model version
- **🎨 Premium Design System** — Dark/light themes, cohesive tokens, fluid responsive layouts, and professional typography
- **🏗️ Versioned Model Artifacts** — Semantic versioning for ML models with companion metadata (hyperparameters, metrics, training date)
- **🔒 Safe Unknown Categories** — OneHotEncoder with `handle_unknown="ignore"` gracefully handles unseen cities or modes
- **👤 User Accounts** — Register, log in, and log out with hashed passwords (Werkzeug scrypt); the analytics dashboard and data explorer are behind a login
- **🔑 Secured REST API** — Per-user API keys (hashed at rest, revocable) required on `POST /api/predict`, with per-key rate limiting and JSON 429 responses
- **💳 Plans & quotas** — Every account starts on the free **Free** plan (50 predictions/month); **Pro ₹299/month** and **Pro+ ₹1,299/year** unlock unlimited predictions. Quotas are enforced on the API (HTTP `402`) and on logged-in web predictions
- **👤 Editable profile** — Update your display name, company (defaults to **Personal** when blank) and profile picture from the Account page
- **🛡️ Same-city guard** — Origin and destination cannot be the same city: the form disables the matching city live and the API rejects it with a clear error
- **🔤 Friendly form hints** — Shipment modes show plain-English labels (Surface *by land*, Express *fast*, Air Cargo *by air*), and the password field explains the 8-character minimum with a text-only strength indicator

---

## 🚀 Live Demo

The application is deployed and running at:

**👉 [https://smart-delivery-prediction.onrender.com](https://smart-delivery-prediction.onrender.com)**

No login required. Open it in any browser and start making predictions immediately.

> ⚠️ **Note:** The free Render tier may take 30-60 seconds to spin up after periods of inactivity (cold start). Once loaded, subsequent requests are fast.

---

## 🗺️ Pages & Access

Every route in the app, who can reach it, and how a visitor gets there. **Public** = no
account; **User** = any registered account; **Admin** = a standalone admin session
(`/admin-login`). The two are strictly separated: a registered account is always a
regular user, and the admin console is always its own session — they never mix.

| Page | Route | Access | Reached from |
|---|---|---|---|
| Home | `/` | Public | Navbar · Footer brand |
| Predict | `/predict` | Public | Navbar · Footer · Homepage CTA |
| AI Prediction Demo | `/demo` | Public | Dashboard · Result · Error pages *(not in navbar — see note below)* |
| Tracking | `/tracking` | Public | Navbar · Footer · Result page |
| Shipment timeline | `/tracking/<id>` | Public | Tracking result · Account history |
| Model Comparison | `/model-comparison` | Public | Navbar · Footer · Homepage |
| API Docs | `/api` | Public | Navbar · Footer · Homepage CTA |
| About | `/about` | Public | Navbar · Footer · Homepage |
| Blog | `/blog` | Public | Footer |
| Contact | `/contact` | Public | Navbar · Footer · Homepage |
| Pricing | `/pricing` | Public | Homepage pricing cards *(not in navbar)* |
| Plans (choose plan) | `/plans` | User | Register redirect · Dashboard “Manage plan” |
| Login | `/login` | Public | Navbar · Footer CTA |
| Register | `/register` | Public | Navbar “Get started” · Login page · API docs |
| Forgot password | `/forgot-password` | Public | Login page · Footer “Privacy & Terms” |
| Admin login | `/admin-login` | Public | Footer · direct URL |
| Admin console | `/admin` | Admin | **Admin ▾ dropdown** (navbar) · `/admin-login` redirect |
| Analytics | `/analytics` | Admin | Redirects to the Admin Console Analytics tab |
| Dashboard | `/dashboard` | User | Navbar (after login) · login redirect |
| Account / profile | `/account` | User | Navbar user chip · Dashboard |
| Prediction result | `/result` | User/Public | POST from the prediction form |
| Health | `/health` | Public | JSON API — no UI link |
| Live metrics | `/metrics` | Public | Homepage “Live metrics JSON” · Demo page |
| API predict | `/api/predict` | API key | JSON API — no UI link |

### 🗂️ Pages that exist but are not linked from the main navigation

These pages are fully functional and reachable by direct URL, but a visitor won't
stumble on them from the navbar/footer:

- **`/demo` (AI Prediction Demo)** — a duplicate of the Predict flow. It is linked only
  from the dashboard, the result page and error pages, **not** from the homepage or
  navbar. Anyone can use `/predict` instead, so this is redundant for users; it exists
  mainly as a shareable demo URL.
- **`/pricing`** — reachable only from the homepage pricing section. The in-app
  upgrade buttons (plans page, account page) point to `/plans`, so `/pricing` is the
  marketing copy of the same offer.
- **`/plans`** — the account-facing plan selector; only reachable after registration
  or from the dashboard's “Manage plan”.
- **`/forgot-password`** — the password-reset request form is a **demo** (no email is
  sent); it's linked from the login page and, unusually, from the footer's “Privacy &
  Terms” link.
- **`/admin-login`** — deliberately footer-only: it is the standalone admin console
  entry point.

> **Note on `/demo`:** since the Predict page in the navbar covers the same workflow,
> `/demo` was left unlinked to avoid duplicate nav entries. If you'd rather have it
> in the navbar (or remove it entirely), it's a one-line change in `templates/base.html`.

---

## 📸 Screenshots

Captured live from a local run at 1440×900 (dark theme). Full gallery:
[`docs/screenshots/`](docs/screenshots/).

| Page | Screenshot |
|---|---|
| Home | ![Home](docs/screenshots/01-home.png) |
| Predict | ![Predict](docs/screenshots/02-predict.png) |
| AI Prediction Demo | ![Demo](docs/screenshots/03-demo.png) |
| Tracking | ![Tracking](docs/screenshots/04-tracking.png) |
| Model Comparison | ![Model comparison](docs/screenshots/05-model-comparison.png) |
| API Docs | ![API docs](docs/screenshots/06-api-docs.png) |
| About | ![About](docs/screenshots/07-about.png) |
| Blog | ![Blog](docs/screenshots/08-blog.png) |
| Contact | ![Contact](docs/screenshots/09-contact.png) |
| Pricing | ![Pricing](docs/screenshots/10-pricing.png) |
| Plans | ![Plans](docs/screenshots/11-plans.png) |
| Login | ![Login](docs/screenshots/12-login.png) |
| Register | ![Register](docs/screenshots/13-register.png) |
| Forgot password | ![Forgot password](docs/screenshots/14-forgot-password.png) |
| Admin login | ![Admin login](docs/screenshots/15-admin-login.png) |
| Dashboard | ![Dashboard](docs/screenshots/16-dashboard.png) |
| Account | ![Account](docs/screenshots/17-account.png) |
| Admin console (Overview) | ![Admin](docs/screenshots/18-admin.png) |

**Admin Console tabs:**

| Tab | Screenshot |
|---|---|
| Users & Plans | ![Admin users](docs/screenshots/20-admin-users.png) |
| System & ML | ![Admin system](docs/screenshots/21-admin-system.png) |
| Demo Mode | ![Admin demo](docs/screenshots/22-admin-demo.png) |
| Analytics | ![Admin analytics](docs/screenshots/23-admin-analytics.png) |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | HTML5, CSS3 (premium design system), JavaScript (ES6), Jinja2 templates |
| **Backend** | Python 3.10+, Flask 3.0 |
| **Machine Learning** | scikit-learn (HistGradientBoostingRegressor), pandas, NumPy |
| **Model Evaluation** | Cross-validation, Grid Search, permutation importance |
| **Database** | SQLite (production), with migration support |
| **Visualization** | Matplotlib (dark-theme charts) |
| **Serialization** | joblib (model artifacts), JSON (metadata) |
| **Deployment** | Render (via `render.yaml`), Gunicorn |
| **Testing** | pytest |
| **Design** | Inter font, CSS custom properties, dark/light theme |

---

## 🏗️ Architecture

```mermaid
graph TB
    subgraph Client
        A[Browser] --> B[Flask Server]
        C[curl / API Client] --> B
    end

    subgraph Flask
        B --> D[Route Handlers]
        D --> E[REST API /api/*]
        D --> F[Web UI /predict]
    end

    subgraph ML
        G[DTDCPredictor Singleton]
        H[dtdc_hgb_v1_0_0.joblib]
        I[dtdc_hgb_v1_0_0.meta.json]
        G --> H
        G --> I
    end

    subgraph Data
        J[(SQLite DB)]
        K[dtdc_predictions Table]
        L[orders Table]
        J --> K
        J --> L
    end

    D --> G
    D --> J
    E --> G
    E --> J
    F --> G
    F --> J

    subgraph Charts
        M[charts.py]
        N[static/plots/*.png]
        M --> N
    end

    F --> M
```

---

## 🤖 ML Pipeline

### Dataset

- **Source**: DTDC Improved Dataset (49,639 real courier records)
- **Target**: `delivery_duration_days` — actual delivery time in days
- **Features**: 9 booking-time fields (no PII, no post-delivery data)

### Preprocessing

| Step | Description |
|---|---|
| Unicode normalisation | NFKC normalization of all category strings |
| Whitespace cleaning | Strip, collapse multiple spaces |
| Case folding | All categories lowercased for consistency |
| Validation | Reject missing/invalid weights, pieces, dates |
| Deduplication | Remove exact duplicate rows |
| Date extraction | Extract weekday name from booking date |

### Feature Engineering

| Feature | Type | Description |
|---|---|---|
| `origin` | Categorical | Origin city (30+ Indian cities) |
| `destination` | Categorical | Destination city (30+ Indian cities) |
| `booking_weekday` | Categorical | Day of week (Monday–Sunday) |
| `mode` | Categorical | Surface, Express, or Air Cargo |
| `nature_of_consignment` | Categorical | Dox (Documents) or Non-Dox (Parcel) |
| `total_pieces` | Numeric | Number of items in shipment |
| `actual_weight` | Numeric | Actual weight in kg |
| `volumetric_weight` | Numeric | Dimensional weight in kg |
| `chargeable_weight` | Numeric | Billing weight (max of actual & volumetric) |

Encoding: OneHotEncoder with `handle_unknown="ignore"` for categoricals.
Numerics: Passed through unchanged (tree-based model handles scale natively).

### Model Comparison

| Model | MAE (days) | RMSE (days) | R² | Training Time |
|---|---|---|---|---|
| **HistGradientBoostingRegressor** (tuned) | **0.5361** | **0.7263** | **0.7466** | 0.8s |
| HistGradientBoostingRegressor (baseline) | 0.5595 | 0.7324 | 0.7424 | 0.8s |
| RandomForestRegressor | 0.5793 | 0.7430 | 0.7349 | 14.1s |

*The table shows the tuned production pipeline's hold-out results. The full base-model / hybrid / stacking comparison — including XGBoost and CatBoost — is runnable from the admin Experiment Lab or `train_experiments.py` (see below).*

### Experiment Harness (`train_experiments.py`)

A full benchmarking suite adapted from the DTDC Kaggle notebook — runnable from the **admin panel** (Experiment Lab) or the CLI. It compares **5 base models** (Random Forest, XGBoost, CatBoost, SVR/SVC, MLP), **voting & stacking hybrids**, and up to **17 stacking combinations** — for both **regression** (delivery days) and **classification** (delayed / on-time) on the same stratified 20% test split.

```bash
# CLI usage
python train_experiments.py --scope smoke    # 600 rows, ~30s (sanity check)
python train_experiments.py --scope quick    # 5,000 rows, ~2 min
python train_experiments.py --scope reduced  # full data, 6 paradigm-covering stack combos
python train_experiments.py --scope full     # full data, all 17 combos
```

Artifacts:
- `models/experiment_results.json` — latest results, consumed by `/model-comparison`
- `dtdc_results/` — per-run progress CSVs (crash-resume) + `status.json` (admin UI polling)

Graceful fallbacks: if `xgboost`/`catboost` aren't installed, those candidates are skipped automatically rather than breaking the run.

### Hyperparameter Tuning

A randomized search over **150 of 960** possible configurations with 5-fold cross-validation (750 fits).

**Search Space:**

| Parameter | Values Tested | Best Value |
|---|---|---|
| `learning_rate` | 0.01, 0.03, 0.05, 0.07, 0.10 | **0.03** |
| `max_iter` | 50, 100, 150, 200 | **150** |
| `max_depth` | None, 6, 8, 10 | **6** |
| `min_samples_leaf` | 20, 30, 50 | **20** |
| `l2_regularization` | 0.0, 0.01, 0.1, 1.0 | **0.0** |

### Final Model

**HistGradientBoostingRegressor** — selected for:
- Lowest holdout MAE (0.5361 days)
- Highest R² (0.7466)
- Near-perfect train/test balance (ratio 0.973)
- Fast inference (~5.8 µs per prediction)
- No overfitting (CV MAE std: ±0.0056)

### Feature Importance

| Feature | Importance |
|---|---|
| `mode_surface` | 0.649 |
| `total_pieces` | 0.207 |
| `nature_of_consignment_dox` | 0.173 |
| `actual_weight` | 0.121 |
| `nature_of_consignment_non-dox` | 0.024 |

**Key insight:** Surface mode dominates — ground transport vs express is the primary driver of delivery duration.

---

## 📈 Final Performance

| Metric | Value |
|---|---|
| **Holdout MAE** | 0.5361 days (~12.9 hours) |
| **Holdout RMSE** | 0.7263 days |
| **Holdout R²** | 0.7466 |
| **CV MAE (5-fold)** | 0.5306 ± 0.0056 days |
| **Training Time** | ~0.8s |
| **Inference Time** | ~5.8 µs / prediction |

---

## 📁 Folder Structure

```
smart-delivery-prediction/
├── app.py                      # Flask application & route handlers
├── dtdc_model.py               # Production DTDC model (training + prediction wrapper)
├── train_experiments.py        # Base/hybrid/stacking experiment harness (admin + CLI)
├── config.py                   # Configuration & environment variables
├── database.py                 # SQLite access layer
├── charts.py                   # Matplotlib chart generation for dashboard
├── ml_model.py                 # Legacy synthetic model (bootstrap/tests only)
├── seed_data.py                # Synthetic data generator (legacy bootstrap)
├── schema.sql                  # Database schema
├── requirements.txt            # Python dependencies
├── render.yaml                 # Render deployment config
├── .gitignore
├── pytest.ini
├── README.md
│
├── data/
│   └── dtdc_preprocessing.py       # Data preprocessing pipeline
│
├── models/
│   ├── dtdc_hgb_v1_0_0.joblib      # Production model artifact (564 KB)
│   ├── dtdc_hgb_v1_0_0.meta.json   # Production model metadata
│   └── .gitkeep
│
├── docs/
│   └── screenshots/                 # Live UI captures (1440×900, dark theme)
│
├── static/
│   ├── css/app.css                  # Design system (tokens, components, themes)
│   ├── js/app.js                    # Interactions (theme, nav, charts, toasts)
│   └── plots/                       # Generated chart images
│
├── templates/
│   ├── base.html                    # Layout (navbar, footer, toast/modal regions)
│   ├── index.html                   # Landing page + live predictor
│   ├── demo.html                    # AI Prediction Demo (shared form)
│   ├── _predict_form.html           # Reusable prediction form partial
│   ├── result.html                  # Prediction result
│   ├── dashboard.html               # Personal dashboard (user)
│   ├── admin_base.html              # Standalone admin shell (own navbar, no public chrome)
│   ├── admin.html                   # Admin Console (5 navbar sections)
│   ├── admin_login.html             # Standalone admin console login
│   ├── predict.html / plans.html / model_comparison.html / api_docs.html
│   ├── account.html / tracking.html / pricing.html / about.html / blog.html / contact.html
│   ├── login.html / register.html / forgot-password.html
│   └── error.html                   # 404 / 500 / 429 pages
│
└── tests/
    └── test_delivery.py             # Integration tests
```

---

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- pip
- Git

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone https://github.com/suman2308/Delivery-time-prediction.git
cd Delivery-time-prediction

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Train the DTDC model on fresh data
python -m dtdc_model train

# 5. Run the application
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

---

## 🔐 Admin Panel

The admin panel (**Admin Console** — overview, demo mode, users & plans, analytics, and system/ML sections) lives at **`/admin`** and can be reached two ways:

### Option A — Standalone admin login (default, no account needed)

1. Open **`/admin-login`** (linked in the footer as **Admin login**).
2. Sign in with the default credentials:

   ```text
   email:    admin@gmail.com
   password: 00000000
   ```

   (Override via the `ADMIN_LOGIN_EMAIL` / `ADMIN_LOGIN_PASSWORD` env vars.)

3. You land on **`/admin`** — a **fully standalone, walled-off console**: its own top
   navbar (CourierAI brand, section links, and an admin profile avatar whose menu
   holds the demo-mode status + **Log out**) and no public site navbar or footer.
   An admin session is confined to the console — **every public page (`/`, `/predict`,
   `/about`, …) bounces back to `/admin`**, and user pages like `/dashboard` or
   `/account` redirect to the panel too.

   The **Admin Console** has five sections in its navbar, all computed from real data:

   - **Overview** — total/active users, predictions today, paid subscriptions,
     system status and quick alerts
   - **Demo Mode** — enable/disable sample data, demo row/user counts, and a
     reset that deletes only demo rows (real predictions untouched)
   - **Users & Plans** — every account with its plan, usage and prediction totals
   - **Analytics** — platform-wide KPIs, charts, recent predictions and activity
   - **System & ML** — model version/status, API & database health, live
     prediction latency (measured), Python/platform info, and the experiment lab

The login is rate-limited (10 attempts/min), CSRF-protected, and compared in constant
time. Non-admin accounts are redirected away from `/admin`.

> **Strict separation:** there is deliberately **no email-allowlist admin channel**.
> A registered account can never become an admin and an admin session never sees
> user pages — the two panels are completely disjoint.

> **Experiment Lab in containers:** the lab trains on `DTDC_Improved_Dataset.csv` from the project root. The CSV is git-ignored (24 MB), so Docker/Render deployments that don't mount it will report a `FileNotFoundError` status when an experiment is started — predictions and everything else keep working. Mount the CSV or run experiments from a local checkout.

---

## 🔧 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `5000` | Server port |
| `FLASK_DEBUG` | `0` | Enable Flask debug mode (`1` to enable) |
| `BOOTSTRAP_ON_START` | `1` | Run legacy bootstrap on startup |
| `BOOTSTRAP_SEED_COUNT` | `300` | Number of legacy demo orders/predictions seeded by the bootstrap |
| `DELIVERY_DB_PATH` | `delivery.db` | SQLite database path |
| `DELIVERY_MODEL_PATH` | `models/delivery_regressor.joblib` | *(Legacy)* Old model path |
| `FREE_PLAN_LIMIT` | `50` | Monthly prediction quota for the free plan |
| `DTDC_DATA_PATH` | `DTDC_Improved_Dataset.csv` | Path to the dataset used by `train_experiments.py` |
| `ADMIN_LOGIN_EMAIL` | `admin@gmail.com` | Email for the standalone `/admin-login` console (**change in production**) |
| `ADMIN_LOGIN_PASSWORD` | `00000000` | Password for the standalone `/admin-login` console (**change in production**) |
| `SECRET_KEY` | *(auto-generated, persisted in `instance/secret_key`)* | Session-signing secret; a random per-install key is generated on first boot when unset — a known default is **never** used. Set it explicitly in production (Render can auto-generate via `generateValue`) |
| `KEY_ENCRYPTION_KEY` | *(derived from `SECRET_KEY`)* | Optional separate key for API-key encryption (Fernet) |
| `COOKIE_SECURE` | `0` | Set to `1` behind HTTPS so session cookies are only sent over TLS |
| `TRUST_PROXY` | `0` | Set to `1` behind a trusted reverse proxy (Render/nginx) so rate limiting keys per real client IP |
| `RATE_LIMIT_STORAGE_URI` | `memory://` | Rate-limit backing store — use Redis (`redis://…`) when running multiple workers |
| `REDIS_URL` | *(none)* | Fallback used as the rate-limit store when `RATE_LIMIT_STORAGE_URI` is unset |
| `API_RATE_LIMIT` | `30 per minute` | Prediction API key rate limit |
| `API_RATE_LIMIT_DAILY` | `1000 per day` | Prediction API daily cap |
| `LOGIN_RATE_LIMIT` | `10 per minute` | Login attempts per IP |
| `REGISTER_RATE_LIMIT` | `5 per minute` | Registrations per IP |
| `REGENERATE_KEY_RATE_LIMIT` | `3 per hour` | API-key regenerations per account |

> The DTDC model path is managed internally by `dtdc_model.py` and is not configurable via environment variables.

---

## 🖥️ Running Locally

```bash
# Start the Flask development server
python app.py

# Or with Gunicorn (recommended for production)
gunicorn app:app --bind 0.0.0.0:5000 --workers 2 --timeout 120
```

### CLI Usage

```bash
# Train the DTDC model
python -m dtdc_model train

# Interactive prediction via CLI
python -m dtdc_model predict

# Batch prediction via JSON lines
echo '{"origin":"Mumbai","destination":"Pune","booking_weekday":"Monday","mode":"Surface","nature_of_consignment":"Dox","total_pieces":1,"actual_weight":0.5,"volumetric_weight":0.8,"chargeable_weight":0.5}' | python -m dtdc_model predict
```

### Running Tests

```bash
pytest
```

---

## ☁️ Deployment on Render

1. Push the repository to GitHub:

```bash
git add .
git commit -m "Ready for deployment"
git push origin main
```

2. In [Render Dashboard](https://dashboard.render.com), click **New +** → **Blueprint**.

3. Select this repository.

4. Render automatically detects `render.yaml` and deploys with:
   - Build: `pip install -r requirements.txt`
   - Start: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`

5. Your app will be live at `https://smart-delivery-prediction.onrender.com`.

> ⚠️ **Important**: The DTDC model artifact (`models/dtdc_hgb_v1_0_0.joblib`) is included in the git repository and will be available after deployment. No additional setup is required.

---

## 📡 API Documentation

### `POST /api/predict`

Predict delivery duration in days. **Authentication required** — send your API key in the
`X-API-Key` header (or `Authorization: Bearer <key>`).

**Getting a key:** register an account — the key is shown once in the confirmation toast,
and can be regenerated anytime from the **Account** page (`/account`), which also offers
**Show / Hide / Copy** buttons for it. Keys are stored hashed (SHA-256) for API auth plus an
**encrypted** copy (Fernet, derived from `SECRET_KEY` or `KEY_ENCRYPTION_KEY`) so the owner
can reveal/copy it on the account page — a database leak alone never exposes keys.
Keep `SECRET_KEY`/`KEY_ENCRYPTION_KEY` stable across deploys, or stored keys become
undecryptable. Keys are revoked immediately on regeneration.

**Rate limits:** `30 requests/minute` and `1000 requests/day` per API key by default
(configurable via `API_RATE_LIMIT` / `API_RATE_LIMIT_DAILY`). Login (`10/min`) and
registration (`5/min`) are rate limited per IP. Exceeding a limit returns `429`.

**Plan quota:** every account starts on the free plan (50 predictions/month,
configurable via `FREE_PLAN_LIMIT`). The counter resets on the 1st of each month and every
successful prediction is charged against it. Exceeding the quota returns `402` with the
current plan/usage in the body; upgrading to Pro (₹299/month) or Pro+ (₹1,299/year — demo billing,
via `/account/upgrade`) unlocks unlimited predictions.

**Request:**

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: scp_live_YOUR_KEY" \
  -d '{
    "origin": "Mumbai",
    "destination": "Pune",
    "booking_weekday": "Monday",
    "mode": "Surface",
    "nature_of_consignment": "Dox",
    "total_pieces": 1,
    "actual_weight": 0.5,
    "volumetric_weight": 0.8,
    "chargeable_weight": 0.5
  }'
```

**Response:**

```json
{
  "predicted_time_days": 3.8444,
  "model_version": "1.0.0",
  "algorithm": "HistGradientBoostingRegressor",
  "tracking_id": "SCP-4F2A9B1C"
}
```

> `tracking_id` is the public lookup key for the shipment — paste it on the
> Tracking page (`/tracking`) to follow the prediction's journey timeline.

**Error Responses:**

```json
{
  "error": "origin is required."
}
```

```json
// 401 — missing/invalid API key
{
  "error": "A valid API key is required. Send it in the X-API-Key header."
}
```

```json
// 429 — rate limit exceeded
{
  "error": "Rate limit exceeded. Please slow down and try again later."
}
```

```json
// 402 — free-plan prediction quota exhausted
{
  "error": "You have used all 50 free predictions this month. Upgrade to Pro for unlimited predictions.",
  "plan": "basic",
  "limit": 50,
  "used": 50
}
```

### `GET /health`

```json
{ "status": "healthy" }
```

### `GET /metrics`

```json
{
  "mae_days": 0.5361,
  "rmse_days": 0.7263,
  "r2": 0.7466,
  "model_version": "1.0.0",
  "algorithm": "HistGradientBoostingRegressor",
  "training_date": "2026-07-30T04:29:48.595917+00:00"
}
```

---

## 🔮 Future Improvements

- [ ] **Geographic clustering** — Group origin/destination cities by region to improve generalization for unseen routes
- [ ] **Real-time features** — Integrate weather API and traffic data for live-adjusted predictions
- [ ] **SHAP explanations** — Add per-prediction feature importance for explainability
- [ ] **Automated retraining** — CI/CD pipeline that retrains on new data and hot-swaps model artifacts
- [ ] **Interactive charts** — Replace static Matplotlib PNGs with interactive Plotly/D3.js visualizations
- [x] **Docker support** — Containerize the application with Docker for consistent deployments (Dockerfile + docker-compose.yml included)
- [x] **CI/CD pipeline** — Automated testing with GitHub Actions on every push
- [x] **User authentication** — Account registration, session-based login/logout, and protected dashboard/admin routes
- [x] **API keys & rate limiting** — Hashed per-user API keys (required on `POST /api/predict`) with per-key rate limiting
- [x] **Plans & prediction quotas** — Free plan (50/month) with Pro upgrades (₹299/mo, ₹1,299/yr) for unlimited predictions
- [x] **Model experiment lab** — Admin-runnable base/hybrid/stacking benchmarks (regression + classification) feeding the Model Comparison page
- [x] **OWASP hardening** — CSRF tokens on every state-changing form, XSS-safe tracking, security headers (CSP, frame-deny, nosniff), hardened session cookies, and stale-session handling
- [x] **Editable profiles** — Display name, company and avatar uploads from the Account page
- [ ] **Expanded test coverage** — Add unit tests for `dtdc_model.py` and edge cases

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for more information.

---

<div align="center">
  <sub>Built with ❤️ using Python, Flask, scikit-learn & modern CSS</sub>
  <br>
  <sub>© 2026 Suman Jash</sub>
</div>
