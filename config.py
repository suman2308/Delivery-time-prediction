"""Configuration settings for Smart Delivery platform."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.environ.get("DELIVERY_DB_PATH", os.path.join(BASE_DIR, "delivery.db"))
# Legacy path — kept for _bootstrap_if_needed which trains the old synthetic model.
# The DTDC model manages its own path internally in dtdc_model.py.
MODEL_PATH = os.environ.get("DELIVERY_MODEL_PATH", os.path.join(BASE_DIR, "models", "delivery_regressor.joblib"))
PLOTS_DIR = os.path.join(BASE_DIR, "static", "plots")
