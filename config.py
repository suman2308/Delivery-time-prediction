import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.environ.get("DELIVERY_DB_PATH", os.path.join(BASE_DIR, "delivery.db"))
MODEL_PATH = os.environ.get("DELIVERY_MODEL_PATH", os.path.join(BASE_DIR, "models", "delivery_regressor.joblib"))
PLOTS_DIR = os.path.join(BASE_DIR, "static", "plots")
