-- Smart Courier Prediction — core schema (SQLite compatible)

CREATE TABLE IF NOT EXISTS orders (
    order_id INTEGER PRIMARY KEY AUTOINCREMENT,
    distance REAL NOT NULL,
    order_time INTEGER NOT NULL,
    traffic_level TEXT NOT NULL,
    weather TEXT NOT NULL,
    delivery_time REAL NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Index to speed up distance-based queries
CREATE INDEX IF NOT EXISTS idx_orders_distance ON orders(distance);

-- DTDC prediction audit log
-- User accounts (authentication)
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT NOT NULL,
    company TEXT NOT NULL DEFAULT '',
    email TEXT NOT NULL COLLATE NOCASE UNIQUE,
    password_hash TEXT NOT NULL,
    api_key_hash TEXT UNIQUE,
    api_key_hint TEXT,
    api_key_enc TEXT,
    avatar TEXT,
    -- Subscription plan ('basic' | 'pro_monthly' | 'pro_yearly') + usage quota
    plan TEXT NOT NULL DEFAULT 'basic',
    plan_expires_at TIMESTAMP,
    usage_month TEXT,
    predictions_used INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);


CREATE TABLE IF NOT EXISTS dtdc_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tracking_id TEXT,
    user_id INTEGER,
    origin TEXT NOT NULL,
    destination TEXT NOT NULL,
    booking_weekday TEXT NOT NULL,
    mode TEXT NOT NULL,
    nature_of_consignment TEXT NOT NULL,
    total_pieces INTEGER NOT NULL,
    actual_weight REAL NOT NULL,
    volumetric_weight REAL NOT NULL,
    chargeable_weight REAL NOT NULL,
    predicted_days REAL NOT NULL,
    model_version TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
