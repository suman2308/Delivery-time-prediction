-- Smart Delivery Time Prediction — core schema (SQLite compatible)

CREATE TABLE IF NOT EXISTS orders (
    order_id INTEGER PRIMARY KEY AUTOINCREMENT,
    distance REAL NOT NULL,
    order_time INTEGER NOT NULL,
    traffic_level TEXT NOT NULL,
    weather TEXT NOT NULL,
    delivery_time REAL NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER,
    predicted_time REAL NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (order_id) REFERENCES orders(order_id) ON DELETE SET NULL
);

-- Index to speed up distance-based queries
CREATE INDEX IF NOT EXISTS idx_orders_distance ON orders(distance);
