# =============================================================================
# Stage 1 — Build dependencies (wheels, compilation)
# =============================================================================
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

# Copy only dependency metadata first for optimal layer caching
COPY requirements.txt .

# Build / fetch binary wheels for all dependencies
RUN pip install --no-cache-dir --user --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --user -r requirements.txt


# =============================================================================
# Stage 2 — Runtime image (tiny, production-only)
# =============================================================================
FROM python:3.11-slim-bookworm AS runtime

# Never buffer stdout / stderr — important for container logs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FLASK_DEBUG=0

# Create a non-root user for safety
RUN groupadd --system --gid 1000 app && \
    useradd --system --gid 1000 --uid 1000 --no-create-home --shell /bin/false app

# Copy installed packages from the builder stage
COPY --from=builder /root/.local /usr/local

WORKDIR /app

# Copy application code — order matters for layer caching:
#  1. Stable paths (models, schema, core modules)
#  2. Volatile paths (templates, static assets)
COPY models/            ./models/
COPY schema.sql         ./schema.sql
COPY config.py          ./
COPY database.py        ./
COPY dtdc_model.py      ./
COPY charts.py          ./
COPY ml_model.py        ./
COPY seed_data.py       ./
COPY train_experiments.py ./
COPY app.py             ./

COPY data/              ./data/
COPY templates/         ./templates/
COPY static/            ./static/

# Ensure runtime directories exist
RUN mkdir -p /app/static/plots && \
    chown -R app:app /app

USER app

EXPOSE 5000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120"]
