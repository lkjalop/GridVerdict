FROM python:3.11-slim

WORKDIR /app

# libgomp1 is required by LightGBM for OpenMP threading on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev libgomp1 && rm -rf /var/lib/apt/lists/*

COPY . .
# Install CPU-only torch first to avoid pulling the 750 MB CUDA wheel;
# subsequent pip install sees torch already satisfied and skips it.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -e ".[dev,ml]"

CMD ["sh", "-c", "alembic upgrade head && uvicorn app.api.main:app --host 0.0.0.0 --port 8000"]
