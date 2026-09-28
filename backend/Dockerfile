FROM python:3.11-slim

WORKDIR /app

# System deps for psycopg (libpq) — kept minimal on purpose for Phase 1;
# Phase 7 will add build tools here for torch/laya if the slim image needs them.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install the CPU-only PyTorch wheel before Laya so pip does not resolve the
# CUDA-enabled PyPI build in this CPU-only API container.
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY tests ./tests
COPY pytest.ini .
COPY alembic.ini .
COPY alembic ./alembic

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
