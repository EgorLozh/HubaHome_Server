# HubaHome Server

Phase 1 skeleton for the `HubaHome_Server` repository.

## Stack

- FastAPI
- Python 3.12
- Pydantic Settings
- Prometheus metrics
- Docker Compose (`server + qdrant`)

## Quick start

1. Start services:
   - `docker compose up --build`
2. Open:
   - `http://localhost:8000/health`
   - `http://localhost:8000/config`
   - `http://localhost:8000/diagnostics`
   - `http://localhost:8000/metrics`
3. Run smoke checks:
   - `python scripts/smoke_check.py`

## Configuration

- You can optionally create `.env` from `.env.example`.
- `docker-compose.yml` already has safe defaults, so `.env` is not mandatory for local run.

## Local run without Docker

1. Install dependencies:
   - `python -m pip install -e ".[dev]"`
2. Start server:
   - `python -m uvicorn src.main:app --host 0.0.0.0 --port 8000`
3. Run tests:
   - `python -m pytest -q`
