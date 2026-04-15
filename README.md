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
- Phase 2 speech toggles:
  - `ENABLE_REAL_STT=true` with local `faster-whisper`.
  - Whisper defaults to model `small` and auto-download on startup.
  - Optional overrides: `WHISPER_MODEL_NAME`, `WHISPER_MODELS_DIR`, `WHISPER_AUTO_DOWNLOAD`, `WHISPER_DEVICE`, `WHISPER_COMPUTE_TYPE`.
  - `ENABLE_REAL_TTS=true` with Piper Python API (`piper-tts` package).
  - Model defaults to `ru_RU-irina-medium` and is auto-downloaded on server startup.
  - Optional overrides: `PIPER_MODEL_NAME`, `PIPER_MODELS_DIR`, `PIPER_AUTO_DOWNLOAD`, `PIPER_USE_CUDA`.
  - `ENABLE_REAL_WEB_SEARCH=true` to use DuckDuckGo web search adapter.

## Local run without Docker

1. Install dependencies:
   - `python -m pip install -e ".[dev]"`
2. Start server:
   - `python -m uvicorn src.main:app --host 0.0.0.0 --port 8000`
3. Run tests:
   - `python -m pytest -q`
