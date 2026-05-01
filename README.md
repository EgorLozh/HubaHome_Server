# HubaHome Server

MVP server runtime for the `HubaHome_Server` repository.

## Stack

- FastAPI
- Python 3.12
- Pydantic Settings
- Prometheus metrics
- LangChain Tool Calling Agent
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
   - `python scripts/llm_smoke_check.py` (basic LLM path)
   - `python scripts/llm_smoke_check.py --full` (LLM + orchestrator + metadata flows)
   - `python scripts/tools_smoke_check.py` (basic metadata tool check)
- `python scripts/tools_smoke_check.py --full` (metadata + knowledge CRUD + weather flow)
4. Interactive terminal chat with agent (transcript simulation):
   - `python scripts/agent_terminal_chat.py`
   - `python scripts/agent_terminal_chat.py --scenario basic`
   - `python scripts/agent_terminal_chat.py --scenario tools`

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
- Agent runtime settings:
  - `AGENT_MAX_STEPS_PER_TURN` (default `4`)
  - `AGENT_MAX_TOOL_CALLS_PER_TURN` (default `3`)
  - `AGENT_TURN_TIMEOUT_MS` (default `8000`)
- Knowledge settings:
  - `KNOWLEDGE_COLLECTION_NAME` (default `knowledge_notes`)
  - `KNOWLEDGE_METADATA_PATH` (default `.data/knowledge/metadata.json`)

## MVP capabilities

- Weather retrieval via `weather_tool`.
- Knowledge document lifecycle via tool-calling flow:
  - create/update/delete/search documents.
- Persistent metadata instructions via tool-calling flow:
  - store and apply user preferences in prompt context (for example user name).

## Not supported in MVP

- Reminder and scheduler workflows (return controlled refusal).
- Production notification transport (will be implemented after MVP).

## Local run without Docker

1. Install dependencies:
   - `python -m pip install -e ".[dev]"`
2. Start server:
   - `python -m uvicorn src.main:app --host 0.0.0.0 --port 8000`
3. Run tests:
   - `python -m pytest -q`
