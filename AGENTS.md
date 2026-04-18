# Agent Instructions

Conventions for AI coding agents working in this repository.

---

## Repo Layout

```
backend/    FastAPI + uv (Python 3.11+)
frontend/   Next.js 15 (TypeScript, CSS custom properties)
scripts/    Standalone learnable Python demos — one file per module
docs/       blog.md + superpowers/ (specs, plans)
Makefile    setup / dev / check targets
```

## How to Run

```bash
make setup    # install all deps (default TTS: chatterbox)
make dev      # backend (port 8000) + frontend (port 3000)
make check    # lint + typecheck before committing
```

Backend requires Python 3.11+. Use `uv` — never `pip` or `poetry`.
Frontend requires Node 18+. Use `npm`.

## Backend Structure

```
backend/app/
  main.py        FastAPI app + routes/websockets
  models.py      Pydantic request/response schemas
  services/      One service per integration (stt.py, llm.py, tts.py)
  agents/        Orchestration — chains services
  prompts/       System prompt constants (never inline prompts)
  tools/         LLM tool/function definitions
  modules/       Reusable domain logic
  utils/         Pure utility functions
config/
  settings.py    Pydantic Settings loaded from .env
  logging.py     Loguru setup (terminal + JSON file sink)
logs/            JSON log files — gitignored, keep latest 5
```

## Config Rules

- All config via `config/settings.py` (pydantic-settings + `.env`)
- Never use `os.getenv()` in application code
- `.env` is gitignored — copy `.env.example` and fill in values
- Secrets stay in `.env`, never in code

## TTS Backend Switching

Four TTS models available as uv dependency groups: `chatterbox` (default), `qwen`, `vibevoice`, `omnivoice`.

```bash
make backend-install TTS_BACKEND=qwen   # switch to qwen
```

Only one group installs at a time. Change `TTS_BACKEND` in `.env`, re-run install, update `services/tts.py` to import the new model class.

## Logging

Two sinks with distinct purposes:

**Terminal** — runtime/human. Colorized stdout. Used for live debugging and development. Not persisted.

**`backend/logs/` JSON files** — persistent record. Used for performance analysis, debugging, and LLM call tracing. Never delete these manually — `retention=5` auto-rotates. Log entries should include `event=`, `latency_ms=`, model info, and request IDs for traceability.

Rules:
- Never use `print()` — always `logger.info / debug / warning / error`
- Structured key=value: `logger.info("event=llm_done latency_ms={}", ms)`
- LLM calls must log: model, token count, latency

## Frontend Theme System

- CSS tokens in `:root` (light) and `[data-theme="dark"]` in `globals.css`
- `data-theme` set on `<html>` element
- localStorage key: `nt-theme` — values `"dark"` | `"light"`
- **Dark is the default** — applied when no preference is saved
- Lazy `useState` initializer reads localStorage synchronously to prevent FOUC

## Deployment

| Layer | Platform |
|-------|----------|
| Frontend | Vercel — root `frontend/`, set `NEXT_PUBLIC_BACKEND_URL` to Railway URL |
| Backend | Railway — root `backend/`, port 8000, set all `.env.example` vars |

## Authentication

Supabase handles user auth.
- Frontend: `@supabase/supabase-js`, env vars `NEXT_PUBLIC_SUPABASE_URL` + `NEXT_PUBLIC_SUPABASE_ANON_KEY`
- Backend: verify JWT via `Authorization: Bearer <token>` using Supabase admin client with `SUPABASE_SERVICE_ROLE_KEY`
- Never expose `SUPABASE_SERVICE_ROLE_KEY` to frontend

## Code Rules

1. **Type-annotate everything** — all signatures and return types
2. **Async for I/O** — `async def` for DB, HTTP, WebSocket, file I/O
3. **Loguru, not print** — `from loguru import logger`
4. **Structured logs** — `logger.info("event=x key=val", ...)`
5. **Specific exceptions** — never bare `except:`
6. **No magic numbers** — constants in `config/settings.py` or named module constants
7. **One responsibility per file** — services do one thing, agents orchestrate

## Testing a Change

```bash
# Backend
cd backend && uv run python -m compileall app config

# Frontend
npx --prefix frontend tsc --noEmit

# Full check
make check
```

## What NOT to Do

- Do not `print()` — use `logger`
- Do not hardcode URLs, keys, model names — use settings
- Do not put business logic in `main.py`
- Do not create top-level Python files outside the defined structure
- Do not edit `uv.lock` manually
- Do not commit `.env`
- Do not add dependencies without `uv add` + `pyproject.toml`

## Scripts Folder

`scripts/*.py` — standalone educational demos. Each:
- Runs with `uv run --project backend python scripts/<name>.py`
- Self-contained (no imports from other scripts)
- Docstring explaining what it teaches
- Prints timing output

`scripts/tts_projects/` — isolated uv envs per TTS model for benchmarking.

Do not refactor scripts into shared modules — standalone clarity is their value.
