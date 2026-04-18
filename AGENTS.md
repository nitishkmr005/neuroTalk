# Agent Instructions

Conventions for AI coding agents (OpenAI Codex, Cursor, etc.) working in this repository.

---

## Repo Layout

```
backend/    Python (FastAPI, uv, pydantic-settings, loguru)
frontend/   Next.js (TypeScript, Tailwind/CSS modules)
scripts/    Standalone learnable Python demos — one file per module
docs/       blog.md (project narrative) + any architecture docs
Makefile    setup / dev / check targets
```

## How to Run

```bash
make setup    # install all deps
make dev      # start backend (port 8000) + frontend (port 3000)
make check    # lint + typecheck before committing
```

Backend requires Python 3.11+. Use `uv` — never `pip` or `poetry`.
Frontend requires Node 18+. Use `npm`.

## Backend Structure

```
backend/app/
  main.py        FastAPI app + routes/websockets
  models.py      Pydantic request/response schemas
  services/      One service per external integration (stt.py, llm.py, …)
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

## Code Rules

1. **Type-annotate everything** — all function signatures, return types
2. **Async for I/O** — use `async def` for DB, HTTP, WebSocket, file I/O
3. **Loguru, not print** — `from loguru import logger; logger.info(...)`
4. **Structured log lines** — `logger.info("event=llm_done latency_ms={}", ms)`
5. **Specific exceptions** — never `except:` alone
6. **No magic numbers** — put constants in `config/settings.py` or named constants
7. **One responsibility per file** — services do one thing, agents orchestrate

## Testing a Change

After any backend change:
```bash
cd backend && uv run python -m compileall app config
```

After any frontend change:
```bash
npm --prefix frontend run build
```

Full check:
```bash
make check
```

## What NOT to Do

- Do not add `print()` — use `logger`
- Do not hardcode URLs, API keys, or model names — use settings
- Do not put business logic in `main.py` — delegate to services/agents
- Do not create new top-level Python files outside the defined folder structure
- Do not modify `uv.lock` manually — let uv manage it
- Do not commit `.env` — only `.env.example`
- Do not add dependencies without updating `pyproject.toml` via `uv add`

## Scripts Folder

`scripts/*.py` are standalone educational demos. Each:
- Runs with `uv run --project backend python scripts/<name>.py`
- Is self-contained (no imports from other scripts)
- Has a docstring explaining what it teaches
- Prints timing output

Do not refactor scripts into shared modules — their value is standalone clarity.
