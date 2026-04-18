# Claude Code — Project Conventions

Generic conventions for AI agent projects. Copy this file to any new project and adjust names/values.

---

## Project Layout

```
project/
├── backend/          Python backend (FastAPI + uv)
├── frontend/         Next.js frontend (TypeScript)
├── scripts/          Standalone learnable .py demos
├── docs/
│   └── blog.md       Narrative explanation of the project
├── README.md         Minimal setup guide
├── Makefile          dev / setup / check targets
├── CLAUDE.md         This file
├── AGENTS.md         Codex / other agent conventions
└── .gitignore
```

---

## Backend Conventions

**Package manager:** uv (`uv add <pkg>`, `uv run python ...`)

**Python version:** 3.11+. Use modern syntax: `X | Y` unions, `match`, `list[str]`.

**Folder roles inside `backend/app/`:**

| Folder | Purpose |
|--------|---------|
| `services/` | External integrations (STT, LLM, TTS, DB). One class or function per file. |
| `agents/` | Orchestration logic — chains services together. |
| `prompts/` | System prompts as Python string constants. Never inline prompts in business logic. |
| `tools/` | Tool definitions for LLM function-calling. |
| `modules/` | Reusable domain modules (audio processing, text chunking, etc.). |
| `utils/` | Pure utility functions with no side effects. |
| `models.py` | Pydantic models for API request/response shapes. |
| `main.py` | FastAPI app, routes, and WebSocket handlers only. |

**Config pattern:** Always use `pydantic-settings` + `.env`. Never use bare `os.getenv()` in application code.

```python
# config/settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    app_name: str = "My Agent"
    some_api_key: str = ""

from functools import lru_cache

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
```

**Never commit `.env`.** Always maintain `.env.example` with safe placeholder values.

---

## Logging

Two sinks — always set both up at startup:

**Terminal (human):**
```python
logger.add(sys.stdout, colorize=True, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan> | <level>{message}</level>")
```

**JSON files (machine):**
```python
logger.add("logs/app_{time}.json", serialize=True, rotation="10 MB", retention=5)
```

Rules:
- Log files go in `backend/logs/` — gitignored, `.gitkeep` commits the folder
- Keep only the latest 5 files (`retention=5`)
- Never use `print()` — always `logger.info / debug / warning / error`
- Log structured key=value pairs: `logger.info("event=llm_done latency_ms={}", ms)`

---

## Scripts Folder

`scripts/` contains standalone, runnable, heavily-commented `.py` files — one per module. They exist to teach how each component works in isolation.

Rules:
- Each script runs with `uv run --project backend python scripts/<name>.py`
- No shared imports between scripts — self-contained
- Has a `"""docstring"""` at the top explaining what it teaches and how to run it
- Has a `if __name__ == "__main__":` block with a working demo
- Prints timing information so the reader can see latency

---

## Docs

`docs/blog.md` — narrative blog post explaining the project. Covers:
- The problem being solved
- ASCII architecture diagram
- Key technical choices and why
- Latency breakdown table
- What was learned
- What's next

`README.md` at root — minimal. Covers: what it is, stack table, quick start, env vars, project structure, makefile commands.

---

## Makefile Targets

Standard targets for every project:

```makefile
make setup    # install all deps (uv sync + npm install)
make dev      # start backend + frontend with hot-reload
make backend  # backend only
make frontend # frontend only
make check    # lint + type check
```

---

## Code Style

- Type-annotate all function signatures
- Prefer `async def` for I/O-bound functions
- No bare `except:` — always catch specific exceptions
- Use `loguru` not `logging`
- Import order: stdlib → third-party → local (enforced by ruff)
- No TODO comments in committed code — create a GitHub issue instead

---

## .gitignore Essentials

Always ignore: `.env`, `.venv`, `__pycache__`, `*.pyc`, `.cache`, `logs/`, `node_modules/`, `.next/`, `.DS_Store`

Always track: `.env.example`, `logs/.gitkeep`
