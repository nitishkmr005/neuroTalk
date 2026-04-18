# Claude Code — Project Conventions

Conventions for the NeuroTalk voice agent. Loaded on every Claude session — keep concise and precise.

---

## Project Layout

```
neuroTalk/
├── backend/          FastAPI + uv (Python 3.11+)
├── frontend/         Next.js 15 (TypeScript)
├── scripts/          Standalone learnable .py demos
├── docs/
│   ├── blog.md       Project narrative
│   └── superpowers/  AI-generated specs and plans
├── README.md
├── Makefile
├── ___testing_CLAUDE.md   This file (Claude conventions)
└── AGENTS.md         Agent conventions
```

---

## Backend Conventions

**Package manager:** uv only. Never `pip` or `poetry`.

**Python:** 3.11+. Use modern syntax: `X | Y` unions, `match`, `list[str]`.

**Folder roles inside `backend/app/`:**

| Folder | Purpose |
|--------|---------|
| `services/` | External integrations (STT, LLM, TTS). One class or function per file. |
| `agents/` | Orchestration — chains services together. |
| `prompts/` | System prompt constants. Never inline prompts in business logic. |
| `tools/` | LLM tool/function definitions. |
| `modules/` | Reusable domain modules. |
| `utils/` | Pure utility functions, no side effects. |
| `models.py` | Pydantic request/response schemas. |
| `main.py` | FastAPI app, routes, WebSocket handlers only. |

**Config pattern:** `pydantic-settings` + `.env` via `config/settings.py`. Never use bare `os.getenv()`.

**Never commit `.env`.** Always maintain `.env.example`.

---

## TTS Backend Switching

Four TTS models are available as uv dependency groups: `chatterbox` (default), `qwen`, `vibevoice`, `omnivoice`.

- Install a specific model: `make backend-install TTS_BACKEND=qwen`
- Default is `chatterbox` — set in Makefile and `.env`
- Only one model group can be installed at a time (uv conflicts enforced)
- `settings.tts_backend` reflects which model is active

To switch: change `TTS_BACKEND` in `.env`, re-run `make backend-install TTS_BACKEND=<value>`, update `backend/app/services/tts.py` to import the new model class.

---

## Logging

Two sinks — distinct purposes, both always set up at startup.

**Terminal sink** (runtime/human): colorized Loguru output to stdout. Used for live debugging, startup messages, and request tracing during development. Not persisted.

```python
logger.add(sys.stdout, colorize=True, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan> | <level>{message}</level>")
```

**JSON file sink** (`backend/logs/`): structured output for performance analysis, debugging, and LLM tracing. These files are the persistent record — used to analyze latency, trace LLM calls, and debug production issues.

```python
logger.add("logs/app_{time}.json", serialize=True, rotation="10 MB", retention=5)
```

Rules:
- `backend/logs/` — gitignored, `.gitkeep` commits the folder
- Keep only latest 5 files (`retention=5`)
- Never use `print()` — always `logger.info / debug / warning / error`
- Log structured key=value pairs: `logger.info("event=llm_done latency_ms={}", ms)`
- LLM calls must log: model, token count, latency, prompt hash

---

## Frontend Conventions

**Stack:** Next.js 15 · TypeScript · CSS custom properties (no Tailwind).

**Theme system:**
- CSS custom property tokens defined in `:root` (light) and `[data-theme="dark"]` blocks in `globals.css`
- `data-theme` attribute set on `<html>` element
- localStorage key: `nt-theme` — values: `"dark"` | `"light"`
- **Dark is the default theme** — applied when no saved preference exists
- Toggle implemented in `VoiceAgentConsole` via `isDark` state + `toggleTheme()`
- Theme initializes synchronously via lazy `useState` initializer to prevent FOUC

---

## Deployment

| Layer | Platform | Notes |
|-------|----------|-------|
| Frontend | Vercel | Auto-deploys from `main`. Set `NEXT_PUBLIC_BACKEND_URL` env var to Railway URL. |
| Backend | Railway | Dockerfile or Nixpacks. Set all `.env` vars as Railway env vars. Expose port 8000. |

**Vercel setup:** Connect GitHub repo → select `frontend/` as root directory → add env var `NEXT_PUBLIC_BACKEND_URL=https://<railway-app>.up.railway.app`.

**Railway setup:** Connect GitHub repo → select `backend/` as root → set all env vars from `.env.example` → service starts with `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.

---

## Authentication (Supabase)

Supabase handles user authentication.

- Client: `@supabase/supabase-js` in frontend
- Env vars: `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY` (frontend); `SUPABASE_SERVICE_ROLE_KEY` (backend only — never expose to client)
- Auth flow: email/password or OAuth via Supabase Auth UI or custom forms
- Backend route protection: verify JWT from `Authorization: Bearer <token>` header using Supabase admin client
- Never store session tokens in localStorage — use Supabase's built-in session management

---

## Scripts Folder

`scripts/` — standalone, runnable `.py` files. One per module. Teach how each component works in isolation.

Rules:
- Runs with `uv run --project backend python scripts/<name>.py`
- No shared imports between scripts — self-contained
- Docstring at top: what it teaches and how to run it
- `if __name__ == "__main__":` block with working demo
- Prints timing output

`scripts/tts_projects/` — isolated uv environments per TTS model for benchmarking. Run via `make tts-report`.

---

## Makefile Targets

```makefile
make setup                              # install all deps
make dev                                # backend + frontend (hot-reload)
make backend                            # backend only
make frontend                           # frontend only
make check                              # lint + type check
make backend-install TTS_BACKEND=qwen   # install specific TTS model group
make tts-envs                           # install all 4 TTS venvs
make tts-report                         # benchmark all TTS models → scripts/speech/
```

---

## Code Style

- Type-annotate all function signatures
- `async def` for all I/O-bound functions
- No bare `except:` — always catch specific exceptions
- Use `loguru` not `logging`
- Import order: stdlib → third-party → local (enforced by ruff)
- No TODO comments in committed code

---

## .gitignore Essentials

Ignore: `.env`, `.venv`, `__pycache__`, `*.pyc`, `.cache`, `logs/`, `node_modules/`, `.next/`, `.DS_Store`

Track: `.env.example`, `logs/.gitkeep`
