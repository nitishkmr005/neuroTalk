UV_BACKEND    = uv --directory backend
NPM_FRONTEND  = npm --prefix frontend
BACKEND_PORT  = 8000
FRONTEND_PORT = 3000
LLM_MODEL     = gemma4:latest

.PHONY: setup backend-install frontend-install backend frontend dev run \
        check free-ports free-backend-port free-frontend-port \
        ollama ollama-pull

# ── Install ───────────────────────────────────────────────────────────────────

setup: backend-install frontend-install

backend-install:
	$(UV_BACKEND) sync

frontend-install:
	$(NPM_FRONTEND) install

# ── Servers ───────────────────────────────────────────────────────────────────

backend: backend-install
	$(UV_BACKEND) run uvicorn app.main:app --reload --host 0.0.0.0 --port $(BACKEND_PORT)

frontend: frontend-install
	$(NPM_FRONTEND) run dev

# ── Ollama ────────────────────────────────────────────────────────────────────

ollama:
	@if pgrep -x ollama > /dev/null; then \
		echo "Ollama already running"; \
	else \
		echo "Starting Ollama..."; \
		ollama serve & \
		sleep 2; \
	fi

ollama-pull: ollama
	@if ollama list | grep -q "$(LLM_MODEL)"; then \
		echo "Model $(LLM_MODEL) already downloaded"; \
	else \
		echo "Pulling $(LLM_MODEL)..."; \
		ollama pull $(LLM_MODEL); \
	fi

# ── Port helpers ──────────────────────────────────────────────────────────────

free-backend-port:
	@PIDS=$$(lsof -ti tcp:$(BACKEND_PORT)); \
	if [ -n "$$PIDS" ]; then \
		echo "Stopping process(es) on port $(BACKEND_PORT): $$PIDS"; \
		kill $$PIDS; \
	else \
		echo "Port $(BACKEND_PORT) is already free"; \
	fi

free-frontend-port:
	@PIDS=$$(lsof -ti tcp:$(FRONTEND_PORT)); \
	if [ -n "$$PIDS" ]; then \
		echo "Stopping process(es) on port $(FRONTEND_PORT): $$PIDS"; \
		kill $$PIDS; \
	else \
		echo "Port $(FRONTEND_PORT) is already free"; \
	fi

free-ports: free-backend-port free-frontend-port

# ── Dev (all-in-one) ──────────────────────────────────────────────────────────

dev: free-ports ollama-pull
	@trap 'kill 0' EXIT; $(MAKE) backend & $(MAKE) frontend & wait

run: dev

# ── Quality ───────────────────────────────────────────────────────────────────

check:
	$(UV_BACKEND) run python -m compileall app config
	$(NPM_FRONTEND) run lint
	$(NPM_FRONTEND) run typecheck
