UV_BACKEND = uv --directory backend
NPM_FRONTEND = npm --prefix frontend
BACKEND_PORT = 8000
FRONTEND_PORT = 3000

.PHONY: setup backend-install frontend-install backend frontend dev check free-ports free-backend-port free-frontend-port

setup: backend-install frontend-install

backend-install:
	$(UV_BACKEND) sync

frontend-install:
	$(NPM_FRONTEND) install

backend: backend-install
	$(UV_BACKEND) run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

frontend: frontend-install
	$(NPM_FRONTEND) run dev

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

dev: free-ports
	@trap 'kill 0' EXIT; $(MAKE) backend & $(MAKE) frontend & wait

check:
	$(UV_BACKEND) run python -m compileall app config
	$(NPM_FRONTEND) run lint
	$(NPM_FRONTEND) run typecheck
