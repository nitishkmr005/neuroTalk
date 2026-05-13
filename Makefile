UV_BACKEND    = uv --directory backend
NPM_FRONTEND  = npm --prefix frontend
BACKEND_PORT  = 8000
FRONTEND_PORT = 3000
LLM_MODEL     = llama3.2:3b #gemma3:1b
TTS_BACKEND  ?= kokoro

.PHONY: setup backend-install frontend-install backend frontend dev run \
        check free-ports free-backend-port free-frontend-port \
        ollama ollama-pull tts-envs tts-report install-llama-cpp \
        meeting-models meeting-llm-ollama meeting-llm-gguf

# ── Install ───────────────────────────────────────────────────────────────────

setup: backend-install frontend-install

backend-install:
	CMAKE_ARGS="-DGGML_METAL=on" $(UV_BACKEND) sync --group $(TTS_BACKEND)_model --group deepfilter --group llama_cpp_llm

install-llama-cpp:
	@if $(UV_BACKEND) run python -c "import llama_cpp" 2>/dev/null; then \
		echo "llama-cpp-python already installed, skipping"; \
	else \
		echo "Installing llama-cpp-python with Metal support..."; \
		CMAKE_ARGS="-DGGML_METAL=on" uv --directory backend pip install "llama-cpp-python>=0.3.0" --no-cache; \
	fi

frontend-install:
	$(NPM_FRONTEND) install

tts-envs:
	uv sync --project scripts/tts_projects/chatterbox
	uv sync --project scripts/tts_projects/qwen
	uv sync --project scripts/tts_projects/vibevoice
	uv sync --project scripts/tts_projects/kokoro

tts-report: tts-envs
	python3 scripts/tts.py

meeting-models:  ## Download Whisper large-v3-turbo to backend/models/meeting_stt/
	$(UV_BACKEND) run python -c "\
from faster_whisper import WhisperModel; \
print('Downloading Whisper large-v3-turbo...'); \
WhisperModel('large-v3-turbo', device='cpu', compute_type='int8', download_root='models/meeting_stt'); \
print('Done -> models/meeting_stt/')"

meeting-llm-gguf:  ## Download Qwen3-8B Q4_K_M GGUF to backend/models/meeting_llm/
	@if [ -f backend/models/meeting_llm/qwen3-8b-q4_k_m.gguf ]; then \
		echo "qwen3-8b-q4_k_m.gguf already downloaded"; \
	else \
		echo "Downloading Qwen3-8B Q4_K_M GGUF (~5 GB) ..."; \
		$(UV_BACKEND) run python -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download(\
    repo_id='Qwen/Qwen3-8B-GGUF', \
    filename='qwen3-8b-q4_k_m.gguf', \
    local_dir='models/meeting_llm', \
); \
print('Done -> backend/models/meeting_llm/qwen3-8b-q4_k_m.gguf')"; \
	fi

meeting-llm-ollama: ollama  ## Pull Ollama model for meeting summarization
	@if ollama list | grep -q "qwen3:8b"; then \
		echo "qwen3:8b already downloaded"; \
	else \
		echo "Pulling qwen3:8b for meeting summarization..."; \
		ollama pull qwen3:8b; \
	fi

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

dev: free-ports ollama-pull meeting-llm-ollama
	@trap 'kill 0' EXIT; $(MAKE) backend & $(MAKE) frontend & wait

run: dev

# ── Quality ───────────────────────────────────────────────────────────────────

check:
	$(UV_BACKEND) run python -m compileall app config
	$(NPM_FRONTEND) run lint
	$(NPM_FRONTEND) run typecheck
