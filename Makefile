# NeuroTalk Makefile
# Automates setup, environment creation, and common tasks

.PHONY: help install dev-install clean test lint format run setup

# Default target
help:
	@echo "NeuroTalk - Voice Activity Detection & Speech-to-Text"
	@echo ""
	@echo "Available commands:"
	@echo "  setup          - Complete project setup (install + env)"
	@echo "  install        - Install dependencies using uv"
	@echo "  dev-install    - Install with development dependencies"
	@echo "  run            - Run the Streamlit application"
	@echo "  test           - Run tests"
	@echo "  lint           - Run linting checks"
	@echo "  format         - Format code with black and isort"
	@echo "  clean          - Clean up temporary files"
	@echo ""

# Complete project setup
setup: install
	@echo "Setting up environment file..."
	@if [ ! -f .env ]; then cp env.example .env; echo "Created .env file from template"; fi
	@echo "Setup complete! Run 'make run' to start the application."

# Install dependencies
install:
	@echo "Installing dependencies with uv..."
	uv sync
	@echo "Dependencies installed successfully!"

# Install with development dependencies
dev-install:
	@echo "Installing with development dependencies..."
	uv sync --extra dev
	@echo "Development dependencies installed successfully!"

# Run the main application
run:
	@echo "Starting NeuroTalk Real-time Voice Processing..."
	@echo "Cleaning up any existing processes..."
	@-pkill -f "streamlit.*app" 2>/dev/null || true
	@-lsof -ti:8501 | xargs kill -9 2>/dev/null || true
	@-lsof -ti:8765 | xargs kill -9 2>/dev/null || true
	@sleep 1
	@echo "Starting application on http://localhost:8501..."
	uv run streamlit run app.py --server.port 8501 --server.address localhost

# Run tests
test:
	@echo "Running tests..."
	uv run pytest tests/ -v

# Run interactive microphone test
test-mic:
	@echo "Running interactive microphone test..."
	@echo "🎤 Get ready to speak when prompted!"
	uv run python tests/test_microphone_interactive.py

# Run end-to-end test
test-e2e:
	@echo "Running end-to-end test..."
	uv run python tests/test_e2e_realtime.py

# Run linting
lint:
	@echo "Running linting checks..."
	uv run flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	uv run flake8 . --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics

# Format code
format:
	@echo "Formatting code..."
	uv run black .
	uv run isort .
	@echo "Code formatting complete!"

# Clean up temporary files
clean:
	@echo "Cleaning up temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	@echo "Cleanup complete!"

# Development server with auto-reload
dev-run:
	@echo "Starting development server with auto-reload..."
	uv run streamlit run app.py --server.port 8501 --server.address localhost --server.runOnSave true

# Install system dependencies (macOS)
install-system-deps-mac:
	@echo "Installing system dependencies for macOS..."
	brew install portaudio
	@echo "System dependencies installed!"

# Install system dependencies (Ubuntu/Debian)
install-system-deps-ubuntu:
	@echo "Installing system dependencies for Ubuntu/Debian..."
	sudo apt-get update
	sudo apt-get install -y portaudio19-dev python3-pyaudio
	@echo "System dependencies installed!"

# Check system requirements
check-system:
	@echo "Checking system requirements..."
	@python3 -c "import sys; print(f'Python version: {sys.version}')"
	@which uv > /dev/null && echo "✓ uv is installed" || echo "✗ uv not found - install from https://docs.astral.sh/uv/"
	@echo "System check complete!"
