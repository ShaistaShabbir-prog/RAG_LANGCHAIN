# RAG_LANGCHAIN — Makefile

.PHONY: setup run test docker-up docker-down clean

setup:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "Copying .env.example to .env..."
	cp -n .env.example .env || true
	@echo "✅ Setup complete. Edit .env with your API keys."

run:
	uvicorn rag_project.app.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v --tb=short

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	@echo "✅ Cleaned"
