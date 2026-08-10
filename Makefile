install:
	pip install -e ".[dev]"
	pre-commit install

install-local-embed:
	pip install -e ".[local-embed]"

install-ui:
	pip install -e ".[ui]"

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

typecheck:
	mypy src/

test-unit:
	pytest tests/unit/ -v --cov-fail-under=0

test-integration:
	pytest tests/integration/ -v -m integration

test-e2e:
	pytest tests/e2e/ -v -m e2e

test: test-unit

eval:
	pytest tests/e2e/test_eval.py -v --tb=short --no-cov

ingest:
	rag-platform-ingest --source $(SOURCE)

api:
	uvicorn rag.api.main:app --reload --host 0.0.0.0 --port 8000

ui:
	streamlit run app.py

docker-up:
	docker compose up -d && docker compose ps

docker-down:
	docker compose down -v

ci: lint typecheck test

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .coverage htmlcov/ dist/ .mypy_cache/ .ruff_cache/ reports/ .chroma/

.PHONY: install install-local-embed install-ui lint format typecheck test-unit \
        test-integration test-e2e test eval ingest api ui docker-up docker-down ci clean
