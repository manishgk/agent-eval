.PHONY: install test lint type check sample dashboard demo

install:
	poetry install --with dev,dashboard

test:
	poetry run pytest -q

lint:
	poetry run ruff check .

type:
	poetry run mypy

check: lint type test

sample:
	poetry run python scripts/generate_sample.py

demo:
	poetry run agent-eval run evalsets/tool_calling.yaml --reps 20 --mock

dashboard:
	poetry run streamlit run dashboard/app.py
