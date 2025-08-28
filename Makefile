.PHONY: install run dev lint format zip

VENV?=.venv
PY?=python3.13

install:
	$(PY) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install -U pip && pip install -r requirements.txt

run:
	. $(VENV)/bin/activate && uvicorn app.main:app --host 0.0.0.0 --port $${PORT:-8080}

dev:
	. $(VENV)/bin/activate && uvicorn app.main:app --reload --host 0.0.0.0 --port $${PORT:-8080}

format:
	. $(VENV)/bin/activate && python -m pip install ruff==0.6.9 && ruff format .

lint:
	. $(VENV)/bin/activate && python -m pip install ruff==0.6.9 && ruff check .

zip:
	python tools/make_zip.py
