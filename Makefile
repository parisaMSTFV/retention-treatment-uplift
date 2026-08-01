PYTHON ?= python

.PHONY: run lint test safety check

run:
	$(PYTHON) -m retention_uplift --project-root .

lint:
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) -m unittest discover -s tests -v

safety:
	$(PYTHON) scripts/check_sensitive.py

check: lint test safety
