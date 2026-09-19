PYTHON ?= python

.PHONY: run external-download external-validate lint test coverage safety seed-stability check

run:
	$(PYTHON) -m retention_uplift --project-root local-runs/latest

external-download:
	@test "$(ACCEPT_CRITEO_LICENSE)" = "CC-BY-NC-SA-4.0" || (echo "Review the Criteo license, then set ACCEPT_CRITEO_LICENSE=CC-BY-NC-SA-4.0" && exit 2)
	$(PYTHON) scripts/download_criteo_uplift.py --accept-license "$(ACCEPT_CRITEO_LICENSE)"

external-validate:
	MPLCONFIGDIR=/tmp/matplotlib $(PYTHON) -m retention_uplift --external-criteo data/external/criteo-research-uplift-v2.1.csv.gz --external-sample-size 300000 --external-target-share 0.30 --seed 42 --project-root local-runs/criteo

lint:
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) -m unittest discover -s tests -v

coverage:
	MPLCONFIGDIR=/tmp/matplotlib $(PYTHON) -m coverage run -m unittest discover -s tests -v
	$(PYTHON) -m coverage report --fail-under=85

safety:
	$(PYTHON) scripts/check_sensitive.py

seed-stability:
	MPLCONFIGDIR=/tmp/matplotlib $(PYTHON) scripts/run_seed_stability.py

check: lint coverage safety
