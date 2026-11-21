#!/usr/bin/env bash
# Repeatable bootstrap from a clean checkout. Usage: ./scripts/bootstrap.sh [--no-models] [--no-tests]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
$PY -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements/runtime.lock -r requirements/dev.lock
if [[ " $* " != *" --no-models "* ]]; then
  pip install -q --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple -r requirements/models.lock
fi
pip install -q --no-deps -e .
pvs fixtures all --n 60 >/dev/null
echo "bootstrap: environment ready ($(python --version))"
if [[ " $* " != *" --no-tests "* ]]; then
  python -m pytest -q -m "not models and not postgres"
fi
