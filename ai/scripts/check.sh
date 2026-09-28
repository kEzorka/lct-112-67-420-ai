#!/usr/bin/env bash
# Проверки ИИ-контура: линтер, тесты, актуальность JSON Schema в /contracts.
# Запуск из любого места: ai/scripts/check.sh
set -euo pipefail
cd "$(dirname "$0")/.."

ruff check src tests tools
ruff format --check src tests tools
python -m pytest
python tools/export_schemas.py --check
