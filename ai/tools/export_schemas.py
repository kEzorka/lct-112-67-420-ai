"""Выгрузка JSON Schema контрактов в /contracts.

python tools/export_schemas.py          # записать
python tools/export_schemas.py --check  # проверить, что схемы актуальны (CI)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import TypeAdapter

from dds_ai.contracts import CONTRACTS_VERSION, SCHEMA_MODELS

OUT = Path(__file__).resolve().parents[2] / "contracts"


def render() -> dict[str, str]:
    files = {}
    for name, model in SCHEMA_MODELS.items():
        schema = (
            model.json_schema() if isinstance(model, TypeAdapter) else model.model_json_schema()
        )
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": f"dds-ai/{name}/{CONTRACTS_VERSION}",
            **schema,
        }
        files[f"{name}.schema.json"] = json.dumps(schema, ensure_ascii=False, indent=2) + "\n"
    return files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    files = render()
    if args.check:
        stale = [
            n
            for n, text in files.items()
            if not (OUT / n).exists() or (OUT / n).read_text("utf-8") != text
        ]
        extra = [p.name for p in OUT.glob("*.schema.json") if p.name not in files]
        if stale or extra:
            print(f"contracts/ is out of date: stale={stale} extra={extra}", file=sys.stderr)
            print("run: python ai/tools/export_schemas.py", file=sys.stderr)
            return 1
        return 0
    OUT.mkdir(exist_ok=True)
    for p in OUT.glob("*.schema.json"):
        if p.name not in files:
            p.unlink()
    for name, text in files.items():
        (OUT / name).write_text(text, "utf-8")
    print(f"wrote {len(files)} schemas to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
