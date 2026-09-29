"""Запуск: `python -m dds_ai.service` (порт — `DDS_AI_PORT`, по умолчанию 8090)."""

import os

import uvicorn

from .app import create_app

if __name__ == "__main__":
    uvicorn.run(
        create_app(),
        host=os.environ.get("DDS_AI_HOST", "0.0.0.0"),
        port=int(os.environ.get("DDS_AI_PORT", "8090")),
    )
