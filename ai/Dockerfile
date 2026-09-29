# ИИ-контур тренажёра диспетчера ДДС — HTTP-сервис (docs/ai/deploy.md).
# Контекст сборки — папка ai/:  docker build -t dds-ai ai/
#
# Аргументы:
#   EXTRAS  — дополнительные зависимости через запятую, например "llm-llamacpp"
#             (для локальной LLM; нужен компилятор, поэтому ставится build-essential).
# Веса моделей в образ не кладутся: монтируйте каталог в /models и задайте
#   DDS_LLM_MODEL_PATH=/models/<файл>.gguf

FROM python:3.11-slim

ARG EXTRAS=""

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DDS_AI_PORT=8090

WORKDIR /app

RUN if [ -n "$EXTRAS" ]; then \
      apt-get update && apt-get install -y --no-install-recommends build-essential cmake \
      && rm -rf /var/lib/apt/lists/*; \
    fi

# Пути к config/ и data/ считаются от исходников, поэтому ставим пакет в режиме editable
# и сохраняем структуру ai/ как есть.
COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config
COPY data ./data
RUN pip install -e ".[service${EXTRAS:+,$EXTRAS}]"

RUN useradd --system --uid 10001 dds && mkdir -p /models && chown dds /models
USER dds

EXPOSE 8090
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"DDS_AI_PORT\"]}/health', timeout=4)"

CMD ["python", "-m", "dds_ai.service"]
