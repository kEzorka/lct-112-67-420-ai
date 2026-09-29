# Запуск ИИ-контура в контейнере

Сервис — `ai/` → образ `dds-ai`, HTTP на порту `8090`. Транспорт — черновик до ответа B03 (gRPC или HTTP + WebSocket); сообщения — контракты из `/contracts`.

## Сборка и запуск

```bash
# из корня репозитория, где лежит папка ai/
docker build -t dds-ai ai/
docker run --rm -p 8090:8090 dds-ai
curl localhost:8090/health
```

Документация API (OpenAPI) — `http://localhost:8090/docs`.

Без модели руководитель отвечает проверенным шаблоном (fallback, C-04), семантические критерии — «не проверено», итог предварительный. Это штатный режим деградации, не ошибка.

### С локальной LLM

```bash
docker build -t dds-ai --build-arg EXTRAS=llm-llamacpp ai/
docker run --rm -p 8090:8090 \
  -v /path/to/models:/models:ro \
  -e DDS_LLM_MODEL_PATH=/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  dds-ai
```

Веса в образ не входят (D-051: поставляются локальным пакетом). Скачать модель: см. `docs/ai/bench/llm-cpu.md`.

## docker-compose (фрагмент для общего репозитория)

```yaml
services:
  ai:
    build:
      context: ./ai
      args:
        EXTRAS: ""            # "llm-llamacpp" — для локальной LLM
    environment:
      DDS_AI_PORT: "8090"
      # DDS_LLM_MODEL_PATH: /models/qwen2.5-1.5b-instruct-q4_k_m.gguf
    # volumes:
    #   - ./models:/models:ro
    ports:
      - "8090:8090"
    restart: unless-stopped
```

Бэкенд обращается к сервису по имени `http://ai:8090` внутри сети compose. Наружу порт можно не публиковать.

## Переменные окружения

| Переменная | По умолчанию | Что |
|---|---|---|
| `DDS_AI_PORT` | `8090` | порт HTTP |
| `DDS_AI_HOST` | `0.0.0.0` | адрес |
| `DDS_LLM_MODEL_PATH` | — | путь к GGUF внутри контейнера; нет — работа без LLM |

## API (v1, черновик)

| Метод | Путь | Что |
|---|---|---|
| GET | `/health` | статус, версия контрактов, подключённая модель |
| GET | `/v1/contracts`, `/v1/contracts/{name}` | список и JSON Schema контрактов |
| GET | `/v1/scenarios` | синтетические сценарии |
| POST | `/v1/attempts` `{scenario_id, card_seed?}` | начать попытку (уведомление) |
| POST | `/v1/attempts/{id}/open` | открыть карточку → видимые поля |
| POST | `/v1/attempts/{id}/card` `{values}` | ручные правки карточки |
| POST | `/v1/attempts/{id}/decision` `{decision}` | `respond` / `refuse` / `redirect` |
| POST | `/v1/attempts/{id}/dial` | звонок руководителю |
| POST | `/v1/attempts/{id}/say` `{text}` | реплика диспетчера → ответ руководителя, `ack_id` |
| POST | `/v1/attempts/{id}/submit` `{incomplete?}` | сдача |
| GET | `/v1/attempts/{id}/evaluation` | критерии с доказательствами, итог C-06, снимок версий |
| GET | `/v1/attempts/{id}/events` | журнал событий попытки |

Каждый ответ возвращает события попытки в формате контракта `attempt-event` — бэкенд может сохранять их в свой журнал.

## Ограничения

- Попытки хранятся в памяти процесса: перезапуск контейнера их теряет. Постоянное хранение — у бэкенда (B04).
- Сейчас только текстовый канал; голосовой (STT/TTS, медиатракт) — после выбора транспорта (B03).
- Нет аутентификации между сервисами — не публиковать порт наружу; сервисный токен или mTLS — после B03.
- Один процесс, без масштабирования; нагрузка не проверялась.
