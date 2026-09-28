# contracts/ — черновики контрактов ИИ-контура ↔ бэкенд

Статус: **черновик `0.4.0-draft`**, не согласован с владельцем бэкенда (вопросы B01–B04 в [`docs/ai/agent-prompt.md`](../docs/ai/agent-prompt.md), раздел 12).

Файлы `*.schema.json` (JSON Schema 2020-12) генерируются из моделей `ai/src/dds_ai/contracts/` командой `python ai/tools/export_schemas.py`. Руками не править: CI сверяет их с кодом.

Схемы не привязаны к транспорту: из них можно получить и REST/WebSocket-сообщения, и `.proto` для gRPC (B03).

Типы совместимы с бэкендом: UUID, время ISO 8601 с часовым поясом (`timestamptz`), произвольные данные — JSON (`jsonb`).

| Схема | Владелец хранения | Кто пишет |
|---|---|---|
| `attempt-event` | бэкенд | клиент, бэкенд, ИИ-воркер (`utterance`, `ack.generated`, `model_failure`), медиатракт (`ack.sent_to_media` / `played_by_client` / `delivery_unconfirmed`) |
| `incident-card`, `card-revision` | бэкенд | ИИ-оператор 112 — исходная карточка; диспетчер — ревизии |
| `criterion-result`, `remark` | бэкенд | ИИ-контур (оцениватель, правила), эксперт |
| `rubric` | бэкенд | преподаватель / администратор; рабочая версия — `ai/config/rubric.w01.json` |
| `score-summary`, `score-version` | бэкенд | модуль оценивания; поправка эксперта — версия N+1 |
| `attempt-version-snapshot` | бэкенд | бэкенд при старте; ИИ-воркер отдаёт версии моделей в preflight |
| `routing-request`, `routing-decision` | движок маршрутизации | только движок; LLM службы не выбирает |
| `interlocutor`, `supervisor-reply` | ИИ-контур | ИИ-руководитель |
| `skill-profile`, `recommendation` | бэкенд | ИИ-контур (`dds_ai.profile`); поправка/аудит C-08 — новая версия, не перезапись |
| `task-pool` | бэкенд (пул назначает преподаватель) | преподаватель публикует; `Recommendation.pool_snapshot_id`/`pool_task_ids` — снимок на момент выбора (C-08) |
| `error-explanation` | бэкенд | `dds_ai.hints.explain_errors` — из замечаний-ошибок ученика после завершения попытки (Q&A 8) |
| `difficulty-vector`, `difficulty-weights`, `difficulty-score`, `difficulty-proposal` | бэкенд (публикует преподаватель) | `dds_ai.difficulty` — ИИ предлагает вектор с объяснением по фактору (`difficulty-proposal`); преподаватель утверждает и публикует `difficulty-vector` |
| `group-error-report` | ИИ-контур (по запросу преподавателя) | `dds_ai.analytics.build_group_error_report` — только из `valid_score`, технические сбои и `not_checked` отдельно |
| `adapter-policy`, `failure-record`, `preflight-report` | ИИ-воркер | ИИ-воркер; бэкенд показывает администратору |
| `scenario` | бэкенд (публикация), ИИ-контур (черновики M4) | преподаватель утверждает; на M1 — синтетика в `ai/data/synthetic/` |

## Известные причины `ack.delivery_unconfirmed`

Пока строка (enum — после согласования с владельцем медиатракта, B04). Используются: `call_ended_by_client` — ученик сам завершил вызов (не технический сбой); `call_dropped`, `client_silent`, `tts_failed`, `empty audio` — технические.
