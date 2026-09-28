# ai/ — ИИ-контур тренажёра диспетчера ДДС

Задание и границы — [`docs/ai/agent-prompt.md`](../docs/ai/agent-prompt.md). Ход работ — [`docs/ai/progress.md`](../docs/ai/progress.md).

## Запуск проверок

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e "ai[dev]"
ai/scripts/check.sh          # ruff + pytest + актуальность JSON Schema в /contracts
```

После изменения контрактов перегенерируй схемы: `python ai/tools/export_schemas.py`.

## Структура

| Путь | Что |
|---|---|
| `src/dds_ai/contracts/` | Контракты (pydantic): события попытки, карточка, критерии, рубрика, оценка, замечания, реплики, профиль, рекомендации, inference-worker, версия корпуса и фрагмент (6.10), выход NLP-извлечения (6.11) |
| `src/dds_ai/ports.py` | Протоколы адаптеров моделей (`STTProvider`, `TTSProvider`, `LLMProvider`, `SemanticJudge`, `Recommender`, `Embedder`) и соседних модулей |
| `src/dds_ai/faults.py` | Инъекция отказов, журнал сбоев, таблица деградации C-04 |
| `src/dds_ai/evaluation.py` | Вызов оценивателя: сбой модели → `not_checked`, не 0 |
| `src/dds_ai/worker/` | Inference-worker: исполнители адаптеров (тайм-аут, очередь, параллелизм), полосы `interactive`/`background`, статус, preflight |
| `src/dds_ai/supervisor/` | ИИ-руководитель: автомат диалога, сценарный fallback, сопоставление доклада, учёт подтверждений C-05; M3 — LLM-формулировка внутри выбранного действия (`phrasing.py`), сборка промпта с изоляцией реплик (`prompting.py`), валидатор выхода (`validator.py`) |
| `src/dds_ai/llm/` | Адаптеры `LLMProvider` (M3): llama.cpp (`llamacpp_provider.py`, extra `ai[llm-llamacpp]`, только локальный GGUF, свой срок генерации) и детерминированный фейк |
| `src/dds_ai/cycle/` | Сквозной учебный цикл на моках, правила критериев M1 по журналу событий, загрузка сценариев |
| `src/dds_ai/mocks/` | Моки бэкенда: маршрутизация, хранилище карточек, журнал событий, оператор 112, медиатракт, эталонный расчёт балла C-06 + W-01 |
| `config/rubric.w01.json` | Рабочая рубрика W-01 (группы с доски Miro); перечень критериев — черновик, утверждает преподаватель |
| `config/worker.cpu.json` | Политики адаптеров CPU-профиля: тайм-ауты, очереди, параллелизм по полосам (до замера) |
| `config/supervisor.fallback.ru.json` | Проверенный шаблон реплик руководителя (версия шаблона пишется в каждую реплику) |
| `config/supervisor.prompt.ru.json` | Версионированный промпт LLM-формулировки (версия промпта пишется в каждую LLM-реплику) |
| `data/synthetic/` | Явно синтетические сценарии M1 — не билеты заказчика |
| `src/dds_ai/voice/` | STT/TTS-адаптеры (M2): нормализация текста для TTS, сигнал неуверенности STT по критическим сущностям, `VoicePipeline` через мок медиатракта; тяжёлые зависимости — extras `ai[voice-...]` |
| `src/dds_ai/knowledge/` | База знаний / RAG (6.10): загрузка → валидация → нормализация → обезличивание → QA → версия корпуса → чанкинг → метаданные → офлайн-индекс (`HashingEmbedder` через порт `Embedder`) |
| `src/dds_ai/nlp/` | NLP-извлечение признаков (6.11): top-k типа, теги, слоты, пропуски, противоречия — правило-ориентированная база для русского языка, `ML hints`, служб не назначает |
| `data/synthetic/knowledge/` | Явно синтетический мини-корпус для pipeline/тестов; путь к настоящему корпусу — конфигурация вызывающего |
| `bench/` | Бенчмарк STT/TTS и стадий LLM-формулировки на CPU (не в `pytest` по умолчанию — отдельные скрипты, см. `bench/README.md`) |
| `tools/export_schemas.py` | Выгрузка JSON Schema в `/contracts` для бэкенда |
| `tests/` | Тесты инвариантов раздела 5 и матрицы раздела 8, накоплены по вехам M0–M7 (441 тест, `ai/scripts/check.sh`); сводная карта строк матрицы раздела 8 по файлам — [`docs/ai/test-matrix.md`](../docs/ai/test-matrix.md), матрица C-04 — [`docs/ai/c04-matrix.md`](../docs/ai/c04-matrix.md) |
