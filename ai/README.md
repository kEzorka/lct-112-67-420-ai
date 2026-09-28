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
| `src/dds_ai/mocks/` | Моки бэкенда: маршрутизация, хранилище карточек, медиатракт, эталонный расчёт балла C-06 + W-01 |
| `src/dds_ai/knowledge/` | База знаний / RAG (6.10): загрузка → валидация → нормализация → обезличивание → QA → версия корпуса → чанкинг → метаданные → офлайн-индекс (`HashingEmbedder` через порт `Embedder`) |
| `src/dds_ai/nlp/` | NLP-извлечение признаков (6.11): top-k типа, теги, слоты, пропуски, противоречия — правило-ориентированная база для русского языка, `ML hints`, служб не назначает |
| `data/synthetic/knowledge/` | Явно синтетический мини-корпус для pipeline/тестов; путь к настоящему корпусу — конфигурация вызывающего |
| `config/rubric.w01.json` | Рабочая рубрика W-01 (группы с доски Miro); перечень критериев — черновик, утверждает преподаватель |
| `tools/export_schemas.py` | Выгрузка JSON Schema в `/contracts` для бэкенда |
| `tests/` | Тесты инвариантов раздела 5 и матрицы раздела 8, покрытых на M0–M4a |
