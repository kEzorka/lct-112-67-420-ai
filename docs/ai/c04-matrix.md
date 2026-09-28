# Матрица C-04: деградация по отдельным отказам

Приёмка 6.1: по отдельности отключить STT, LLM, TTS, семантическую модель, рекомендатель;
вызвать тайм-аут и переполнение очереди; невалидный JSON от модели. Для каждого случая —
сохранённые события, возможность продолжения, статус оценки, исключение из рейтинга.

Тесты: [`ai/tests/test_c04_matrix.py`](../../ai/tests/test_c04_matrix.py).

| Отказ | Событие сохранено | Попытка продолжается | Статус оценки | Рейтинг |
|---|---|---|---|---|
| STT (`error`, preflight) | `ModelFailure(stt)` в журнале попытки | да, текстовый режим | `provisional` | исключена (`valid_score=False`) |
| TTS (`error`, preflight) | `ModelFailure(tts)` в журнале попытки | да, текстовый режим | `provisional` | исключена |
| LLM (`error`/`timeout`/`overflow`) | `ModelFailure(llm)` в журнале попытки, вид сбоя сохранён | да, сценарный fallback (6.1) | `provisional` | исключена |
| Семантический оцениватель (`error`/`timeout`/`overflow`) | запись в журнале сбоев воркера (`FailureRecord`) — критерий не создаёт событие звонка | да, критерий → `not_checked` | `provisional` | исключена |
| Семантический оцениватель, невалидный JSON | `FailureRecord(kind=invalid_output)` | да, критерий → `not_checked` | `provisional` | исключена |
| Рекомендатель (`error`/`timeout`/`overflow`) | запись в журнале сбоев воркера | да, `recommend()` возвращает решение, не исключение | рекомендация «недоступна», причина явная | профиль не ухудшается (не мутирует) |

## Почему тайм-аут/переполнение не проверяются отдельно для STT/TTS

Preflight — это снимок готовности (D-3), не отдельный вызов модели: `InferenceWorker.available()`
теряет вид внедрённого сбоя (любой `FaultInjector.inject` на компонент даёт `available=False`),
а `preflight_failures()` сводит недоступность к `error` (`overflow` — только когда исполнитель
реально перегружен очередью, что отдельно проверено на уровне `InferenceWorker` в
`ai/tests/test_worker.py::test_queue_overflow_is_explicit_and_logged` и
`test_timeout_is_finite_and_logged`, реальным тайм-аутом и очередью, без подмены).
Тайм-аут и переполнение очереди с сохранением вида сбоя на уровне попытки проверены там, где
сбой ловится в момент вызова модели, а не в preflight-снимке: на LLM, оценивателе и
рекомендателе (`FaultInjector.call()` сохраняет `kind` без изменений).

## Не проверено на этом узле

- Реальная (не внедрённая через `FaultInjector`) очередь/тайм-аут на полном цикле
  `TrainingSession` — механика уже доказана изолированно на `InferenceWorker`
  (`test_worker.py`), но не прогонялась сквозь весь учебный цикл с боевыми таймаутами.
- Невалидный JSON для LLM/рекомендателя на уровне попытки (для LLM это `invalid_output`,
  уже покрыт `test_faults.py::test_plain_value_error_is_runtime_error_not_invalid_output` и
  `test_cycle.py::test_invalid_llm_output_in_cycle_is_model_failure_and_technical_flag`; для
  рекомендателя — `test_m6_adaptation.py::test_recommender_output_outside_candidates_is_rejected_not_silently_accepted`)
  не дублируется здесь во избежание повторения уже существующих тестов.
