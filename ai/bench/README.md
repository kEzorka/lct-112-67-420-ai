# Бенчмарк STT/TTS на CPU (M2)

Не запускается в `pytest` по умолчанию (`ai/scripts/check.sh` его не вызывает) — это
отдельные скрипты, не тесты. Результаты полного замера — [`docs/ai/bench/voice-cpu.md`](../../docs/ai/bench/voice-cpu.md).

**Это не целевое железо заказчика.** Любой прогон обязан печатать `platform.platform()`
и процессор; переноси эти данные в отчёт вместе с квантизацией и конкурентностью (раздел 9
промпта). Кандидаты — предмет замера, не решение (раздел 11: «не объявлять модели
утверждёнными до замера и проверки лицензий»); лицензии — [`docs/ai/licenses.md`](../../docs/ai/licenses.md).

## Порядок запуска

```bash
pip install -e "ai[voice-fasterwhisper,voice-vosk]"   # STT-кандидаты
pip install -e "ai[voice-piper,voice-silero]"          # TTS-кандидаты
# Веса моделей — отдельно, руками (huggingface.co/alphacephei.com/models.silero.ai
# могут быть недоступны в изолированном окружении; см. docs/ai/bench/voice-cpu.md).

cd ai/bench
python selfcheck.py                                    # проверка обвязки на фейках, без весов

python synthesize_dataset_audio.py --tts-candidate piper-irina --out-dir /tmp/bench-audio
python run_stt_bench.py --audio-dir /tmp/bench-audio --synthetic-audio

python run_tts_bench.py --candidates piper-irina piper-denis silero-baya silero-aidar
```

## Ограничения замера

- Аудио для STT из `synthesize_dataset_audio.py` — синтетическая речь, а не запись
  живого голоса; WER будет смещён в лучшую сторону (совпадение доменов синтеза и
  распознавания). Отмечай `--synthetic-audio`, не выдавай за замер на реальной речи.
- `run_tts_bench.py` меряет время до **полного** буфера синтеза, не до первого звука в
  потоковом смысле — `TTSProvider.synthesize()` не потоковый API.
- Датасет (`dataset.py`) — синтетические учебные реплики, не билеты (D-050) и не эталон.

## LLM-формулировка реплик руководителя (M3)

Отчёт — [`docs/ai/bench/llm-cpu.md`](../../docs/ai/bench/llm-cpu.md). Стадии меряются
раздельно: генерация, валидация, формулировка целиком (с очередью воркера и повтором),
ход руководителя целиком. TTS и транспорт сюда не входят.

```bash
cd ai/bench
python run_llm_bench.py --runs 20                          # фейк: проверка обвязки, не модели
python run_llm_bench.py --runs 5 --fake-latency-ms 50

pip install -e "ai[llm-llamacpp]"                          # runtime llama.cpp
python run_llm_bench.py --provider llamacpp --model /models/candidate.gguf \
    --model-version <из манифеста> --n-threads 6 --runs 20

# Дымовой прогон адаптера на настоящем llama.cpp без весов-кандидатов:
pip install gguf numpy
python make_tiny_gguf.py /tmp/tiny.gguf                    # случайные веса, не модель
python run_llm_bench.py --provider llamacpp --model /tmp/tiny.gguf --n-ctx 8192 --runs 5
```

Замер на крошечной модели показывает, что путь через runtime работает. О задержке или
качестве настоящей LLM он ничего не говорит.
