"""Адаптер TTS на Silero (torch, CPU). Extras: `pip install "dds-ai[voice-silero]"`.

Не запускался в этом окружении: веса раздаются с models.silero.ai, хост заблокирован
сетевой политикой контейнера (см. docs/ai/bench/voice-cpu.md). Лицензия Silero — CC BY-NC-SA
4.0 (некоммерческая) — риск для поставки заказчику отмечен в docs/ai/licenses.md, модель
не утверждена.

Silero не даёт явного параметра «интонация»; управление ограничено скоростью (`rate`) и,
для part моделей, ударением/паузами через SSML-подобную разметку модели `v3_1_ru`. Здесь
интонация транслируется в изменение темпа — то же ограничение и та же оговорка, что и для
Piper: это не «громкость», но и не полноценная просодия.
"""

from __future__ import annotations

from ...contracts.common import ModelRef

_INTONATION_RATE: dict[str, float] = {"neutral": 1.0, "tense": 1.15}


class SileroTTS:
    def __init__(self, speaker: str, *, language: str = "ru", model_version: str = "v4_ru"):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - зависит от опциональной зависимости
            raise ImportError('torch не установлен: pip install "dds-ai[voice-silero]"') from exc
        self._model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language=language,
            speaker=model_version,
        )
        self._speaker = speaker
        self.model_ref = ModelRef(
            component="tts", model_name=f"silero/{speaker}", model_version=model_version
        )

    def synthesize(self, text: str, *, voice: str, intonation: str) -> bytes:
        if voice != self._speaker:
            raise ValueError(
                f"этот адаптер обслуживает голос {self._speaker!r}, запрошен {voice!r}"
            )
        if intonation not in _INTONATION_RATE:
            raise ValueError(
                f"неизвестная интонация {intonation!r}, доступны: {sorted(_INTONATION_RATE)}"
            )
        # Точный параметр темпа/просодии в apply_tts зависит от версии Silero и не проверен
        # здесь без весов модели (см. docstring модуля); rate передаётся, если API его поддерживает.
        kwargs = {"text": text, "speaker": self._speaker, "sample_rate": 48000}
        try:
            audio = self._model.apply_tts(**kwargs, rate=_INTONATION_RATE[intonation])
        except TypeError:
            audio = self._model.apply_tts(**kwargs)
        return audio.numpy().tobytes()
