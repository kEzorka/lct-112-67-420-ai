"""Адаптер TTS на Silero (torch, CPU). Extras: `pip install "dds-ai[voice-silero]"`.

Реально прогонялся в бенчмарке (`docs/ai/bench/voice-cpu.md`) на модели `v4_ru` (спикеры
`aidar` — мужской, `baya` — женский), веса которой скачаны напрямую с models.silero.ai в
файл `.pt` (не через `torch.hub.load(repo_or_dir="snakers4/silero-models", ...)`: этот путь
дважды не работает в этом окружении — во-первых, `torch.hub` обращается к `github.com` за
метаданными репозитория, а доступ к GitHub в этой сессии ограничен только собственным
репозиторием проекта (не связано с сетевой политикой моделей); во-вторых, раздел 11 промпта
запрещает скачивание моделей при запуске — конструктор берёт уже скачанный локальный файл
весов, как и `PiperTTS`). Лицензия Silero — CC BY-NC-SA 4.0 (некоммерческая) — риск для
поставки заказчику отмечен в `docs/ai/licenses.md`, модель не утверждена.

Модель `v4_ru` не принимает `rate=` в `apply_tts`, но поддерживает SSML-разметку
(`model.valid_tags`: `prosody rate=`, `prosody pitch=`, `break strength=`). Интонация здесь
транслируется в `<prosody rate=... pitch=...>` — это ближе к настоящей просодии, чем чистое
изменение темпа, но всё ещё не нативный параметр «эмоция»/«интонация» модели.
"""

from __future__ import annotations

from xml.sax.saxutils import escape

from ...contracts.common import ModelRef

_INTONATION_SSML: dict[str, dict[str, str]] = {
    "neutral": {"rate": "medium", "pitch": "medium"},
    "tense": {"rate": "fast", "pitch": "high"},
}


class SileroTTS:
    def __init__(
        self,
        model_path: str,
        *,
        speaker: str,
        sample_rate: int = 48000,
        model_version: str = "v4_ru",
    ):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - зависит от опциональной зависимости
            raise ImportError('torch не установлен: pip install "dds-ai[voice-silero]"') from exc
        self._model = torch.package.PackageImporter(model_path).load_pickle("tts_models", "model")
        self._speaker = speaker
        self.sample_rate = sample_rate
        self.model_ref = ModelRef(
            component="tts", model_name=f"silero/{speaker}", model_version=model_version
        )

    def synthesize(self, text: str, *, voice: str, intonation: str) -> bytes:
        if voice != self._speaker:
            raise ValueError(
                f"этот адаптер обслуживает голос {self._speaker!r}, запрошен {voice!r}"
            )
        params = _INTONATION_SSML.get(intonation)
        if params is None:
            raise ValueError(
                f"неизвестная интонация {intonation!r}, доступны: {sorted(_INTONATION_SSML)}"
            )
        ssml = (
            f'<speak><prosody rate="{params["rate"]}" pitch="{params["pitch"]}">'
            f"{escape(text)}</prosody></speak>"
        )
        import torch

        audio = self._model.apply_tts(
            ssml_text=ssml, speaker=self._speaker, sample_rate=self.sample_rate
        )
        pcm16 = (audio.clamp(-1.0, 1.0) * 32767.0).to(dtype=torch.int16)
        return pcm16.numpy().tobytes()
