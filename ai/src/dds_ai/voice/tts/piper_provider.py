"""Адаптер TTS на Piper (VITS/onnxruntime, CPU). Extras: `pip install "dds-ai[voice-piper]"`.

Реально прогонялся в бенчмарке (`docs/ai/bench/voice-cpu.md`) на пакете `piper-tts==1.8.0`
(форк `OHF-Voice/piper1-gpl`, GPL-зависимость `espeak-ng` — см. `docs/ai/licenses.md`) и
голосах `ru_RU-irina-medium`/`ru_RU-denis-medium` с huggingface.co (`rhasspy/piper-voices`).
API этой версии — `PiperVoice.synthesize(text, syn_config) -> Iterable[AudioChunk]`, не
`synthesize_stream_raw(...)` из более старого `rhasspy/piper` (MIT), под который был
написан этот файл изначально; переписан под фактически проверенную версию.

Piper не имеет отдельного параметра «интонация» — модель обучена на одном стиле речи.
Две различимые интонации получены варьированием `length_scale`/`noise_scale` (темп и
вариативность произношения), а не только громкостью (D-025 явно исключает громкость).
Это рабочее приближение, не нативная просодия — риск отмечен в docs/ai/licenses.md.
"""

from __future__ import annotations

from ...contracts.common import ModelRef

_INTONATION_PARAMS: dict[str, dict[str, float]] = {
    "neutral": {"length_scale": 1.0, "noise_scale": 0.667, "noise_w_scale": 0.8},
    "tense": {"length_scale": 0.85, "noise_scale": 0.9, "noise_w_scale": 0.8},
}


class PiperTTS:
    def __init__(self, voice_model_path: str, *, voice_name: str, model_version: str | None = None):
        try:
            from piper import PiperVoice
        except ImportError as exc:  # pragma: no cover - зависит от опциональной зависимости
            raise ImportError('piper-tts не установлен: pip install "dds-ai[voice-piper]"') from exc
        self._voice = PiperVoice.load(voice_model_path)
        self._voice_name = voice_name
        self.sample_rate = self._voice.config.sample_rate
        self.model_ref = ModelRef(
            component="tts", model_name=f"piper/{voice_name}", model_version=model_version or "1"
        )

    def synthesize(self, text: str, *, voice: str, intonation: str) -> bytes:
        if voice != self._voice_name:
            raise ValueError(
                f"этот адаптер обслуживает голос {self._voice_name!r}, запрошен {voice!r}"
            )
        params = _INTONATION_PARAMS.get(intonation)
        if params is None:
            raise ValueError(
                f"неизвестная интонация {intonation!r}, доступны: {sorted(_INTONATION_PARAMS)}"
            )
        from piper.config import SynthesisConfig

        syn_config = SynthesisConfig(**params)
        chunks = list(self._voice.synthesize(text, syn_config=syn_config))
        return b"".join(chunk.audio_int16_bytes for chunk in chunks)
