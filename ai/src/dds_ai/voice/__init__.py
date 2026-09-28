"""Голосовой контур M2: адаптеры STT/TTS (ports.py), нормализация текста, сигнал неуверенности,
пайплайн через мок медиатракта (mocks/media.py). Модели не утверждены до замера
(docs/ai/bench/voice-cpu.md).
"""

from .confidence import UncertainEntity, detect_uncertain_entities
from .normalize import normalize_for_tts
from .pipeline import SttOutcome, TtsOutcome, VoicePipeline

__all__ = [
    "SttOutcome",
    "TtsOutcome",
    "UncertainEntity",
    "VoicePipeline",
    "detect_uncertain_entities",
    "normalize_for_tts",
]
