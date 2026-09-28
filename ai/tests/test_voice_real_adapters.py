"""Логика реальных STT/TTS-адаптеров (bench/voice-cpu) без весов моделей.

Веса и опциональные пакеты (`vosk`/`piper`/`torch`) не устанавливаются для этих тестов —
`piper`/`torch` подменяются фейковыми модулями через `sys.modules` (стандартный приём: `from
X import Y` берёт `sys.modules['X']`, если он уже зарегистрирован, не трогая диск), поэтому
тесты работают и на машине без `ai[voice-*]` extras. Реальный прогон с весами — в
`docs/ai/bench/voice-cpu.md`.
"""

from __future__ import annotations

import struct
import sys
import types

import pytest

from dds_ai.voice.stt.vosk_provider import _pcm_and_rate


def _make_wav_bytes(*, framerate: int, frames: bytes) -> bytes:
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(framerate)
        wav_file.writeframes(frames)
    return buf.getvalue()


def test_pcm_and_rate_extracts_frames_and_rate_from_wav_container() -> None:
    frames = struct.pack("<4h", 1, -1, 2, -2)
    wav_bytes = _make_wav_bytes(framerate=16000, frames=frames)

    pcm, rate = _pcm_and_rate(wav_bytes, default_sample_rate=8000)

    assert pcm == frames
    assert rate == 16000


def test_pcm_and_rate_passes_through_raw_pcm_with_default_rate() -> None:
    raw = struct.pack("<2h", 5, -5)

    pcm, rate = _pcm_and_rate(raw, default_sample_rate=16000)

    assert pcm == raw
    assert rate == 16000


class _FakePiperVoice:
    def __init__(self, sample_rate: int = 22050):
        self.config = types.SimpleNamespace(sample_rate=sample_rate)
        self.synth_calls: list[tuple[str, object]] = []

    @classmethod
    def load(cls, path: str) -> _FakePiperVoice:
        return cls()

    def synthesize(self, text: str, *, syn_config):
        self.synth_calls.append((text, syn_config))
        chunk = types.SimpleNamespace(audio_int16_bytes=b"chunk-bytes")
        return [chunk, chunk]


@pytest.fixture
def fake_piper_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    piper_module = types.ModuleType("piper")
    piper_module.PiperVoice = _FakePiperVoice
    piper_config_module = types.ModuleType("piper.config")

    class _FakeSynthesisConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    piper_config_module.SynthesisConfig = _FakeSynthesisConfig
    monkeypatch.setitem(sys.modules, "piper", piper_module)
    monkeypatch.setitem(sys.modules, "piper.config", piper_config_module)
    return piper_module


def test_piper_tts_reads_sample_rate_from_voice_config(fake_piper_module) -> None:
    from dds_ai.voice.tts.piper_provider import PiperTTS

    provider = PiperTTS("dummy.onnx", voice_name="irina")

    assert provider.sample_rate == 22050
    assert provider.model_ref.model_name == "piper/irina"


def test_piper_tts_synthesize_joins_chunks_and_passes_intonation_params(
    fake_piper_module,
) -> None:
    from dds_ai.voice.tts.piper_provider import PiperTTS

    provider = PiperTTS("dummy.onnx", voice_name="irina")
    audio = provider.synthesize("Пожар", voice="irina", intonation="tense")

    assert audio == b"chunk-bytes" * 2
    _text, syn_config = provider._voice.synth_calls[-1]
    assert syn_config.kwargs == {
        "length_scale": 0.85,
        "noise_scale": 0.9,
        "noise_w_scale": 0.8,
    }


def test_piper_tts_rejects_unknown_voice(fake_piper_module) -> None:
    from dds_ai.voice.tts.piper_provider import PiperTTS

    provider = PiperTTS("dummy.onnx", voice_name="irina")
    with pytest.raises(ValueError, match="irina"):
        provider.synthesize("текст", voice="denis", intonation="neutral")


def test_piper_tts_rejects_unknown_intonation(fake_piper_module) -> None:
    from dds_ai.voice.tts.piper_provider import PiperTTS

    provider = PiperTTS("dummy.onnx", voice_name="irina")
    with pytest.raises(ValueError, match="интонация"):
        provider.synthesize("текст", voice="irina", intonation="angry")


class _FakeTensor:
    def __init__(self, values: list[float]):
        self.values = values

    def clamp(self, lo: float, hi: float) -> _FakeTensor:
        return _FakeTensor([max(lo, min(hi, v)) for v in self.values])

    def __mul__(self, scalar: float) -> _FakeTensor:
        return _FakeTensor([v * scalar for v in self.values])

    def to(self, dtype: object) -> _FakeTensor:
        return _FakeTensor([int(v) for v in self.values])

    def numpy(self) -> _FakeTensor:
        return self

    def tobytes(self) -> bytes:
        return struct.pack(f"<{len(self.values)}h", *self.values)


class _FakeSileroModel:
    def __init__(self):
        self.apply_tts_calls: list[dict] = []

    def apply_tts(self, *, ssml_text: str, speaker: str, sample_rate: int) -> _FakeTensor:
        self.apply_tts_calls.append(
            {"ssml_text": ssml_text, "speaker": speaker, "sample_rate": sample_rate}
        )
        return _FakeTensor([0.5, -2.0, 1.5])


@pytest.fixture
def fake_torch_module(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    fake_model = _FakeSileroModel()

    class _FakePackageImporter:
        def __init__(self, path: str):
            self.path = path

        def load_pickle(self, package: str, resource: str) -> _FakeSileroModel:
            return fake_model

    torch_module = types.ModuleType("torch")
    torch_module.package = types.SimpleNamespace(PackageImporter=_FakePackageImporter)
    torch_module.int16 = "int16"
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    return types.SimpleNamespace(module=torch_module, model=fake_model)


def test_silero_tts_builds_ssml_with_prosody_for_each_intonation(fake_torch_module) -> None:
    from dds_ai.voice.tts.silero_provider import SileroTTS

    provider = SileroTTS("dummy.pt", speaker="baya")
    provider.synthesize("Пожар & дым", voice="baya", intonation="tense")

    call = fake_torch_module.model.apply_tts_calls[-1]
    assert call["speaker"] == "baya"
    assert call["sample_rate"] == 48000
    assert '<prosody rate="fast" pitch="high">' in call["ssml_text"]
    assert "Пожар &amp; дым" in call["ssml_text"]


def test_silero_tts_clamps_and_converts_to_int16_pcm(fake_torch_module) -> None:
    from dds_ai.voice.tts.silero_provider import SileroTTS

    provider = SileroTTS("dummy.pt", speaker="baya")
    audio = provider.synthesize("текст", voice="baya", intonation="neutral")

    # Модель отдала [0.5, -2.0, 1.5] -> clamp в [-1, 1] -> [0.5, -1.0, 1.0] * 32767
    assert audio == struct.pack("<3h", 16383, -32767, 32767)


def test_silero_tts_rejects_unknown_voice(fake_torch_module) -> None:
    from dds_ai.voice.tts.silero_provider import SileroTTS

    provider = SileroTTS("dummy.pt", speaker="baya")
    with pytest.raises(ValueError, match="baya"):
        provider.synthesize("текст", voice="aidar", intonation="neutral")


def test_silero_tts_rejects_unknown_intonation(fake_torch_module) -> None:
    from dds_ai.voice.tts.silero_provider import SileroTTS

    provider = SileroTTS("dummy.pt", speaker="baya")
    with pytest.raises(ValueError, match="интонация"):
        provider.synthesize("текст", voice="baya", intonation="angry")
