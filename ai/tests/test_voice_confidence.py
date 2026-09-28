"""Искажённое распознавание адреса/числа даёт сигнал неуверенности, не молчаливое принятие (6.2)."""

from dds_ai.ports import TranscriptSegment
from dds_ai.voice.confidence import detect_uncertain_entities


def seg(text: str, *, confidence=None, final=True) -> TranscriptSegment:
    return TranscriptSegment(text=text, start_ms=0, end_ms=1000, final=final, confidence=confidence)


def test_clean_address_with_high_confidence_is_not_flagged():
    segments = [seg("улица Ленина, дом 12", confidence=0.95)]
    assert detect_uncertain_entities(segments) == ()


def test_low_confidence_address_is_flagged():
    segments = [seg("улица Ленина, дом 12", confidence=0.3)]
    uncertain = detect_uncertain_entities(segments)
    assert any(e.kind == "address" and e.reason == "low_confidence" for e in uncertain)


def test_address_without_house_number_is_flagged_even_without_confidence():
    """Модель может не отдавать confidence вообще (инвариант 5) — сигнал всё равно нужен."""
    segments = [seg("улица Ленина", confidence=None)]
    uncertain = detect_uncertain_entities(segments)
    assert any(e.kind == "address" and e.reason == "no_house_number" for e in uncertain)


def test_low_confidence_number_is_flagged():
    segments = [seg("пострадавших 3 человека", confidence=0.4)]
    uncertain = detect_uncertain_entities(segments)
    assert any(e.kind == "number" for e in uncertain)


def test_low_confidence_service_name_is_flagged():
    segments = [seg("вызвана скорая помощь", confidence=0.2)]
    uncertain = detect_uncertain_entities(segments)
    assert any(e.kind == "service" for e in uncertain)


def test_partial_segments_are_ignored():
    segments = [seg("улица Лен", confidence=0.1, final=False)]
    assert detect_uncertain_entities(segments) == ()


def test_no_invented_confidence_value():
    """Сигнал неуверенности — это флаг, а не выдуманное число (инвариант 5)."""
    segments = [seg("улица Ленина", confidence=None)]
    for entity in detect_uncertain_entities(segments):
        assert not hasattr(entity, "confidence")
