"""Синтетический набор учебных реплик для бенчмарка STT/TTS (раздел 6.2/6.3 промпта).

Каждая реплика — синтетический текст для замера моделей, не билет и не эталон сценария
(D-050 запрещает выдумывать содержание отсутствующего билета). Покрывает адреса, номера
домов, числа, отрицания и названия служб — категории из требования замера. Аудио для этого
набора в репозиторий не входит (`*.wav` в `.gitignore`): генерируется локально перед прогоном
(см. `synthesize_dataset_audio.py`) или записывается вручную.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchUtterance:
    id: str
    text: str
    has_address: bool = False
    has_number: bool = False
    has_negation: bool = False
    has_service: bool = False


DATASET: tuple[BenchUtterance, ...] = (
    BenchUtterance(
        id="addr-1",
        text="Пожар по адресу улица Ленина, дом 12, квартира 5.",
        has_address=True,
        has_number=True,
    ),
    BenchUtterance(
        id="addr-2",
        text="ДТП на проспекте Мира, дом 9А, рядом с остановкой.",
        has_address=True,
        has_number=True,
    ),
    BenchUtterance(
        id="neg-1",
        text="Пострадавших нет, есть только повреждённый автомобиль.",
        has_negation=True,
    ),
    BenchUtterance(
        id="neg-2",
        text="Пожара не было, сработала пожарная сигнализация ложно.",
        has_negation=True,
        has_service=True,
    ),
    BenchUtterance(
        id="num-1",
        text="Пострадавших трое, один без сознания.",
        has_number=True,
    ),
    BenchUtterance(
        id="num-2",
        text="Дежурная бригада прибудет через пятнадцать минут.",
        has_number=True,
    ),
    BenchUtterance(
        id="svc-1",
        text="Вызвана скорая помощь и пожарная охрана.",
        has_service=True,
    ),
    BenchUtterance(
        id="svc-2",
        text="Нужен наряд ГИБДД на место дорожно-транспортного происшествия.",
        has_service=True,
    ),
    BenchUtterance(
        id="mixed-1",
        text="Утечка газа по адресу улица Садовая, дом 3, корпус 2, вызваны МЧС и газовая служба.",
        has_address=True,
        has_number=True,
        has_service=True,
    ),
    BenchUtterance(
        id="mixed-2",
        text="Обрушение части кровли, пострадавших нет, вызвана служба спасения.",
        has_negation=True,
        has_service=True,
    ),
)
