"""Нормализация текста перед синтезом: адреса, номера домов, числа, аббревиатуры (6.3)."""

from dds_ai.voice.normalize import normalize_for_tts
from dds_ai.voice.numerals import num2ru


def test_address_abbreviations_expand_to_words():
    assert (
        normalize_for_tts("ул. Ленина, д. 12, кв. 5")
        == "улица Ленина, дом двенадцать, квартира пять"
    )


def test_house_number_with_letter_reads_letter_name():
    assert normalize_for_tts("д. 9А") == "дом девять а"


def test_service_acronym_is_spelled_out():
    assert normalize_for_tts("Вызовите МЧС") == "Вызовите эм-че-эс"


def test_plain_number_is_converted():
    assert normalize_for_tts("Пострадавших 3 человека") == "Пострадавших три человека"


def test_num2ru_basic_values():
    assert num2ru(0) == "ноль"
    assert num2ru(5) == "пять"
    assert num2ru(21) == "двадцать один"
    assert num2ru(100) == "сто"
    assert num2ru(125) == "сто двадцать пять"
    assert num2ru(1000) == "одна тысяча"
    assert num2ru(2000) == "две тысячи"
    assert num2ru(1984) == "одна тысяча девятьсот восемьдесят четыре"


def test_normalize_is_idempotent_on_plain_text():
    text = "диспетчер уточнил обстоятельства"
    assert normalize_for_tts(text) == text
