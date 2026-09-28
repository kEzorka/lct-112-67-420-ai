"""Число → русские слова (для нормализации TTS, 6.3). Без внешних зависимостей.

Поддержан диапазон 0..999_999_999. Согласование рода: мужская форма по умолчанию
(«дом один», «квартира пять» — так читаются номера в адресах), «тысяча/тысячи» —
в женском роде («одна тысяча», «две тысячи»), как того требует русская грамматика
независимо от рода счётного слова. Порядковые числительные («двенадцатый») и падежное
согласование не поддержаны — вне объёма нормализации адресов/чисел для TTS.
"""

from __future__ import annotations

_ONES = (
    "",
    "один",
    "два",
    "три",
    "четыре",
    "пять",
    "шесть",
    "семь",
    "восемь",
    "девять",
)
_ONES_FEMININE = ("", "одна", "две", *_ONES[3:])
_TEENS = (
    "десять",
    "одиннадцать",
    "двенадцать",
    "тринадцать",
    "четырнадцать",
    "пятнадцать",
    "шестнадцать",
    "семнадцать",
    "восемнадцать",
    "девятнадцать",
)
_TENS = (
    "",
    "",
    "двадцать",
    "тридцать",
    "сорок",
    "пятьдесят",
    "шестьдесят",
    "семьдесят",
    "восемьдесят",
    "девяносто",
)
_HUNDREDS = (
    "",
    "сто",
    "двести",
    "триста",
    "четыреста",
    "пятьсот",
    "шестьсот",
    "семьсот",
    "восемьсот",
    "девятьсот",
)


def _plural(n: int, one: str, few: str, many: str) -> str:
    n = abs(n) % 100
    if 11 <= n <= 19:
        return many
    tail = n % 10
    if tail == 1:
        return one
    if 2 <= tail <= 4:
        return few
    return many


def _under_1000(n: int, *, feminine: bool = False) -> str:
    if n == 0:
        return ""
    words: list[str] = []
    hundreds, rem = divmod(n, 100)
    if hundreds:
        words.append(_HUNDREDS[hundreds])
    if 10 <= rem < 20:
        words.append(_TEENS[rem - 10])
    else:
        tens, ones = divmod(rem, 10)
        if tens:
            words.append(_TENS[tens])
        if ones:
            words.append((_ONES_FEMININE if feminine else _ONES)[ones])
    return " ".join(words)


_SCALES = (
    (1_000_000_000, "миллиард", "миллиарда", "миллиардов", False),
    (1_000_000, "миллион", "миллиона", "миллионов", False),
    (1_000, "тысяча", "тысячи", "тысяч", True),
)


def num2ru(n: int) -> str:
    """Целое число → русские слова, например 125 → «сто двадцать пять»."""
    if n == 0:
        return "ноль"
    if n < 0:
        return "минус " + num2ru(-n)

    parts: list[str] = []
    remainder = n
    for scale, one, few, many, feminine in _SCALES:
        count, remainder = divmod(remainder, scale)
        if count:
            parts.append(
                f"{_under_1000(count, feminine=feminine)} {_plural(count, one, few, many)}"
            )
    if remainder or not parts:
        parts.append(_under_1000(remainder))
    return " ".join(p.strip() for p in parts if p.strip())
