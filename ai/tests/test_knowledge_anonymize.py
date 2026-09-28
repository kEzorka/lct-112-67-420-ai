"""Обезличивание текста корпуса (6.10): телефоны, ФИО, адреса физлиц, e-mail."""

from __future__ import annotations

from dds_ai.knowledge.anonymize import anonymize_text


def test_phone_is_redacted():
    for phone in ("+7 (912) 345-67-89", "8-912-345-67-89", "89123456789"):
        clean, hits = anonymize_text(f"Позвоните по номеру {phone} для уточнения.")
        assert phone not in clean
        assert "[PHONE]" in clean
        assert any(h.category == "phone" for h in hits)


def test_email_is_redacted():
    clean, hits = anonymize_text("Пишите на test.user@example.com по вопросам.")
    assert "test.user@example.com" not in clean
    assert "[EMAIL]" in clean
    assert any(h.category == "email" for h in hits)


def test_full_name_is_redacted():
    for text in ("Смирнова Анна Петровна", "Кузнецов К.К."):
        clean, hits = anonymize_text(f"Заявитель — {text}, обратился лично.")
        assert text not in clean
        assert "[NAME]" in clean
        assert any(h.category == "full_name" for h in hits)


def test_individual_address_is_redacted():
    text = "Проживает по адресу: ул. Тестовая, д. 5, кв. 12."
    clean, hits = anonymize_text(text)
    assert "Тестовая, д. 5, кв. 12" not in clean
    assert "[ADDRESS]" in clean
    assert any(h.category == "address" for h in hits)


def test_text_without_pii_is_untouched():
    text = "Тип происшествия «пожар» — главная служба пожарная охрана."
    clean, hits = anonymize_text(text)
    assert clean == text
    assert hits == ()


def test_combined_document_has_no_pii_left():
    text = (
        "Заявитель Смирнова Анна Петровна, телефон +7 (912) 345-67-89, "
        "e-mail smirnova.test@example.com, адрес: ул. Тестовая, д. 5, кв. 12."
    )
    clean, hits = anonymize_text(text)
    for leak in ("Смирнова", "345-67-89", "smirnova.test", "Тестовая, д. 5"):
        assert leak not in clean
    assert len(hits) == 4
