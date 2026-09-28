"""Адаптеры LLMProvider (ports.py). Runtime llama.cpp — опциональный extra `ai[llm-llamacpp]`."""

from .fake import EchoDraftLLM, draft_of

__all__ = ["EchoDraftLLM", "draft_of"]
