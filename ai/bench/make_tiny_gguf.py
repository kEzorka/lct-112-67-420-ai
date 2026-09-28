#!/usr/bin/env python3
"""Крошечная GGUF-модель со случайными весами — только для дымового прогона адаптера llama.cpp.

Это не модель-кандидат и не замер качества или задержки настоящей LLM. Файл нужен, чтобы
проверить путь `LlamaCppProvider` → llama.cpp: загрузку локального файла, грамматику JSON,
срок генерации и работу валидатора на реальном runtime, когда настоящие веса недоступны
(docs/ai/bench/llm-cpu.md). Словарь — байтовый SentencePiece, архитектура `llama`.
Веса нигде не публикуются и в git не кладутся.

Зависимости: `pip install gguf numpy` (не входят в extras пакета).
"""

from __future__ import annotations

import argparse
from pathlib import Path

N_EMBD, N_HEAD, N_LAYER, N_FF, N_CTX = 64, 4, 2, 128, 8192


def build(path: Path, seed: int = 0) -> None:
    import gguf
    import numpy as np

    rng = np.random.default_rng(seed)
    tokens = ["<unk>", "<s>", "</s>", *(f"<0x{b:02X}>" for b in range(256)), "▁"]
    types = [
        gguf.TokenType.UNKNOWN,
        gguf.TokenType.CONTROL,
        gguf.TokenType.CONTROL,
        *([gguf.TokenType.BYTE] * 256),
        gguf.TokenType.NORMAL,
    ]
    n_vocab = len(tokens)

    w = gguf.GGUFWriter(str(path), "llama")
    w.add_name("tiny-random-smoke")
    w.add_context_length(N_CTX)
    w.add_embedding_length(N_EMBD)
    w.add_block_count(N_LAYER)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD)
    w.add_rope_dimension_count(N_EMBD // N_HEAD)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_file_type(gguf.LlamaFileType.ALL_F32)
    w.add_tokenizer_model("llama")
    w.add_token_list(tokens)
    w.add_token_scores([0.0] * n_vocab)
    w.add_token_types(types)
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)
    w.add_unk_token_id(0)

    def t(name: str, *shape: int) -> None:
        w.add_tensor(name, (rng.standard_normal(shape) * 0.02).astype(np.float32))

    t("token_embd.weight", n_vocab, N_EMBD)
    w.add_tensor("output_norm.weight", np.ones(N_EMBD, dtype=np.float32))
    t("output.weight", n_vocab, N_EMBD)
    for i in range(N_LAYER):
        w.add_tensor(f"blk.{i}.attn_norm.weight", np.ones(N_EMBD, dtype=np.float32))
        w.add_tensor(f"blk.{i}.ffn_norm.weight", np.ones(N_EMBD, dtype=np.float32))
        for name in ("attn_q", "attn_k", "attn_v", "attn_output"):
            t(f"blk.{i}.{name}.weight", N_EMBD, N_EMBD)
        t(f"blk.{i}.ffn_gate.weight", N_FF, N_EMBD)
        t(f"blk.{i}.ffn_up.weight", N_FF, N_EMBD)
        t(f"blk.{i}.ffn_down.weight", N_EMBD, N_FF)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    build(args.out, args.seed)
    print(f"{args.out}: {args.out.stat().st_size} байт, случайные веса (не модель-кандидат)")


if __name__ == "__main__":
    main()
