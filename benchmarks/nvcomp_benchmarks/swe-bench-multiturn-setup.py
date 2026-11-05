#!/usr/bin/env python3
"""Prepare SWE-bench Oracle as a multi-turn JSON conversations file.

Output: JSON list of {"id", "messages": [user, assistant, user, ...]} in the
shape vLLM's multi_turn benchmark expects. The same file works against
sglang server via any OpenAI-chat-completions client (e.g. multi_turn_client.py).

--isl N: target turn-0 prompt length in tokens. The script walks the shuffled
dataset, accumulating oracle texts (concatenated with a visible separator)
until the running token count reaches N, then encodes the stitched text and
HARD-TRUNCATES to exactly N tokens (decode back to text). Every emitted conv
is therefore exactly N tokens (may cut a single oracle mid-sentence, but
keeps the distribution tight).
"""

# python3 benchmarks/nvcomp_benchmarks/swe-bench-multiturn-setup.py     --tokenizer /tmp/nvidia-mps/GLM-5.1-FP8     --output-file benchmarks/nvcomp_benchmarks/swe-bench-oracle-ISL128k-5turns-conversations.json     --isl 128000     --num-turns 5    --num-conversations 40 --seed 0


from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

SUFFIXES = [
    "Summarize the bug in 3 bullet points.",
    "List the relevant files that need to be modified and explain why.",
    "Write a unit test that reproduces this bug.",
    "Propose a minimal patch that fixes this bug. Output the diff only.",
    "What edge cases should the fix handle? List them.",
    "Are there other places in the codebase that may have the same bug pattern?",
    "Write a short PR description for this fix.",
    "Explain the root cause to a new contributor in plain English.",
]
PREAMBLE = (
    "You are an expert software engineer. Read the following SWE-bench "
    "oracle issue carefully; you will be asked several follow-up "
    "questions about it.\n\n"
)
ASSISTANT_PLACEHOLDER = "(assistant response will be filled in at runtime)"

_STITCH_SEPARATOR = "\n\n---\n\nNEXT ISSUE:\n\n---\n\n"

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def log(msg: str) -> None:
    print(f"[prep] {msg}", flush=True)


class TokenizersAdapter:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.tokenizer.encode(
            text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)


def load_tokenizer(tokenizer_path: str) -> TokenizersAdapter:
    path = Path(tokenizer_path)
    tokenizer_file = path if path.is_file() else path / "tokenizer.json"
    if not tokenizer_file.is_file():
        raise FileNotFoundError(
            f"tokenizer.json not found: {tokenizer_file}. "
            "Pass either a tokenizer.json file or a model directory containing it.")

    log(f"loading tokenizer: {tokenizer_file}")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    log("tokenizer loaded")
    return TokenizersAdapter(tokenizer)


def _resolve_dataset(dataset: str) -> tuple[str, str | None]:
    # Accept HF cache dirs like .../datasets--<org>--<name> by splitting into
    # (hf_id, cache_root); load_dataset itself can't open that path directly.
    p = Path(dataset)
    if p.is_dir() and p.name.startswith("datasets--"):
        parts = p.name[len("datasets--"):].split("--", 1)
        if len(parts) == 2:
            return f"{parts[0]}/{parts[1]}", str(p.parent)
    return dataset, None


def prepare_swe_dataset(
    tokenizer: Any,
    *,
    dataset: str = "princeton-nlp/SWE-bench_oracle",
    split: str = "test",
    isl_tokens: int = 16000,
    num_conversations: int | None = None,
    seed: int = 0,
) -> list[tuple[str, str, int]]:
    """Return [(conv_id, oracle_text, n_tokens), ...].

    Walks the shuffled dataset and accumulates oracle texts (joined by a
    visible separator) until the running token count crosses ``isl_tokens``;
    the stitched text is then encoded and hard-truncated to exactly
    ``isl_tokens`` tokens. Stops after ``num_conversations`` convs (or when
    the dataset is exhausted, if None).
    """
    hf_id, cache_dir = _resolve_dataset(dataset)
    log("importing datasets.load_dataset")
    from datasets import load_dataset

    log(f"loading dataset={hf_id!r} split={split!r} cache_dir={cache_dir!r}")
    ds = load_dataset(hf_id, split=split, cache_dir=cache_dir)
    log(f"loaded dataset with {len(ds)} rows")
    order = list(range(len(ds)))
    random.Random(seed).shuffle(order)
    log(f"shuffled dataset with seed={seed}; target_isl={isl_tokens}")

    sep_tokens = len(tokenizer.encode(_STITCH_SEPARATOR, add_special_tokens=False))
    log(f"separator token count={sep_tokens}")

    kept: list[tuple[str, str, int]] = []
    buf_pieces: list[str] = []
    buf_tokens = 0
    for scanned, idx in enumerate(order, start=1):
        if num_conversations is not None and len(kept) >= num_conversations:
            break
        if scanned == 1 or scanned % 100 == 0:
            limit = "all" if num_conversations is None else str(num_conversations)
            log(f"scan progress: rows={scanned} conversations={len(kept)}/{limit} buffer_tokens~{buf_tokens}")
        text = ds[idx].get("text") or ""
        if not text:
            continue
        n = len(tokenizer.encode(text, add_special_tokens=False))
        if buf_pieces:
            buf_tokens += sep_tokens
        buf_pieces.append(text)
        buf_tokens += n
        if buf_tokens >= isl_tokens:
            stitched = _STITCH_SEPARATOR.join(buf_pieces)
            # Hard-truncate: re-encode the joined text (per-piece counts are
            # only an estimate since boundary tokens can merge), then slice
            # to exactly isl_tokens and decode back.
            ids = tokenizer.encode(stitched, add_special_tokens=False)
            if len(ids) >= isl_tokens:
                ids = ids[:isl_tokens]
                stitched = tokenizer.decode(ids)
                kept.append((f"swe_{len(kept):05d}", stitched, len(ids)))
                log(f"emitted conversation {len(kept)} with {len(ids)} tokens after scanning {scanned} rows")
                buf_pieces = []
                buf_tokens = 0
            # else: boundary merging shaved us below ISL — keep accumulating
    log(f"finished scan: emitted {len(kept)} conversations")
    return kept


def build_conversations(
    kept: list[tuple[str, str, int]], num_turns: int
) -> list[dict]:
    # Oracle lives in turn 1; later user turns are just the short questions.
    out = []
    for conv_id, oracle, n_tokens in kept:
        messages = [
            {"role": "user", "content": f"{PREAMBLE}{oracle}\n\nQuestion 1: {SUFFIXES[0]}"},
            {"role": "assistant", "content": ASSISTANT_PLACEHOLDER},
        ]
        for i in range(2, num_turns + 1):
            messages.append({"role": "user", "content": f"Question {i}: {SUFFIXES[i - 1]}"})
            messages.append({"role": "assistant", "content": ASSISTANT_PLACEHOLDER})
        out.append({
            "id": conv_id,
            "oracle_tokens": n_tokens,
            "messages": messages,
        })
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--output-file", required=True)
    p.add_argument("--dataset", default="princeton-nlp/SWE-bench_oracle")
    p.add_argument("--split", default="test")
    p.add_argument("--isl", type=int, required=True,
                   help="target turn-0 oracle length in tokens; pieces are "
                        "accumulated until running total >= ISL, then emitted "
                        "as one conv.")
    p.add_argument("--num-conversations", type=int, default=0,
                   help="0 (default) keeps producing convs until dataset runs out")
    p.add_argument("--num-turns", type=int, default=5, help=f"1..{len(SUFFIXES)}")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if not 1 <= args.num_turns <= len(SUFFIXES):
        p.error(f"--num-turns must be in [1, {len(SUFFIXES)}]")

    tokenizer = load_tokenizer(args.tokenizer)

    kept = prepare_swe_dataset(
        tokenizer,
        dataset=args.dataset, split=args.split,
        isl_tokens=args.isl,
        num_conversations=args.num_conversations or None,
        seed=args.seed,
    )
    if args.num_conversations and len(kept) < args.num_conversations:
        log(f"WARNING: requested {args.num_conversations}, got {len(kept)}")

    if kept:
        lens = sorted(n for _, _, n in kept)
        log("oracle tokens over "
            f"{len(lens)} convs: min={lens[0]} p50={lens[len(lens)//2]} "
            f"max={lens[-1]} (target ISL={args.isl})")

    log(f"building {len(kept)} conversations with {args.num_turns} turns each")
    records = build_conversations(kept, args.num_turns)

    out_path = Path(args.output_file)
    log(f"writing output to {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
    try:
        os.chmod(out_path, 0o666)
    except OSError:
        pass

    log(f"Wrote {len(records)} conversations x {args.num_turns} turns -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
