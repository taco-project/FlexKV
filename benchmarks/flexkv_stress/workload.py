from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
import hashlib
import random
from typing import Any, Iterable


def sample_blocks(spec: Any, turn_id: int, rng: random.Random) -> int:
    if isinstance(spec, int):
        return spec
    if isinstance(spec, list):
        if not spec:
            return 0
        if len(spec) == 2:
            return rng.randint(int(spec[0]), int(spec[1]))
        return int(spec[turn_id % len(spec)])
    if isinstance(spec, dict):
        mode = spec.get("mode", "fixed")
        if mode == "fixed":
            return int(spec.get("value", spec.get("blocks", 0)))
        if mode == "range":
            low, high = spec.get("values", spec.get("blocks"))
            return rng.randint(int(low), int(high))
        if mode == "list":
            values = spec.get("values", spec.get("blocks"))
            return int(values[turn_id % len(values)])
    raise ValueError(f"Unsupported block length spec: {spec!r}")


def align_down(value: int, alignment: int) -> int:
    return value // alignment * alignment


@dataclass
class Turn:
    conversation_id: int
    turn_id: int
    get_tokens: tuple[int, ...]
    put_tokens: tuple[int, ...]
    added_input_tokens: int
    output_tokens: int
    expected_hit_tokens: int

    @property
    def total_tokens(self) -> int:
        return len(self.put_tokens)


@dataclass
class Conversation:
    conversation_id: int
    history: list[int]
    committed_prefix_tokens: int = 0
    turns: list[Turn] = field(default_factory=list)


class WorkloadGenerator:
    def __init__(self, config, tokens_per_block: int, seed: int):
        self.config = config
        self.tokens_per_block = tokens_per_block
        self.rng = random.Random(seed)
        shared_count = config.system_prompt_blocks * tokens_per_block
        self.shared_system = self._tokens(shared_count, salt=0) if config.shared_system_prompt else ()

    def _tokens(self, count: int, salt: int) -> tuple[int, ...]:
        # Keep token IDs deterministic while making unrelated conversations distinct.
        return tuple(self.rng.randrange(1, 2**30) ^ salt for _ in range(count))

    def _new_conversation(self, conversation_id: int) -> Conversation:
        if self.config.shared_system_prompt:
            system = list(self.shared_system)
        else:
            count = self.config.system_prompt_blocks * self.tokens_per_block
            system = list(self._tokens(count, salt=conversation_id << 8))
        return Conversation(conversation_id=conversation_id, history=system)

    def generate_round(self, round_id: int) -> list[Conversation]:
        conversations = []
        base_id = round_id * self.config.conversations_per_round
        for offset in range(self.config.conversations_per_round):
            conv = self._new_conversation(base_id + offset)
            num_turns = self.rng.randint(self.config.turns_min, self.config.turns_max)
            for turn_id in range(num_turns):
                input_spec = self.config.first_input_blocks if turn_id == 0 else self.config.added_input_blocks
                input_blocks = sample_blocks(input_spec, turn_id, self.rng)
                output_blocks = sample_blocks(self.config.output_blocks, turn_id, self.rng)
                partial = self.config.partial_block_tokens if turn_id == 0 else 0
                added_count = input_blocks * self.tokens_per_block + partial
                output_count = output_blocks * self.tokens_per_block
                added = self._tokens(added_count, salt=(conv.conversation_id << 12) ^ turn_id)
                output = self._tokens(output_count, salt=(conv.conversation_id << 12) ^ turn_id ^ 0xA5)
                get_tokens = tuple(conv.history) + added
                put_tokens = get_tokens + output
                conv.turns.append(Turn(
                    conversation_id=conv.conversation_id,
                    turn_id=turn_id,
                    get_tokens=get_tokens,
                    put_tokens=put_tokens,
                    added_input_tokens=added_count,
                    output_tokens=output_count,
                    expected_hit_tokens=conv.committed_prefix_tokens,
                ))
                conv.history[:] = put_tokens
                conv.committed_prefix_tokens = align_down(len(put_tokens), self.tokens_per_block)
            conversations.append(conv)
        return conversations


class SlotAllocator:
    def __init__(self, num_blocks: int, tokens_per_block: int):
        self.num_blocks = num_blocks
        self.tokens_per_block = tokens_per_block
        self.cursor = 0

    def allocate(self, num_tokens: int):
        import numpy as np
        num_blocks = (num_tokens + self.tokens_per_block - 1) // self.tokens_per_block
        if num_blocks > self.num_blocks:
            raise ValueError(
                f"Request needs {num_blocks} GPU blocks, pool only has {self.num_blocks}"
            )
        blocks = (np.arange(num_blocks, dtype=np.int64) + self.cursor) % self.num_blocks
        self.cursor = int((self.cursor + num_blocks) % self.num_blocks)
        slots = (
            blocks[:, None] * self.tokens_per_block
            + np.arange(self.tokens_per_block, dtype=np.int64)[None, :]
        ).reshape(-1)
        return blocks, slots[:num_tokens]


class PrefixOracle:
    """Block-prefix trie for sequences successfully committed through PUT."""

    @dataclass
    class _Node:
        children: dict[int, "PrefixOracle._Node"] = field(default_factory=dict)
        references: int = 0

    def __init__(self, tokens_per_block: int, capacity_blocks: int = 0):
        self.tokens_per_block = tokens_per_block
        self.capacity_blocks = capacity_blocks
        self.root = self._Node()
        self.sequences: deque[tuple[int, ...]] = deque()
        self.logical_blocks = 0

    def _keys(self, tokens: Iterable[int]) -> tuple[int, ...]:
        import numpy as np
        values = np.asarray(tuple(tokens), dtype=np.int64)
        aligned = align_down(len(values), self.tokens_per_block)
        keys = []
        for start in range(0, aligned, self.tokens_per_block):
            digest = hashlib.blake2b(
                values[start:start + self.tokens_per_block].tobytes(), digest_size=8
            ).digest()
            keys.append(int.from_bytes(digest, "little"))
        return tuple(keys)

    def match(self, tokens: Iterable[int]) -> int:
        node = self.root
        matched = 0
        for key in self._keys(tokens):
            node = node.children.get(key)
            if node is None:
                break
            matched += 1
        return matched * self.tokens_per_block

    def put(self, tokens: Iterable[int]) -> None:
        keys = self._keys(tokens)
        if not keys:
            return
        node = self.root
        node.references += 1
        for key in keys:
            node = node.children.setdefault(key, self._Node())
            node.references += 1
        self.sequences.append(keys)
        self.logical_blocks += len(keys)
        while self.capacity_blocks and self.logical_blocks > self.capacity_blocks and self.sequences:
            self._remove(self.sequences.popleft())

    def _remove(self, keys: tuple[int, ...]) -> None:
        node = self.root
        path = []
        node.references -= 1
        for key in keys:
            child = node.children.get(key)
            if child is None:
                return
            path.append((node, key, child))
            child.references -= 1
            node = child
        for parent, key, child in reversed(path):
            if child.references == 0:
                del parent.children[key]
            else:
                break
        self.logical_blocks -= len(keys)
