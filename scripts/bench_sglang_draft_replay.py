#!/usr/bin/env python3
"""Matched-input speculative decoding replay benchmark for a dedicated server.

Run ``prepare`` once, then reuse that JSONL for every baseline/fixed run. The
benchmark never changes model startup configuration. /flush_cache must retain
FlexKV host/storage caches; L3 additionally requires the task-only, idle-checked
control bridge. A cache-tier mismatch invalidates a sample rather than silently
turning a cold computation into a replay result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def request(base, path, payload=None, timeout=600):
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base.rstrip("/") + path, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{path}: HTTP {exc.code}: {exc.read().decode()}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def prepare(args):
    from transformers import AutoTokenizer

    source = Path(args.dataset)
    raw = source.read_bytes()
    try:
        dataset = json.loads(raw)
    except json.JSONDecodeError:
        dataset = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not isinstance(dataset, list):
        raise ValueError("dataset must be a JSON list or JSONL")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False)
    random.Random(args.seed).shuffle(dataset)
    records = []
    seen = set()
    for item in dataset:
        if "input_ids" in item:
            ids = list(item["input_ids"])
        else:
            turns = item.get("conversations", item.get("messages", []))
            if turns:
                first = next((turn for turn in turns if turn.get("from", turn.get("role")) in ("human", "user")), None)
                if first is None:
                    continue
                text = first.get("value", first.get("content", ""))
            else:
                text = item.get("prompt", item.get("text", ""))
            if not isinstance(text, str) or not text.strip():
                continue
            if args.chat_template:
                ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True
                )
                if hasattr(ids, "keys"):
                    ids = ids["input_ids"]
                if ids and isinstance(ids[0], list):
                    ids = ids[0]
            else:
                ids = tokenizer.encode(text, add_special_tokens=True)
        if args.replay_page_size:
            end = (len(ids) - args.replay_suffix_tokens) // args.replay_page_size * args.replay_page_size
            if end <= 0:
                continue
            ids = ids[: end + args.replay_suffix_tokens]
        if not args.min_tokens <= len(ids) <= args.max_tokens:
            continue
        fingerprint = digest(ids)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        records.append({"id": len(records), "input_ids": ids, "input_sha256": fingerprint})
        if len(records) == args.samples:
            break
    if len(records) != args.samples:
        raise ValueError(f"only {len(records)} eligible unique prompts; requested {args.samples}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in records))
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(
            {
                "dataset_sha256": hashlib.sha256(raw).hexdigest(),
                "tokenizer": args.tokenizer,
                "seed": args.seed,
                "samples": len(records),
                "input_sha256": digest([r["input_ids"] for r in records]),
                "chat_template": args.chat_template,
                "replay_page_size": args.replay_page_size,
                "replay_suffix_tokens": args.replay_suffix_tokens if args.replay_page_size else None,
                "min_tokens": min(len(r["input_ids"]) for r in records),
                "max_tokens": max(len(r["input_ids"]) for r in records),
            },
            indent=2,
        )
        + "\n"
    )


def summarize(rows):
    valid = [row for row in rows if row.get("valid")]
    totals = {
        name: sum(row[name] for row in valid)
        for name in ("completion_tokens", "verify_ct", "correct_drafts", "proposed_drafts")
    }
    seconds = [row["client_elapsed_s"] for row in valid]
    return {
        "samples": len(rows),
        "valid_samples": len(valid),
        "invalid_samples": len(rows) - len(valid),
        "all_samples_valid": bool(rows) and len(valid) == len(rows),
        "totals": totals,
        "accept_length": totals["completion_tokens"] / totals["verify_ct"] if totals["verify_ct"] else None,
        "accept_rate": totals["correct_drafts"] / totals["proposed_drafts"] if totals["proposed_drafts"] else None,
        "mean_request_accept_length": statistics.mean(row["completion_tokens"] / row["verify_ct"] for row in valid)
        if valid
        else None,
        "mean_client_elapsed_s": statistics.mean(seconds) if seconds else None,
        "output_sha256": digest([row.get("output_ids") for row in rows]),
    }


def measure(args):
    records = [json.loads(line) for line in Path(args.inputs).read_text().splitlines() if line.strip()]
    if not records or any(digest(row["input_ids"]) != row["input_sha256"] for row in records):
        raise ValueError("empty or corrupted frozen input file")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    run_salt = f"flexkv-draft-replay-{time.time_ns()}"
    manifest = {
        "arguments": vars(args),
        "inputs_sha256": digest([r["input_ids"] for r in records]),
        "server_info": request(args.url, "/get_server_info"),
        "started_at": time.time(),
        "run_cache_salt": run_salt,
        "scope": "acceptance and cache-tier replay; output equality is reported separately",
        "run_metadata": json.loads(Path(args.run_metadata).read_text()) if args.run_metadata else None,
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    rows = []

    def prefix_length(record):
        length = len(record["input_ids"])
        raw_end = min(length - 1, length - args.reprefill_tail)
        return (raw_end - int(args.key_mode == "bigram")) // args.page_size * args.page_size

    def replay_salt(record, repeat):
        return f"{run_salt}-replay-{repeat}-{record['id']}"

    def control(action, payload=None):
        if not args.control_url:
            raise ValueError(f"{action} needs --control-url for this dedicated benchmark server")
        result = request(args.control_url, "/" + action, payload or {})
        if not isinstance(result, dict) or not result.get("valid"):
            raise RuntimeError(f"control {action} failed: {result}")
        with (output / "controls.jsonl").open("a") as stream:
            stream.write(json.dumps({"action": action, "result": result}) + "\n")
        if action == "reset_all" and args.store_control_url:
            cleared = request(args.store_control_url, "/clear", {})
            if not isinstance(cleared, dict) or not cleared.get("valid"):
                raise RuntimeError(f"dedicated Store clear failed: {cleared}")
            with (output / "controls.jsonl").open("a") as stream:
                stream.write(json.dumps({"action": "dedicated_store_clear", "result": cleared}) + "\n")
        return result

    def flush():
        result = request(args.url, "/flush_cache", {})
        if isinstance(result, dict) and result.get("success") is False:
            raise RuntimeError(f"flush refused: {result}")

    def generate(record, phase, repeat):
        l3_before = control("snapshot") if phase == "l3" else None
        begin = time.monotonic()
        payload = {
            "input_ids": record["input_ids"],
            "sampling_params": {"max_new_tokens": args.output_tokens, "temperature": 0, "ignore_eos": True},
            "cache_salt": (
                f"{run_salt}-cold-{repeat}-{record['id']}" if phase == "cold" else replay_salt(record, repeat)
            ),
            "return_logprob": False,
        }
        response = request(args.url, "/generate", payload)
        meta = response.get("meta_info", {})
        row = {
            "id": record["id"],
            "input_sha256": record["input_sha256"],
            "phase": phase,
            "repeat": repeat,
            "client_elapsed_s": time.monotonic() - begin,
            "output_ids": response.get("output_ids"),
            "meta_info": meta,
            "errors": [],
        }
        for dest, source in (
            ("completion_tokens", "completion_tokens"),
            ("verify_ct", "spec_verify_ct"),
            ("correct_drafts", "spec_num_correct_drafts"),
            ("proposed_drafts", "spec_num_proposed_drafts"),
        ):
            value = meta.get(source)
            if not isinstance(value, int) or value < 0:
                row["errors"].append(f"missing/invalid {source}")
                value = 0
            row[dest] = value
        if row["completion_tokens"] != args.output_tokens or row["verify_ct"] <= 0:
            row["errors"].append("incomplete output or no speculative verify steps")
        if row["proposed_drafts"] <= 0 or row["correct_drafts"] > row["proposed_drafts"]:
            row["errors"].append("inconsistent speculative draft counters")
        if not isinstance(row["output_ids"], list) or len(row["output_ids"]) != row["completion_tokens"]:
            row["errors"].append("complete output token IDs are required for paired output checks")
        details = meta.get("cached_tokens_details") or {}
        device, host, storage = (int(details.get(key, 0)) for key in ("device", "host", "storage"))
        row["cache_tokens"] = dict(device=device, host=host, storage=storage)
        minimum = prefix_length(record)
        if phase == "cold" and int(meta.get("cached_tokens", 0)) != 0:
            row["errors"].append("cold request reused a prefix")
        if phase == "l1" and (device < minimum or host or storage):
            row["errors"].append("L1 tier precondition failed")
        if phase == "l2" and (device or host < minimum or storage):
            row["errors"].append("L2 tier precondition failed")
        if phase == "l3" and (device or storage < minimum):
            row["errors"].append("L3 tier precondition failed")
        if l3_before is not None:
            l3_after = control("drain")
            name = "flexkv_benchmark_REMOTE2H_committed_bytes_total"
            row["l3_committed_bytes"] = l3_after["transfers"][name] - l3_before["transfers"][name]
            if row["l3_committed_bytes"] <= 0:
                row["errors"].append("L3 did not publish any remote payload bytes")
        row["valid"] = not row["errors"]
        return row

    def record_batch(batch):
        rows.extend(batch)
        with (output / "requests.jsonl").open("a") as stream:
            for row in batch:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
        print(
            json.dumps({"phase": batch[0]["phase"], "repeat": batch[0]["repeat"], "summary": summarize(batch)}),
            flush=True,
        )
        if any(not row["valid"] for row in batch):
            raise RuntimeError("invalid benchmark sample; see requests.jsonl")

    try:
        # One prompt at a time keeps the working set bounded for full-size GLM.
        # Each record is its own cold -> L1 -> L2/L3 experiment. Distinct frozen
        # prompts plus reset_all remove contamination across formal repeats.
        for repeat in range(args.repeats):
            for record in records:
                flush()
                if args.control_url:
                    control("reset_all")
                record_batch([generate(record, "cold", repeat)])
                # A hybrid cache stores its exact SWA snapshot at checkpoint
                # boundaries. Seed the intended replay boundary explicitly;
                # a full-length cold generation can end on a later snapshot.
                flush()
                if args.control_url:
                    control("reset_all")
                prefix = prefix_length(record)
                if prefix <= 0:
                    raise ValueError("input has no reusable page after the reprefill tail")
                seed = request(
                    args.url,
                    "/generate",
                    {
                        "input_ids": record["input_ids"][: prefix + int(args.key_mode == "bigram")],
                        "cache_salt": replay_salt(record, repeat),
                        "sampling_params": {"max_new_tokens": 1, "temperature": 0, "ignore_eos": True},
                    },
                )
                with (output / "seeds.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "id": record["id"],
                                "repeat": repeat,
                                "prefix_tokens": prefix,
                                "meta_info": seed.get("meta_info"),
                            }
                        )
                        + "\n"
                    )
                if args.control_url:
                    control("drain")
                if "l1" in args.tiers:
                    record_batch([generate(record, "l1", repeat)])
                if args.control_url:
                    control("drain")
                for tier in ("l2", "l3"):
                    if tier not in args.tiers:
                        continue
                    flush()
                    if args.churn:
                        # Overwrite reused device slots with a different real
                        # prompt before replay; otherwise a freshly reset
                        # allocator may return exactly the old draft slots and
                        # conceal the missing-backup bug.
                        alternate = records[(records.index(record) + 1) % len(records)]["input_ids"]
                        if alternate == record["input_ids"]:
                            raise ValueError("--churn requires at least two distinct inputs")
                        alternate = (alternate * ((len(record["input_ids"]) + len(alternate) - 1) // len(alternate)))[
                            : len(record["input_ids"])
                        ]
                        churn = request(
                            args.url,
                            "/generate",
                            {
                                "input_ids": alternate,
                                "cache_salt": f"{run_salt}-churn-{repeat}-{record['id']}-{tier}",
                                "sampling_params": {"max_new_tokens": 1, "temperature": 0, "ignore_eos": True},
                            },
                        )
                        with (output / "churn.jsonl").open("a") as stream:
                            stream.write(json.dumps({"before_tier": tier, "meta_info": churn.get("meta_info")}) + "\n")
                        control("drain")
                        flush()
                    if tier == "l3":
                        control(
                            "clear_l2",
                            {
                                "prefixes": [
                                    {
                                        "input_ids": record["input_ids"],
                                        "cache_salt": replay_salt(record, repeat),
                                        "cached_prefix_tokens": prefix,
                                    }
                                ]
                            },
                        )
                    record_batch([generate(record, tier, repeat)])
                    if args.control_url:
                        control("drain")
        summaries = {phase: summarize([r for r in rows if r["phase"] == phase]) for phase in ("cold", *args.tiers)}
        pairs = []
        for repeat in range(args.repeats):
            for phase in args.tiers:
                cold = {r["id"]: r for r in rows if r["phase"] == "cold" and r["repeat"] == repeat}
                replay = [r for r in rows if r["phase"] == phase and r["repeat"] == repeat]
                comparable = [
                    r
                    for r in replay
                    if isinstance(r["output_ids"], list) and isinstance(cold[r["id"]]["output_ids"], list)
                ]
                pairs.append(
                    {
                        "repeat": repeat,
                        "phase": phase,
                        "output_pairs": len(comparable),
                        "identical_output_pairs": sum(
                            r["output_ids"] == cold[r["id"]]["output_ids"] for r in comparable
                        ),
                    }
                )
        result = {"complete": True, "summaries": summaries, "output_pair_checks": pairs}
    except BaseException as exc:
        result = {
            "complete": False,
            "error": repr(exc),
            "summaries": {
                phase: summarize([r for r in rows if r["phase"] == phase]) for phase in ("cold", *args.tiers)
            },
        }
        raise
    finally:
        (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


def compare(args):
    roots = [Path(args.before), Path(args.after)]
    manifests = [json.loads((root / "manifest.json").read_text()) for root in roots]
    summaries = [json.loads((root / "summary.json").read_text()) for root in roots]
    if not all(summary.get("complete") for summary in summaries):
        raise ValueError("both runs must be complete; invalid samples cannot form a comparison")
    if manifests[0]["inputs_sha256"] != manifests[1]["inputs_sha256"]:
        raise ValueError("frozen inputs differ")
    for field in ("page_size", "key_mode", "reprefill_tail", "output_tokens", "repeats", "tiers", "churn"):
        if manifests[0]["arguments"][field] != manifests[1]["arguments"][field]:
            raise ValueError(f"benchmark argument differs: {field}")
    for field in (
        "model_path",
        "revision",
        "dtype",
        "quantization",
        "kv_cache_dtype",
        "context_length",
        "tp_size",
        "ep_size",
        "moe_ep_size",
        "pp_size",
        "attn_dp_size",
        "attn_cp_size",
        "dcp_size",
        "chunked_prefill_size",
        "max_running_requests",
        "max_total_tokens",
        "speculative_algorithm",
        "speculative_num_steps",
        "speculative_eagle_topk",
        "speculative_num_draft_tokens",
        "speculative_dspark_block_size",
        "speculative_draft_model_path",
        "speculative_draft_kv_cache_dtype",
        "enable_decoder_swa_bounded_replay",
        "enable_encoder_swa_bounded_replay",
        "attention_backend",
        "dsa_prefill_backend",
        "dsa_decode_backend",
        "dsv4_prefill_backend",
        "sampling_backend",
        "random_seed",
        "enable_deterministic_inference",
        "enable_tf32_matmul",
        "disable_cuda_graph",
        "cpu_offload_gb",
        "offload_group_size",
        "offload_num_in_group",
        "offload_prefetch_step",
    ):
        if manifests[0].get("server_info", {}).get(field) != manifests[1].get("server_info", {}).get(field):
            raise ValueError(f"resolved model configuration differs: {field}")
    launch_metadata = [manifest.get("run_metadata") for manifest in manifests]
    if all(launch_metadata):
        for field in (
            "FLEXKV_ENABLE_LAYERWISE_TRANSFER",
            "FLEXKV_ENABLE_MPS",
            "SGLANG_DSV4_KV_LAYOUT",
            "SGLANG_DSV4_COMPRESSED_KV_LAYOUT",
            "SGLANG_DSV4_UNIFIED_KV_FP8",
            "SGLANG_MOE_COPY_WEIGHT_VIEWS_BEFORE_H2D",
        ):
            if launch_metadata[0].get("environment", {}).get(field) != launch_metadata[1].get("environment", {}).get(
                field
            ):
                raise ValueError(f"launch environment differs: {field}")
    requests = [[json.loads(line) for line in (root / "requests.jsonl").read_text().splitlines()] for root in roots]
    tables = [{(row["phase"], row["repeat"], row["id"]): row for row in rows} for rows in requests]
    if any(len(table) != len(rows) for table, rows in zip(tables, requests)):
        raise ValueError("duplicate request identity")
    if tables[0].keys() != tables[1].keys():
        raise ValueError("request pairing differs")
    for key, before in tables[0].items():
        after = tables[1][key]
        if not before["valid"] or not after["valid"] or before["input_sha256"] != after["input_sha256"]:
            raise ValueError(f"invalid matched request: {key}")
    result = {
        "inputs_sha256": manifests[0]["inputs_sha256"],
        "before": str(roots[0]),
        "after": str(roots[1]),
        "paired_requests": len(tables[0]),
        "phases": {},
    }
    for phase, first in summaries[0]["summaries"].items():
        second = summaries[1]["summaries"][phase]
        if not first["all_samples_valid"] or not second["all_samples_valid"]:
            raise ValueError(f"invalid phase: {phase}")
        pairs = [(before, tables[1][key]) for key, before in tables[0].items() if key[0] == phase]
        comparable = [
            (before, after)
            for before, after in pairs
            if isinstance(before["output_ids"], list) and isinstance(after["output_ids"], list)
        ]
        ids = sorted({before["id"] for before, _ in pairs})
        clusters = {index: [(before, after) for before, after in pairs if before["id"] == index] for index in ids}
        rng = random.Random(42)
        deltas = []
        for _ in range(args.bootstrap):
            draw = [pair for index in rng.choices(ids, k=len(ids)) for pair in clusters[index]]
            lengths = [
                sum(pair[arm]["completion_tokens"] for pair in draw) / sum(pair[arm]["verify_ct"] for pair in draw)
                for arm in (0, 1)
            ]
            deltas.append(lengths[1] - lengths[0])
        deltas.sort()
        result["phases"][phase] = {
            "before": first,
            "after": second,
            "accept_length_delta": second["accept_length"] - first["accept_length"],
            "accept_rate_delta": second["accept_rate"] - first["accept_rate"],
            "accept_length_delta_prompt_bootstrap_95pct": [
                deltas[int(0.025 * len(deltas))],
                deltas[min(len(deltas) - 1, int(0.975 * len(deltas)))],
            ]
            if deltas
            else None,
            "bootstrap_unit": "prompt id, retaining all repeated requests in a sampled cluster",
            "output_pairs": len(comparable),
            "identical_output_pairs": sum(before["output_ids"] == after["output_ids"] for before, after in comparable),
        }
        if phase != "cold":
            gaps = {
                "before": first["accept_length"] - summaries[0]["summaries"]["cold"]["accept_length"],
                "after": second["accept_length"] - summaries[1]["summaries"]["cold"]["accept_length"],
            }
            result["phases"][phase]["accept_length_gap_vs_cold"] = gaps
            result["phases"][phase]["change_in_gap_vs_cold"] = gaps["after"] - gaps["before"]
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--dataset", required=True)
    prep.add_argument("--tokenizer", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--samples", type=int, default=200)
    prep.add_argument("--min-tokens", type=int, default=512)
    prep.add_argument("--max-tokens", type=int, default=4096)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--chat-template", action=argparse.BooleanOptionalAction, default=False)
    prep.add_argument(
        "--replay-page-size",
        type=int,
        default=0,
        help="optional reproduction workload: truncate prompts to whole pages plus a short suffix",
    )
    prep.add_argument("--replay-suffix-tokens", type=int, default=16)
    run = sub.add_parser("measure")
    run.add_argument("--url", required=True)
    run.add_argument("--control-url")
    run.add_argument(
        "--store-control-url", help="optional, exclusively owned Store cleanup endpoint between experiments"
    )
    run.add_argument("--run-metadata", help="optional launch/source/hardware JSON retained with the results")
    run.add_argument("--inputs", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--page-size", type=int, required=True)
    run.add_argument("--key-mode", choices=["token", "bigram"], default="token")
    run.add_argument(
        "--reprefill-tail",
        type=int,
        default=0,
        help="resolved window re-prefill tail; DSv4.1 decoder bounded replay uses its window",
    )
    run.add_argument("--output-tokens", type=int, default=128)
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--tiers", nargs="+", choices=("l1", "l2", "l3"), default=["l1", "l2", "l3"])
    run.add_argument(
        "--churn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="overwrite reused device slots with another real prompt before replay",
    )
    diff = sub.add_parser("compare")
    diff.add_argument("--before", required=True)
    diff.add_argument("--after", required=True)
    diff.add_argument("--output", required=True)
    diff.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.replay_page_size < 0 or (
            args.replay_page_size and not 2 <= args.replay_suffix_tokens < args.replay_page_size
        ):
            parser.error("replay suffix must be at least 2 tokens and shorter than its positive page size")
        prepare(args)
    elif args.command == "compare":
        if args.bootstrap < 1:
            parser.error("--bootstrap must be positive")
        compare(args)
    else:
        if args.repeats < 1 or args.output_tokens < 2 or args.page_size < 1:
            parser.error("positive repeat/page counts and at least 2 output tokens required")
        if "l3" in args.tiers and not args.control_url:
            parser.error("--control-url is required to establish cold/L3 preconditions")
        measure(args)


if __name__ == "__main__":
    main()
