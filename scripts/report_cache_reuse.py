"""Offline block-level GET attribution; never equate lookup totals with prefill.

Requires put_trace.cache_details=true from a fresh cache. Optional SGLang trace
adds source/session/input digests. Unknown or incomplete evidence stays unknown.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path


def read_rows(paths, events):
    rows, invalid = [], 0
    for path in paths:
        with Path(path).open() as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                except ValueError:
                    invalid += 1
                    continue
                if row.get("event") in events:
                    rows.append(dict(row, file=str(path), line=line_number))
    return rows, invalid


def task_key(row):
    return row["host"], f"{row['run']}/{row['pid']}/{row['task']}"


def analyze(rows, sglang_rows=()):
    identities, inputs, links = {}, {}, {}
    for row in sglang_rows:
        key = row["host"], row.get("rid")
        if row["event"] == "request_identity": identities[key] = row
        elif row["event"] == "request_input": inputs[key] = row
        elif row["event"] == "flexkv_task_link" and row.get("flexkv_request"):
            links[(row["host"], row["flexkv_request"])] = key
    matches = {task_key(r): r for r in rows if r["event"] == "cache_match"}
    if not matches:
        raise ValueError("No cache_match records; enable put_trace.cache_details and rerun from a fresh cache")
    plans = {task_key(r): r for r in rows if r["event"] == "h2d_diag"}
    launched = {task_key(r) for r in rows if r["event"] == "graph_launch"}
    children = {task_key(r): r.get("children", []) for r in rows if r["event"] == "planned"}
    # Layerwise GETs launch a merged parent graph, not each lookup child.
    pending = list(launched)
    while pending:
        parent = pending.pop()
        for request in children.get(parent, []):
            child = parent[0], request
            if child not in launched:
                launched.add(child)
                pending.append(child)
    terminal = {task_key(r): r.get("status") for r in rows if r["event"] == "task_end"}
    admitted = Counter()
    for row in rows:
        if row["event"] == "h2d_layerwise_launch" and row.get("tp_rank") == 0:
            admitted[(row["host"], row["rid"])] += row["slots"]
    history, previous_source, lookups = defaultdict(dict), {}, []
    totals, launched_totals = Counter(), Counter()
    incomplete_evictions = 0
    ordered = sorted(rows, key=lambda r: (r["host"], r["run"], r["ns"]))
    for row in ordered:
        event = row["event"]
        scope = row["host"], row["run"], row["pid"]
        state = history[scope]
        hashes = row.get("block_hashes", [])
        if event == "cache_reset":
            for key in state: state[key] = "reset"
        elif event == "cache_evict" and row.get("scope") == "full":
            if row.get("hashes_available") is False:
                incomplete_evictions += 1
            for value in hashes: state[(row["tier"], value)] = "evicted"
        elif event == "cache_ready":
            for value in hashes: state[(row["tier"], value)] = "ready"
        elif event == "cache_store_plan":
            match = matches.get(task_key(row))
            if match:
                lo = row["block_start"] + row["skipped_blocks"]
                for value in match["block_hashes"][lo:lo+row["new_blocks"]]:
                    state[("cpu", value)] = "submitted"
        elif event == "cache_match":
            key = task_key(row)
            for tier, result in row["tiers"].items():
                # A peer match does not describe this process's local index.
                if result.get("matched_pos") not in (None, "local"):
                    continue
                for value in hashes[:result["ready_blocks"]]: state[(tier, value)] = "ready"
            if row["operation"] != "get": continue
            rid_key = links.get(key)
            identity = identities.get(rid_key, {})
            source = identity.get("source_request_id")
            source_key = (scope, source) if source else None
            previous = previous_source.get(source_key)
            ready = max((v["ready_blocks"] for v in row["tiers"].values()), default=0)
            matched = max((v["matched_blocks"] for v in row["tiers"].values()), default=0)
            plan = plans.get(key)
            planned_end = row["block_start"] + (plan["num_gpu_blocks_to_transfer"] if plan else 0)
            reasons = Counter()
            for index in range(row["block_start"], row["block_end"]):
                value = hashes[index]
                if plan is None: reason = "plan_missing"
                elif index < planned_end: reason = "planned_restore"
                elif index < ready: reason = "ready_not_planned"
                elif index < matched: reason = "unready_prefix"
                else:
                    states = {state.get((tier, value)) for tier in row["tiers"]}
                    if "evicted" in states: reason = "evicted"
                    elif "submitted" in states: reason = "store_unpublished"
                    elif "reset" in states: reason = "cache_reset"
                    elif "ready" in states: reason = "unexplained_missing"
                    elif previous is not None and index < len(previous) and previous[index] != value:
                        reason = "prefix_changed"
                    else: reason = "no_store_observed"
                reasons[reason] += row["tokens_per_block"]
            if source_key is not None: previous_source[source_key] = hashes
            totals.update(reasons)
            if key in launched: launched_totals.update(reasons)
            lookups.append(dict(
                request=key[1], host=key[0], rid=rid_key[1] if rid_key else None,
                source_request_id=source, session=identity.get("source_session_id"),
                source_start_time=identity.get("source_start_time"),
                input_digest=inputs.get(rid_key, {}).get("token_digest"),
                layerwise_admitted_tokens_for_rid=admitted.get(rid_key, 0),
                file=row["file"], line=row["line"],
                launched=key in launched, terminal_status=terminal.get(key),
                requested_tokens=(row["block_end"]-row["block_start"])*row["tokens_per_block"],
                **reasons,
            ))
    return dict(
        lookup_count=len(lookups), lookup_tokens_by_reason=dict(totals),
        launched_lookup_tokens_by_reason=dict(launched_totals),
        evictions_without_hashes=incomplete_evictions,
        layerwise_admitted_tokens=sum(admitted.values()),
        note="Lookup-level counts include retries; planned_restore is not GPU completion or actual prefill saved. "
             "no_store_observed needs a complete capture from an empty cache. "
             "SWA/peer/remote restrictions can appear as ready_not_planned.",
    ), lookups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", nargs="+", required=True)
    parser.add_argument("--sglang-trace", nargs="*", default=[])
    parser.add_argument("--output", required=True, help="Output prefix for .json and .csv")
    args = parser.parse_args()
    rows, invalid = read_rows(args.trace, {
        "cache_match", "cache_store_plan", "cache_ready", "cache_evict", "cache_reset",
        "h2d_diag", "h2d_layerwise_launch", "planned", "graph_launch", "task_end",
    })
    sglang, invalid_sglang = read_rows(args.sglang_trace, {
        "request_identity", "request_input", "flexkv_task_link",
    })
    summary, lookups = analyze(rows, sglang)
    summary["invalid_lines"] = invalid + invalid_sglang
    prefix = Path(args.output)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    fields = list(dict.fromkeys(key for row in lookups for key in row))
    with prefix.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(lookups)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
