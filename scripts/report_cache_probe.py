#!/usr/bin/env python3
"""Read probe JSONL without importing torch/FlexKV; never treat missing data as PASS."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def analyze(rows):
    counts = Counter(row.get("event", "unknown") for row in rows)
    issues = [r for r in rows if r.get("event") == "coverage_gap" or
              r.get("status") in ("mismatch", "error", "io_failed", "transfer_failed")]
    puts = defaultdict(list)
    sends = defaultdict(set)
    for row in rows:
        if row.get("event") == "l3_page" and row.get("stage") == "put_complete" and row.get("io_success"):
            puts[(row["key"], row["bytes"])].append(row)
        if row.get("event") == "pd_page" and row.get("stage") == "send":
            sends[(row["room"], row["session"], row["dst_ptr"], row["bytes"])].add(row["sha256"])
    l3, pd = [], []
    received = set()
    for row in rows:
        if row.get("event") == "l3_page" and row.get("stage") == "get_complete":
            prior = [p for p in puts[(row["key"], row["bytes"])] if p["host"] == row["host"]
                     and p["wall_ns"] <= row["wall_ns"]]
            hashes = {p["sha256"] for p in prior}
            status = ("io_failed" if not row.get("io_success") else
                      "unverified_no_prior_put" if not hashes else
                      "unverified_multiple_versions" if len(hashes) > 1 else
                      "pass" if row["sha256"] in hashes else "mismatch")
            l3.append(dict(key=row["key"], status=status, sha256=row.get("sha256"),
                           expected=sorted(hashes), host=row["host"], pid=row["pid"]))
        if row.get("event") == "pd_page" and row.get("stage") == "receive":
            key = (row["room"], row["session"], row["dst_ptr"], row["bytes"])
            received.add(key)
            expected = sends[key]
            status = ("unverified_no_sender" if not expected else
                      "unverified_multiple_senders" if len(expected) > 1 else
                      "pass" if row["sha256"] in expected else "mismatch")
            pd.append(dict(room=row["room"], dst_ptr=row["dst_ptr"], state=row["state"],
                           status=status, sha256=row["sha256"], expected=sorted(expected)))
    for key in sends.keys() - received:
        pd.append(dict(room=key[0], session=key[1], dst_ptr=key[2], bytes=key[3],
                       status="unverified_no_receiver"))
    return dict(events=dict(counts), issues=issues, l3_checks=l3, pd_checks=pd,
                server_l3_checks=[r for r in rows if r.get('event') == 'l3_check'],
                note="No overall PASS: absent operations, ranks, groups or baselines are unverified. Hash equality is byte integrity, not model accuracy.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    rows = []
    for path in args.paths:
        # checks-*.jsonl mirrors verdicts already present in probe files.
        files = sorted(path.rglob("probe-*.jsonl")) if path.is_dir() else [path]
        for file in files:
            with file.open() as stream:
                for line_no, line in enumerate(stream, 1):
                    try:
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise ValueError("record is not an object")
                        rows.append(row)
                    except (ValueError, TypeError) as exc:
                        rows.append(dict(event="coverage_gap", file=str(file), line=line_no, error=str(exc)))
    result = analyze(rows)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    bad = result["issues"] or any(r["status"] != "pass" for r in
                                 result["l3_checks"] + result["pd_checks"] + result["server_l3_checks"])
    return 1 if bad or not rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
