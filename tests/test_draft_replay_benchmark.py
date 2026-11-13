"""Reject misleading benchmark comparisons and retain verify-weighted metrics."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "draft_replay_benchmark", Path(__file__).parents[1] / "scripts/bench_sglang_draft_replay.py"
)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def _row(index, steps):
    return dict(
        id=index,
        phase="cold",
        repeat=0,
        input_sha256=str(index),
        valid=True,
        completion_tokens=12,
        verify_ct=steps,
        correct_drafts=12 - steps,
        proposed_drafts=steps * 5,
        client_elapsed_s=1,
        output_ids=[index] * 12,
    )


def _run(root, rows):
    root.mkdir()
    arguments = dict(
        page_size=256, key_mode="token", reprefill_tail=128, output_tokens=12, repeats=1, tiers=[], churn=True
    )
    (root / "manifest.json").write_text(json.dumps(dict(inputs_sha256="frozen", arguments=arguments)))
    (root / "requests.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (root / "summary.json").write_text(json.dumps(dict(complete=True, summaries={"cold": bench.summarize(rows)})))


def test_weighted_length_does_not_average_per_request_ratios():
    summary = bench.summarize([_row(0, 2), _row(1, 6)])
    assert summary["accept_length"] == 24 / 8
    assert summary["mean_request_accept_length"] == (6 + 2) / 2


def test_paired_bootstrap_retains_inputs_and_reports_output_changes(tmp_path):
    _run(tmp_path / "before", [_row(0, 6), _row(1, 6)])
    _run(tmp_path / "after", [_row(0, 3), {**_row(1, 3), "output_ids": [9] * 12}])
    args = SimpleNamespace(
        before=tmp_path / "before", after=tmp_path / "after", output=tmp_path / "comparison.json", bootstrap=100
    )
    bench.compare(args)
    phase = json.loads(args.output.read_text())["phases"]["cold"]
    assert phase["accept_length_delta"] == 2
    assert phase["accept_length_delta_prompt_bootstrap_95pct"] == [2, 2]
    assert phase["identical_output_pairs"] == 1


@pytest.mark.parametrize(
    "failure", ["incomplete", "different_inputs", "different_workload", "different_dtype", "duplicates"]
)
def test_unmatched_or_incomplete_runs_cannot_produce_a_comparison(tmp_path, failure):
    for arm in ("before", "after"):
        _run(tmp_path / arm, [_row(0, 3)])
    root = tmp_path / "after"
    if failure == "duplicates":
        path = root / "requests.jsonl"
        path.write_text(path.read_text() * 2)
    else:
        path = root / ("summary.json" if failure == "incomplete" else "manifest.json")
        data = json.loads(path.read_text())
        if failure == "incomplete":
            data["complete"] = False
        elif failure == "different_inputs":
            data["inputs_sha256"] = "other"
        elif failure == "different_dtype":
            data["server_info"] = {"kv_cache_dtype": "bf16"}
        else:
            data["arguments"]["reprefill_tail"] = 0
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        bench.compare(
            SimpleNamespace(before=tmp_path / "before", after=root, output=tmp_path / "comparison.json", bootstrap=100)
        )
