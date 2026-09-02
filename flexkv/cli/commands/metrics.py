"""flexkv metrics - Scrape and display local FlexKV Prometheus metrics."""

import argparse
import json as jsonlib
import re
import sys
import urllib.request
import urllib.error

from flexkv.cli.commands.base import BaseCommand, print_table, print_json


def _fetch(url, timeout=3.0):
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def _parse_prometheus(raw):
    metrics = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)((?:\{[^}]*\})?)\s+(\S+)", line)
        if not m:
            continue
        name, labels_str, value_str = m.group(1), m.group(2), m.group(3)
        labels = {}
        if labels_str:
            for lm in re.finditer(r'(\w+)="([^"]*)"', labels_str):
                labels[lm.group(1)] = lm.group(2)
        try:
            value = float(value_str)
        except ValueError:
            value = value_str
        metrics.append({"name": name, "labels": labels, "value": value})
    return metrics


def _fmt_value(v):
    if isinstance(v, float):
        if v == int(v):
            return str(int(v))
        return "{:.4f}".format(v)
    return str(v)


def _fmt_labels(labels):
    if not labels:
        return ""
    parts = []
    for k, v in sorted(labels.items()):
        parts.append("{}={}".format(k, v))
    return "{" + ",".join(parts) + "}"


class MetricsCommand(BaseCommand):
    def name(self):
        return "metrics"

    def help(self):
        return "Scrape and display local FlexKV Prometheus metrics."

    def add_arguments(self, parser):
        parser.add_argument(
            "--py-port", type=int, default=None,
            help="Override Python metrics port (default: FLEXKV_PY_METRICS_PORT or 8080).",
        )
        parser.add_argument(
            "--cpp-port", type=int, default=None,
            help="Override C++ metrics port (default: FLEXKV_CPP_METRICS_PORT or 8081).",
        )

    def execute(self, args):
        from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV

        py_port = getattr(args, "py_port", None) or getattr(
            GLOBAL_CONFIG_FROM_ENV, "py_metrics_port", 8080
        )
        cpp_port = getattr(args, "cpp_port", None) or getattr(
            GLOBAL_CONFIG_FROM_ENV, "cpp_metrics_port", 8081
        )

        sections = []
        for label, port in [("Python", py_port), ("C++", cpp_port)]:
            url = "http://127.0.0.1:{}/metrics".format(port)
            raw = _fetch(url)
            metrics = _parse_prometheus(raw)
            sections.append({
                "runtime": label,
                "url": url,
                "reachable": bool(raw),
                "metrics": metrics,
            })

        if getattr(args, "format", "terminal") == "json":
            print_json(sections)
            return
        if getattr(args, "quiet", False):
            return

        has_data = False
        for sec in sections:
            title = "FlexKV {} Metrics ({})".format(sec["runtime"], sec["url"])
            if not sec["reachable"]:
                print("\n{}\n{}\n  (endpoint not reachable)\n".format(title, "=" * 48))
                continue
            if not sec["metrics"]:
                print("\n{}\n{}\n  (no metrics)\n".format(title, "=" * 48))
                continue
            has_data = True
            rows = []
            for m in sec["metrics"]:
                label_str = _fmt_labels(m["labels"])
                name = m["name"] + label_str if label_str else m["name"]
                rows.append((name, _fmt_value(m["value"])))
            print_table(title, rows)

        if not has_data:
            print(
                "No metrics endpoints reachable. "
                "Is FLEXKV_ENABLE_METRICS=1 set and the runtime running?",
                file=sys.stderr,
            )
            sys.exit(1)
