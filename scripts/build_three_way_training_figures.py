#!/usr/bin/env python3
"""Build Baseline vs X-gram vs Engram training comparison figures."""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/home/bcjiang/X-gram")
LOG_DIR = ROOT / "logs"
OUT_DIR = ROOT / "docs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STEP_RE = re.compile(r"\[step=(\d+)/2385,epoch=1")
CE_RE = re.compile(r"train/CE loss=([0-9.,eE+-]+)")
PPL_RE = re.compile(r"train/PPL=([0-9.,eE+-]+)")

MODELS = {
    "Baseline": {
        "pattern": ["*faircompare_baseline_360m_rank0.log"],
        "color": "#1f77b4",
    },
    "X-gram": {
        "pattern": ["*faircompare_xgram_360m_rank0.log"],
        "color": "#d62728",
    },
    "Engram": {
        "pattern": ["*faircompare_engram_360m_rank0.log"],
        "color": "#2ca02c",
    },
}


def to_float(value: str) -> float:
    return float(value.replace(",", ""))


def parse_log(path: Path) -> dict[int, dict[str, float]]:
    text = path.read_text()
    matches = list(STEP_RE.finditer(text))
    if not matches:
        return {}
    spans = [(m.start(), int(m.group(1))) for m in matches]
    spans.append((len(text), -1))

    records: dict[int, dict[str, float]] = {}
    for i in range(len(spans) - 1):
        start, step = spans[i]
        end = spans[i + 1][0]
        block = text[start:end]
        ce = CE_RE.search(block)
        ppl = PPL_RE.search(block)
        if not ce:
            continue
        records[step] = {"ce_loss": to_float(ce.group(1))}
        if ppl:
            records[step]["ppl"] = to_float(ppl.group(1))
    return records


def collect(patterns: str | list[str]) -> dict[int, dict[str, float]]:
    merged: dict[int, dict[str, float]] = {}
    if isinstance(patterns, str):
        patterns = [patterns]
    for pattern in patterns:
        for path in sorted(LOG_DIR.glob(pattern)):
            merged.update(parse_log(path))
    return merged


def series(records: dict[int, dict[str, float]], key: str) -> tuple[list[int], list[float]]:
    steps = sorted(s for s, v in records.items() if key in v)
    return steps, [records[s][key] for s in steps]


all_records = {name: collect(cfg["pattern"]) for name, cfg in MODELS.items()}


fig, ax = plt.subplots(figsize=(8.6, 5.3))
for name, cfg in MODELS.items():
    xs, ys = series(all_records[name], "ce_loss")
    ax.plot(xs, ys, label=name, color=cfg["color"], lw=1.7)
ax.set_xlabel("step")
ax.set_ylabel("train CE loss")
ax.set_title("Training CE Loss: Baseline vs X-gram vs Engram")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "ce_loss_three_way.png", dpi=140)
plt.close(fig)


fig, ax = plt.subplots(figsize=(8.6, 5.3))
for name, cfg in MODELS.items():
    xs, ys = series(all_records[name], "ppl")
    ax.plot(xs, ys, label=name, color=cfg["color"], lw=1.7)
ax.set_yscale("log")
ax.set_xlabel("step")
ax.set_ylabel("train PPL (log scale)")
ax.set_title("Training PPL: Baseline vs X-gram vs Engram")
ax.grid(alpha=0.3, which="both")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "ppl_three_way.png", dpi=140)
plt.close(fig)


for name in MODELS:
    records = all_records[name]
    if not records:
        continue
    last = max(records)
    metrics = records[last]
    print(name, last, metrics.get("ce_loss"), metrics.get("ppl"))
