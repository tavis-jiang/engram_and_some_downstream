#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/home/bcjiang/X-gram")
REPORTS = ROOT / "reports"
LOGS = ROOT / "logs"
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            step = int(row["step"])
            if not row.get("ce_loss") or not row.get("ppl"):
                continue
            out[step] = {
                "ce": float(row["ce_loss"]),
                "ppl": float(row["ppl"]),
            }
    return out


def load_engram_log(path: Path) -> dict[int, dict[str, float]]:
    text = path.read_text()
    step_re = re.compile(r"\[step=(\d+)/2385,epoch=1")
    ce_re = re.compile(r"train/CE loss=([0-9.,eE+-]+)")
    ppl_re = re.compile(r"train/PPL=([0-9.,eE+-]+)")
    spans = [(m.start(), int(m.group(1))) for m in step_re.finditer(text)]
    spans.append((len(text), -1))
    out: dict[int, dict[str, float]] = {}
    for i in range(len(spans) - 1):
        start, step = spans[i]
        end = spans[i + 1][0]
        block = text[start:end]
        ce = ce_re.search(block)
        ppl = ppl_re.search(block)
        if not ce or not ppl:
            continue
        out[step] = {
            "ce": float(ce.group(1).replace(",", "")),
            "ppl": float(ppl.group(1).replace(",", "")),
        }
    return out


baseline = load_csv(REPORTS / "baseline_full_metrics.csv")
xgram = load_csv(REPORTS / "xgram_full_metrics.csv")
engram = load_engram_log(LOGS / "20260718-205008_faircompare_engram_360m_rank0.log")

baseline_steps = [s for s in sorted(baseline) if s >= 1980]
xgram_steps = [s for s in sorted(xgram) if s >= 1980]
engram_steps = [s for s in sorted(engram) if s >= 2240]


fig, ax = plt.subplots(figsize=(8.8, 5.4))
ax.plot(baseline_steps, [baseline[s]["ce"] for s in baseline_steps], label="Baseline", color="#2f6db3", lw=1.8)
ax.plot(xgram_steps, [xgram[s]["ce"] for s in xgram_steps], label="X-gram", color="#c73a2a", lw=1.8)
ax.plot(engram_steps, [engram[s]["ce"] for s in engram_steps], label="Engram", color="#2f9a52", lw=1.8)
ax.set_xlabel("step")
ax.set_ylabel("train CE loss")
ax.set_title("Faircompare Tail CE Loss (final visible stage)")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "faircompare_tail_ce_loss_three_way.png", dpi=150)
plt.close(fig)


fig, ax = plt.subplots(figsize=(8.8, 5.4))
ax.plot(baseline_steps, [baseline[s]["ppl"] for s in baseline_steps], label="Baseline", color="#2f6db3", lw=1.8)
ax.plot(xgram_steps, [xgram[s]["ppl"] for s in xgram_steps], label="X-gram", color="#c73a2a", lw=1.8)
ax.plot(engram_steps, [engram[s]["ppl"] for s in engram_steps], label="Engram", color="#2f9a52", lw=1.8)
ax.set_xlabel("step")
ax.set_ylabel("train PPL")
ax.set_title("Faircompare Tail PPL (final visible stage)")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "faircompare_tail_ppl_three_way.png", dpi=150)
plt.close(fig)

print("wrote", OUT / "faircompare_tail_ce_loss_three_way.png")
print("wrote", OUT / "faircompare_tail_ppl_three_way.png")
