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
    rows: dict[int, dict[str, float]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            step = int(row["step"])
            entry: dict[str, float] = {}
            if row.get("ce_loss"):
                entry["ce"] = float(row["ce_loss"])
            if row.get("ppl"):
                try:
                    entry["ppl"] = float(row["ppl"])
                except ValueError:
                    pass
            if entry:
                rows[step] = entry
    return rows


def load_engram_log(path: Path) -> dict[int, dict[str, float]]:
    text = path.read_text()
    step_re = re.compile(r"\[step=(\d+)/2385,epoch=1")
    ce_re = re.compile(r"train/CE loss=([0-9.,eE+-]+)")
    ppl_re = re.compile(r"train/PPL=([0-9.,eE+-]+)")
    spans = [(m.start(), int(m.group(1))) for m in step_re.finditer(text)]
    spans.append((len(text), -1))
    rows: dict[int, dict[str, float]] = {}
    for i in range(len(spans) - 1):
        start, step = spans[i]
        end = spans[i + 1][0]
        block = text[start:end]
        ce = ce_re.search(block)
        ppl = ppl_re.search(block)
        if not ce:
            continue
        entry = {"ce": float(ce.group(1).replace(",", ""))}
        if ppl:
            entry["ppl"] = float(ppl.group(1).replace(",", ""))
        rows[step] = entry
    return rows


def plot_metric(series_map: dict[str, dict[int, dict[str, float]]], key: str, out_name: str, title: str, ylabel: str, logy: bool = False) -> None:
    colors = {
        "Baseline": "#2f6db3",
        "X-gram": "#c73a2a",
        "Engram": "#2f9a52",
    }
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, records in series_map.items():
        xs = sorted(s for s, v in records.items() if key in v)
        ys = [records[s][key] for s in xs]
        ax.plot(xs, ys, label=name, color=colors[name], lw=1.7)
        if xs and ys:
            last_x, last_y = xs[-1], ys[-1]
            label = f"{name} {last_y:.2f}" if key == "ppl" else f"{name} {last_y:.3f}"
            ax.scatter([last_x], [last_y], color=colors[name], s=24, zorder=3)
            ax.text(last_x, last_y, f"  {label}", color=colors[name], va="center", fontsize=9)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    if logy:
        ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / out_name, dpi=150)
    plt.close(fig)


series_map = {
    "Baseline": load_csv(REPORTS / "baseline_full_metrics.csv"),
    "X-gram": load_csv(REPORTS / "xgram_full_metrics.csv"),
    "Engram": load_engram_log(LOGS / "20260718-205008_faircompare_engram_360m_rank0.log"),
}

plot_metric(
    series_map,
    key="ce",
    out_name="faircompare_full_ce_loss_three_way_v2.png",
    title="Faircompare Full Training CE Loss (Baseline/X-gram CSV + Engram log)",
    ylabel="train CE loss",
)

plot_metric(
    series_map,
    key="ppl",
    out_name="faircompare_full_ppl_three_way_v2.png",
    title="Faircompare Full Training PPL (Baseline/X-gram CSV + Engram log)",
    ylabel="train PPL",
    logy=True,
)

print("wrote", OUT / "faircompare_full_ce_loss_three_way_v2.png")
print("wrote", OUT / "faircompare_full_ppl_three_way_v2.png")
