#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
REPORTS = ROOT / "reports"
FIGS = ROOT / "docs" / "25_Engram最新Faircompare对比结果_20260730" / "images"
FIGS.mkdir(parents=True, exist_ok=True)


FINAL = {
    "Baseline": {
        "global": 0.3944,
        "mmlu": 0.2504,
        "sciq": 0.6480,
        "arc_c": 0.2321,
        "arc_e": 0.4211,
        "boolq": 0.6095,
        "commonsenseqa": 0.2539,
        "csqa_val": 0.2981,
        "hellaswag": 0.2766,
        "openbookqa": 0.2740,
        "piqa": 0.5647,
        "socialiqa": 0.3910,
        "winogrande": 0.5138,
    },
    "X-gram": {
        "global": 0.4167,
        "mmlu": 0.2634,
        "sciq": 0.7010,
        "arc_c": 0.2466,
        "arc_e": 0.4509,
        "boolq": 0.6190,
        "commonsenseqa": 0.2752,
        "csqa_val": 0.3260,
        "hellaswag": 0.3020,
        "openbookqa": 0.2960,
        "piqa": 0.6045,
        "socialiqa": 0.4058,
        "winogrande": 0.5099,
    },
    "Engram 91976": {
        "global": 0.4073,
        "mmlu": 0.2427,
        "sciq": 0.6820,
        "arc_c": 0.2398,
        "arc_e": 0.4544,
        "boolq": 0.6055,
        "commonsenseqa": 0.2768,
        "csqa_val": 0.3342,
        "hellaswag": 0.2870,
        "openbookqa": 0.2700,
        "piqa": 0.5713,
        "socialiqa": 0.3987,
        "winogrande": 0.5249,
    },
    "Engram 2g3g": {
        "global": 0.4142,
        "mmlu": 0.2451,
        "sciq": 0.6840,
        "arc_c": 0.2483,
        "arc_e": 0.5158,
        "boolq": 0.5985,
        "commonsenseqa": 0.2973,
        "csqa_val": 0.3219,
        "hellaswag": 0.2923,
        "openbookqa": 0.2740,
        "piqa": 0.5892,
        "socialiqa": 0.3905,
        "winogrande": 0.5138,
    },
    "Old Engram": {
        "global": 0.3968,
        "mmlu": 0.2447,
        "sciq": 0.6590,
        "arc_c": 0.2457,
        "arc_e": 0.4421,
        "boolq": 0.5804,
        "commonsenseqa": 0.2686,
        "csqa_val": 0.3071,
        "hellaswag": 0.2860,
        "openbookqa": 0.2680,
        "piqa": 0.5658,
        "socialiqa": 0.3966,
        "winogrande": 0.4972,
    },
    "Engram 91067": {
        "global": 0.3980,
        "mmlu": 0.2563,
        "sciq": 0.6410,
        "arc_c": 0.2381,
        "arc_e": 0.4175,
        "boolq": 0.6031,
        "commonsenseqa": 0.2621,
        "csqa_val": 0.3170,
        "hellaswag": 0.2859,
        "openbookqa": 0.2700,
        "piqa": 0.5789,
        "socialiqa": 0.4012,
        "winogrande": 0.5051,
    },
    "Engram e2": {
        "global": 0.3985,
        "mmlu": 0.2371,
        "sciq": 0.6510,
        "arc_c": 0.2346,
        "arc_e": 0.4684,
        "boolq": 0.5361,
        "commonsenseqa": 0.2899,
        "csqa_val": 0.3022,
        "hellaswag": 0.2877,
        "openbookqa": 0.2800,
        "piqa": 0.5979,
        "socialiqa": 0.3915,
        "winogrande": 0.5051,
    },
    "Engram e1": {
        "global": 0.3939,
        "mmlu": 0.2333,
        "sciq": 0.6410,
        "arc_c": 0.2278,
        "arc_e": 0.4211,
        "boolq": 0.5725,
        "commonsenseqa": 0.2654,
        "csqa_val": 0.3342,
        "hellaswag": 0.2907,
        "openbookqa": 0.2640,
        "piqa": 0.5696,
        "socialiqa": 0.3966,
        "winogrande": 0.5107,
    },
    "Engram 91347": {
        "global": 0.4016,
        "mmlu": 0.2522,
        "sciq": 0.6510,
        "arc_c": 0.2457,
        "arc_e": 0.4281,
        "boolq": 0.6086,
        "commonsenseqa": 0.2727,
        "csqa_val": 0.3104,
        "hellaswag": 0.2856,
        "openbookqa": 0.2700,
        "piqa": 0.5778,
        "socialiqa": 0.4053,
        "winogrande": 0.5122,
    },
}

PARAMS = {
    "Baseline": (401.2, 354.0),
    "X-gram": (694.5, 354.0),
    "Engram 91976": (697.9, 650.7),
    "Engram 2g3g": (704.4, 393.5),
    "Engram 91067": (655.8, 377.7),
    "Engram e1": (689.8, 374.7),
    "Engram e2": (675.1, 373.7),
    "Engram 91347": (520.0, 369.8),
    "Engram v2": (697.9, 650.7),
}

INTERMEDIATE = {
    "Old step596 Jul19": {"step": 596, "global": 0.3471, "mmlu": 0.2352, "sciq": 0.4630},
    "Old step596 Jul23": {"step": 596, "global": 0.3453, "mmlu": 0.2356, "sciq": 0.4590},
    "91347 step596": {"step": 596, "global": 0.3431, "mmlu": 0.2470, "sciq": 0.4430},
    "Old step1192 Jul19": {"step": 1192, "global": 0.3864, "mmlu": 0.2466, "sciq": 0.5960},
    "Old step1192 Jul23": {"step": 1192, "global": 0.3768, "mmlu": 0.2461, "sciq": 0.5800},
    "91347 step1192": {"step": 1192, "global": 0.3890, "mmlu": 0.2484, "sciq": 0.6080},
    "91347 step2385": {"step": 2385, "global": 0.4016, "mmlu": 0.2522, "sciq": 0.6510},
}


COLORS = {
    "Baseline": "#6b7280",
    "X-gram": "#2563eb",
    "Engram 91976": "#dc2626",
    "Engram 2g3g": "#16a34a",
    "Old Engram": "#64748b",
    "Engram 91067": "#f59e0b",
    "Engram e1": "#a855f7",
    "Engram e2": "#ec4899",
    "Engram 91347": "#0f766e",
    "Engram v2": "#dc2626",
}

TRAINING = {
    "Baseline": {
        "sources": [REPORTS / "baseline_full_metrics.csv"],
        "kind": "csv",
        "style": "-",
        "width": 1.8,
    },
    "X-gram": {
        "sources": [REPORTS / "xgram_full_metrics.csv"],
        "kind": "csv",
        "style": "-",
        "width": 2.0,
    },
    "Engram 2g3g": {
        "sources": [LOGS / "20260724-125126_faircompare_engram_2g3g_xgrammatch_360m_rank0.log"],
        "kind": "log",
        "style": "-",
        "width": 2.0,
    },
    "Engram 91067": {
        "sources": [LOGS / "20260727-115348_faircompare_engram_2gram_xgrammatch_rerun_360m_rank0.log"],
        "kind": "log",
        "style": "--",
        "width": 1.8,
    },
    "Engram e1": {
        "sources": [LOGS / "20260727-124252_engram_faircompare_e1_28l_2g_d384_b49152_rank0.log"],
        "kind": "log",
        "style": ":",
        "width": 1.6,
    },
    "Engram e2": {
        "sources": [LOGS / "20260727-124252_engram_faircompare_e2_20l_2g_d512_b49152_rank0.log"],
        "kind": "log",
        "style": ":",
        "width": 1.6,
    },
    "Engram 91347": {
        "sources": [LOGS / "20260729-121510_engram_improved_v1_360m_rank0.log"],
        "kind": "log",
        "style": "-.",
        "width": 2.2,
    },
    "Engram 91976": {
        "sources": [LOGS / "20260731-132524_faircompare_engram_vpath_xgrammatch_360m_rank0.log"],
        "kind": "log",
        "style": "-",
        "width": 2.4,
    },
    "Engram v2": {
        "sources": [
            LOGS / "20260729-142210_engram_v2_360m_rank0.log",
            LOGS / "20260729-173857_engram_v2_360m_rank0.log",
        ],
        "kind": "log",
        "style": "-.",
        "width": 2.2,
    },
}

STEP_RE = re.compile(r"\[step=(\d+)/2385,epoch=1")
CE_RE = re.compile(r"train/CE loss=([0-9.,eE+-]+)")
PPL_RE = re.compile(r"train/PPL=([0-9.,eE+-]+)")


def save(fig: plt.Figure, name: str) -> None:
    path = FIGS / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(path)


def parse_float(value: str) -> float:
    return float(value.replace(",", ""))


def load_training_csv(path: Path) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            step = int(row["step"])
            entry: dict[str, float] = {}
            if row.get("ce_loss"):
                entry["ce"] = parse_float(row["ce_loss"])
            if row.get("ppl"):
                try:
                    entry["ppl"] = parse_float(row["ppl"])
                except ValueError:
                    pass
            if entry:
                rows[step] = entry
    return rows


def parse_training_log(path: Path) -> dict[int, dict[str, float]]:
    text = path.read_text()
    matches = list(STEP_RE.finditer(text))
    if not matches:
        return {}

    spans = [(match.start(), int(match.group(1))) for match in matches]
    spans.append((len(text), -1))
    rows: dict[int, dict[str, float]] = {}
    for idx in range(len(spans) - 1):
        start, step = spans[idx]
        end = spans[idx + 1][0]
        block = text[start:end]
        ce = CE_RE.search(block)
        ppl = PPL_RE.search(block)
        if not ce:
            continue
        entry = {"ce": parse_float(ce.group(1))}
        if ppl:
            entry["ppl"] = parse_float(ppl.group(1))
        rows[step] = entry
    return rows


def load_training_series() -> dict[str, dict[int, dict[str, float]]]:
    all_rows: dict[str, dict[int, dict[str, float]]] = {}
    for name, cfg in TRAINING.items():
        rows: dict[int, dict[str, float]] = {}
        for source in cfg["sources"]:
            path = Path(source)
            if not path.exists():
                print(f"missing training source for {name}: {path}")
                continue
            parsed = load_training_csv(path) if cfg["kind"] == "csv" else parse_training_log(path)
            rows.update(parsed)
        all_rows[name] = dict(sorted(rows.items()))
    return all_rows


def last_metric(rows: dict[int, dict[str, float]]) -> tuple[int | None, float | None, float | None]:
    if not rows:
        return None, None, None
    step = max(rows)
    return step, rows[step].get("ce"), rows[step].get("ppl")


def draw_training_curve(
    training: dict[str, dict[int, dict[str, float]]],
    metric: str,
    name: str,
    ylabel: str,
    *,
    min_step: int | None = None,
    rolling_window: int | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.2))
    all_xs: list[int] = []
    endpoints: list[tuple[int, float, str, str]] = []
    for model_name, rows in training.items():
        xs = sorted(
            step
            for step, row in rows.items()
            if metric in row and (min_step is None or step >= min_step)
        )
        if not xs:
            continue
        ys = [rows[step][metric] for step in xs]
        if rolling_window and len(ys) >= rolling_window:
            kernel = np.ones(rolling_window) / rolling_window
            ys = np.convolve(ys, kernel, mode="valid").tolist()
            xs = xs[rolling_window - 1 :]
        cfg = TRAINING[model_name]
        color = COLORS.get(model_name, "#999999")
        label = model_name if xs[-1] >= 2380 else f"{model_name} (partial)"
        ax.plot(
            xs,
            ys,
            label=label,
            color=color,
            lw=float(cfg["width"]),
            linestyle=str(cfg["style"]),
            alpha=0.90,
        )
        ax.scatter([xs[-1]], [ys[-1]], color=color, s=28, zorder=3)
        value = f"{ys[-1]:.3f}" if metric == "ce" else f"{ys[-1]:.2f}"
        endpoints.append((xs[-1], ys[-1], color, value))
        all_xs.extend(xs)

    if all_xs:
        x_min = min(all_xs)
        x_max = max(all_xs)
        left = min_step if min_step is not None else max(0, x_min - 25)
        ax.set_xlim(left, x_max + max(85, int((x_max - left) * 0.06)))

    groups: list[list[tuple[int, float, str, str]]] = []
    for endpoint in sorted(endpoints, key=lambda item: item[0]):
        if groups and abs(groups[-1][0][0] - endpoint[0]) <= 35:
            groups[-1].append(endpoint)
        else:
            groups.append([endpoint])
    for group in groups:
        ordered = sorted(group, key=lambda item: item[1])
        if len(ordered) == 1:
            offsets = [0.0]
        else:
            offsets = np.linspace(-18, 18, len(ordered))
        for (x, yv, color, value), y_offset in zip(ordered, offsets):
            ax.annotate(
                value,
                xy=(x, yv),
                xytext=(8, y_offset),
                textcoords="offset points",
                color=color,
                va="center",
                fontsize=8,
                clip_on=False,
            )

    suffix_parts = []
    if min_step is not None:
        suffix_parts.append(f"step >= {min_step}")
    if rolling_window:
        suffix_parts.append(f"rolling-{rolling_window} logs")
    suffix = "" if not suffix_parts else f" ({', '.join(suffix_parts)})"
    ax.set_title(f"Training {ylabel} Comparison{suffix}", fontsize=15, weight="bold")
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    if metric == "ppl":
        ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncols=2, fontsize=8)
    save(fig, name)


def draw_training_curves() -> dict[str, tuple[int | None, float | None, float | None]]:
    training = load_training_series()
    for model_name, rows in training.items():
        step, ce, ppl = last_metric(rows)
        print(f"{model_name}: step={step} ce={ce} ppl={ppl}")

    draw_training_curve(training, "ce", "engram_25_full_ce_loss_20260730.png", "CE Loss")
    draw_training_curve(training, "ppl", "engram_25_full_ppl_20260730.png", "PPL")
    draw_training_curve(training, "ce", "engram_25_tail_ce_loss_20260730.png", "CE Loss", min_step=500)
    draw_training_curve(training, "ppl", "engram_25_tail_ppl_20260730.png", "PPL", min_step=500)
    draw_training_curve(
        training,
        "ce",
        "engram_25_tail_ce_loss_smooth_20260730.png",
        "CE Loss",
        min_step=500,
        rolling_window=5,
    )
    draw_training_curve(
        training,
        "ppl",
        "engram_25_tail_ppl_smooth_20260730.png",
        "PPL",
        min_step=500,
        rolling_window=5,
    )
    return {model_name: last_metric(rows) for model_name, rows in training.items()}


def draw_param_comparison() -> None:
    names = list(PARAMS)
    total = [PARAMS[name][0] for name in names]
    non_emb = [PARAMS[name][1] for name in names]
    x = np.arange(len(names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12, 5.6))
    bars1 = ax.bar(x - width / 2, total, width, label="Total params", color="#60a5fa")
    bars2 = ax.bar(x + width / 2, non_emb, width, label="Non-embedding params", color="#34d399")
    ax.set_title("Parameter Comparison", fontsize=15, weight="bold")
    ax.set_ylabel("Parameters (M)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=2)
    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 8, f"{height:.0f}", ha="center", va="bottom", fontsize=8)
    save(fig, "engram_25_param_comparison_20260730.png")


def draw_final_summary() -> None:
    names = ["Baseline", "X-gram", "Engram 91976", "Old Engram", "Engram 2g3g", "Engram 91067", "Engram 91347", "Engram e2", "Engram e1"]
    metrics = [("global", "Global Avg"), ("mmlu", "MMLU Avg"), ("sciq", "SciQ")]
    x = np.arange(len(metrics))
    width = 0.08

    fig, ax = plt.subplots(figsize=(12, 5.6))
    offsets = (np.arange(len(names)) - (len(names) - 1) / 2) * width
    for offset, name in zip(offsets, names):
        values = [FINAL[name][key] for key, _ in metrics]
        ax.bar(x + offset, values, width, label=name, color=COLORS.get(name, "#999999"))
    ax.set_title("Final step2385 Summary", fontsize=15, weight="bold")
    ax.set_ylim(0.20, 0.74)
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=4, fontsize=8)
    save(fig, "engram_25_final_summary_20260730.png")


def draw_final_downstream_tasks() -> None:
    tasks = [
        ("hellaswag", "HellaSwag"),
        ("arc_e", "ARC-E"),
        ("arc_c", "ARC-C"),
        ("mmlu", "MMLU"),
        ("boolq", "BoolQ"),
        ("openbookqa", "OpenBookQA"),
        ("socialiqa", "SocialIQA"),
        ("sciq", "SciQ"),
        ("piqa", "PIQA"),
        ("commonsenseqa", "CSQA"),
        ("csqa_val", "CSQA Val"),
        ("winogrande", "Winogrande"),
    ]
    names = ["Baseline", "X-gram", "Engram 91976", "Old Engram", "Engram 2g3g", "Engram 91067", "Engram 91347", "Engram e2", "Engram e1"]
    y = np.arange(len(tasks))
    height = 0.082

    fig, ax = plt.subplots(figsize=(13, 8.4))
    offsets = (np.arange(len(names)) - (len(names) - 1) / 2) * height
    for offset, name in zip(offsets, names):
        values = [FINAL[name][key] for key, _ in tasks]
        ax.barh(y + offset, values, height, label=name, color=COLORS.get(name, "#999999"))
    ax.set_title("Final step2385 Downstream Tasks", fontsize=15, weight="bold")
    ax.set_xlabel("Accuracy")
    ax.set_yticks(y)
    ax.set_yticklabels([label for _, label in tasks])
    ax.set_xlim(0.18, 0.73)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, ncols=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.invert_yaxis()
    save(fig, "engram_25_final_downstream_tasks_20260730.png")


def draw_intermediate_progression() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharex=True)
    metrics = [("global", "Global Avg"), ("mmlu", "MMLU Avg"), ("sciq", "SciQ")]
    series = {
        "Old Engram Jul19": {
            596: INTERMEDIATE["Old step596 Jul19"],
            1192: INTERMEDIATE["Old step1192 Jul19"],
        },
        "Old Engram Jul23": {
            596: INTERMEDIATE["Old step596 Jul23"],
            1192: INTERMEDIATE["Old step1192 Jul23"],
        },
        "91347 improved-v1": {
            596: INTERMEDIATE["91347 step596"],
            1192: INTERMEDIATE["91347 step1192"],
            2385: INTERMEDIATE["91347 step2385"],
        },
    }
    colors = ["#94a3b8", "#64748b", "#0f766e"]
    for ax, (metric, title) in zip(axes, metrics):
        for (name, points), color in zip(series.items(), colors):
            xs = sorted(points)
            ys = [points[x][metric] for x in xs]
            ax.plot(xs, ys, marker="o", linewidth=2.2, label=name, color=color)
            for x, yv in zip(xs, ys):
                ax.text(x, yv + 0.003, f"{yv:.4f}", ha="center", fontsize=8)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xticks([596, 1192, 2385])
        ax.set_xlabel("Checkpoint step")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("Intermediate Checkpoint Progression", fontsize=15, weight="bold")
    save(fig, "engram_25_intermediate_progression_20260730.png")


def draw_step1192_breakdown() -> None:
    tasks = [
        ("global", "Global"),
        ("mmlu", "MMLU"),
        ("sciq", "SciQ"),
    ]
    names = ["Old step1192 Jul19", "Old step1192 Jul23", "91347 step1192"]
    x = np.arange(len(tasks))
    width = 0.24

    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = ["#94a3b8", "#64748b", "#0f766e"]
    for i, (name, color) in enumerate(zip(names, colors)):
        values = [INTERMEDIATE[name][key] for key, _ in tasks]
        bars = ax.bar(x + (i - 1) * width, values, width, label=name, color=color)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.006, f"{height:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Step1192 Checkpoint Comparison", fontsize=15, weight="bold")
    ax.set_ylim(0.20, 0.66)
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in tasks])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    save(fig, "engram_25_step1192_summary_20260730.png")


def main() -> None:
    draw_training_curves()
    draw_param_comparison()
    draw_final_summary()
    draw_final_downstream_tasks()
    draw_intermediate_progression()
    draw_step1192_breakdown()


if __name__ == "__main__":
    main()
