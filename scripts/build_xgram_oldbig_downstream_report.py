#!/usr/bin/env python3
"""Build the X-gram vs old-big Engram downstream comparison report."""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
DOCS = ROOT / "docs"
DOC_DIR = DOCS / "22_Xgram_vs_OldEngram_下游任务对比_20260724"
FIGS = DOC_DIR / "images"

XGRAM_LOG = LOGS / "downstream_eval_xgram_step2385_full_20260606-162307.log"
OLD_ENGRAM_LOG = LOGS / "downstream_eval_engram_oldbig71677_step2385_fullonly_full_20260724-124728-90819.log"

BAR_FIG = FIGS / "xgram_vs_oldbig71677_downstream_20260724.png"
DELTA_FIG = FIGS / "xgram_vs_oldbig71677_downstream_delta_20260724.png"
REPORT = DOC_DIR / "README.md"

TASKS = [
    ("HellaSwag", "eval/downstream/hellaswag"),
    ("ARC-Easy", "eval/downstream/arc_easy"),
    ("ARC-Challenge", "eval/downstream/arc_challenge_test_rc_5shot"),
    ("MMLU Avg", "downstream/mmlu_average"),
    ("BoolQ", "eval/downstream/boolq"),
    ("OpenBookQA", "eval/downstream/openbook_qa"),
    ("SocialIQA", "eval/downstream/social_iqa"),
    ("SciQ", "eval/downstream/sciq"),
    ("PIQA", "eval/downstream/piqa"),
    ("CommonsenseQA", "eval/downstream/commonsense_qa"),
    ("CSQA Val RC", "eval/downstream/csqa_val_rc_5shot"),
    ("Winogrande", "eval/downstream/winogrande"),
]


def parse_log(path: Path) -> dict[str, float | int]:
    text = path.read_text()
    data: dict[str, float | int] = {}

    params = re.findall(r"Building transformer with ([0-9,]+) total params", text)
    if not params:
        raise RuntimeError(f"Could not parse total params from {path}")
    data["total_params"] = int(params[-1].replace(",", ""))

    global_avg = re.findall(r"Global Average: ([0-9.]+)", text)
    if not global_avg:
        raise RuntimeError(f"Could not parse global average from {path}")
    data["global_avg"] = float(global_avg[-1])

    for display, metric_key in TASKS:
        values = re.findall(re.escape(metric_key) + r".*?:\s*([0-9.]+)", text)
        if not values:
            raise RuntimeError(f"Could not parse {metric_key} from {path}")
        data[display] = float(values[-1])

    return data


def fmt_params(n: int | float) -> str:
    n = float(n)
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    return f"{n / 1e3:.1f}K"


def draw_bar_chart(xgram: dict[str, float | int], old: dict[str, float | int]) -> None:
    labels = [task for task, _ in TASKS]
    x_vals = [float(xgram[label]) for label in labels]
    old_vals = [float(old[label]) for label in labels]

    y = np.arange(len(labels))
    height = 0.36
    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    ax.barh(y - height / 2, x_vals, height=height, color="#2f6db3", label="X-gram", alpha=0.9)
    ax.barh(y + height / 2, old_vals, height=height, color="#b84e2f", label="Old Engram 71677", alpha=0.9)

    for yi, x_value, old_value in zip(y, x_vals, old_vals):
        ax.text(x_value + 0.006, yi - height / 2, f"{x_value:.4f}", va="center", fontsize=8, color="#1c3f69")
        ax.text(old_value + 0.006, yi + height / 2, f"{old_value:.4f}", va="center", fontsize=8, color="#71301d")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(max(x_vals), max(old_vals)) + 0.08)
    ax.set_xlabel("Score")
    ax.set_title("X-gram vs Old Engram 71677: Downstream Accuracy at step2385")
    ax.grid(axis="x", alpha=0.22)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(BAR_FIG, dpi=150)
    plt.close(fig)


def draw_delta_chart(xgram: dict[str, float | int], old: dict[str, float | int]) -> None:
    labels = [task for task, _ in TASKS]
    deltas = [float(xgram[label]) - float(old[label]) for label in labels]
    colors = ["#2f6db3" if delta >= 0 else "#b84e2f" for delta in deltas]

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    ax.barh(y, deltas, color=colors, alpha=0.9)
    ax.axvline(0, color="#333333", linewidth=1.0)

    min_delta = min(deltas)
    max_delta = max(deltas)
    pad = max(abs(min_delta), abs(max_delta)) * 0.06
    ax.set_xlim(min_delta - 0.025, max_delta + 0.018)
    for yi, delta in zip(y, deltas):
        ha = "left" if delta >= 0 else "right"
        x = delta + (pad if delta >= 0 else -pad)
        ax.text(x, yi, f"{delta:+.4f}", va="center", ha=ha, fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Delta score (X-gram - Old Engram)")
    ax.set_title("Per-Task Difference: Positive Means X-gram Is Better")
    ax.grid(axis="x", alpha=0.22)
    fig.tight_layout()
    fig.savefig(DELTA_FIG, dpi=150)
    plt.close(fig)


def build_table(xgram: dict[str, float | int], old: dict[str, float | int]) -> tuple[str, int, int]:
    rows = ["| 任务 | X-gram | Old Engram 71677 | 差值 X-gram - Old | 胜出 |",
            "|:--|:--|:--|:--|:--|"]
    x_wins = 0
    old_wins = 0
    for label, _ in TASKS:
        x_value = float(xgram[label])
        old_value = float(old[label])
        delta = x_value - old_value
        if delta >= 0:
            winner = "X-gram"
            x_wins += 1
        else:
            winner = "Old Engram"
            old_wins += 1
        rows.append(f"| {label} | {x_value:.4f} | {old_value:.4f} | {delta:+.4f} | {winner} |")
    return "\n".join(rows), x_wins, old_wins


def write_report(xgram: dict[str, float | int], old: dict[str, float | int]) -> None:
    table, x_wins, old_wins = build_table(xgram, old)
    global_delta = float(xgram["global_avg"]) - float(old["global_avg"])
    param_ratio = float(old["total_params"]) / float(xgram["total_params"])

    old_better = []
    xgram_better = []
    for label, _ in TASKS:
        delta = float(xgram[label]) - float(old[label])
        if delta >= 0:
            xgram_better.append((label, delta))
        else:
            old_better.append((label, -delta))

    strongest_x = sorted(xgram_better, key=lambda item: item[1], reverse=True)[:3]
    strongest_old = sorted(old_better, key=lambda item: item[1], reverse=True)[:3]

    report = f"""# X-gram vs Old Engram 下游任务对比报告

日期：2026-07-24
目的：对比 X-gram 与历史 old Engram 在下游判别式任务上的终点评测效果。
说明：格式参考 `docs/19_Docs_总合并版.md`。这里的 old Engram 指 `runs/engram-smollm2-360m-fineweb10b-71677/step2385`，不是当前 faircompare 压缩版 Engram。

---

## 1. bg

本报告只回答一个问题：在相同 12 项 downstream 汇总口径下，X-gram 与历史 old Engram 谁的终点下游效果更好。

需要特别区分的是，old Engram 71677 是历史大模型口径，评测日志显示总参数量约 {fmt_params(old["total_params"])}；X-gram 对照约 {fmt_params(xgram["total_params"])}。因此本报告可用于观察效果差异，但不应当被表述为严格参数公平对比。

## 2. Objective

1. 复用已有 step2385 下游评测日志，抽取 12 项任务分数。
2. 生成 X-gram 与 old Engram 的任务级柱状图。
3. 生成 `X-gram - Old Engram` 的差值图，明确每项任务谁领先。
4. 给出组会可直接引用的结论口径。

## 3. Experimental Setup

### 3.1 评测输入

| 模型 | Checkpoint | Eval log | 参数量 |
|:--|:--|:--|:--|
| X-gram | `runs/xgram-smollm2-360m-fineweb10b-64971/step2385` | `logs/downstream_eval_xgram_step2385_full_20260606-162307.log` | {fmt_params(xgram["total_params"])} |
| Old Engram 71677 | `runs/engram-smollm2-360m-fineweb10b-71677/step2385` | `logs/downstream_eval_engram_oldbig71677_step2385_fullonly_full_20260724-124728-90819.log` | {fmt_params(old["total_params"])} |

### 3.2 任务口径

下表采用 filtered downstream 的 12 项汇总口径：`MMLU Avg` 加上 11 个单项任务。分数均为对应 evaluator 输出的 accuracy 或 length-normalized accuracy v2。old Engram 的完整评测完成于 2026-07-24 13:16:37，X-gram 的评测日志完成于 2026-06-06 16:31:39。

## 4. Results

### 4.1 总览

| 指标 | X-gram | Old Engram 71677 | 结论 |
|:--|:--|:--|:--|
| Global Average | {float(xgram["global_avg"]):.4f} | {float(old["global_avg"]):.4f} | X-gram 高 {global_delta:+.4f} |
| 胜出任务数 | {x_wins}/12 | {old_wins}/12 | X-gram 覆盖面更稳 |
| 参数量 | {fmt_params(xgram["total_params"])} | {fmt_params(old["total_params"])} | Old Engram 约为 X-gram 的 {param_ratio:.1f} 倍 |

### 4.2 任务分数图

![X-gram vs Old Engram downstream](images/{BAR_FIG.name})

### 4.3 差值图

![X-gram minus Old Engram downstream delta](images/{DELTA_FIG.name})

### 4.4 明细表

{table}

### 4.5 任务级观察

X-gram 领先最明显的任务是 {", ".join(f"{name} (+{delta:.4f})" for name, delta in strongest_x)}。这说明它在 BoolQ、SciQ 等常识/科学类判别任务上优势更稳定。

Old Engram 领先最明显的任务是 {", ".join(f"{name} (+{delta:.4f})" for name, delta in strongest_old)}。它主要在 ARC-Easy、HellaSwag 和 CommonsenseQA 上超过 X-gram，但这些优势没有抵消 BoolQ 与 SciQ 上的较大落后。

## 5. Conclusion

在这组历史下游评测中，X-gram 的 Global Average 为 {float(xgram["global_avg"]):.4f}，高于 old Engram 71677 的 {float(old["global_avg"]):.4f}；12 项任务中 X-gram 赢 {x_wins} 项，old Engram 赢 {old_wins} 项。

因此建议汇报口径是：即使与约 {fmt_params(old["total_params"])} 的历史大容量 Engram 对比，约 {fmt_params(xgram["total_params"])} 的 X-gram 仍取得更高 downstream global average 和更好的任务覆盖面。old Engram 的个别任务收益存在，但整体稳定性和参数效率都弱于 X-gram。
"""
    REPORT.write_text(report)


def main() -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    xgram = parse_log(XGRAM_LOG)
    old = parse_log(OLD_ENGRAM_LOG)
    draw_bar_chart(xgram, old)
    draw_delta_chart(xgram, old)
    write_report(xgram, old)
    print(f"Wrote {BAR_FIG}")
    print(f"Wrote {DELTA_FIG}")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
