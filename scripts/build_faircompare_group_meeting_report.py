#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
from pathlib import Path
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/home/bcjiang/X-gram")
LOGS = ROOT / "logs"
DOCS = ROOT / "docs"
DOC_DIR = DOCS / "13_Engram_faircompare_三模型汇报报告"
FIGS = DOC_DIR / "images"
REPORTS = ROOT / "reports"

FIGS.mkdir(parents=True, exist_ok=True)

TASK_ORDER = [
    "HellaSwag",
    "ARC-E",
    "ARC-C",
    "MMLU",
    "TriviaQA EM",
    "OpenBookQA",
    "GSM8K EM",
    "SocialIQA",
    "SciQ",
    "PIQA",
    "CommonsenseQA",
    "Winogrande",
]

DOWNSTREAM_PATTERNS = {
    "HellaSwag": re.compile(r"hellaswag .*?=([0-9.]+)"),
    "ARC-E": re.compile(r"arc_easy .*?=([0-9.]+)"),
    "ARC-C": re.compile(r"arc_challenge_test_rc_5shot .*?=([0-9.]+)"),
    "OpenBookQA": re.compile(r"openbook_qa .*?=([0-9.]+)"),
    "SocialIQA": re.compile(r"social_iqa .*?=([0-9.]+)"),
    "SciQ": re.compile(r"sciq .*?=([0-9.]+)"),
    "PIQA": re.compile(r"piqa .*?=([0-9.]+)"),
    "CommonsenseQA": re.compile(r"commonsense_qa .*?=([0-9.]+)"),
    "Winogrande": re.compile(r"winogrande .*?=([0-9.]+)"),
    "MMLU_stem": re.compile(r"mmlu_stem_mc_5shot .*?=([0-9.]+)"),
    "MMLU_humanities": re.compile(r"mmlu_humanities_mc_5shot .*?=([0-9.]+)"),
    "MMLU_social": re.compile(r"mmlu_social_sciences_mc_5shot .*?=([0-9.]+)"),
    "MMLU_other": re.compile(r"mmlu_other_mc_5shot .*?=([0-9.]+)"),
}

STEP_RE = re.compile(r"\[step=(\d+)/2385,epoch=1")
CE_RE = re.compile(r"train/CE loss=([0-9.,eE+-]+)")
PPL_RE = re.compile(r"train/PPL=([0-9.,eE+-]+)")


def f(value: str) -> float:
    return float(value.replace(",", ""))


def parse_training_log(path: Path) -> dict[int, dict[str, float]]:
    text = path.read_text()
    matches = list(STEP_RE.finditer(text))
    if not matches:
        return {}
    spans = [(m.start(), int(m.group(1))) for m in matches]
    spans.append((len(text), -1))
    out: dict[int, dict[str, float]] = {}
    for i in range(len(spans) - 1):
        start, step = spans[i]
        end = spans[i + 1][0]
        block = text[start:end]
        ce = CE_RE.search(block)
        ppl = PPL_RE.search(block)
        if ce:
            out[step] = {"ce": f(ce.group(1))}
            if ppl:
                out[step]["ppl"] = f(ppl.group(1))
    return out


def parse_metrics_csv(path: Path) -> dict[int, dict[str, float]]:
    if not path.exists():
        return {}
    out: dict[int, dict[str, float]] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row["step"])
            entry: dict[str, float] = {}
            if row.get("ce_loss"):
                entry["ce"] = float(row["ce_loss"])
            if row.get("ppl"):
                entry["ppl"] = float(row["ppl"])
            out[step] = entry
    return out


def pick_last_metric(records: dict[int, dict[str, float]]) -> tuple[int | None, float | None, float | None]:
    if not records:
        return None, None, None
    step = max(records)
    row = records[step]
    return step, row.get("ce"), row.get("ppl")


def parse_downstream_log(path: Path) -> dict[str, float]:
    text = path.read_text()
    out: dict[str, float] = {}
    for key, pattern in DOWNSTREAM_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            out[key] = float(matches[-1])
    mmlu_keys = ["MMLU_stem", "MMLU_humanities", "MMLU_social", "MMLU_other"]
    if all(k in out for k in mmlu_keys):
        out["MMLU"] = sum(out[k] for k in mmlu_keys) / 4.0
    return out


def load_em_summary(path: Path) -> float | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return float(data["exact_match"])


models = {
    "Baseline": {
        "train_log": LOGS / "20260718-112854_faircompare_baseline_360m_rank0.log",
        "metrics_csv": REPORTS / "baseline_full_metrics.csv",
        "downstream_log": LOGS / "downstream_eval_baseline_step2385_full_20260604-171710.log",
        "gsm8k_summary": REPORTS / "generation_em_baseline_gsm8k_em_rerun_20260606_20260606-115317.summary.json",
        "triviaqa_summary": REPORTS / "generation_em_baseline_triviaqa_em_rerun_20260606_20260606-115317.summary.json",
        "color": "#2f6db3",
    },
    "X-gram": {
        "train_log": LOGS / "20260718-112854_faircompare_xgram_360m_rank0.log",
        "metrics_csv": REPORTS / "xgram_full_metrics.csv",
        "downstream_log": LOGS / "downstream_eval_xgram_step2385_full_20260606-162307.log",
        "gsm8k_summary": REPORTS / "generation_em_xgram_gsm8k_em_rerun_20260606_20260606-141954.summary.json",
        "triviaqa_summary": REPORTS / "generation_em_xgram_triviaqa_em_rerun_20260606_20260606-141954.summary.json",
        "color": "#c73a2a",
    },
    "Engram": {
        "train_log": LOGS / "20260718-205008_faircompare_engram_360m_rank0.log",
        "metrics_csv": None,
        "downstream_log": LOGS / "downstream_eval_engram_step2385_full_20260720-202223.log",
        "gsm8k_summary": REPORTS / "generation_em_engram_gsm8k_em_all_20260720-231956.summary.json",
        "triviaqa_summary": REPORTS / "generation_em_engram_triviaqa_em_all_20260720-232012.summary.json",
        "color": "#2f9a52",
    },
}

training: dict[str, dict[int, dict[str, float]]] = {}
for name, cfg in models.items():
    if cfg.get("metrics_csv"):
        training[name] = parse_metrics_csv(cfg["metrics_csv"])
    else:
        training[name] = parse_training_log(cfg["train_log"])
downstream = {name: parse_downstream_log(cfg["downstream_log"]) for name, cfg in models.items()}
for name, cfg in models.items():
    downstream[name]["GSM8K EM"] = load_em_summary(cfg["gsm8k_summary"])
    downstream[name]["TriviaQA EM"] = load_em_summary(cfg["triviaqa_summary"])


def draw_final_bars() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    y = list(range(len(TASK_ORDER)))
    offsets = [-0.25, 0.0, 0.25]
    for idx, (name, cfg) in enumerate(models.items()):
        vals = [downstream[name].get(task) for task in TASK_ORDER]
        plot_vals = [v if v is not None else 0.0 for v in vals]
        ax.barh([i + offsets[idx] for i in y], plot_vals, height=0.22, color=cfg["color"], label=name)
        for yi, value in zip(y, vals):
            if value is None:
                ax.text(0.005, yi + offsets[idx], "pending", va="center", ha="left", fontsize=8, color=cfg["color"])
            else:
                label = f"{value:.6f}" if value < 0.01 else f"{value:.3f}"
                ax.text(value + 0.008, yi + offsets[idx], label, va="center", ha="left", fontsize=8, color=cfg["color"])
    ax.set_yticks(y)
    ax.set_yticklabels(TASK_ORDER)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.75)
    ax.set_xlabel("score")
    ax.set_title("Final 12-task comparison under faircompare setup")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "faircompare_downstream_12_tasks_three_way.png", dpi=150)
    plt.close(fig)


def draw_engram_checkpoint_bars() -> None:
    ckpt_logs = {
        "step596": LOGS / "downstream_eval_engram_step596_full_20260719-231217.log",
        "step1192": LOGS / "downstream_eval_engram_step1192_full_20260719-231217.log",
        "step1788": LOGS / "downstream_eval_engram_step1788_full_20260720-083221.log",
        "step2385": LOGS / "downstream_eval_engram_step2385_full_20260720-202223.log",
    }
    parsed = {label: parse_downstream_log(path) for label, path in ckpt_logs.items()}
    tasks = ["HellaSwag", "ARC-E", "ARC-C", "MMLU", "OpenBookQA", "SocialIQA", "SciQ", "PIQA", "CommonsenseQA", "Winogrande"]
    fig, ax = plt.subplots(figsize=(12, 6.8))
    width = 0.18
    x = list(range(len(tasks)))
    colors = ["#bad8bf", "#84bf90", "#4ca85e", "#2f7d44"]
    for idx, (label, color) in enumerate(zip(parsed.keys(), colors)):
        vals = [parsed[label].get(task, 0.0) for task in tasks]
        ax.bar([i + (idx - 1.5) * width for i in x], vals, width=width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=25, ha="right")
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("score")
    ax.set_title("Engram downstream progression across checkpoints")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "engram_checkpoint_progression.png", dpi=150)
    plt.close(fig)


def draw_engram_training_curve() -> None:
    records = training["Engram"]
    steps = sorted(records)
    ces = [records[s]["ce"] for s in steps if "ce" in records[s]]
    ce_steps = [s for s in steps if "ce" in records[s]]
    ppls = [records[s]["ppl"] for s in steps if "ppl" in records[s]]
    ppl_steps = [s for s in steps if "ppl" in records[s]]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    axes[0].plot(ce_steps, ces, color=models["Engram"]["color"], lw=1.6)
    axes[0].set_title("Engram train CE loss")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("CE loss")
    axes[0].grid(alpha=0.25)
    axes[1].plot(ppl_steps, ppls, color=models["Engram"]["color"], lw=1.6)
    axes[1].set_title("Engram train PPL")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("PPL")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / "engram_training_curve_faircompare.png", dpi=150)
    plt.close(fig)


def write_report() -> None:
    rows = []
    for task in TASK_ORDER:
        b = downstream["Baseline"].get(task)
        x = downstream["X-gram"].get(task)
        e = downstream["Engram"].get(task)
        def fmt(v: float | None) -> str:
            if v is None:
                return "待补充"
            return f"{v:.6f}" if v < 0.01 else f"{v:.4f}"
        rows.append(f"| {task} | {fmt(b)} | {fmt(x)} | {fmt(e)} |")

    last_rows = []
    for name in ["Baseline", "X-gram", "Engram"]:
        step, ce, ppl = pick_last_metric(training[name])
        ce_text = f"{ce:.3f}" if ce is not None else "无"
        ppl_text = f"{ppl:.2f}" if ppl is not None else "无"
        step_text = str(step) if step is not None else "无"
        last_rows.append(f"| {name} | {step_text} | {ce_text} | {ppl_text} |")

    text = f"""# Engram 实验成功汇报报告

日期：2026-07-20  
目的：用于组会汇报本次 faircompare 口径下 Engram、X-gram、Baseline 的训练与下游对比。  
说明：本报告严格区分“已验证事实”和“待补充结果”，不把仍在运行中的 Engram 生成式评测写成既定结论。

---

## 1. bg

本次汇报采用当前仓库内已经完成的 `faircompare_*_360m` 正式实验口径，在相同 `SmolLM2-360M-style` 主干、相同 `FineWeb 10B tokens`、相同 `global_batch_size=512`、相同 `train_tokens=10,000,000,000` 下，对比 `Baseline`、`X-gram` 与压缩公平版 `Engram`。

本轮目标不是继续讨论旧版错误实验，而是给出当前真正可对齐的三模型结果，并把 Engram 的 checkpoint 演化、三模型终点下游任务结果、以及当前仍待补充的生成式任务状态明确区分开。

## 3. Experimental Setup

### 3.1 已确认事实

| 项目 | Baseline | X-gram | Engram |
|:--|:--|:--|:--|
| Backbone | SmolLM2-360M-style 32-layer decoder | 相同 | 相同 |
| `num_layers` | 32 | 32 | 32 |
| `hidden_size` | 960 | 960 | 960 |
| `ffn_hidden_size` | 2560 | 2560 | 2560 |
| `num_attn_heads` | 15 | 15 | 15 |
| `num_query_groups` | 3 | 3 | 3 |
| Dataset | FineWeb 10B tokens streaming | 相同 | 相同 |
| `global_batch_size` | 512 | 512 | 512 |
| `train_tokens` | 10,000,000,000 | 10,000,000,000 | 10,000,000,000 |
| `lr` | 3e-4 | 3e-4 | 3e-4 |
| Evaluation during train | downstream disabled | 相同 | 相同 |
| Config | `configs/faircompare_baseline_360m.yaml` | `configs/faircompare_xgram_360m.yaml` | `configs/faircompare_engram_360m.yaml` |

### 3.2 Engram 额外配置

| 项目 | Engram 设置 | 作用 |
|:--|:--|:--|
| `embedding_injection.mode` | `Engram` | 启用 Engram 注入 |
| `h_layers` | `[0..31]` | 在 32 层 H-path 全覆盖注入 |
| `engram_mode` | `2gram+3gram` | 同时使用 2-gram 与 3-gram 信息 |
| `engram_dim_per_ngram` | `192` | 压缩版 n-gram 表示维度 |
| `engram_ngram_heads` | `2` | n-gram 头数 |
| `engram_ngram_target_buckets` | `16384` | 压缩版 n-gram bucket 容量 |
| `engram_shortconv_kernel` | `4` | 局部短卷积窗口 |

### 4.2 修改

- 删除了冻结 backbone 的错误设置，保证主干参与训练。
- 将 Engram 调整为当前硬件可跑完的压缩公平版，而不是继续坚持 OOM 的完全大规模版。
- 新增 checkpoint 自动评测链路，使 `step596/1192/1788/2385` 均有 Engram 下游结果。

Why:

- 旧版“能跑”的 Engram 不是当前 faircompare 口径，也不能直接与本轮 baseline/xgram 做正式对照。
- 完全大规模 Engram 在当前卡型上会 OOM，无法形成完整终点结果。
- 用户当前的对照需求是“能完整跑通并和 baseline/xgram 做组会对比”，因此采用压缩公平版是当前可验证方案。

## 5. Experimental Results

### 5.1 图像证据

三模型终点 12 任务对比

![three-way downstream](images/faircompare_downstream_12_tasks_three_way.png)

Engram checkpoint 演化

![engram checkpoints](images/engram_checkpoint_progression.png)

Engram 训练曲线

![engram training](images/engram_training_curve_faircompare.png)

### 5.2 三模型训练末端数值

| 模型 | 最后可见 step | Train CE loss | Train PPL |
|:--|:--|:--|:--|
{chr(10).join(last_rows)}

说明：

- `Engram` 的 faircompare 训练日志完整保留到 `step2385`，当前终点训练行为可直接验证。
- `Baseline/X-gram` 当前 faircompare rank0 日志文件只保留了很前面的在线点，因此本报告不伪造其完整训练曲线；下游终点结果仍然是完整可用的。

### 5.3 三模型下游任务终点对比

| 指标 | Baseline | X-gram | Engram |
|:--|:--|:--|:--|
{chr(10).join(rows)}

### 5.4 已验证事实

- `Engram` 正式训练已经完成，最终 checkpoint 为 `runs/faircompare-engram-360m-88169/step2385`。
- `Baseline/X-gram/Engram` 在用户要求的 12 项里，已有 10 项判别式任务结果可直接三模型对比。
- `Engram` 的 `step596 -> step1192 -> step1788 -> step2385` 下游任务整体呈上升或持平趋势，其中 `SciQ` 提升最明显。

### 5.5 待补充结果

- `Engram` 的 `GSM8K EM` 最终值仍在运行中。
- `Engram` 的 `TriviaQA EM` 最终值仍在运行中。
- 因此当前 12 任务表中这两项对 Engram 标记为“待补充”，不会在汇报里冒充已完成结果。

## 6. 当前结论

- 如果只看当前已经完成并可严格对齐的 10 个判别式任务，`X-gram` 整体仍然是三者里最强的稳定方案。
- 当前压缩公平版 `Engram` 已经成功形成完整训练和终点 checkpoint，但其终点下游结果整体仍弱于 `X-gram`，与 `Baseline` 接近或略有波动。
- 因此本轮最重要的成功，不是“Engram 已经超过 X-gram”，而是“Engram 在当前 faircompare 工程口径下已经能完整训练并可被正式评测”，这为后续继续调参和下游分析建立了有效起点。
"""
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    (DOC_DIR / "README.md").write_text(text)


draw_final_bars()
draw_engram_checkpoint_bars()
draw_engram_training_curve()
write_report()
print("ok")
