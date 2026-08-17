#!/usr/bin/env python3
"""Build a docs/20-style faircompare report for Baseline, X-gram, and new Engram."""
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
DOCS = ROOT / "docs"
DOC_DIR = DOCS / "23_Baseline_Xgram_NewEngram_下游任务对比_20260727"
FIGS = DOC_DIR / "images"
REPORTS = ROOT / "reports"

REPORT = DOC_DIR / "README.md"
PARAM_FIG = FIGS / "faircompare_23_param_comparison_20260727.png"
CE_FIG = FIGS / "faircompare_23_full_ce_loss_three_way_20260727.png"
PPL_FIG = FIGS / "faircompare_23_full_ppl_three_way_20260727.png"
DOWNSTREAM_FIG = FIGS / "faircompare_23_downstream_12_tasks_three_way_20260727.png"
RADAR_FIG = FIGS / "faircompare_23_downstream_radar_20260727.png"

BASELINE_LOG = LOGS / "downstream_eval_baseline_step2385_full_20260606-104406.log"
XGRAM_LOG = LOGS / "downstream_eval_xgram_step2385_full_20260606-162307.log"
NEW_ENGRAM_LOG = LOGS / "downstream_eval_engram_2g3gxgrammatch_step2385_full_20260727-105716-91063.log"

BASELINE_TRAIN = REPORTS / "baseline_full_metrics.csv"
XGRAM_TRAIN = REPORTS / "xgram_full_metrics.csv"
NEW_ENGRAM_TRAIN = LOGS / "20260724-125126_faircompare_engram_2g3g_xgrammatch_360m_rank0.log"

VOCAB_SIZE = 49152
D_MODEL = 960
N_LAYERS = 32
D_FF = 2560
N_HEADS = 15
N_KV_HEADS = 3
D_HEAD = D_MODEL // N_HEADS

TASKS = [
    ("HellaSwag", "eval/downstream/hellaswag"),
    ("ARC-E", "eval/downstream/arc_easy"),
    ("ARC-C", "eval/downstream/arc_challenge_test_rc_5shot"),
    ("MMLU", "downstream/mmlu_average"),
    ("BoolQ", "eval/downstream/boolq"),
    ("OpenBookQA", "eval/downstream/openbook_qa"),
    ("SocialIQA", "eval/downstream/social_iqa"),
    ("SciQ", "eval/downstream/sciq"),
    ("PIQA", "eval/downstream/piqa"),
    ("CommonsenseQA", "eval/downstream/commonsense_qa"),
    ("CSQA Val RC", "eval/downstream/csqa_val_rc_5shot"),
    ("Winogrande", "eval/downstream/winogrande"),
]

MODELS = {
    "Baseline": {
        "eval_log": BASELINE_LOG,
        "train_source": BASELINE_TRAIN,
        "train_kind": "csv",
        "checkpoint": "runs/baseline-smollm2-360m-fineweb10b-64984/step2385",
        "config": "faircompare_baseline_360m.yaml",
        "color": "#6f7782",
        "line_style": "-",
    },
    "X-gram": {
        "eval_log": XGRAM_LOG,
        "train_source": XGRAM_TRAIN,
        "train_kind": "csv",
        "checkpoint": "runs/xgram-smollm2-360m-fineweb10b-64971/step2385",
        "config": "faircompare_xgram_360m.yaml",
        "color": "#2f6db3",
        "line_style": "-",
    },
    "New Engram": {
        "eval_log": NEW_ENGRAM_LOG,
        "train_source": NEW_ENGRAM_TRAIN,
        "train_kind": "log",
        "checkpoint": "runs/faircompare-engram-2g3g-xgrammatch-360m/step2385",
        "config": "faircompare_engram_2g3g_xgrammatch_360m.yaml",
        "color": "#4c8f5a",
        "line_style": "--",
    },
}

STEP_RE = re.compile(r"\[step=(\d+)/2385,epoch=1")
CE_RE = re.compile(r"train/CE loss=([0-9.,eE+-]+)")
PPL_RE = re.compile(r"train/PPL=([0-9.,eE+-]+)")


def compute_backbone_params() -> dict[str, int]:
    embed = VOCAB_SIZE * D_MODEL
    q = D_MODEL * N_HEADS * D_HEAD
    k = D_MODEL * N_KV_HEADS * D_HEAD
    v = D_MODEL * N_KV_HEADS * D_HEAD
    o = N_HEADS * D_HEAD * D_MODEL
    attn = q + k + v + o
    ffn = 3 * D_MODEL * D_FF
    rms = 2 * D_MODEL
    per_layer = attn + ffn + rms
    all_layers = N_LAYERS * per_layer
    final_norm = D_MODEL
    lm_head = VOCAB_SIZE * D_MODEL
    return {
        "embedding": embed,
        "per_layer_attn": attn,
        "per_layer_ffn": ffn,
        "per_layer_norm": rms,
        "per_layer_total": per_layer,
        "all_layers": all_layers,
        "final_norm": final_norm,
        "lm_head": lm_head,
        "total_backbone": embed + all_layers + final_norm + lm_head,
    }


def fmt_params(n: int | float, digits: int = 1) -> str:
    n = float(n)
    if abs(n) >= 1e9:
        return f"{n / 1e9:.2f}B"
    if abs(n) >= 1e6:
        return f"{n / 1e6:.{digits}f}M"
    if abs(n) >= 1e3:
        return f"{n / 1e3:.1f}K"
    return f"{n:.0f}"


def parse_eval_log(path: Path) -> dict[str, float | int | str]:
    text = path.read_text()
    data: dict[str, float | int | str] = {}

    params = re.findall(
        r"Building transformer with ([0-9,]+) total params, ([0-9,]+) non-embedding params",
        text,
    )
    if not params:
        raise RuntimeError(f"Could not parse params from {path}")
    total, non_emb = params[-1]
    data["total_params"] = int(total.replace(",", ""))
    data["non_embedding_params"] = int(non_emb.replace(",", ""))

    avg = re.findall(r"Global Average: ([0-9.]+)", text)
    if not avg:
        raise RuntimeError(f"Could not parse global average from {path}")
    data["global_avg"] = float(avg[-1])

    completion = re.findall(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.\d+.*Global Average",
        text,
    )
    data["eval_time"] = completion[-1] if completion else "unknown"

    checkpoints = re.findall(r"Loading checkpoint from '([^']+)'", text)
    if checkpoints:
        data["checkpoint"] = checkpoints[-1].removesuffix("/model_and_optim")

    for label, key in TASKS:
        matches = re.findall(re.escape(key) + r".*?:\s*([0-9.]+)", text)
        if not matches:
            raise RuntimeError(f"Could not parse {key} from {path}")
        data[label] = float(matches[-1])
    return data


def load_training_csv(path: Path) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
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


def parse_training_log(path: Path) -> dict[int, dict[str, float]]:
    text = path.read_text()
    matches = list(STEP_RE.finditer(text))
    if not matches:
        return {}
    spans = [(m.start(), int(m.group(1))) for m in matches]
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
        entry = {"ce": float(ce.group(1).replace(",", ""))}
        if ppl:
            entry["ppl"] = float(ppl.group(1).replace(",", ""))
        rows[step] = entry
    return rows


def load_training(model_name: str, cfg: dict[str, str | Path]) -> dict[int, dict[str, float]]:
    source = Path(cfg["train_source"])
    if cfg["train_kind"] == "csv":
        return load_training_csv(source)
    return parse_training_log(source)


def last_metric(rows: dict[int, dict[str, float]]) -> tuple[int | None, float | None, float | None]:
    if not rows:
        return None, None, None
    step = max(rows)
    return step, rows[step].get("ce"), rows[step].get("ppl")


def winner_counts(eval_data: dict[str, dict[str, float | int | str]]) -> tuple[dict[str, int], dict[str, list[str]]]:
    counts = {name: 0 for name in MODELS}
    tasks = {name: [] for name in MODELS}
    for label, _key in TASKS:
        values = {name: float(data[label]) for name, data in eval_data.items()}
        best = max(values.values())
        for name, value in values.items():
            if abs(value - best) < 1e-12:
                counts[name] += 1
                tasks[name].append(label)
    return counts, tasks


def draw_param_comparison(eval_data: dict[str, dict[str, float | int | str]]) -> None:
    baseline_total = float(eval_data["Baseline"]["total_params"])
    baseline_non_emb = float(eval_data["Baseline"]["non_embedding_params"])

    names = list(MODELS)
    colors = [MODELS[name]["color"] for name in names]
    totals = [float(eval_data[name]["total_params"]) / 1e6 for name in names]
    non_emb = [float(eval_data[name]["non_embedding_params"]) / 1e6 for name in names]
    extra_total = [(float(eval_data[name]["total_params"]) - baseline_total) / 1e6 for name in names]
    extra_non_emb = [(float(eval_data[name]["non_embedding_params"]) - baseline_non_emb) / 1e6 for name in names]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))
    bars = axes[0].bar(names, totals, color=colors, alpha=0.88, edgecolor="#333333", linewidth=0.5)
    for bar, value in zip(bars, totals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 7, f"{value:.1f}M", ha="center", fontsize=10)
    axes[0].set_ylabel("Parameters (Millions)")
    axes[0].set_title("Total Parameters from Eval Logs")
    axes[0].grid(axis="y", alpha=0.25)

    x = np.arange(len(names))
    width = 0.34
    axes[1].bar(x - width / 2, extra_total, width=width, color=colors, alpha=0.80, label="Total extra vs Baseline")
    axes[1].bar(x + width / 2, extra_non_emb, width=width, color=colors, alpha=0.45, label="Non-embedding extra")
    for xpos, value in zip(x - width / 2, extra_total):
        axes[1].text(xpos, value + 5, f"{value:.1f}M", ha="center", fontsize=8)
    for xpos, value in zip(x + width / 2, extra_non_emb):
        axes[1].text(xpos, value + 5, f"{value:.1f}M", ha="center", fontsize=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    axes[1].set_ylabel("Parameters (Millions)")
    axes[1].set_title("Injection Extra Parameters")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.suptitle("Model Parameter Comparison: Baseline vs X-gram vs New Engram", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PARAM_FIG, dpi=150)
    plt.close(fig)


def draw_training_curve(training: dict[str, dict[int, dict[str, float]]], metric: str, out_path: Path, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for name, rows in training.items():
        xs = sorted(step for step, row in rows.items() if metric in row)
        ys = [rows[step][metric] for step in xs]
        color = MODELS[name]["color"]
        ax.plot(xs, ys, label=name, color=color, lw=1.6, linestyle=MODELS[name]["line_style"])
        if xs:
            ax.scatter([xs[-1]], [ys[-1]], color=color, s=28, zorder=3)
            suffix = f"{ys[-1]:.3f}" if metric == "ce" else f"{ys[-1]:.2f}"
            ax.text(xs[-1], ys[-1], f"  {name} {suffix}", color=color, va="center", fontsize=9)
    ax.set_xlabel("Training Step")
    ax.set_ylabel(ylabel)
    if metric == "ppl":
        ax.set_yscale("log")
    ax.set_title(f"Faircompare {ylabel}: Baseline vs X-gram vs New Engram")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def draw_downstream(eval_data: dict[str, dict[str, float | int | str]]) -> None:
    labels = [label for label, _key in TASKS]
    y = np.arange(len(labels))
    offsets = [-0.25, 0.0, 0.25]

    fig, ax = plt.subplots(figsize=(12.5, 7.8))
    for offset, (name, cfg) in zip(offsets, MODELS.items()):
        values = [float(eval_data[name][label]) for label in labels]
        ax.barh(y + offset, values, height=0.22, color=cfg["color"], label=name, alpha=0.88)
        for yi, value in zip(y + offset, values):
            ax.text(value + 0.007, yi, f"{value:.3f}", va="center", fontsize=7, color=cfg["color"])
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(float(eval_data[name][label]) for name in MODELS for label in labels) + 0.08)
    ax.set_xlabel("Score")
    ax.set_title("Final 12-Task Downstream Comparison (Filtered Evaluator)")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(DOWNSTREAM_FIG, dpi=150)
    plt.close(fig)


def draw_radar(eval_data: dict[str, dict[str, float | int | str]]) -> None:
    labels = [label for label, _key in TASKS]
    max_by_task = {
        label: max(float(eval_data[name][label]) for name in MODELS)
        for label in labels
    }
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8.4, 8.4), subplot_kw={"polar": True})
    for name, cfg in MODELS.items():
        values = [float(eval_data[name][label]) / max_by_task[label] for label in labels]
        values += values[:1]
        ax.fill(angles, values, color=cfg["color"], alpha=0.10)
        ax.plot(angles, values, "o-", lw=1.7, label=name, color=cfg["color"])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.10)
    ax.set_title("Relative Downstream Performance (Normalized per Task)", fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.08))
    fig.tight_layout()
    fig.savefig(RADAR_FIG, dpi=150)
    plt.close(fig)


def fmt_score(value: float | int | str) -> str:
    return f"{float(value):.4f}"


def top_deltas(
    eval_data: dict[str, dict[str, float | int | str]],
    left: str,
    right: str,
    *,
    positive: bool,
    limit: int = 4,
) -> list[tuple[str, float]]:
    rows = []
    for label, _key in TASKS:
        delta = float(eval_data[left][label]) - float(eval_data[right][label])
        if positive and delta > 0:
            rows.append((label, delta))
        elif not positive and delta < 0:
            rows.append((label, -delta))
    return sorted(rows, key=lambda item: item[1], reverse=True)[:limit]


def fmt_delta_list(rows: list[tuple[str, float]]) -> str:
    return ", ".join(f"{label} (+{delta:.4f})" for label, delta in rows)


def build_report(eval_data: dict[str, dict[str, float | int | str]], training: dict[str, dict[int, dict[str, float]]]) -> str:
    bp = compute_backbone_params()
    baseline_total = float(eval_data["Baseline"]["total_params"])
    baseline_non_emb = float(eval_data["Baseline"]["non_embedding_params"])
    counts, win_tasks = winner_counts(eval_data)

    param_rows = []
    for name in MODELS:
        total = float(eval_data[name]["total_params"])
        non_emb = float(eval_data[name]["non_embedding_params"])
        param_rows.append(
            f"| {name} | {fmt_params(total)} | {fmt_params(non_emb)} | "
            f"{fmt_params(total - baseline_total)} | {fmt_params(non_emb - baseline_non_emb)} |"
        )

    train_rows = []
    for name in MODELS:
        step, ce, ppl = last_metric(training[name])
        train_rows.append(
            f"| {name} | {step if step is not None else 'N/A'} | "
            f"{ce:.3f} | {ppl:.2f} |"
        )

    ds_rows = []
    for label, _key in TASKS:
        values = {name: float(eval_data[name][label]) for name in MODELS}
        best = max(values.values())
        winners = [name for name, value in values.items() if abs(value - best) < 1e-12]
        ds_rows.append(
            f"| {label} | {values['Baseline']:.4f} | {values['X-gram']:.4f} | "
            f"{values['New Engram']:.4f} | {' / '.join(winners)} |"
        )

    global_rows = []
    for name in MODELS:
        global_rows.append(f"| {name} | {float(eval_data[name]['global_avg']):.4f} |")

    x_minus_base = float(eval_data["X-gram"]["global_avg"]) - float(eval_data["Baseline"]["global_avg"])
    e_minus_base = float(eval_data["New Engram"]["global_avg"]) - float(eval_data["Baseline"]["global_avg"])
    e_minus_x = float(eval_data["New Engram"]["global_avg"]) - float(eval_data["X-gram"]["global_avg"])

    step_b, ce_b, ppl_b = last_metric(training["Baseline"])
    step_x, ce_x, ppl_x = last_metric(training["X-gram"])
    step_e, ce_e, ppl_e = last_metric(training["New Engram"])

    e_over_x = fmt_delta_list(top_deltas(eval_data, "New Engram", "X-gram", positive=True))
    x_over_e = fmt_delta_list(top_deltas(eval_data, "New Engram", "X-gram", positive=False))
    e_over_b = fmt_delta_list(top_deltas(eval_data, "New Engram", "Baseline", positive=True))

    report = f"""# New Engram / X-gram / Baseline 三模型 Faircompare 对比实验报告

日期：2026-07-27
目的：组会汇报用，包含模型参数量、训练曲线、12 项 filtered downstream 评测的完整三模型对比。
说明：格式仿照 `docs/20_EngramFaircompare_三模型对比实验报告_20260724/README.md`；本报告把原 Engram 替换为最新 `2g3g+xgrammatch` Engram。

---

## 1. 背景

本报告在 **faircompare** 统一口径下对比三个模型：
- **Baseline**: SmolLM2-360M 标准 backbone，无任何注入
- **X-gram**: Backbone + X-gram 注入（hash v-path + shortconv）
- **New Engram**: Backbone + 最新 2gram+3gram xgrammatch Engram 注入（32 层全覆盖，shortconv enabled）

三个模型使用相同的 backbone 架构、FineWeb 10B 训练数据、global_batch_size=512、train_tokens=10B、lr=3e-4。需要注意的是，New Engram 因显存压力使用 `micro_batch_size=1` 和 full activation checkpointing；Baseline/X-gram 使用 `micro_batch_size=4`。

---

## 2. 模型参数量

### 2.1 Backbone (SmolLM2-360M 风格)

| 组件 | 参数量 |
|:--|:--|
| Token Embedding (vocab={VOCAB_SIZE}) | {fmt_params(bp['embedding'], 2)} |
| 每层 Attention (Q/K/V/O, GQA {N_HEADS}/{N_KV_HEADS}) | {fmt_params(bp['per_layer_attn'], 2)} |
| 每层 SwiGLU FFN (gate/up/down) | {fmt_params(bp['per_layer_ffn'], 2)} |
| 每层 RMSNorm | {fmt_params(bp['per_layer_norm'])} |
| 每层总参数 | {fmt_params(bp['per_layer_total'], 2)} |
| {N_LAYERS} 层合计 | {fmt_params(bp['all_layers'], 2)} |
| LM Head (untied) | {fmt_params(bp['lm_head'], 2)} |
| **Backbone 手算总计** | **{fmt_params(bp['total_backbone'], 2)}** |
| **Baseline eval log 实测总计** | **{fmt_params(eval_data['Baseline']['total_params'], 2)}** |

### 2.2 X-gram 额外参数

| 组件 | 说明 | 参数量 |
|:--|:--|:--|
| X-gram total params | eval log 实测 | {fmt_params(eval_data['X-gram']['total_params'], 2)} |
| X-gram non-embedding params | eval log 实测 | {fmt_params(eval_data['X-gram']['non_embedding_params'], 2)} |
| Total extra vs Baseline | total params 差值 | {fmt_params(float(eval_data['X-gram']['total_params']) - baseline_total, 2)} |
| Non-embedding extra vs Baseline | non-embedding params 差值 | {fmt_params(float(eval_data['X-gram']['non_embedding_params']) - baseline_non_emb, 2)} |
| 注入配置 | `targets=[v]`, `v_layers=20`, shortconv kernels `[3,5,7]`, hash token map Cap=75968 | - |

### 2.3 New Engram 额外参数

| 组件 | 说明 | 参数量 |
|:--|:--|:--|
| New Engram total params | eval log 实测 | {fmt_params(eval_data['New Engram']['total_params'], 2)} |
| New Engram non-embedding params | eval log 实测 | {fmt_params(eval_data['New Engram']['non_embedding_params'], 2)} |
| Total extra vs Baseline | total params 差值 | {fmt_params(float(eval_data['New Engram']['total_params']) - baseline_total, 2)} |
| Non-embedding extra vs Baseline | non-embedding params 差值 | {fmt_params(float(eval_data['New Engram']['non_embedding_params']) - baseline_non_emb, 2)} |
| Engram n-gram mode | `2gram+3gram`, `dim_per_ngram=320`, `ngram_heads=2`, `buckets=24576` | - |
| ShortConv | `engram_shortconv_enabled=true`, kernel=4 | - |

### 2.4 参数对比图

![param comparison](images/{PARAM_FIG.name})

| 模型 | Total 参数 | Non-embedding 参数 | Total extra vs Baseline | Non-emb extra vs Baseline |
|:--|:--|:--|:--|:--|
{chr(10).join(param_rows)}

> **注意**：本节最终对比采用 eval log 实测参数量，而不是沿用 `docs/20` 的早期手算估计。原因是 X-gram 与 Engram 的注入实现中存在 embedding/hash table 计数细节，实测值更适合作为汇报口径。

---

## 3. 实验配置

### 3.1 三模型共同配置

| 项目 | 设置 |
|:--|:--|
| Backbone | SmolLM2-360M-style: {N_LAYERS} layers, hidden={D_MODEL}, FFN={D_FF}, heads={N_HEADS}, GQA={N_KV_HEADS} |
| Dataset | FineWeb 10B tokens (streaming) |
| `global_batch_size` | 512 |
| `train_tokens` | 10,000,000,000 (~2385 steps) |
| `lr` | 3e-4 |
| `seq_len` | 4096 |
| Evaluation during train | downstream disabled |
| 评测口径 | filtered downstream 12-task global average |

### 3.2 各模型差异配置

| 配置项 | Baseline | X-gram | New Engram |
|:--|:--|:--|:--|
| Config 文件 | `{MODELS['Baseline']['config']}` | `{MODELS['X-gram']['config']}` | `{MODELS['New Engram']['config']}` |
| Checkpoint | `{MODELS['Baseline']['checkpoint']}` | `{MODELS['X-gram']['checkpoint']}` | `{MODELS['New Engram']['checkpoint']}` |
| Eval log | `logs/{BASELINE_LOG.name}` | `logs/{XGRAM_LOG.name}` | `logs/{NEW_ENGRAM_LOG.name}` |
| `micro_batch_size` | 4 | 4 | 1 (AC full) |
| 注入模式 | 无 | X-gram (v-path) | Engram (H-path) |
| 注入层数 | 0 | 20 (`v_layers`) | 32 (`h_layers`, 全覆盖) |
| Engram n-gram mode | - | - | `2gram+3gram` |
| Engram dim_per_ngram | - | - | 320 |
| Engram buckets | - | - | 24576 |
| ShortConv | - | kernels `[3,5,7]`, 20 layers | enabled, kernel=4 |

---

## 4. 训练结果

### 4.1 训练曲线

三模型 CE Loss 对比：

![three-way ce](images/{CE_FIG.name})

三模型 PPL 对比：

![three-way ppl](images/{PPL_FIG.name})

> 注：Baseline 和 X-gram 曲线来自 `reports/*_full_metrics.csv`；New Engram 曲线来自 `logs/{NEW_ENGRAM_TRAIN.name}` 逐 step 解析。

### 4.2 训练终点数值 (step ~2385)

| 模型 | 最后 step | Train CE Loss | Train PPL |
|:--|:--|:--|:--|
{chr(10).join(train_rows)}

**分析**：
- 训练 CE/PPL 上，X-gram 仍然最低：CE={ce_x:.3f}, PPL={ppl_x:.2f}。
- New Engram 的终点训练 CE={ce_e:.3f}, PPL={ppl_e:.2f}，高于 Baseline 的 CE={ce_b:.3f}, PPL={ppl_b:.2f}。
- 因此如果目标是“同配置下把 Engram 训练到更低 CE”，这次 New Engram run 是失败的；它没有达到 Engram 理论预期，也不能被称为当前最优 Engram。
- New Engram 的 downstream Global Average 高于 Baseline，只说明它在部分判别任务上有收益；这不能抵消训练 CE/PPL 比 Baseline 更差的问题。

---

## 5. 下游评测结果

### 5.1 Global Average

| 模型 | Global Average |
|:--|:--|
{chr(10).join(global_rows)}

### 5.2 12 项任务终点对比

![downstream bars](images/{DOWNSTREAM_FIG.name})

| 指标 | Baseline | X-gram | New Engram | 最高 |
|:--|:--|:--|:--|:--|
{chr(10).join(ds_rows)}

### 5.3 相对性能雷达图

![downstream radar](images/{RADAR_FIG.name})

### 5.4 各模型获胜统计（12 项任务，含并列）

| 模型 | 获胜任务数 | 获胜任务 |
|:--|:--|:--|
| Baseline | {counts['Baseline']} | {', '.join(win_tasks['Baseline']) or '-'} |
| X-gram | {counts['X-gram']} | {', '.join(win_tasks['X-gram']) or '-'} |
| New Engram | {counts['New Engram']} | {', '.join(win_tasks['New Engram']) or '-'} |

**总结**：X-gram 的 Global Average 为 {float(eval_data['X-gram']['global_avg']):.4f}，仍是最高；New Engram 为 {float(eval_data['New Engram']['global_avg']):.4f}，比 Baseline 高 {e_minus_base:+.4f}，但比 X-gram 低 {abs(e_minus_x):.4f}。

---

## 6. New Engram 结果分析

### 6.1 New Engram 强项

New Engram 相对 X-gram 领先的任务是 {e_over_x}。其中 ARC-E 的提升最大，说明恢复 2gram+3gram、增大 `dim_per_ngram` 并打开 shortconv 后，Engram 在部分 ARC/commonsense 类任务上确实能超过 X-gram。

### 6.2 New Engram 短板

X-gram 相对 New Engram 领先最明显的任务是 {x_over_e}。这些差距主要集中在 OpenBookQA、BoolQ、MMLU 和 SciQ 上，因此 X-gram 仍靠更均衡的任务覆盖面拿到最高 Global Average。

### 6.3 相对 Baseline

New Engram 相对 Baseline 提升最明显的任务是 {e_over_b}。这说明 New Engram 不是只在 n-gram 训练目标或 CE loss 上有变化，它的 downstream global average 也从 Baseline 的 {float(eval_data['Baseline']['global_avg']):.4f} 提升到 {float(eval_data['New Engram']['global_avg']):.4f}。

---

## 7. 当前状态总结 (2026-07-27)

### 7.1 已完成

- Baseline faircompare 训练 + step2385 12 项 filtered downstream 评测
- X-gram faircompare 训练 + step2385 12 项 filtered downstream 评测
- New Engram 2g3g+xgrammatch 训练完成，checkpoint 为 `runs/faircompare-engram-2g3g-xgrammatch-360m/step2385`
- New Engram step2385 完整 filtered downstream 评测完成，评测完成时间为 {eval_data['New Engram']['eval_time']}

### 7.2 Runs 状态

| Run / Log | 名称 | 最终 Checkpoint | 状态 |
|:--|:--|:--|:--|
| `baseline-smollm2-360m-fineweb10b-64984` | Baseline | step2385 | 完成 |
| `xgram-smollm2-360m-fineweb10b-64971` | X-gram | step2385 | 完成 |
| `faircompare-engram-2g3g-xgrammatch-360m` | New Engram | step2385 | 完成 |

### 7.3 核心发现

1. **X-gram 仍是当前 downstream 最强**：Global Average={float(eval_data['X-gram']['global_avg']):.4f}，比 New Engram 高 {abs(e_minus_x):.4f}。

2. **New Engram 的训练目标失败**：终点 CE={ce_e:.3f}，高于 Baseline 的 CE={ce_b:.3f}。这说明本次 2g3g+xgrammatch 配置没有把 Engram 优化到我们预期的状态。

3. **New Engram 的 downstream 仍有局部收益**：Global Average={float(eval_data['New Engram']['global_avg']):.4f}，比 Baseline 高 {e_minus_base:+.4f}；它在 ARC-E、ARC-C、CommonsenseQA、Winogrande 上超过或追平 X-gram，但在 BoolQ、MMLU、OpenBookQA、SciQ 上落后。

4. **不能用 downstream 局部提升包装训练失败**：这次结果最多说明 Engram 注入带来了任务偏置收益，不说明语言建模能力更强。后续必须先解决 CE/PPL 劣化，再谈是否能超过 X-gram。

### 7.4 后续方向

- 针对 BoolQ、MMLU、OpenBookQA、SciQ 分析 New Engram 相对 X-gram 的短板。
- 补跑 New Engram 的 GSM8K/TriviaQA generation EM，使报告可完全覆盖 `docs/20` 的生成式任务口径。
- 继续探索 Engram 的容量、bucket、shortconv 和注入层配置，验证是否能把 ARC/commonsense 的收益扩展到更多任务。

---

## 8. 生成图像清单

| 图像 | 文件 |
|:--|:--|
| 三模型参数对比 | `images/{PARAM_FIG.name}` |
| 三模型 CE Loss 对比 | `images/{CE_FIG.name}` |
| 三模型 PPL 对比 | `images/{PPL_FIG.name}` |
| 12 任务终点对比 | `images/{DOWNSTREAM_FIG.name}` |
| 相对性能雷达图 | `images/{RADAR_FIG.name}` |
"""
    return report


def main() -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    eval_data = {name: parse_eval_log(cfg["eval_log"]) for name, cfg in MODELS.items()}
    training = {name: load_training(name, cfg) for name, cfg in MODELS.items()}

    draw_param_comparison(eval_data)
    draw_training_curve(training, "ce", CE_FIG, "Train CE Loss")
    draw_training_curve(training, "ppl", PPL_FIG, "Train PPL")
    draw_downstream(eval_data)
    draw_radar(eval_data)

    REPORT.write_text(build_report(eval_data, training))
    print(f"Wrote {PARAM_FIG}")
    print(f"Wrote {CE_FIG}")
    print(f"Wrote {PPL_FIG}")
    print(f"Wrote {DOWNSTREAM_FIG}")
    print(f"Wrote {RADAR_FIG}")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
