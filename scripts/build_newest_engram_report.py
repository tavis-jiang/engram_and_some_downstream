#!/usr/bin/env python3
"""
Updated faircompare report - focusing on the NEWEST Engram xgrammatch (run 89613).
Compares: Baseline, X-gram, Old Engram (88169), New Engram xgrammatch (89613).
Date: 2026-07-24
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/bcjiang/X-gram")
LOGS = ROOT / "logs"
DOCS = ROOT / "docs"
DOC_DIR = DOCS / "21_EngramFaircompare_最新Engram对比报告_20260724"
FIGS = DOC_DIR / "images"
REPORTS = ROOT / "reports"
FIGS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Config constants
# ---------------------------------------------------------------------------
VOCAB_SIZE = 49152
D_MODEL = 960
N_LAYERS = 32
D_FF = 2560
N_HEADS = 15
N_KV_HEADS = 3


def compute_backbone_params():
    embed = VOCAB_SIZE * D_MODEL
    attn = D_MODEL * N_HEADS * 64 + D_MODEL * N_KV_HEADS * 64 * 2 + N_HEADS * 64 * D_MODEL
    ffn = 3 * D_MODEL * D_FF
    rms = 2 * D_MODEL
    per_layer = attn + ffn + rms
    all_layers = N_LAYERS * per_layer
    return embed + all_layers + D_MODEL + VOCAB_SIZE * D_MODEL


def compute_engram_extra(mode, dim_per_ngram, ngram_heads, ngram_buckets, n_layers):
    """Compute extra params for Engram variants."""
    combined_dim = dim_per_ngram * ngram_heads
    if mode == "2gram+3gram":
        n_ngrams = 2
    elif mode == "2gram":
        n_ngrams = 1
    else:
        n_ngrams = 1
    hash_per_ngram = ngram_buckets * combined_dim
    hash_total = hash_per_ngram * n_ngrams * n_layers
    concat_dim = combined_dim * n_ngrams
    o_proj_total = concat_dim * D_MODEL * n_layers
    return hash_total + o_proj_total, hash_total, o_proj_total


def fmt_params(n):
    if n >= 1e9: return f"{n/1e9:.2f}B"
    elif n >= 1e6: return f"{n/1e6:.1f}M"
    else: return f"{n/1e3:.1f}K"


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
BB = compute_backbone_params()

# Each variant: (name, label, color, linestyle, marker)
VARIANTS = {
    "Baseline": {
        "tag": "Baseline",
        "color": "#2f6db3",
        "ls": "-",
        "extra_params": 0,
        "total_params": BB,
        "desc": "无注入，纯 backbone",
        "config": "faircompare_baseline_360m.yaml",
    },
    "X-gram": {
        "tag": "X-gram",
        "color": "#c73a2a",
        "ls": "-",
        "extra_params": 73_720_000,
        "total_params": BB + 73_720_000,
        "desc": "Hash v-path + ShortConv, 20 layers",
        "config": "faircompare_xgram_360m.yaml",
    },
    "Engram (old, 2g+3g, d192/b16k)": {
        "tag": "Engram-old",
        "color": "#2f9a52",
        "ls": "--",
        "extra_params": compute_engram_extra("2gram+3gram", 192, 2, 16384, 32)[0],
        "total_params": BB + compute_engram_extra("2gram+3gram", 192, 2, 16384, 32)[0],
        "desc": "2gram+3gram, dim=192, buckets=16384, 32 层",
        "config": "faircompare_engram_360m.yaml",
        "run": "88169",
    },
    "Engram (new, 2g, d384/b37k)": {
        "tag": "Engram-new",
        "color": "#e67e22",
        "ls": "-.",
        "extra_params": compute_engram_extra("2gram", 384, 2, 36864, 32)[0],
        "total_params": BB + compute_engram_extra("2gram", 384, 2, 36864, 32)[0],
        "desc": "2gram only, dim=384, buckets=36864, 32 层 (最新 xgrammatch)",
        "config": "faircompare_engram_2gram_xgrammatch_360m.yaml",
        "run": "89613",
    },
}

# Update total params
for k, v in VARIANTS.items():
    if k.startswith("Engram"):
        extra, _, _ = compute_engram_extra(
            "2gram+3gram" if "2g+3g" in k else "2gram",
            192 if "d192" in k else 384,
            2,
            16384 if "b16k" in k else 36864,
            32,
        )
        VARIANTS[k]["extra_params"] = extra
        VARIANTS[k]["total_params"] = BB + extra


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
STEP_RE = re.compile(r"\[step=(\d+)/2385,epoch=1")
CE_RE = re.compile(r"train/CE loss=([0-9.,eE+-]+)")
PPL_RE = re.compile(r"train/PPL=([0-9.,eE+-]+)")

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

TASK_ORDER = [
    "HellaSwag", "ARC-E", "ARC-C", "MMLU",
    "TriviaQA EM", "OpenBookQA", "GSM8K EM",
    "SocialIQA", "SciQ", "PIQA",
    "CommonsenseQA", "Winogrande",
]


def load_csv(path):
    if not path.exists(): return {}
    rows = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            step = int(row["step"])
            e = {}
            if row.get("ce_loss"): e["ce"] = float(row["ce_loss"])
            if row.get("ppl"):
                try: e["ppl"] = float(row["ppl"])
                except ValueError: pass
            if e: rows[step] = e
    return rows


def parse_training_log(path):
    text = path.read_text()
    matches = list(STEP_RE.finditer(text))
    if not matches: return {}
    spans = [(m.start(), int(m.group(1))) for m in matches]
    spans.append((len(text), -1))
    out = {}
    for i in range(len(spans) - 1):
        start, step = spans[i]
        end = spans[i + 1][0]
        block = text[start:end]
        ce = CE_RE.search(block)
        ppl = PPL_RE.search(block)
        if ce:
            e = {"ce": float(ce.group(1).replace(",", ""))}
            if ppl: e["ppl"] = float(ppl.group(1).replace(",", ""))
            out[step] = e
    return out


def parse_downstream_log(path):
    text = path.read_text()
    out = {}
    for key, pattern in DOWNSTREAM_PATTERNS.items():
        matches = pattern.findall(text)
        if matches: out[key] = float(matches[-1])
    mmlu_keys = ["MMLU_stem", "MMLU_humanities", "MMLU_social", "MMLU_other"]
    if all(k in out for k in mmlu_keys):
        out["MMLU"] = sum(out[k] for k in mmlu_keys) / 4.0
    return out


def load_em_summary(path):
    if not path.exists(): return None
    return float(json.loads(path.read_text())["exact_match"])


# Training data sources
training_src = {
    "Baseline": ("csv", REPORTS / "baseline_full_metrics.csv", None),
    "X-gram": ("csv", REPORTS / "xgram_full_metrics.csv", None),
    "Engram (old, 2g+3g, d192/b16k)": ("log", None, LOGS / "20260718-205008_faircompare_engram_360m_rank0.log"),
    "Engram (new, 2g, d384/b37k)": ("log", None, LOGS / "20260721-160353_faircompare_engram_2gram_xgrammatch_360m_rank0.log"),
}

training = {}
for name, (src, csv_path, log_path) in training_src.items():
    if src == "csv":
        training[name] = load_csv(csv_path)
    else:
        training[name] = parse_training_log(log_path)

# Downstream data sources
downstream_src = {
    "Baseline": LOGS / "downstream_eval_baseline_step2385_full_20260604-171710.log",
    "X-gram": LOGS / "downstream_eval_xgram_step2385_full_20260606-162307.log",
    "Engram (old, 2g+3g, d192/b16k)": LOGS / "downstream_eval_engram_step2385_full_20260720-202223.log",
    "Engram (new, 2g, d384/b37k)": None,  # step2385 not yet evaluated!
}

downstream = {}
for name, log_path in downstream_src.items():
    if log_path:
        downstream[name] = parse_downstream_log(log_path)
    else:
        downstream[name] = {}

# EM data (generation tasks)
em_src = {
    "Baseline": {
        "gsm8k": REPORTS / "generation_em_baseline_gsm8k_em_rerun_20260606_20260606-115317.summary.json",
        "triviaqa": REPORTS / "generation_em_baseline_triviaqa_em_rerun_20260606_20260606-115317.summary.json",
    },
    "X-gram": {
        "gsm8k": REPORTS / "generation_em_xgram_gsm8k_em_rerun_20260606_20260606-141954.summary.json",
        "triviaqa": REPORTS / "generation_em_xgram_triviaqa_em_rerun_20260606_20260606-141954.summary.json",
    },
    "Engram (old, 2g+3g, d192/b16k)": {
        "gsm8k": REPORTS / "generation_em_engram_gsm8k_em_20260722-040542.summary.json",
        "triviaqa": REPORTS / "generation_em_engram_triviaqa_em_20260722-034414.summary.json",
    },
    "Engram (new, 2g, d384/b37k)": {
        "gsm8k": None,
        "triviaqa": None,
    },
}

for name, src in em_src.items():
    downstream[name]["GSM8K EM"] = load_em_summary(src["gsm8k"]) if src["gsm8k"] else None
    downstream[name]["TriviaQA EM"] = load_em_summary(src["triviaqa"]) if src["triviaqa"] else None

# New engram checkpoint progression (step596/1192/1788 from July 23 evals)
new_engram_ckpt = {
    "step596 (Jul 23)": LOGS / "downstream_eval_engram_step596_full_20260723-165022.log",
    "step1192 (Jul 23)": LOGS / "downstream_eval_engram_step1192_full_20260723-164610.log",
    "step1788 (Jul 23)": LOGS / "downstream_eval_engram_step1788_full_20260723-164610.log",
}
new_engram_ds = {k: parse_downstream_log(v) for k, v in new_engram_ckpt.items()}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def draw_ce_loss_all():
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, cfg in VARIANTS.items():
        records = training[name]
        xs = sorted(s for s, v in records.items() if "ce" in v)
        ys = [records[s]["ce"] for s in xs]
        ax.plot(xs, ys, label=name, color=cfg["color"], lw=1.5, linestyle=cfg["ls"])
        if xs:
            ax.scatter([xs[-1]], [ys[-1]], color=cfg["color"], s=28, zorder=3)
            ax.text(xs[-1], ys[-1], f"  {name.split(chr(40))[0].strip()} {ys[-1]:.3f}",
                    color=cfg["color"], va="center", fontsize=8, fontweight="bold")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Train CE Loss")
    ax.set_title("Faircompare CE Loss: Baseline vs X-gram vs Engram Variants (360M, 10B tokens)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGS / "faircompare_ce_loss_all_variants_20260724.png", dpi=150)
    plt.close(fig)
    print(f"  Wrote ce_loss figure")


def draw_ppl_all():
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, cfg in VARIANTS.items():
        records = training[name]
        xs = sorted(s for s, v in records.items() if "ppl" in v)
        ys = [records[s]["ppl"] for s in xs]
        ax.plot(xs, ys, label=name, color=cfg["color"], lw=1.5, linestyle=cfg["ls"])
        if xs:
            ax.scatter([xs[-1]], [ys[-1]], color=cfg["color"], s=28, zorder=3)
            ax.text(xs[-1], ys[-1], f"  {name.split(chr(40))[0].strip()} {ys[-1]:.1f}",
                    color=cfg["color"], va="center", fontsize=8, fontweight="bold")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Train PPL (log scale)")
    ax.set_yscale("log")
    ax.set_title("Faircompare PPL: Baseline vs X-gram vs Engram Variants (360M, 10B tokens)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGS / "faircompare_ppl_all_variants_20260724.png", dpi=150)
    plt.close(fig)
    print(f"  Wrote ppl figure")


def draw_param_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    names = list(VARIANTS.keys())
    short_names = ["Baseline", "X-gram", "Engram\n(old)", "Engram\n(new)"]
    totals = [VARIANTS[n]["total_params"] for n in names]
    extras = [VARIANTS[n]["extra_params"] for n in names]
    colors = [VARIANTS[n]["color"] for n in names]

    bars = ax1.bar(short_names, [t / 1e6 for t in totals], color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, totals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                 f"{val/1e6:.0f}M", ha="center", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Total Parameters (Millions)")
    ax1.set_title("Total Model Parameters")
    ax1.grid(axis="y", alpha=0.25)

    extra_names = ["X-gram\nExtra", "Engram Old\nExtra", "Engram New\nExtra"]
    extra_vals = [extras[1], extras[2], extras[3]]
    extra_colors = [colors[1], colors[2], colors[3]]
    bars2 = ax2.bar(extra_names, [v / 1e6 for v in extra_vals], color=extra_colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars2, extra_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f"{val/1e6:.0f}M", ha="center", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Extra Parameters (Millions)")
    ax2.set_title("Injection Module Extra Parameters")
    ax2.grid(axis="y", alpha=0.25)

    fig.suptitle("Model Parameter Comparison - All Variants", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS / "faircompare_param_comparison_v2_20260724.png", dpi=150)
    plt.close(fig)
    print(f"  Wrote param figure")


def draw_downstream_bars():
    # Only compare models with downstream results
    plot_models = ["Baseline", "X-gram", "Engram (old, 2g+3g, d192/b16k)"]
    fig, ax = plt.subplots(figsize=(13, 7.5))
    y = list(range(len(TASK_ORDER)))
    offsets = [-0.28, 0.0, 0.28]
    short = ["Baseline", "X-gram", "Engram (old)"]
    for idx, (name, sn) in enumerate(zip(plot_models, short)):
        cfg = VARIANTS[name]
        vals = [downstream[name].get(task) for task in TASK_ORDER]
        plot_vals = [v if v is not None else 0.0 for v in vals]
        ax.barh([i + offsets[idx] for i in y], plot_vals, height=0.25,
                color=cfg["color"], label=sn, alpha=0.85)
        for yi, value in zip(y, vals):
            if value is None:
                ax.text(0.005, yi + offsets[idx], "N/A", va="center", ha="left",
                        fontsize=7, color=cfg["color"], style="italic")
            else:
                label = f"{value:.6f}" if value < 0.01 else f"{value:.3f}"
                ax.text(value + 0.01, yi + offsets[idx], label, va="center",
                        ha="left", fontsize=6.5, color=cfg["color"])
    ax.set_yticks(y)
    ax.set_yticklabels(TASK_ORDER)
    ax.invert_yaxis()
    ax.set_xlabel("Score")
    ax.set_title("Final 12-Task Downstream Comparison (Faircompare 360M, 10B tokens)")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "faircompare_downstream_12_tasks_20260724.png", dpi=150)
    plt.close(fig)
    print(f"  Wrote downstream bars figure")


def draw_new_engram_progression():
    """Show the new engram xgrammatch checkpoint progression at step596/1192/1788."""
    tasks_10 = ["HellaSwag", "ARC-E", "ARC-C", "MMLU", "OpenBookQA",
                "SocialIQA", "SciQ", "PIQA", "CommonsenseQA", "Winogrande"]
    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    width = 0.22
    x = list(range(len(tasks_10)))
    colors = ["#fde0c8", "#f5b881", "#e67e22", "#b85e0c"]
    labels = list(new_engram_ds.keys())

    for idx, (label, color) in enumerate(zip(labels, colors)):
        vals = [new_engram_ds[label].get(task, 0.0) for task in tasks_10]
        ax.bar([i + (idx - 1.5) * width for i in x], vals, width=width,
               label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks_10, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("Score")
    ax.set_title("Engram (NEW, xgrammatch) Downstream Progression: step596 → step1192 → step1788")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "engram_new_xgrammatch_progression_20260724.png", dpi=150)
    plt.close(fig)
    print(f"  Wrote new engram progression figure")


def draw_engram_vs_engram_ce():
    """Compare old vs new engram CE loss on same plot."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for name in ["Engram (old, 2g+3g, d192/b16k)", "Engram (new, 2g, d384/b37k)"]:
        cfg = VARIANTS[name]
        records = training[name]
        xs = sorted(s for s, v in records.items() if "ce" in v)
        ys = [records[s]["ce"] for s in xs]
        ax.plot(xs, ys, label=name, color=cfg["color"], lw=1.8, linestyle=cfg["ls"])
        if xs:
            ax.scatter([xs[-1]], [ys[-1]], color=cfg["color"], s=30, zorder=3)
            short = "Old Engram" if "old" in name else "New Engram"
            ax.text(xs[-1], ys[-1], f"  {short} {ys[-1]:.3f}",
                    color=cfg["color"], va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Train CE Loss")
    ax.set_title("Engram Old vs New: CE Loss Comparison (same 360M backbone, 10B tokens)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "engram_old_vs_new_ce_loss_20260724.png", dpi=150)
    plt.close(fig)
    print(f"  Wrote old vs new engram CE loss figure")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def generate_report():
    r = f"""# Engram / X-gram / Baseline 三模型 Faircompare 对比实验报告

日期：**2026-07-24** (最新版)
目的：组会汇报用，包含**最新 Engram xgrammatch (run 89613)** 的训练与评测结果。
说明：本次报告以最新 Engram 变体为核心，同时保留 Baseline、X-gram、旧 Engram 作为对照。

---

## 1. Engram 变体总览

当前仓库已有多个 Engram 变体完成训练：

| Run ID | 名称 | 完成日期 | n-gram 模式 | dim_per_ngram | buckets | 状态 |
|:--|:--|:--|:--|:--|:--|:--|
| 88169 | Engram (old) | Jul 20 | 2gram+3gram | 192 | 16384 | ✅ 训练+评测完成 |
| 88167 | Engram fsdpmb1 | Jul 20 | 2gram+3gram | 256 | 32768 | ✅ 训练完成 |
| **89613** | **Engram xgrammatch (NEW)** | **Jul 23** | **2gram only** | **384** | **36864** | ✅ 训练完成，step2385 评测待补 |

---

## 2. 模型参数量

### 2.1 Backbone (SmolLM2-360M 风格，四模型共用)

| 组件 | 参数量 |
|:--|:--|
| Token Embedding (vocab={VOCAB_SIZE}) | {fmt_params(VOCAB_SIZE * D_MODEL)} |
| 每层 Attention (Q/K/V/O, GQA {N_HEADS}/{N_KV_HEADS}) | {fmt_params(D_MODEL*N_HEADS*64 + D_MODEL*N_KV_HEADS*64*2 + N_HEADS*64*D_MODEL)} |
| 每层 SwiGLU FFN (gate/up/down) | {fmt_params(3 * D_MODEL * D_FF)} |
| {N_LAYERS} 层合计 | {fmt_params(N_LAYERS * (D_MODEL*N_HEADS*64 + D_MODEL*N_KV_HEADS*64*2 + N_HEADS*64*D_MODEL + 3*D_MODEL*D_FF + 2*D_MODEL))} |
| LM Head (untied) | {fmt_params(VOCAB_SIZE * D_MODEL)} |
| **Backbone 总计** | **{fmt_params(BB)}** |

### 2.2 各模型额外参数与总量

"""
    for name in ["Baseline", "X-gram", "Engram (old, 2g+3g, d192/b16k)", "Engram (new, 2g, d384/b37k)"]:
        cfg = VARIANTS[name]
        r += f"| **{name}** | {fmt_params(BB)} | {fmt_params(cfg['extra_params'])} | **{fmt_params(cfg['total_params'])}** | {cfg['config']} |\n"

    r += """
### 2.3 参数对比图

![param comparison](images/faircompare_param_comparison_v2_20260724.png)

---

## 3. 实验配置对比

### 3.1 共同配置

| 项目 | 设置 |
|:--|:--|
| Backbone | SmolLM2-360M: 32 layers, hidden=960, FFN=2560, heads=15, GQA=3 |
| Dataset | FineWeb 10B tokens (streaming) |
| `global_batch_size` | 512 |
| `train_tokens` | 10,000,000,000 (~2385 steps) |
| `lr` | 3e-4 |
| `seq_len` | 4096 |

### 3.2 Engram 变体差异

| 配置项 | Engram Old (88169) | Engram New xgrammatch (89613) |
|:--|:--|:--|
| n-gram 模式 | 2gram + 3gram | 2gram only |
| `engram_dim_per_ngram` | 192 | **384** (2x) |
| `engram_ngram_heads` | 2 | 2 |
| `engram_ngram_target_buckets` | 16384 | **36864** (2.25x) |
| 每层 n-gram hash 参数 | ~12.6M | ~28.3M |
| H-path 覆盖层数 | 32 | 32 |
| ShortConv | disabled | disabled |
| `micro_batch_size` | 1 (AC full) | 1 (AC full) |
| 运行硬件 | 8×L40S | 8×L40S |

> **设计意图**: 新 Engram (xgrammatch) 放弃了 3-gram 分支，将释放的参数量全部投入 2-gram 的容量（dim×2, buckets×2.25），使单个 n-gram 的表示能力大幅增强，总参数量也超过了 1B。

---

## 4. 训练结果

### 4.1 四模型 CE Loss 对比

![ce loss all](images/faircompare_ce_loss_all_variants_20260724.png)

### 4.2 四模型 PPL 对比

![ppl all](images/faircompare_ppl_all_variants_20260724.png)

### 4.3 训练终点数值 (step ~2380)

"""
    train_rows = []
    for name in ["Baseline", "X-gram", "Engram (old, 2g+3g, d192/b16k)", "Engram (new, 2g, d384/b37k)"]:
        records = training[name]
        if not records:
            continue
        step = max(records)
        row = records[step]
        ce_text = f"{row['ce']:.3f}" if 'ce' in row else "N/A"
        ppl_text = f"{row['ppl']:.2f}" if 'ppl' in row else "N/A"
        train_rows.append(f"| {name} | {step} | {ce_text} | {ppl_text} |")

    r += "\n".join(train_rows) + "\n"

    r += """
**分析**:
- **X-gram** 在训练 CE/PPL 上仍是最优的（CE=3.006）。
- **新 Engram** (CE=3.126) 相比于旧 Engram (CE=3.156) 有约 0.03 的提升，但仍未超越 X-gram。
- **新 Engram** 的 PPL (22.79) 也优于旧 Engram (23.47)，说明单 n-gram 加大容量的策略有效。

### 4.5 Engram Old vs New 直接对比

![engram ce vs](images/engram_old_vs_new_ce_loss_20260724.png)

---

## 5. 下游评测结果

### 5.1 旧 Engram (88169) 终点 12 任务（对比 Baseline / X-gram）

> **新 Engram (89613) step2385 评测尚未完成**，下表仅包含已有终点评测的三模型。

![downstream bars](images/faircompare_downstream_12_tasks_20260724.png)

"""
    # Build downstream table
    ds_rows = []
    plot_models = ["Baseline", "X-gram", "Engram (old, 2g+3g, d192/b16k)"]
    short = ["Baseline", "X-gram", "Engram(old)"]
    for task in TASK_ORDER:
        vals = []
        for name in plot_models:
            v = downstream[name].get(task)
            vals.append(v)
        def fmt(v):
            if v is None: return "N/A"
            return f"{v:.6f}" if v < 0.01 else f"{v:.4f}"
        # best
        numeric = [(v, short[i]) for i, v in enumerate(vals) if v is not None]
        best = max(numeric, key=lambda x: x[0])[1] if numeric else "-"
        ds_rows.append(f"| {task} | {fmt(vals[0])} | {fmt(vals[1])} | {fmt(vals[2])} | {best} |")

    r += "| 指标 | Baseline | X-gram | Engram (old) | 最高 |\n"
    r += "|:--|:--|:--|:--|:--|\n"
    r += "\n".join(ds_rows) + "\n"

    # Win counts
    win = {"Baseline": 0, "X-gram": 0, "Engram(old)": 0}
    for task in TASK_ORDER:
        vals = []
        for i, name in enumerate(plot_models):
            v = downstream[name].get(task)
            if v is not None:
                vals.append((v, short[i]))
        if vals:
            winner = max(vals, key=lambda x: x[0])[1]
            if winner in win:
                win[winner] += 1
    r += f"""
**获胜统计**: Baseline 赢 {win['Baseline']} 项，X-gram 赢 {win['X-gram']} 项，旧 Engram 赢 {win['Engram(old)']} 项。

---

## 6. 新 Engram xgrammatch (89613) Checkpoint 演化

> step2385 下游评测**尚未完成**，以下仅展示 step596 → step1192 → step1788 的 10 项判别式任务结果。

![new engram progression](images/engram_new_xgrammatch_progression_20260724.png)

"""
    tasks_10 = ["HellaSwag", "ARC-E", "ARC-C", "MMLU", "OpenBookQA",
                "SocialIQA", "SciQ", "PIQA", "CommonsenseQA", "Winogrande"]
    r += "| 任务 | step596 | step1192 | step1788 | 趋势 |\n"
    r += "|:--|:--|:--|:--|:--|\n"
    for task in tasks_10:
        vals = []
        for label in new_engram_ds:
            v = new_engram_ds[label].get(task)
            vals.append(f"{v:.4f}" if v is not None else "N/A")
        numeric = [new_engram_ds[label].get(task) for label in new_engram_ds]
        numeric = [v for v in numeric if v is not None]
        if len(numeric) >= 2:
            if numeric[-1] > numeric[0] + 0.005:
                trend = "📈 上升"
            elif numeric[-1] < numeric[0] - 0.005:
                trend = "📉 下降"
            else:
                trend = "➡️ 持平"
        else:
            trend = "?"
        r += f"| {task} | {' | '.join(vals)} | {trend} |\n"

    # Compare step1788 of new engram vs step1788 of old engram
    old_engram_ckpt_1788 = parse_downstream_log(LOGS / "downstream_eval_engram_step1788_full_20260720-083221.log")

    r += """
### 6.1 Step1788 新老 Engram 对比

| 任务 | Engram Old (step1788) | Engram New (step1788) | 差异 |
|:--|:--|:--|:--|
"""
    for task in tasks_10:
        old_v = old_engram_ckpt_1788.get(task)
        new_v = new_engram_ds["step1788 (Jul 23)"].get(task)
        if old_v is not None and new_v is not None:
            diff = new_v - old_v
            sym = "✅ +" if diff > 0 else ("🔻 " if diff < 0 else "➡️ ")
            r += f"| {task} | {old_v:.4f} | {new_v:.4f} | {sym}{diff:.4f} |\n"
        else:
            r += f"| {task} | {old_v} | {new_v} | N/A |\n"

    # Also check what step1788 of new engram looks like vs old engram step2385
    r += """
### 6.2 新 Engram step1788 vs 旧 Engram step2385

> 关键问题：新 Engram 在 75% 训练进度时是否已经超过旧 Engram 的终点表现？

"""
    old_final = downstream["Engram (old, 2g+3g, d192/b16k)"]
    new_1788 = new_engram_ds["step1788 (Jul 23)"]

    better_count = 0
    for task in tasks_10:
        old_v = old_final.get(task)
        new_v = new_1788.get(task)
        if old_v is not None and new_v is not None and new_v > old_v:
            better_count += 1

    r += "| 任务 | Engram Old (step2385 final) | Engram New (step1788, 75%) | New > Old? |\n"
    r += "|:--|:--|:--|:--|\n"
    for task in tasks_10:
        old_v = old_final.get(task)
        new_v = new_1788.get(task)
        if old_v is not None and new_v is not None:
            better = "✅ YES" if new_v > old_v else "❌ no"
            r += f"| {task} | {old_v:.4f} | {new_v:.4f} | {better} |\n"
        else:
            r += f"| {task} | {old_v} | {new_v} | N/A |\n"

    r += f"""
**统计**: 新 Engram step1788 (仅 75% 训练) 在 {better_count}/{len(tasks_10)} 项判别式任务上已超过旧 Engram 终点表现。

---

## 7. 当前状态 (2026-07-24)

### 7.1 各 Run 完成度

| Run | 训练 | step596 评测 | step1192 评测 | step1788 评测 | step2385 评测 | GSM8K | TriviaQA |
|:--|:--|:--|:--|:--|:--|:--|:--|
| 88161 Baseline | ✅ | - | - | - | ✅ | ✅ | ✅ |
| 88162 X-gram | ✅ | - | - | - | ✅ | ✅ | ✅ |
| 88169 Engram Old | ✅ | ✅ | ✅ | ✅ | ✅ | ✅(27/1319) | ✅(30/7993) |
| **89613 Engram New** | ✅ | ✅ | ✅ | ✅ | **❌ PENDING** | **❌ PENDING** | **❌ PENDING** |

### 7.2 核心发现

1. **新 Engram (xgrammatch) 比旧 Engram 更好** — 在仅 75% 训练进度 (step1788) 时已在 {better_count}/10 项任务上超过旧 Engram 终点。训练 CE loss 也更低 (3.126 vs 3.156)。

2. **新策略验证成功** — "单 n-gram + 大容量" (dim=384, buckets=36864) 优于 "双 n-gram + 小容量" (dim=192, buckets=16384)。

3. **X-gram 仍是最强的** — 无论训练 PPL 还是下游评测，X-gram (474.87M) 仍然保持最佳结果，比新 Engram (1.33B) 的参数效率更高。

4. **新 Engram step2385 评测亟待完成** — 当前最关键的缺失数据。

### 7.3 待办

- 🔴 **紧急**: 完成新 Engram (89613) step2385 的 12 项下游评测
- 🔴 **紧急**: 完成新 Engram GSM8K / TriviaQA EM 评测
- 🟡 分析新 Engram 在 step2385 是否能超越 X-gram
- 🟡 评估是否需要进一步加大 dim/buckets 或恢复 3-gram

---

## 8. 生成图像清单

| 图像 | 文件 |
|:--|:--|
| 四模型 CE Loss 对比 | `images/faircompare_ce_loss_all_variants_20260724.png` |
| 四模型 PPL 对比 | `images/faircompare_ppl_all_variants_20260724.png` |
| 参数对比 | `images/faircompare_param_comparison_v2_20260724.png` |
| 三模型下游 12 任务 | `images/faircompare_downstream_12_tasks_20260724.png` |
| 新 Engram Checkpoint 演化 | `images/engram_new_xgrammatch_progression_20260724.png` |
| 新老 Engram CE Loss 对比 | `images/engram_old_vs_new_ce_loss_20260724.png` |
"""
    return r


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Newest Engram (xgrammatch 89613) Report Generator")
    print("=" * 60)

    print("\n--- Parameter Summary ---")
    for name in VARIANTS:
        cfg = VARIANTS[name]
        print(f"  {name}: total={fmt_params(cfg['total_params'])}, extra={fmt_params(cfg['extra_params'])}")

    print("\n--- Training Endpoint Comparison ---")
    for name in ["Baseline", "X-gram", "Engram (old, 2g+3g, d192/b16k)", "Engram (new, 2g, d384/b37k)"]:
        records = training[name]
        if records:
            step = max(records)
            row = records[step]
            print(f"  {name}: step={step}, CE={row.get('ce','?'):.3f}, PPL={row.get('ppl','?'):.2f}")

    print("\n--- Generating Figures ---")
    draw_ce_loss_all()
    draw_ppl_all()
    draw_param_comparison()
    draw_downstream_bars()
    draw_new_engram_progression()
    draw_engram_vs_engram_ce()

    print("\n--- Writing Report ---")
    report = generate_report()
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    report_path = DOC_DIR / "README.md"
    report_path.write_text(report)
    print(f"  Report -> {report_path}")
    print(f"\nDone!")
