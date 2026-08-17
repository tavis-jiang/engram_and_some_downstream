#!/usr/bin/env python3
"""Generate annotated PNGs for the docs results section."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/home/bcjiang/X-gram/docs/_unassigned_images")
OUT.mkdir(parents=True, exist_ok=True)

def load(p):
    rows = list(csv.DictReader(open(p)))
    return [int(r["step"]) for r in rows], rows

xs_x, rx = load("/home/bcjiang/X-gram/reports/xgram_full_metrics.csv")
xs_b, rb = load("/home/bcjiang/X-gram/reports/baseline_full_metrics.csv")

def col(rows, k):
    out = []
    for r in rows:
        v = r[k]
        out.append(float(v) if v not in ("", None) else None)
    return out

def filt(xs, ys):
    return zip(*[(x, y) for x, y in zip(xs, ys) if y is not None]) if any(y is not None for y in ys) else ([], [])

X_COLOR, B_COLOR = "#d62728", "#1f77b4"

# 1. Full CE loss
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(*filt(xs_x, col(rx, "ce_loss")), label="X-gram", color=X_COLOR, lw=1.6)
ax.plot(*filt(xs_b, col(rb, "ce_loss")), label="Baseline", color=B_COLOR, lw=1.6)
ax.annotate(f"final 3.006", xy=(2380, 3.006), xytext=(1700, 4.5),
            arrowprops=dict(arrowstyle="->", color=X_COLOR), color=X_COLOR, fontsize=10)
ax.annotate(f"final 3.074", xy=(2380, 3.074), xytext=(1700, 5.5),
            arrowprops=dict(arrowstyle="->", color=B_COLOR), color=B_COLOR, fontsize=10)
ax.set_xlabel("step"); ax.set_ylabel("train CE loss")
ax.set_title("Full Training CE Loss — X-gram vs Baseline (2385 steps)")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "ce_loss_full.png", dpi=130); plt.close(fig)

# 2. CE loss zoom
fig, ax = plt.subplots(figsize=(8, 5))
xs, ys = filt(xs_x, col(rx, "ce_loss"))
xs2, ys2 = zip(*[(s, y) for s, y in zip(xs, ys) if s >= 200])
ax.plot(xs2, ys2, label="X-gram", color=X_COLOR, lw=1.4)
xs, ys = filt(xs_b, col(rb, "ce_loss"))
xs2, ys2 = zip(*[(s, y) for s, y in zip(xs, ys) if s >= 200])
ax.plot(xs2, ys2, label="Baseline", color=B_COLOR, lw=1.4)
ax.axhline(3.2, ls="--", color="gray", alpha=0.5)
ax.annotate("X-gram reaches loss≈3.2\nat step ~700", xy=(700, 3.2), xytext=(900, 3.9),
            arrowprops=dict(arrowstyle="->", color=X_COLOR), color=X_COLOR, fontsize=9)
ax.annotate("Baseline reaches loss≈3.2\nat step ~2000  (≈2.8× slower)",
            xy=(2000, 3.2), xytext=(1200, 4.5),
            arrowprops=dict(arrowstyle="->", color=B_COLOR), color=B_COLOR, fontsize=9)
ax.set_xlabel("step"); ax.set_ylabel("train CE loss")
ax.set_title("CE Loss zoomed (step ≥ 200) — sample-efficiency gap")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "ce_loss_zoom.png", dpi=130); plt.close(fig)

# 3. PPL log
fig, ax = plt.subplots(figsize=(8, 5))
xs, ys = filt(xs_x, col(rx, "ppl"))
xs2, ys2 = zip(*[(s, y) for s, y in zip(xs, ys) if y > 0])
ax.plot(xs2, ys2, label="X-gram", color=X_COLOR, lw=1.4)
xs, ys = filt(xs_b, col(rb, "ppl"))
xs2, ys2 = zip(*[(s, y) for s, y in zip(xs, ys) if y > 0])
ax.plot(xs2, ys2, label="Baseline", color=B_COLOR, lw=1.4)
ax.set_yscale("log")
ax.annotate("PPL 20.20", xy=(2380, 20.20), xytext=(1500, 60),
            arrowprops=dict(arrowstyle="->", color=X_COLOR), color=X_COLOR, fontsize=10)
ax.annotate("PPL 21.63 (+6.6%)", xy=(2380, 21.63), xytext=(1500, 200),
            arrowprops=dict(arrowstyle="->", color=B_COLOR), color=B_COLOR, fontsize=10)
ax.set_xlabel("step"); ax.set_ylabel("perplexity (log scale)")
ax.set_title("Training Perplexity (log scale)")
ax.legend(); ax.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(OUT / "ppl_log.png", dpi=130); plt.close(fig)

# 4. MFU
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(*filt(xs_x, col(rx, "mfu")), label="X-gram", color=X_COLOR, lw=1.2, alpha=0.9)
ax.plot(*filt(xs_b, col(rb, "mfu")), label="Baseline", color=B_COLOR, lw=1.2, alpha=0.9)
ax.axvline(1300, ls=":", color="green", alpha=0.6)
ax.annotate("Baseline switched\nfrom RTX4090 → ADA6000\n(node handoff, NOT a method change)",
            xy=(1300, 27), xytext=(1400, 38),
            arrowprops=dict(arrowstyle="->", color="green"), color="green", fontsize=9)
ax.annotate("X-gram stable ~26%", xy=(2000, 26), xytext=(1500, 18),
            arrowprops=dict(arrowstyle="->", color=X_COLOR), color=X_COLOR, fontsize=9)
ax.set_xlabel("step"); ax.set_ylabel("Device MFU (%)")
ax.set_title("Device MFU — note Baseline node-switch at step ~1300")
ax.legend(loc="lower right"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "mfu.png", dpi=130); plt.close(fig)

# 5. Requested 12-task downstream/generation result comparison
tasks = [
    ("HellaSwag", 0.2766, 0.3020),
    ("ARC-E", 0.4211, 0.4509),
    ("ARC-C", 0.2321, 0.2466),
    ("MMLU avg", 0.2504, 0.2634),
    ("TriviaQA EM", 0.000250, 0.000125),
    ("OpenBookQA", 0.2740, 0.2960),
    ("GSM8K EM", 0.021228, 0.015921),
    ("SocialIQA", 0.3910, 0.4058),
    ("SciQ", 0.6480, 0.7010),
    ("PIQA", 0.5647, 0.6045),
    ("CommonsenseQA", 0.2539, 0.2752),
    ("Winogrande", 0.5138, 0.5099),
]

labels = [t[0] for t in tasks]
baseline_scores = [t[1] for t in tasks]
xgram_scores = [t[2] for t in tasks]
ypos = list(range(len(tasks)))
bar_h = 0.36

fig, ax = plt.subplots(figsize=(9.5, 7))
ax.barh([y - bar_h / 2 for y in ypos], baseline_scores, height=bar_h, color=B_COLOR, label="Baseline")
ax.barh([y + bar_h / 2 for y in ypos], xgram_scores, height=bar_h, color=X_COLOR, label="X-gram")
ax.set_yticks(ypos)
ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlim(0, 0.70)
ax.set_xlabel("score")
ax.set_title("Final Scores on the 12 Requested Tasks")
ax.grid(axis="x", alpha=0.25)
ax.legend(loc="lower right")
for y, b, x in zip(ypos, baseline_scores, xgram_scores):
    b_txt = f"{b:.6f}" if b < 0.01 else f"{b:.3f}"
    x_txt = f"{x:.6f}" if x < 0.01 else f"{x:.3f}"
    ax.text(max(b + 0.008, 0.015), y - bar_h / 2, b_txt, va="center", fontsize=8, color=B_COLOR)
    ax.text(max(x + 0.008, 0.015), y + bar_h / 2, x_txt, va="center", fontsize=8, color=X_COLOR)
fig.tight_layout(); fig.savefig(OUT / "downstream_12_tasks.png", dpi=140); plt.close(fig)

# 6. Delta view: X-gram minus baseline
deltas = [(name, x - b) for name, b, x in tasks]
deltas.sort(key=lambda item: item[1])
fig, ax = plt.subplots(figsize=(9, 6.5))
ypos = list(range(len(deltas)))
colors = ["#2ca02c" if d > 0 else "#d62728" for _, d in deltas]
ax.barh(ypos, [d for _, d in deltas], color=colors, alpha=0.88)
ax.axvline(0, color="#222222", lw=1)
ax.set_yticks(ypos)
ax.set_yticklabels([name for name, _ in deltas])
ax.set_xlabel("X-gram score - Baseline score")
ax.set_title("Per-task Delta: Positive Means X-gram Is Higher")
ax.grid(axis="x", alpha=0.25)
for y, (_, d) in zip(ypos, deltas):
    x = d + (0.006 if d >= 0 else -0.006)
    ha = "left" if d >= 0 else "right"
    ax.text(x, y, f"{d:+.4f}", va="center", ha=ha, fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "downstream_delta.png", dpi=140); plt.close(fig)

# 7. Generation EM correct counts from merged summaries
summary_files = {
    ("Baseline", "GSM8K EM"): OUT.parent.parent / "reports/generation_em_baseline_gsm8k_em_rerun_20260606_20260606-115317.summary.json",
    ("X-gram", "GSM8K EM"): OUT.parent.parent / "reports/generation_em_xgram_gsm8k_em_rerun_20260606_20260606-141954.summary.json",
    ("Baseline", "TriviaQA EM"): OUT.parent.parent / "reports/generation_em_baseline_triviaqa_em_rerun_20260606_20260606-115317.summary.json",
    ("X-gram", "TriviaQA EM"): OUT.parent.parent / "reports/generation_em_xgram_triviaqa_em_rerun_20260606_20260606-141954.summary.json",
}

em = {}
for key, path in summary_files.items():
    with path.open() as f:
        em[key] = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, task_name in zip(axes, ["GSM8K EM", "TriviaQA EM"]):
    vals = [em[("Baseline", task_name)]["correct"], em[("X-gram", task_name)]["correct"]]
    totals = [em[("Baseline", task_name)]["num_examples"], em[("X-gram", task_name)]["num_examples"]]
    rates = [em[("Baseline", task_name)]["exact_match"], em[("X-gram", task_name)]["exact_match"]]
    ax.bar([0, 1], vals, color=[B_COLOR, X_COLOR], width=0.58)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Baseline", "X-gram"])
    ax.set_ylabel("correct answers")
    ax.set_title(task_name)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, max(vals) * 1.35 + 1)
    for i, (v, n, rate) in enumerate(zip(vals, totals, rates)):
        ax.text(i, v + max(vals) * 0.05 + 0.2, f"{v}/{n}\nEM={rate:.6f}", ha="center", va="bottom", fontsize=9)
fig.suptitle("Generation Exact Match Results After Shard Merge")
fig.tight_layout(); fig.savefig(OUT / "generation_em_counts.png", dpi=140); plt.close(fig)

# 8. Engram status summary
fig, ax = plt.subplots(figsize=(10, 5.6))
ax.axis("off")
ax.set_title("Engram Reproduction Status (2026-06-07)", fontsize=14, loc="left", pad=14)
rows = [
    ("Formal Engram, 4 GPU attempt", "9.50B trainable params; 32-layer injection; bucket=303872", "Dry-run reached, then CUDA OOM; no final checkpoint", "#d62728"),
    ("Formal Engram, 8 GPU rerun", "Job 68078; node8; 8 GPUs; ran 2026-06-06 17:19:47 to 21:12:39", "OUT_OF_MEMORY near step33 checkpoint; last complete checkpoint is step22", "#d62728"),
    ("Resume from step22", "Job 70593; ADA6000; 8 GPUs; 900G host memory; qos=normal", "Submitted after fixing QoS; pending Resources until a full 8-GPU node frees", "#ff7f0e"),
    ("Engram smoke training", "1.03M trainable params; 1 injection layer; bucket=4096", "Training complete; step1 and step2 checkpoints written", "#2ca02c"),
    ("Engram smoke SciQ eval", "Loaded runs/engram-smollm2-360m-fineweb10b-67438/step2", "Completed; SciQ accuracy v2 = 0.2000", "#2ca02c"),
]
for i, (name, setup, result, color) in enumerate(rows):
    y = 0.84 - i * 0.155
    ax.add_patch(plt.Rectangle((0.02, y - 0.047), 0.035, 0.094, color=color, transform=ax.transAxes, clip_on=False))
    ax.text(0.075, y + 0.035, name, transform=ax.transAxes, fontsize=11, weight="bold", va="center")
    ax.text(0.075, y - 0.005, setup, transform=ax.transAxes, fontsize=9, color="#333333", va="center")
    ax.text(0.075, y - 0.045, result, transform=ax.transAxes, fontsize=9, color="#333333", va="center")
ax.text(
    0.02,
    0.045,
    "Interpretation: the full Engram rerun passed dry-run and trained briefly, but did not reach step2385.\n"
    "Use step22 only as a resume point; step33 is incomplete. Job 70593 is now waiting for resources.",
    transform=ax.transAxes,
    fontsize=9,
    color="#333333",
)
fig.tight_layout(); fig.savefig(OUT / "engram_status.png", dpi=140); plt.close(fig)

print("Wrote figures to", OUT)
