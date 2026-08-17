#!/usr/bin/env python3
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/home/bcjiang/X-gram")
OUT = ROOT / "docs" / "21_ComEmbed2v_1v对比报告_20260724" / "images"
OUT.mkdir(parents=True, exist_ok=True)

LOG_1V = ROOT / "logs" / "20260711-171733_our_comembed_faaddqr_1v_overnight5000_rank0.log"
LOG_2V = ROOT / "logs" / "20260717-135026_our_comembed_fanormqr_2v_overnight5000_stabilized_rank0.log"


STEP_RE = re.compile(r"\[step=(\d+)/5000,epoch=1")
CE_RE = re.compile(r"train/CE loss=([0-9.E+-]+)")
PPL_RE = re.compile(r"train/PPL=([0-9,\.E+-]+)")
GRAD_RE = re.compile(r"optim/total grad norm=([0-9.E+-]+)")
SANITIZE_RE = re.compile(r"Sanitized non-finite gradients on (\d+) parameter tensors")


def parse_log(path: Path):
    rows = []
    current = None
    sanitize_events = 0

    for line in path.read_text().splitlines():
        m = SANITIZE_RE.search(line)
        if m:
            sanitize_events += 1

        m = STEP_RE.search(line)
        if m:
            step = int(m.group(1))
            current = {"step": step}
            rows.append(current)
            continue

        if current is None:
            continue

        m = CE_RE.search(line)
        if m:
            current["ce_loss"] = float(m.group(1))
            continue

        m = PPL_RE.search(line)
        if m:
            current["ppl"] = float(m.group(1).replace(",", ""))
            continue

        m = GRAD_RE.search(line)
        if m:
            current["grad_norm"] = float(m.group(1))
            continue

    metric_rows = [r for r in rows if "ce_loss" in r]
    return metric_rows, sanitize_events


rows_1v, sanitize_1v = parse_log(LOG_1V)
rows_2v, sanitize_2v = parse_log(LOG_2V)


def xs(rows):
    return [r["step"] for r in rows]


def ys(rows, key):
    return [r[key] for r in rows]


def with_key(rows, key):
    return [r for r in rows if key in r]


plt.style.use("seaborn-v0_8-whitegrid")
color_1v = "#1f77b4"
color_2v = "#d62728"


fig, ax = plt.subplots(figsize=(8.4, 5.2))
ax.plot(xs(rows_1v), ys(rows_1v, "ce_loss"), label="1v", color=color_1v, lw=2)
ax.plot(xs(rows_2v), ys(rows_2v, "ce_loss"), label="2v", color=color_2v, lw=2)
ax.scatter([rows_1v[-1]["step"]], [rows_1v[-1]["ce_loss"]], color=color_1v, s=36)
ax.scatter([rows_2v[-1]["step"]], [rows_2v[-1]["ce_loss"]], color=color_2v, s=36)
ax.annotate(
    f"1v final {rows_1v[-1]['ce_loss']:.3f}",
    xy=(rows_1v[-1]["step"], rows_1v[-1]["ce_loss"]),
    xytext=(3800, rows_1v[-1]["ce_loss"] + 0.20),
    arrowprops=dict(arrowstyle="->", color=color_1v),
    color=color_1v,
)
ax.annotate(
    f"2v final {rows_2v[-1]['ce_loss']:.3f}",
    xy=(rows_2v[-1]["step"], rows_2v[-1]["ce_loss"]),
    xytext=(3800, rows_2v[-1]["ce_loss"] - 0.35),
    arrowprops=dict(arrowstyle="->", color=color_2v),
    color=color_2v,
)
ax.set_title("1v vs 2v: Train CE Loss")
ax.set_xlabel("step")
ax.set_ylabel("CE loss")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "comembed_1v_vs_2v_ce_loss.png", dpi=150)
plt.close(fig)


fig, ax = plt.subplots(figsize=(8.4, 5.2))
ppl_1v = with_key(rows_1v, "ppl")
ppl_2v = with_key(rows_2v, "ppl")
ax.plot(xs(ppl_1v), ys(ppl_1v, "ppl"), label="1v", color=color_1v, lw=2)
ax.plot(xs(ppl_2v), ys(ppl_2v, "ppl"), label="2v", color=color_2v, lw=2)
ax.scatter([ppl_1v[-1]["step"]], [ppl_1v[-1]["ppl"]], color=color_1v, s=36)
ax.scatter([ppl_2v[-1]["step"]], [ppl_2v[-1]["ppl"]], color=color_2v, s=36)
ax.annotate(
    f"1v final {ppl_1v[-1]['ppl']:.1f}",
    xy=(ppl_1v[-1]["step"], ppl_1v[-1]["ppl"]),
    xytext=(3650, ppl_1v[-1]["ppl"] + 25),
    arrowprops=dict(arrowstyle="->", color=color_1v),
    color=color_1v,
)
ax.annotate(
    f"2v final {ppl_2v[-1]['ppl']:.1f}",
    xy=(ppl_2v[-1]["step"], ppl_2v[-1]["ppl"]),
    xytext=(3650, ppl_2v[-1]["ppl"] - 18),
    arrowprops=dict(arrowstyle="->", color=color_2v),
    color=color_2v,
)
ax.set_title("1v vs 2v: Train PPL")
ax.set_xlabel("step")
ax.set_ylabel("PPL")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "comembed_1v_vs_2v_ppl.png", dpi=150)
plt.close(fig)


fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.8))

axes[0].bar(["1v", "2v"], [rows_1v[-1]["ce_loss"], rows_2v[-1]["ce_loss"]], color=[color_1v, color_2v])
axes[0].set_title("Final CE Loss")
for i, v in enumerate([rows_1v[-1]["ce_loss"], rows_2v[-1]["ce_loss"]]):
    axes[0].text(i, v + 0.03, f"{v:.3f}", ha="center")

axes[1].bar(["1v", "2v"], [ppl_1v[-1]["ppl"], ppl_2v[-1]["ppl"]], color=[color_1v, color_2v])
axes[1].set_title("Final PPL")
for i, v in enumerate([ppl_1v[-1]["ppl"], ppl_2v[-1]["ppl"]]):
    axes[1].text(i, v + 1.5, f"{v:.1f}", ha="center")

fig.suptitle("Final Result Comparison at Step 5000")
fig.tight_layout()
fig.savefig(OUT / "comembed_1v_vs_2v_final_bars.png", dpi=150)
plt.close(fig)


print("1v points:", len(rows_1v), "final CE:", rows_1v[-1]["ce_loss"], "final PPL:", ppl_1v[-1]["ppl"])
print("2v points:", len(rows_2v), "final CE:", rows_2v[-1]["ce_loss"], "final PPL:", ppl_2v[-1]["ppl"])
print("2v sanitize warnings:", sanitize_2v)
