#!/usr/bin/env python3
"""Parse all xgram & baseline training logs and produce a comparison plot."""
import re
import sys
from pathlib import Path
import csv

LOG_DIR = Path("/home/bcjiang/X-gram/logs")
OUT_DIR = Path("/home/bcjiang/X-gram/reports")

# Match a step header like:  [step=300/2385,epoch=1,eta=...]   OR   [step=300/2385,epoch=1]
STEP_RE = re.compile(r"\[step=(\d+)/(\d+),epoch=(\d+)")
CE_RE = re.compile(r"train/CE loss=([0-9.,]+)")
PPL_RE = re.compile(r"train/PPL=([0-9.,eE+]+)")
GRAD_RE = re.compile(r"optim/total grad norm=([0-9.,eE+\-]+)")
TPS_RE = re.compile(r"throughput/device/TPS \(actual avg\)=([0-9.,eE+]+)")
MFU_RE = re.compile(r"throughput/device/MFU \(actual avg\)=([0-9.,]+)")
TOK_RE = re.compile(r"throughput/total tokens=([0-9.,eE+]+)")


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def parse_log(path: Path):
    """Yield dicts {step, ce_loss, ppl, grad, tps, mfu, total_tokens} from one log file."""
    with path.open() as f:
        text = f.read()
    # Split by step headers — each "block" between two consecutive step lines
    # contains the metrics printed for the previous step (if any).
    indices = [(m.start(), int(m.group(1))) for m in STEP_RE.finditer(text)]
    if not indices:
        return
    indices.append((len(text), -1))
    for i in range(len(indices) - 1):
        start, step = indices[i]
        end = indices[i + 1][0]
        block = text[start:end]
        ce = CE_RE.search(block)
        if not ce:
            continue
        rec = {"step": step, "ce_loss": _num(ce.group(1))}
        for key, regex in [
            ("ppl", PPL_RE), ("grad", GRAD_RE),
            ("tps", TPS_RE), ("mfu", MFU_RE), ("total_tokens", TOK_RE),
        ]:
            m = regex.search(block)
            if m:
                try:
                    rec[key] = _num(m.group(1))
                except ValueError:
                    pass
        yield rec


def collect(pattern: str) -> dict:
    merged: dict[int, dict] = {}
    for log in sorted(LOG_DIR.glob(pattern)):
        for rec in parse_log(log):
            # Later logs (resumes) overwrite earlier records for the same step.
            merged[rec["step"]] = rec
    return merged


xgram = collect("*our_xgram_config_rank0.log")
baseline = collect("*our_baseline_config_rank0.log")

print(f"X-gram   : {len(xgram)} unique step records, max step = {max(xgram) if xgram else 'N/A'}")
print(f"Baseline : {len(baseline)} unique step records, max step = {max(baseline) if baseline else 'N/A'}")

# Save CSVs
for name, recs in [("xgram_full", xgram), ("baseline_full", baseline)]:
    out = OUT_DIR / f"{name}_metrics.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "ce_loss", "ppl", "grad", "tps", "mfu", "total_tokens"])
        for step in sorted(recs):
            r = recs[step]
            w.writerow([
                step, r.get("ce_loss"), r.get("ppl"), r.get("grad"),
                r.get("tps"), r.get("mfu"), r.get("total_tokens"),
            ])
    print(f"Wrote {out}")

# Plot
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not available — skipping plot", file=sys.stderr)
    sys.exit(0)

def series(recs, key):
    steps = sorted(recs)
    return steps, [recs[s].get(key) for s in steps]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# (1) CE loss vs step
ax = axes[0, 0]
xs, ys = series(xgram, "ce_loss"); ax.plot(xs, ys, label="X-gram", color="#d62728", lw=1.4)
xs, ys = series(baseline, "ce_loss"); ax.plot(xs, ys, label="Baseline", color="#1f77b4", lw=1.4)
ax.set_xlabel("step"); ax.set_ylabel("train CE loss")
ax.set_title("Training CE Loss"); ax.legend(); ax.grid(alpha=0.3)

# (2) CE loss zoom (after step 200, log-y not used; linear)
ax = axes[0, 1]
xs, ys = series(xgram, "ce_loss")
xs2, ys2 = zip(*[(s, y) for s, y in zip(xs, ys) if s >= 200 and y is not None]) if xs else ([], [])
ax.plot(xs2, ys2, label="X-gram", color="#d62728", lw=1.4)
xs, ys = series(baseline, "ce_loss")
xs2, ys2 = zip(*[(s, y) for s, y in zip(xs, ys) if s >= 200 and y is not None]) if xs else ([], [])
ax.plot(xs2, ys2, label="Baseline", color="#1f77b4", lw=1.4)
ax.set_xlabel("step"); ax.set_ylabel("train CE loss")
ax.set_title("CE Loss (step ≥ 200, zoomed)"); ax.legend(); ax.grid(alpha=0.3)

# (3) PPL log scale
ax = axes[1, 0]
xs, ys = series(xgram, "ppl")
xs2, ys2 = zip(*[(s, y) for s, y in zip(xs, ys) if y is not None and y > 0]) if xs else ([], [])
ax.plot(xs2, ys2, label="X-gram", color="#d62728", lw=1.4)
xs, ys = series(baseline, "ppl")
xs2, ys2 = zip(*[(s, y) for s, y in zip(xs, ys) if y is not None and y > 0]) if xs else ([], [])
ax.plot(xs2, ys2, label="Baseline", color="#1f77b4", lw=1.4)
ax.set_yscale("log")
ax.set_xlabel("step"); ax.set_ylabel("perplexity (log)")
ax.set_title("Training Perplexity"); ax.legend(); ax.grid(alpha=0.3, which="both")

# (4) MFU
ax = axes[1, 1]
xs, ys = series(xgram, "mfu")
filt = [(s, y) for s, y in zip(xs, ys) if y is not None]
if filt:
    xs2, ys2 = zip(*filt)
    ax.plot(xs2, ys2, label="X-gram", color="#d62728", lw=1.0, alpha=0.85)
xs, ys = series(baseline, "mfu")
filt = [(s, y) for s, y in zip(xs, ys) if y is not None]
if filt:
    xs2, ys2 = zip(*filt)
    ax.plot(xs2, ys2, label="Baseline", color="#1f77b4", lw=1.0, alpha=0.85)
ax.set_xlabel("step"); ax.set_ylabel("MFU (%)")
ax.set_title("Device MFU (actual, avg)"); ax.legend(); ax.grid(alpha=0.3)

fig.suptitle(
    "X-gram vs Baseline — SmolLM2-360M × FineWeb-10B (full training, 2385 steps)",
    fontsize=13,
)
fig.tight_layout()
out = OUT_DIR / "comparison_xgram_vs_baseline.png"
fig.savefig(out, dpi=140)
print(f"Wrote {out}")
