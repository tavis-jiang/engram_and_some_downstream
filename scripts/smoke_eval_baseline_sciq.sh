#!/usr/bin/env bash
set -euo pipefail

cd /home/bcjiang/X-gram
mkdir -p logs out reports runs_eval

job_id="$(sbatch --parsable python_slurm/eval_downstream.slurm baseline sciq)"

echo "Submitted baseline SciQ smoke test: job ${job_id}"
echo "Check queue: squeue -u \$USER"
echo "After it finishes, inspect the latest log with:"
echo "  rg \"Task: sciq|sciq|Run canceled\" \"\$(ls -t logs/downstream_eval_baseline_step2385_sciq_* | head -1)\""
