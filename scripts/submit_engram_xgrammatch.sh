#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CONFIG="${1:-configs/faircompare_engram_vpath_xgrammatch_360m.yaml}"
RUN_NAME="${2:-faircompare-engram-vpath-xgrammatch}"

if [[ "$#" -eq 0 ]]; then
  DEFAULT_VALIDATE_MODE="strict"
else
  DEFAULT_VALIDATE_MODE="variant"
fi
VALIDATE_MODE="${VALIDATE_ENGRAM_XGRAMMATCH:-${DEFAULT_VALIDATE_MODE}}"

case "${VALIDATE_MODE}" in
  1|strict)
    python scripts/validate_engram_xgrammatch_config.py --config "${CONFIG}" --strict-xgrammatch
    ;;
  variant)
    python scripts/validate_engram_xgrammatch_config.py --config "${CONFIG}" --allow-engram-variant
    ;;
  0|skip)
    echo "Skipping Engram xgrammatch validation"
    ;;
  *)
    echo "Unsupported VALIDATE_ENGRAM_XGRAMMATCH=${VALIDATE_MODE}" >&2
    exit 2
    ;;
esac

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "sbatch --export=ALL,VALIDATE_ENGRAM_XGRAMMATCH=${VALIDATE_MODE} python_slurm/train_engram_xgrammatch.slurm ${CONFIG} ${RUN_NAME}"
else
  sbatch --export="ALL,VALIDATE_ENGRAM_XGRAMMATCH=${VALIDATE_MODE}" python_slurm/train_engram_xgrammatch.slurm "${CONFIG}" "${RUN_NAME}"
fi
