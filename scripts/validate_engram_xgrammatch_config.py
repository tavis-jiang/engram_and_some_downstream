#!/usr/bin/env python3
"""Validate Engram xgrammatch configs against the existing controls."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "configs/faircompare_baseline_360m.yaml"
XGRAM = ROOT / "configs/faircompare_xgram_360m.yaml"
ENGRAM = ROOT / "configs/faircompare_engram_vpath_xgrammatch_360m.yaml"


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = yaml.safe_load(f)
    if not isinstance(value, dict):
        raise AssertionError(f"{path} did not parse to a mapping")
    return value


def resolved_config(path: Path) -> Dict[str, Any]:
    output = subprocess.check_output(
        [sys.executable, "scripts/train/olmo_train.py", "--config", str(path), "--print-resolved-config"],
        cwd=ROOT,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return json.loads(output)["config"]


def assert_equal(label: str, lhs: Any, rhs: Any) -> None:
    if lhs != rhs:
        raise AssertionError(f"{label} mismatch:\nleft={lhs!r}\nright={rhs!r}")


def validate_common_controls(
    *,
    config_path: Path,
    candidate_yaml: Dict[str, Any],
    baseline_yaml: Dict[str, Any],
    xgram_yaml: Dict[str, Any],
    reference_engram_yaml: Dict[str, Any],
) -> None:
    for section in ("data", "model", "evaluation"):
        assert_equal(f"Engram/X-gram raw {section}", candidate_yaml.get(section), xgram_yaml.get(section))
        assert_equal(f"Engram/baseline raw {section}", candidate_yaml.get(section), baseline_yaml.get(section))

    candidate_train = candidate_yaml["training"]
    reference_train = reference_engram_yaml["training"]
    for key in (
        "seq_len",
        "global_batch_size",
        "train_tokens",
        "warmup_fraction",
        "save_tokens",
        "lr",
        "min_lr",
        "ac_mode",
        "log_interval",
    ):
        assert_equal(f"training.{key}", candidate_train.get(key), reference_train.get(key))

    if int(candidate_train.get("micro_batch_size", 0)) != 1:
        raise AssertionError("Engram variants must keep training.micro_batch_size=1 to avoid changing memory behavior")

    resolved = resolved_config(config_path)
    x_resolved = resolved_config(XGRAM)
    for key in ("ub_global_batch_size", "seq_len", "text_chunk_size", "prefetch_queue_size", "pack_method"):
        assert_equal(f"resolved data_loader.{key}", resolved["data_loader"].get(key), x_resolved["data_loader"].get(key))
    for key in ("d_model", "n_layers", "vocab_size"):
        assert_equal(f"resolved model.{key}", resolved["model"].get(key), x_resolved["model"].get(key))
    assert_equal(
        "resolved optimizer base lr",
        resolved["train_module"]["optim"].get("lr"),
        x_resolved["train_module"]["optim"].get("lr"),
    )
    assert_equal(
        "resolved scheduler",
        resolved["train_module"]["scheduler"],
        x_resolved["train_module"]["scheduler"],
    )


def validate_strict_xgrammatch(candidate_yaml: Dict[str, Any]) -> None:
    xgram_yaml = load_yaml(XGRAM)
    einj = candidate_yaml["embedding_injection"]
    xinj = xgram_yaml["embedding_injection"]
    if einj.get("mode") != "Engram":
        raise AssertionError("Engram xgrammatch config must use mode: Engram")
    if einj.get("targets") != xinj.get("targets") or einj.get("targets") != ["v"]:
        raise AssertionError("Engram xgrammatch must use the existing X-gram v-path target")
    assert_equal("v_layers", einj.get("v_layers"), xinj.get("v_layers"))
    assert_equal("lambda_init", einj.get("lambda_init"), xinj.get("lambda_init"))
    assert_equal("lambda_warmup_enabled", einj.get("lambda_warmup_enabled"), xinj.get("lambda_warmup_enabled"))
    assert_equal("lambda_warmup_steps", einj.get("lambda_warmup_steps"), xinj.get("lambda_warmup_steps"))
    assert_equal("depth_scale_disabled", einj.get("depth_scale_disabled"), xinj.get("depth_scale_disabled"))
    assert_equal("shortconv kernels", einj.get("engram_shortconv_kernels"), xinj.get("shortconv_kernels"))
    assert_equal("shortconv enabled", einj.get("engram_shortconv_enabled"), xinj.get("shortconv_enabled"))
    if "h_layers" in einj:
        raise AssertionError("Engram xgrammatch must not set h_layers")

    train = candidate_yaml["training"]
    expected_warmup_steps = math.ceil(
        int(train["train_tokens"])
        * float(train["warmup_fraction"])
        / (int(train["global_batch_size"]) * int(train["seq_len"]))
    )
    assert_equal("lambda warmup formula", einj.get("lambda_warmup_steps"), expected_warmup_steps)


def validate_engram_variant(candidate_yaml: Dict[str, Any]) -> None:
    xgram_yaml = load_yaml(XGRAM)
    einj = candidate_yaml["embedding_injection"]
    xinj = xgram_yaml["embedding_injection"]
    if einj.get("mode") != "Engram":
        raise AssertionError("Engram variant config must use mode: Engram")
    if einj.get("targets") != ["v"]:
        raise AssertionError("Engram variants in this sweep must keep targets: [v]")
    if "h_layers" in einj:
        raise AssertionError("Engram v-path variants must not set h_layers")
    assert_equal("depth_scale_disabled", einj.get("depth_scale_disabled"), xinj.get("depth_scale_disabled"))
    assert_equal("shortconv kernels", einj.get("engram_shortconv_kernels"), xinj.get("shortconv_kernels"))
    assert_equal("shortconv enabled", einj.get("engram_shortconv_enabled"), xinj.get("shortconv_enabled"))

    v_layers = einj.get("v_layers")
    if not isinstance(v_layers, list) or not v_layers:
        raise AssertionError("Engram variants must specify non-empty v_layers")
    if any(not isinstance(layer_idx, int) or layer_idx < 0 or layer_idx >= 32 for layer_idx in v_layers):
        raise AssertionError(f"Engram variant v_layers out of supported 0..31 range: {v_layers!r}")

    if bool(einj.get("lambda_warmup_enabled")) and int(einj.get("lambda_warmup_steps", 0) or 0) <= 0:
        raise AssertionError("Engram variants with lambda_warmup_enabled must set positive lambda_warmup_steps")
    if float(einj.get("lambda_init", 0.0)) <= 0.0:
        raise AssertionError("Engram variants must set positive lambda_init")

    mode_parts = {part.strip() for part in str(einj.get("engram_mode", "")).split("+") if part.strip()}
    if not mode_parts:
        raise AssertionError("Engram variants must set engram_mode")
    for part in mode_parts:
        if not part.endswith("gram"):
            raise AssertionError(f"Unsupported engram_mode part: {part!r}")
        level = int(part.replace("gram", ""))
        if level < 2:
            raise AssertionError("This Engram sweep only supports >=2gram modes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ENGRAM,
        help="Engram config to validate. Defaults to the canonical v-path xgrammatch config.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--strict-xgrammatch",
        action="store_true",
        help="Require Engram injection knobs to match the X-gram v-path control exactly.",
    )
    mode.add_argument(
        "--allow-engram-variant",
        action="store_true",
        help="Allow Engram injection knobs to differ while enforcing non-injection controls.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = ROOT / config_path

    baseline_yaml = load_yaml(BASELINE)
    xgram_yaml = load_yaml(XGRAM)
    reference_engram_yaml = load_yaml(ENGRAM)
    candidate_yaml = load_yaml(config_path)

    validate_common_controls(
        config_path=config_path,
        candidate_yaml=candidate_yaml,
        baseline_yaml=baseline_yaml,
        xgram_yaml=xgram_yaml,
        reference_engram_yaml=reference_engram_yaml,
    )

    if args.allow_engram_variant:
        validate_engram_variant(candidate_yaml)
        validation_label = "Engram variant"
    else:
        validate_strict_xgrammatch(candidate_yaml)
        validation_label = "Engram strict xgrammatch"

    print(f"{validation_label} config validation passed")
    print(f"engram: {config_path.relative_to(ROOT)}")
    print(f"controls: {BASELINE.relative_to(ROOT)}, {XGRAM.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
