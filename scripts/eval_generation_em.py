#!/usr/bin/env python3
"""Generation-based EM evaluation for GSM8K and TriviaQA.

This is intentionally separate from the in-loop downstream evaluator. The
downstream evaluator scores existing continuations with logits; EM tasks require
the model to generate text first, then compare the generated answer to a target.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import re
import string
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import yaml
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from olmo_core.generate import GenerationConfig, TransformerGenerationModule
from olmo_core.nn.transformer import TransformerConfig


REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL_SPECS = {
    "baseline": {
        "config": REPO_ROOT / "configs/our_baseline_config.yaml",
        "checkpoint_root": REPO_ROOT
        / "runs/baseline-smollm2-360m-fineweb10b-64984/step2385",
    },
    "xgram": {
        "config": REPO_ROOT / "configs/our_xgram_config.yaml",
        "checkpoint_root": REPO_ROOT
        / "runs/xgram-smollm2-360m-fineweb10b-64971/step2385",
    },
    "engram": {
        "config": REPO_ROOT / "configs/our_engram_config.yaml",
        "checkpoint_root": REPO_ROOT / "runs/engram-smollm2-360m-fineweb10b/step2385",
    },
}

GSM8K_REQUESTS = (
    REPO_ROOT
    / "packages/olmo_in_loop_evals/src/olmo_eval/oe_eval_tasks/gsm8k/gold_bpb_5shot/requests.jsonl.gz"
)

TASK_DEFAULTS = {
    "gsm8k_em": {
        "max_new_tokens": 512,
        "stop_sequences": ["Question:", "</s>", "<|im_end|>", "\n\n"],
    },
    "triviaqa_em": {
        "max_new_tokens": 32,
        "stop_sequences": ["\n", "Question:", "<|im_end|>"],
    },
}


@dataclass
class EvalExample:
    task: str
    example_id: str
    prompt: str
    references: List[str]
    metadata: Dict[str, Any]


def parse_limit(raw: str) -> Optional[int]:
    if raw.lower() in {"all", "none", "-1"}:
        return None
    value = int(raw)
    if value < 0:
        return None
    return value


def load_public_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def resolve_model_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    spec = MODEL_SPECS[args.model]
    model_env_prefix = args.model.upper()
    config_path = Path(
        args.config
        or os.environ.get(f"{model_env_prefix}_CONFIG")
        or os.environ.get("CONFIG")
        or spec["config"]
    ).expanduser().resolve()
    checkpoint_root = Path(
        args.checkpoint_root
        or os.environ.get(f"{model_env_prefix}_CHECKPOINT_ROOT")
        or os.environ.get("CHECKPOINT_ROOT")
        or spec["checkpoint_root"]
    ).expanduser().resolve()

    if args.tokenizer:
        tokenizer_path = Path(args.tokenizer).expanduser().resolve()
    elif os.environ.get(f"{model_env_prefix}_TOKENIZER") or os.environ.get("TOKENIZER"):
        tokenizer_path = Path(
            os.environ.get(f"{model_env_prefix}_TOKENIZER")
            or os.environ.get("TOKENIZER")
            or ""
        ).expanduser().resolve()
    else:
        cfg = load_public_yaml(config_path)
        try:
            tokenizer_path = Path(cfg["data"]["streaming_tokenizer_model"]).expanduser().resolve()
        except KeyError as e:
            raise KeyError(
                f"{config_path} must define data.streaming_tokenizer_model, or pass --tokenizer"
            ) from e

    return config_path, checkpoint_root, tokenizer_path


def load_tokenizer(tokenizer_path: Path) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)


def choose_pad_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    eos = tokenizer.eos_token_id
    if tokenizer.pad_token_id is not None and tokenizer.pad_token_id != eos:
        return int(tokenizer.pad_token_id)

    # SmolLM2 uses <|endoftext|> as BOS/EOS/UNK and has no separate PAD token.
    # ID 1 is <|im_start|> in this tokenizer; it is only used for left padding
    # while attention_mask tells the generation module to ignore it.
    candidates = [1, tokenizer.unk_token_id, tokenizer.bos_token_id, 0]
    for candidate in candidates:
        if candidate is not None and int(candidate) >= 0 and int(candidate) != eos:
            return int(candidate)

    raise ValueError("Could not choose a pad_token_id different from eos_token_id")


def load_transformer_config(checkpoint_root: Path) -> TransformerConfig:
    config_json = checkpoint_root / "config.json"
    if not config_json.is_file():
        raise FileNotFoundError(
            f"Missing {config_json}. Pass the checkpoint root, e.g. runs/.../step2385, "
            "not only step2385/model_and_optim."
        )
    with config_json.open("r") as f:
        config_dict = json.load(f)
    return TransformerConfig.from_dict(config_dict["model"])


def build_generation_module(
    checkpoint_root: Path,
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_new_tokens: int,
    stop_sequences: Sequence[str],
    use_cache: bool,
) -> TransformerGenerationModule:
    if not torch.cuda.is_available():
        raise RuntimeError("Generation EM eval must run on a GPU node via sbatch/srun.")

    transformer_config = load_transformer_config(checkpoint_root)
    stop_token_ids = single_token_stop_ids(tokenizer, stop_sequences)
    generation_config = GenerationConfig(
        pad_token_id=choose_pad_token_id(tokenizer),
        eos_token_id=int(tokenizer.eos_token_id),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        use_cache=use_cache,
        stop_token_ids=stop_token_ids or None,
    )
    return TransformerGenerationModule.from_checkpoint(
        str(checkpoint_root),
        transformer_config=transformer_config,
        generation_config=generation_config,
        device=torch.device("cuda"),
    )


def flash_attn_available() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


def single_token_stop_ids(
    tokenizer: PreTrainedTokenizerBase, stop_sequences: Sequence[str]
) -> List[int]:
    ids: List[int] = []
    eos = tokenizer.eos_token_id
    for stop in stop_sequences:
        encoded = tokenizer.encode(stop, add_special_tokens=False)
        if len(encoded) == 1 and encoded[0] != eos and encoded[0] not in ids:
            ids.append(int(encoded[0]))
    return ids


def load_gsm8k_examples(limit: Optional[int]) -> List[EvalExample]:
    if not GSM8K_REQUESTS.is_file():
        raise FileNotFoundError(f"Missing GSM8K request file: {GSM8K_REQUESTS}")

    examples: List[EvalExample] = []
    with gzip.open(GSM8K_REQUESTS, "rt") as f:
        for line in f:
            obj = json.loads(line)
            doc = obj["doc"]
            request = obj["request"]
            examples.append(
                EvalExample(
                    task="gsm8k_em",
                    example_id=str(doc.get("id", obj.get("idx", len(examples)))),
                    prompt=request["context"],
                    references=[str(doc["short_answer"])],
                    metadata={"question": doc["question"]},
                )
            )
            if limit is not None and len(examples) >= limit:
                break
    return examples


def load_triviaqa_examples(args: argparse.Namespace, limit: Optional[int]) -> List[EvalExample]:
    from datasets import load_dataset, load_from_disk

    from olmo_eval.util import load_hf_dataset

    if args.triviaqa_dataset_dir:
        ds = load_from_disk(str(Path(args.triviaqa_dataset_dir).expanduser()))
    else:
        try:
            ds = load_hf_dataset("trivia_qa", "rc.wikipedia.nocontext", args.triviaqa_split)
        except (FileNotFoundError, NotADirectoryError):
            load_kwargs: Dict[str, Any] = {
                "path": "trivia_qa",
                "name": "rc.wikipedia.nocontext",
                "split": args.triviaqa_split,
            }
            if args.dataset_cache_dir:
                load_kwargs["cache_dir"] = str(Path(args.dataset_cache_dir).expanduser())
            ds = load_dataset(**load_kwargs)

    examples: List[EvalExample] = []
    for doc in ds:
        answer = doc["answer"]
        aliases = triviaqa_aliases(answer)
        examples.append(
            EvalExample(
                task="triviaqa_em",
                example_id=str(doc.get("question_id", len(examples))),
                prompt="\nQuestion: " + doc["question"] + "\nAnswer:",
                references=aliases,
                metadata={"question": doc["question"]},
            )
        )
        if limit is not None and len(examples) >= limit:
            break
    return examples


def triviaqa_aliases(answer: Dict[str, Any]) -> List[str]:
    aliases: List[str] = []
    for key in ("normalized_aliases", "aliases", "normalized_value", "value"):
        value = answer.get(key)
        if isinstance(value, str):
            aliases.append(value)
        elif isinstance(value, Iterable):
            aliases.extend(str(x) for x in value)

    deduped: List[str] = []
    seen = set()
    for alias in aliases:
        norm = normalize_text_answer(alias)
        if norm and norm not in seen:
            seen.add(norm)
            deduped.append(alias)
    return deduped


def load_examples(args: argparse.Namespace, task: str, limit: Optional[int]) -> List[EvalExample]:
    if task == "gsm8k_em":
        return load_gsm8k_examples(limit)
    if task == "triviaqa_em":
        return load_triviaqa_examples(args, limit)
    raise NotImplementedError(task)


def select_example_slice(
    examples: Sequence[EvalExample],
    *,
    limit: Optional[int],
    offset: int,
    num_shards: Optional[int],
    shard_index: Optional[int],
) -> Tuple[List[EvalExample], str]:
    if offset < 0:
        raise ValueError("--offset must be >= 0")

    if num_shards is None and shard_index is not None:
        raise ValueError("--shard-index requires --num-shards")
    if num_shards is not None:
        if num_shards <= 0:
            raise ValueError("--num-shards must be > 0")
        if shard_index is None:
            raise ValueError("--num-shards requires --shard-index")
        if shard_index < 0 or shard_index >= num_shards:
            raise ValueError("--shard-index must be in [0, num_shards)")
        if offset != 0:
            raise ValueError("--offset cannot be combined with --num-shards")

        total = len(examples)
        start = (total * shard_index) // num_shards
        end = (total * (shard_index + 1)) // num_shards
        selected = list(examples[start:end])
        if limit is not None:
            selected = selected[:limit]
        return selected, f"shard {shard_index}/{num_shards} original_range=[{start}, {end})"

    selected = list(examples[offset:])
    if limit is not None:
        selected = selected[:limit]
    end = offset + len(selected)
    return selected, f"offset_range=[{offset}, {end})"


def encode_prompt(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    max_prompt_tokens: int,
) -> List[int]:
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    bos = tokenizer.bos_token_id
    if bos is not None and (not token_ids or token_ids[0] != bos):
        token_ids = [int(bos)] + token_ids
    if len(token_ids) > max_prompt_tokens:
        token_ids = token_ids[-max_prompt_tokens:]
    return [int(x) for x in token_ids]


def make_left_padded_batch(
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    *,
    max_prompt_tokens: int,
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    encoded = [encode_prompt(tokenizer, prompt, max_prompt_tokens) for prompt in prompts]
    max_len = max(len(x) for x in encoded)
    input_ids = torch.full((len(encoded), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(encoded), max_len), dtype=torch.bool)
    for row, token_ids in enumerate(encoded):
        length = len(token_ids)
        input_ids[row, -length:] = torch.tensor(token_ids, dtype=torch.long)
        attention_mask[row, -length:] = True
    return input_ids, attention_mask


def generate_completions(
    module: TransformerGenerationModule,
    tokenizer: PreTrainedTokenizerBase,
    examples: Sequence[EvalExample],
    *,
    batch_size: int,
    max_prompt_tokens: int,
    max_new_tokens: int,
    stop_sequences: Sequence[str],
    use_cache: bool,
    progress_interval: int,
) -> List[str]:
    if not use_cache and batch_size != 1:
        raise ValueError("--no-cache only supports --batch-size 1 because the model has no padding mask path.")

    completions: List[str] = []
    start_time = time.monotonic()
    if not use_cache:
        for idx, ex in enumerate(examples, start=1):
            completions.append(
                generate_one_without_cache(
                    module,
                    tokenizer,
                    ex.prompt,
                    max_prompt_tokens=max_prompt_tokens,
                    max_new_tokens=max_new_tokens,
                    stop_sequences=stop_sequences,
                )
            )
            if progress_interval > 0 and (idx % progress_interval == 0 or idx == len(examples)):
                elapsed = time.monotonic() - start_time
                print(
                    f"Generated {idx}/{len(examples)} examples "
                    f"({elapsed:.1f}s elapsed, {idx / max(elapsed, 1e-6):.2f} ex/s)",
                    flush=True,
                )
        return completions

    pad_token_id = choose_pad_token_id(tokenizer)
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        input_ids, attention_mask = make_left_padded_batch(
            tokenizer,
            [ex.prompt for ex in batch],
            max_prompt_tokens=max_prompt_tokens,
            pad_token_id=pad_token_id,
        )
        generated_ids, _, _ = module.generate_batch(
            input_ids,
            attention_mask=attention_mask,
            completions_only=True,
            log_timing=False,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )
        decoded = tokenizer.batch_decode(generated_ids.detach().cpu().tolist(), skip_special_tokens=True)
        completions.extend(truncate_at_stop(text, stop_sequences) for text in decoded)
        done = min(start + len(batch), len(examples))
        if progress_interval > 0 and (done % progress_interval == 0 or done == len(examples)):
            elapsed = time.monotonic() - start_time
            print(
                f"Generated {done}/{len(examples)} examples "
                f"({elapsed:.1f}s elapsed, {done / max(elapsed, 1e-6):.2f} ex/s)",
                flush=True,
            )
    return completions


@torch.inference_mode()
def generate_one_without_cache(
    module: TransformerGenerationModule,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    *,
    max_prompt_tokens: int,
    max_new_tokens: int,
    stop_sequences: Sequence[str],
) -> str:
    prompt_ids = encode_prompt(tokenizer, prompt, max_prompt_tokens)
    generated_ids: List[int] = []
    eos = int(tokenizer.eos_token_id)

    for _ in range(max_new_tokens):
        input_ids = torch.tensor([prompt_ids + generated_ids], dtype=torch.long, device=module.device)
        logits = module.model_forward(input_ids, logits_to_keep=1)
        next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
        if next_id == eos:
            break
        generated_ids.append(next_id)

        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        if any(stop and stop in text for stop in stop_sequences):
            break

    return truncate_at_stop(tokenizer.decode(generated_ids, skip_special_tokens=True), stop_sequences)


def truncate_at_stop(text: str, stop_sequences: Sequence[str]) -> str:
    text = text.lstrip()
    indices = [text.find(stop) for stop in stop_sequences if stop and text.find(stop) >= 0]
    if indices:
        text = text[: min(indices)]
    return text.strip()


def normalize_text_answer(text: str) -> str:
    text = text.lower()
    text = "".join(" " if ch in string.punctuation else ch for ch in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def extract_gsm8k_answer(completion: str) -> str:
    if "####" in completion:
        completion = completion.split("####")[-1]
    matches = NUMBER_RE.findall(completion)
    if not matches:
        return completion.strip()
    return normalize_number(matches[-1])


def normalize_number(text: str) -> str:
    cleaned = text.strip().replace(",", "").replace("$", "")
    cleaned = cleaned.rstrip(".")
    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return cleaned
    if value == value.to_integral():
        return str(value.to_integral())
    return format(value.normalize(), "f")


def score_example(task: str, completion: str, references: Sequence[str]) -> Tuple[str, int]:
    if task == "gsm8k_em":
        prediction = extract_gsm8k_answer(completion)
        gold = {normalize_number(ref) for ref in references}
        return prediction, int(normalize_number(prediction) in gold)

    if task == "triviaqa_em":
        prediction = re.sub(r"^\s*answer\s*:\s*", "", completion, flags=re.IGNORECASE).strip()
        norm_pred = normalize_text_answer(prediction)
        norm_refs = {normalize_text_answer(ref) for ref in references}
        return prediction, int(norm_pred in norm_refs)

    raise NotImplementedError(task)


def write_results(
    output_dir: Path,
    *,
    model: str,
    task: str,
    run_label: Optional[str],
    examples: Sequence[EvalExample],
    completions: Sequence[str],
) -> Tuple[Path, Path, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    label = f"_{run_label}" if run_label else ""
    records_path = output_dir / f"generation_em_{model}_{task}{label}_{timestamp}.jsonl"
    summary_path = output_dir / f"generation_em_{model}_{task}{label}_{timestamp}.summary.json"

    correct = 0
    with records_path.open("w") as f:
        for ex, completion in zip(examples, completions):
            prediction, em = score_example(task, completion, ex.references)
            correct += em
            record = {
                "task": task,
                "model": model,
                "id": ex.example_id,
                "question": ex.metadata.get("question"),
                "completion": completion,
                "prediction": prediction,
                "references": ex.references,
                "exact_match": em,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    em_score = correct / len(examples) if examples else 0.0
    summary = {
        "model": model,
        "task": task,
        "run_label": run_label,
        "num_examples": len(examples),
        "correct": correct,
        "exact_match": em_score,
        "records": str(records_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return records_path, summary_path, em_score


def preview_examples(examples: Sequence[EvalExample], max_items: int) -> None:
    for ex in examples[:max_items]:
        print("=" * 80)
        print(f"id={ex.example_id} task={ex.task}")
        print("prompt tail:")
        print(ex.prompt[-1000:])
        print("references:", ex.references[:8])


def run_task(
    args: argparse.Namespace,
    task: str,
    tokenizer: PreTrainedTokenizerBase,
    checkpoint_root: Path,
    limit: Optional[int],
) -> None:
    defaults = TASK_DEFAULTS[task]
    max_new_tokens = args.max_new_tokens or int(defaults["max_new_tokens"])
    stop_sequences = list(defaults["stop_sequences"])
    load_limit = None if (args.offset or args.num_shards is not None) else limit
    all_examples = load_examples(args, task, load_limit)
    examples, slice_desc = select_example_slice(
        all_examples,
        limit=limit,
        offset=args.offset,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    if not examples:
        raise RuntimeError(f"No examples loaded for {task}")

    print(f"Task: {task}")
    print(f"Model: {args.model}")
    print(f"Example slice: {slice_desc}")
    print(f"Examples: {len(examples)}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Stop sequences: {stop_sequences!r}")
    if args.print_samples:
        preview_examples(examples, args.print_samples)
    if args.dry_run:
        print("Dry run complete: examples loaded, model generation skipped.")
        return

    use_cache = not args.no_cache
    if use_cache and not flash_attn_available():
        print("flash-attn is not installed; falling back to slower no-cache generation.")
        use_cache = False

    module = build_generation_module(
        checkpoint_root,
        tokenizer,
        max_new_tokens=max_new_tokens,
        stop_sequences=stop_sequences,
        use_cache=use_cache,
    )
    completions = generate_completions(
        module,
        tokenizer,
        examples,
        batch_size=args.batch_size,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=max_new_tokens,
        stop_sequences=stop_sequences,
        use_cache=use_cache,
        progress_interval=args.progress_interval,
    )
    records_path, summary_path, em_score = write_results(
        Path(args.output_dir),
        model=args.model,
        task=task,
        run_label=args.run_label,
        examples=examples,
        completions=completions,
    )
    print(f"{task} exact_match={em_score:.4f}")
    print(f"Records: {records_path}")
    print(f"Summary: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--task", choices=["gsm8k_em", "triviaqa_em", "all"], required=True)
    parser.add_argument("--limit", default="5", help="Number of examples, or 'all'. Default: 5")
    parser.add_argument("--offset", type=int, default=0, help="Start index before applying --limit")
    parser.add_argument("--num-shards", type=int, default=None, help="Split all examples into this many contiguous shards")
    parser.add_argument("--shard-index", type=int, default=None, help="0-based shard index to run")
    parser.add_argument("--run-label", default=None, help="Extra label inserted into output filenames")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--progress-interval", type=int, default=50)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-samples", type=int, default=0)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "reports"))

    parser.add_argument("--config", default=None, help="Override model YAML config path")
    parser.add_argument("--checkpoint-root", default=None, help="Override checkpoint root path")
    parser.add_argument("--tokenizer", default=None, help="Override local HF tokenizer/model folder")

    parser.add_argument("--triviaqa-split", default="validation")
    parser.add_argument("--triviaqa-dataset-dir", default=None, help="Optional load_from_disk path")
    parser.add_argument("--dataset-cache-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path, checkpoint_root, tokenizer_path = resolve_model_paths(args)
    limit = parse_limit(args.limit)

    print(f"Config: {config_path}")
    print(f"Checkpoint root: {checkpoint_root}")
    print(f"Tokenizer: {tokenizer_path}")
    tokenizer = load_tokenizer(tokenizer_path)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id")

    tasks = ["gsm8k_em", "triviaqa_em"] if args.task == "all" else [args.task]
    for task in tasks:
        run_task(args, task, tokenizer, checkpoint_root, limit)


if __name__ == "__main__":
    main()
