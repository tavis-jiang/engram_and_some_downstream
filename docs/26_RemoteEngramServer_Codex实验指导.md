# Remote Engram Server Codex 实验指导

本文档给另一台服务器上的 Codex 使用。目标是在新服务器从零复现
Baseline / X-gram / Engram，并继续跑无内存限制条件下的 Engram fair
compare。不要使用本机旧 `runs/` 作为结论来源；新服务器需要重新训练
Baseline 和 X-gram 作为同机对照。

## 1. 实验目标

最终要产出一张同机、同数据、同训练预算的表：

| 模型 | Config | step2385 Train CE/PPL | Full downstream Global | MMLU | SciQ |
| --- | --- | --- | --- | --- | --- |
| Baseline | `configs/faircompare_baseline_360m.yaml` | 待跑 | 待跑 | 待跑 | 待跑 |
| X-gram | `configs/faircompare_xgram_360m.yaml` | 待跑 | 待跑 | 待跑 | 待跑 |
| Engram strict v-path | `configs/faircompare_engram_vpath_xgrammatch_360m.yaml` 的远端 full-microbatch 版 | 待跑 | 待跑 | 待跑 | 待跑 |
| Engram 2g3g v-path | `configs/faircompare_engram_2g3g_vpath_d512_lam05_warm240_360m.yaml` 的远端 full-microbatch 版 | 待跑 | 待跑 | 待跑 | 待跑 |
| Engram single-v v-path | `configs/faircompare_engram_vpath_single_lam05_warm240_360m.yaml` 的远端 full-microbatch 版 | 待跑 | 待跑 | 待跑 | 待跑 |

核心要求：

- 除了 injection 机制本身，Baseline / X-gram / Engram 的数据、tokenizer、backbone、训练 token、global batch、LR、scheduler、保存点、eval 任务都要对齐。
- 新服务器没有内存限制，所以 Engram 不要继承旧机器为了避免 OOM 设置的 `micro_batch_size=1` / `prefetch_queue_size=4`，除非实际硬件仍然需要。
- 旧机器上 `93378/93379` 已证明降低 injection 强度能把 CE 拉近 X-gram，但 downstream 没有同步提升。因此远端优先验证“同机、full microbatch 后 CE/downstream 是否改善”，不要只重复旧 OOM 设置。

## 2. 当前已知背景

旧 Engram 的主要问题不是 X-gram setting，而是 Engram 旧 H-path 实现和配置不公平：

- 旧 Engram 没有和 X-gram 对齐 warm-up。
- 旧 H-path 没有统一乘 `lambda * depth_scale * warmup`。
- 旧 v-path、参数统计、optimizer override 覆盖不完整。
- 因为 OOM，旧机器上的新版 v-path Engram 还使用了较保守的 microbatch/prefetch 设置。

当前代码已经修复：

- Engram 支持 lambda warm-up。
- H-path 统一使用外部 `lambda * depth_scale * warmup`。
- Engram 支持 X-gram 风格 v-path。
- Engram eval 对 post-warmup checkpoint 默认关闭 eval-time warmup，避免 model-only eval 从 step 0 开始导致 injection 被关掉。
- `scripts/validate_engram_xgrammatch_config.py` 可检查 Engram config 是否和 controls 对齐。

本机最近结果仅作参考，不作为远端最终结论：

| Run | 关键设置 | Train CE/PPL | Full downstream |
| --- | --- | --- | --- |
| `91976` | strict v-path, 2gram, lambda 1.0, warm120 | CE 3.195, PPL 24.41 | Global 0.4073 |
| `93378` | 2g+3g, d512, lambda 0.5, warm240 | CE 3.055, PPL 21.22 | Global 0.3995 |
| `93379` | single v-layer, 2gram, lambda 0.5, warm240 | CE 3.048, PPL 21.08 | Global 0.3962 |

解释：`93378/93379` 的 CE 明显改善，但 downstream 不好，说明“降低注入强度”保护了 LM loss，却不一定让 Engram 学到更好的可迁移特征。远端要优先排除旧机器内存策略和硬件/数据环境造成的影响。

## 3. 获取代码和环境安装

推荐从你的私有 GitHub 仓库拉完整镜像；如果代码、配置、数据、tokenizer、runs、logs 都已经在仓库里，新服务器只需要 clone 一次：

```bash
git clone <your-github-repo-url> X-gram
cd /path/to/X-gram
git status --short
```

如果你还没把全量内容放进私有仓库，再看 `docs/26_RemoteEngramServer_复制清单.md` 里的 tar/rsync 方案。

然后安装环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e ./packages/streaming
pip install -e ".[train]"
pip install -e ./packages/olmo_in_loop_evals
```

建议确认：

```bash
python - <<'PY'
import torch
import olmo_core
import ubdataloader
import olmo_eval
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("olmo_core", olmo_core.__file__)
print("ubdataloader", ubdataloader.__file__)
print("olmo_eval", olmo_eval.__file__)
PY
```

如果训练需要 Slurm，先把 `python_slurm/*.slurm` 里的以下硬编码路径改成远端路径：

- `cd /home/bcjiang/X-gram`
- `#SBATCH --output=/home/bcjiang/X-gram/out/%j.out`
- `#SBATCH --error=/home/bcjiang/X-gram/out/%j.err`
- `source /home/bcjiang/miniconda3/etc/profile.d/conda.sh`
- `conda activate xgram_env`
- `STREAMING_TOKENIZER_MODEL=/home/bcjiang/.../SmolLM2-360M/...`
- `PYTHONPATH=/home/bcjiang/X-gram/...`

如果不用 Slurm，直接用 `scripts/train/olmo_train.sh`。

## 4. 远端路径统一

设置远端本地路径：

```bash
export ROOT=/path/to/X-gram
export TOKENIZER=$ROOT/assets/tokenizers/SmolLM2-360M
export DATA=$ROOT/data/fineweb_10b_streaming
export PYTHONPATH="$ROOT/OLMo-core/src:$ROOT/packages/ubdataloader/src:$ROOT/packages/olmo_in_loop_evals/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

关键：Engram YAML 里的 `embedding_injection.engram_tokenizer_id` 可以用环境变量覆盖：

```bash
export ENGRAM_TOKENIZER_ID="$TOKENIZER"
export STREAMING_TOKENIZER_MODEL="$TOKENIZER"
```

但 Slurm 脚本里如果写死了 `STREAMING_TOKENIZER_MODEL`，仍然要改 Slurm 脚本。

## 5. 创建远端 full-microbatch Engram configs

不要直接改原始 configs，复制出远端专用版本：

```bash
cp configs/faircompare_engram_vpath_xgrammatch_360m.yaml \
  configs/remote_fullmb_engram_vpath_xgrammatch_360m.yaml
cp configs/faircompare_engram_2g3g_vpath_d512_lam05_warm240_360m.yaml \
  configs/remote_fullmb_engram_2g3g_vpath_d512_lam05_warm240_360m.yaml
cp configs/faircompare_engram_vpath_single_lam05_warm240_360m.yaml \
  configs/remote_fullmb_engram_vpath_single_lam05_warm240_360m.yaml
```

在这三个远端专用 YAML 里改：

```yaml
training:
  micro_batch_size: 4

data:
  prefetch_queue_size: 16
  streaming_tokenizer_model: /path/to/X-gram/assets/tokenizers/SmolLM2-360M

embedding_injection:
  engram_tokenizer_id: /path/to/X-gram/assets/tokenizers/SmolLM2-360M
```

如果远端 GPU 数或显存允许，也可以把 Baseline/X-gram/Engram 都设成同样的更大 `micro_batch_size`，但必须三类模型一起改，并保持 `global_batch_size: 512` 不变。

## 6. Config 验证

先验证原始 Engram configs 是否和 X-gram 对齐：

```bash
python scripts/validate_engram_xgrammatch_config.py \
  --config configs/faircompare_engram_vpath_xgrammatch_360m.yaml \
  --strict-xgrammatch

python scripts/validate_engram_xgrammatch_config.py \
  --config configs/faircompare_engram_2g3g_vpath_d512_lam05_warm240_360m.yaml \
  --allow-engram-variant

python scripts/validate_engram_xgrammatch_config.py \
  --config configs/faircompare_engram_vpath_single_lam05_warm240_360m.yaml \
  --allow-engram-variant
```

注意：这个 validator 是旧服务器 OOM 口径，会强制要求 Engram `training.micro_batch_size=1`。远端 full-microbatch 配置把 Engram 改成 `micro_batch_size=4` 后，不要直接用这个 validator，除非先 patch 掉该断言并改成和 Baseline/X-gram 对齐检查。

远端 full-microbatch 配置用 resolved config 检查：

```bash
python scripts/train/olmo_train.py \
  --config configs/faircompare_baseline_360m.yaml \
  --print-resolved-config > /tmp/baseline_resolved.json

python scripts/train/olmo_train.py \
  --config configs/faircompare_xgram_360m.yaml \
  --print-resolved-config > /tmp/xgram_resolved.json

python scripts/train/olmo_train.py \
  --config configs/remote_fullmb_engram_2g3g_vpath_d512_lam05_warm240_360m.yaml \
  --print-resolved-config > /tmp/engram_resolved.json
```

最少检查这些字段：

```bash
python - <<'PY'
import json
from pathlib import Path

def load(path):
    return json.loads(Path(path).read_text())["config"]

b = load("/tmp/baseline_resolved.json")
x = load("/tmp/xgram_resolved.json")
e = load("/tmp/engram_resolved.json")

for key in ("d_model", "n_layers", "vocab_size"):
    assert b["model"][key] == x["model"][key] == e["model"][key], key

for key in ("seq_len", "ub_global_batch_size", "text_chunk_size", "prefetch_queue_size", "pack_method"):
    assert b["data_loader"][key] == x["data_loader"][key] == e["data_loader"][key], key

assert b["train_module"]["rank_microbatch_size"] == x["train_module"]["rank_microbatch_size"] == e["train_module"]["rank_microbatch_size"]
assert e["data_loader"]["ub_global_batch_size"] == 512
assert e["train_module"]["rank_microbatch_size"] == 4 * 8192
assert e["model"]["embedding_injection"]["mode"] == "Engram"
assert e["model"]["embedding_injection"]["lambda_warmup_enabled"] is True
print("remote fullmb controls look aligned")
PY
```

还要人工确认：

- `data_loader.data_path` 指向远端 `data/fineweb_10b_streaming`。
- `data_loader.tokenizer_path` 是远端 tokenizer。
- `model.embedding_injection.engram_tokenizer_id` 是远端 tokenizer。
- post-warmup eval 时会关闭 warmup。

## 7. 训练顺序

先跑 controls，再跑 Engram。

直接 `torchrun`/shell 方式：

```bash
mkdir -p logs out runs runs_eval reports

export GPUS_PER_NODE=8
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=localhost
export SAVE_ROOT=./runs
export STREAMING_TOKENIZER_MODEL="$TOKENIZER"
export ENGRAM_TOKENIZER_ID="$TOKENIZER"
export STREAMING_USE_TOKEN_COLUMN=tokens
export TEXT_CHUNK_SIZE=4096
export PREFETCH_QUEUE_SIZE=16
export STREAMING_PACK_METHOD=native:truncate

RUN_NAME=remote-faircompare-baseline \
STREAMING_DATA_PATH="1 /tmp/remote-baseline-cache ./data/fineweb_10b_streaming" \
MASTER_PORT=6101 \
bash scripts/train/olmo_train.sh configs/faircompare_baseline_360m.yaml

RUN_NAME=remote-faircompare-xgram \
STREAMING_DATA_PATH="1 /tmp/remote-xgram-cache ./data/fineweb_10b_streaming" \
MASTER_PORT=6102 \
bash scripts/train/olmo_train.sh configs/faircompare_xgram_360m.yaml

RUN_NAME=remote-engram-vpath-xgrammatch-fullmb \
STREAMING_DATA_PATH="1 /tmp/remote-engram-strict-cache ./data/fineweb_10b_streaming" \
MASTER_PORT=6103 \
bash scripts/train/olmo_train.sh configs/remote_fullmb_engram_vpath_xgrammatch_360m.yaml

RUN_NAME=remote-engram-2g3g-vpath-d512-lam05-warm240-fullmb \
STREAMING_DATA_PATH="1 /tmp/remote-engram-2g3g-cache ./data/fineweb_10b_streaming" \
MASTER_PORT=6104 \
bash scripts/train/olmo_train.sh configs/remote_fullmb_engram_2g3g_vpath_d512_lam05_warm240_360m.yaml

RUN_NAME=remote-engram-single-v-lam05-warm240-fullmb \
STREAMING_DATA_PATH="1 /tmp/remote-engram-single-cache ./data/fineweb_10b_streaming" \
MASTER_PORT=6105 \
bash scripts/train/olmo_train.sh configs/remote_fullmb_engram_vpath_single_lam05_warm240_360m.yaml
```

Slurm 方式：

```bash
sbatch python_slurm/train_faircompare.slurm \
  configs/faircompare_baseline_360m.yaml remote-faircompare-baseline

sbatch python_slurm/train_faircompare.slurm \
  configs/faircompare_xgram_360m.yaml remote-faircompare-xgram

sbatch --export=ALL,VALIDATE_ENGRAM_XGRAMMATCH=skip,PREFETCH_QUEUE_SIZE=16 \
  python_slurm/train_engram_xgrammatch.slurm \
  configs/remote_fullmb_engram_vpath_xgrammatch_360m.yaml \
  remote-engram-vpath-xgrammatch-fullmb

sbatch --export=ALL,VALIDATE_ENGRAM_XGRAMMATCH=skip,PREFETCH_QUEUE_SIZE=16 \
  python_slurm/train_engram_xgrammatch.slurm \
  configs/remote_fullmb_engram_2g3g_vpath_d512_lam05_warm240_360m.yaml \
  remote-engram-2g3g-vpath-d512-lam05-warm240-fullmb

sbatch --export=ALL,VALIDATE_ENGRAM_XGRAMMATCH=skip,PREFETCH_QUEUE_SIZE=16 \
  python_slurm/train_engram_xgrammatch.slurm \
  configs/remote_fullmb_engram_vpath_single_lam05_warm240_360m.yaml \
  remote-engram-single-v-lam05-warm240-fullmb
```

这里用 `VALIDATE_ENGRAM_XGRAMMATCH=skip` 是因为旧 validator 会拒绝 `micro_batch_size=4`。提交前必须已经完成本节 resolved-config 检查。

不要用 `python_slurm/train_faircompare.slurm` 跑 full-microbatch Engram，除非先删掉里面的 Engram OOM policy；那个脚本会把 Engram `MICRO_BATCH_SIZE` 默认压回 1。

## 8. 训练完成判定

每个 run 必须有：

- `runs/<run_name>/step596`
- `runs/<run_name>/step1192`
- `runs/<run_name>/step1788`
- `runs/<run_name>/step2384`
- `runs/<run_name>/step2385`
- rank0 日志里出现 `Training complete`
- rank0 日志里出现 `Cleanup process completed successfully`

提取 final 训练点：

```bash
rg -n "\\[step=2380/2385|\\[step=2385/2385|train/CE loss|train/PPL|Training complete" logs/*rank0.log
```

记录 step2380 附近的 CE/PPL。不要只看 `step2385`，因为最后一行可能只保存 checkpoint，不一定打印 metrics。

## 9. Downstream eval

每个模型先跑 SciQ smoke，再跑 full downstream。post-warmup X-gram / Engram checkpoint 评测时必须关闭 eval-time warmup；`python_slurm/eval_downstream.slurm` 已默认处理。

示例：

```bash
BASELINE_CONFIG=configs/faircompare_baseline_360m.yaml \
BASELINE_CHECKPOINT_ROOT=runs/remote-faircompare-baseline/step2385 \
CHECKPOINT_LABEL=remote_baseline_step2385 \
sbatch python_slurm/eval_downstream.slurm baseline sciq

BASELINE_CONFIG=configs/faircompare_baseline_360m.yaml \
BASELINE_CHECKPOINT_ROOT=runs/remote-faircompare-baseline/step2385 \
CHECKPOINT_LABEL=remote_baseline_step2385 \
sbatch python_slurm/eval_downstream.slurm baseline

XGRAM_CONFIG=configs/faircompare_xgram_360m.yaml \
XGRAM_CHECKPOINT_ROOT=runs/remote-faircompare-xgram/step2385 \
CHECKPOINT_LABEL=remote_xgram_step2385 \
sbatch python_slurm/eval_downstream.slurm xgram sciq

XGRAM_CONFIG=configs/faircompare_xgram_360m.yaml \
XGRAM_CHECKPOINT_ROOT=runs/remote-faircompare-xgram/step2385 \
CHECKPOINT_LABEL=remote_xgram_step2385 \
sbatch python_slurm/eval_downstream.slurm xgram

ENGRAM_CONFIG=configs/remote_fullmb_engram_2g3g_vpath_d512_lam05_warm240_360m.yaml \
ENGRAM_CHECKPOINT_ROOT=runs/remote-engram-2g3g-vpath-d512-lam05-warm240-fullmb/step2385 \
CHECKPOINT_LABEL=remote_engram_2g3g_fullmb_step2385 \
sbatch python_slurm/eval_downstream.slurm engram sciq

ENGRAM_CONFIG=configs/remote_fullmb_engram_2g3g_vpath_d512_lam05_warm240_360m.yaml \
ENGRAM_CHECKPOINT_ROOT=runs/remote-engram-2g3g-vpath-d512-lam05-warm240-fullmb/step2385 \
CHECKPOINT_LABEL=remote_engram_2g3g_fullmb_step2385 \
sbatch python_slurm/eval_downstream.slurm engram
```

对另外两个 Engram config 重复上面的 Engram eval 命令。

评测完成后从日志提取：

```bash
rg -n "MMLU Average|Global Average|eval/downstream/sciq|eval/downstream/arc_easy|eval/downstream/hellaswag" logs/downstream_eval_*remote*.log
```

## 10. 可选 Generation EM

Downstream full 完成后再跑 GSM8K / TriviaQA EM。先 smoke：

```bash
TOKENIZER="$TOKENIZER" \
CHECKPOINT_ROOT=runs/remote-faircompare-xgram/step2385 \
CONFIG=configs/faircompare_xgram_360m.yaml \
sbatch python_slurm/eval_generation_em.slurm xgram gsm8k_em 5
```

全量建议分 shard：

```bash
NUM_SHARDS=8 sbatch --array=0-7 python_slurm/eval_generation_em.slurm xgram gsm8k_em all
NUM_SHARDS=16 sbatch --array=0-15 python_slurm/eval_generation_em.slurm xgram triviaqa_em all
```

Baseline 和 Engram 同样跑。TriviaQA 如果本地没有缓存，可以设置 `ALLOW_ONLINE_DATASETS=1`，但为了复现最好复制原来的 eval dataset cache。

## 11. 结果判断

远端最终报告必须至少包含：

- 每个模型的 config 路径和 checkpoint 路径。
- 训练是否完整到 `step2385`。
- step2380 附近 Train CE/PPL。
- SciQ smoke 是否通过。
- Full downstream 12 项明细、MMLU Average、Global Average。
- X-gram 相对 Baseline 提升。
- 每个 Engram 相对 Baseline / X-gram 的差值。

判定口径：

- 官方主对比：Baseline vs X-gram vs strict v-path Engram full-microbatch。
- Engram 优化主线：`2g3g d512 lambda0.5 warm240 full-microbatch` 和 `single-v lambda0.5 warm240 full-microbatch`。
- 如果 Engram CE 接近或低于 X-gram，但 Global 仍低，说明问题不再主要是训练 loss，而是 Engram 特征对 downstream 迁移不够好。
- 如果 Engram CE 和 Global 都低于 Baseline，先检查 warmup、eval-time warmup disable、tokenizer 路径、data path、checkpoint load path，不要直接下机制结论。

## 12. 禁止事项

- 不要把旧服务器 `runs/` 复制过去当最终结果。
- 不要用旧 H-path / `engram_legacy_h_path=true` 作为新版 fair Engram 结论。
- 不要在 eval post-warmup checkpoint 时保留 `lambda_warmup_enabled=true`。
- 不要只重跑 Engram 而不重跑 Baseline/X-gram；新服务器是新环境，controls 必须同机重跑。
- 不要把 `micro_batch_size=1` 当作无内存限制服务器的默认公平设置。
