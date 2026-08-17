# Remote Engram Server 复制清单

推荐方案是：**把实验相关内容做成私有 GitHub 全量镜像**，新服务器直接 `git clone`。代码、配置、脚本、文档、数据、tokenizer、旧日志、历史结果都放进同一个 repo。

如果仓库里有单文件超过 GitHub 普通 Git 限制，先用 Git LFS 处理。其余实验相关内容都可以直接进库，不需要再分散复制。

## 0. GitHub 发布前检查

如果按私有全量镜像来做，直接把实验相关文件一次性提交进去即可；只保留 `.venv/`、`__pycache__/`、`*.pyc` 这类机器产物不进库。单文件过大时先走 Git LFS。

```bash
git status --short

git add -A

git status --short
git commit -m "Add remote Engram faircompare experiment handoff"
git push
```

发布后，新服务器执行：

```bash
git clone <your-github-repo-url> X-gram
cd X-gram
git rev-parse HEAD
```

`packages/olmo_in_loop_evals/` 也建议直接放进主仓库；只有你明确要拆分时，才考虑 submodule 或另一个私有 repo。远端必须能拿到它，否则 downstream eval 会缺任务和缓存。

## 1. 必须复制的代码和配置

如果走 GitHub，这一节的内容应该已经在 repo 里。必须包含这些文件/目录：

```text
OLMo-core/
packages/streaming/
packages/ubdataloader/
packages/olmo_in_loop_evals/
scripts/
configs/
python_slurm/
tools/
assets/
pyproject.toml
README.md
README_zh.md
LICENSE
METHOD.md
REPRODUCTION.md
docs/25_Engram最新Faircompare对比结果_20260730/
docs/26_RemoteEngramServer_Codex实验指导.md
docs/26_RemoteEngramServer_复制清单.md
```

这些目录里最关键的文件包括：

```text
OLMo-core/src/olmo_core/nn/embedding_injection/engram.py
OLMo-core/src/olmo_core/nn/embedding_injection/xgram.py
OLMo-core/src/olmo_core/nn/embedding_injection/ops/hash_injection.py
OLMo-core/src/olmo_core/nn/transformer/config.py
OLMo-core/src/olmo_core/nn/transformer/model.py
OLMo-core/src/olmo_core/train/train_module/transformer/train_module.py
OLMo-core/src/olmo_core/train/trainer.py
scripts/train/olmo_train.py
scripts/train/olmo_train.sh
scripts/validate_engram_xgrammatch_config.py
python_slurm/train_faircompare.slurm
python_slurm/train_engram_xgrammatch.slurm
python_slurm/eval_downstream.slurm
python_slurm/eval_generation_em.slurm
configs/faircompare_baseline_360m.yaml
configs/faircompare_xgram_360m.yaml
configs/faircompare_engram_vpath_xgrammatch_360m.yaml
configs/faircompare_engram_2g3g_vpath_d512_lam05_warm240_360m.yaml
configs/faircompare_engram_vpath_single_lam05_warm240_360m.yaml
configs/faircompare_engram_vpath_lam05_warm240_360m.yaml
configs/faircompare_engram_vpath_lam025_warm480_360m.yaml
```

## 2. 必须放进 repo 的数据和资产

训练数据：

```text
data/fineweb_10b_streaming/
```

当前大小约 `5.1G`。必须包含：

```text
data/fineweb_10b_streaming/index.json
data/fineweb_10b_streaming/shard.*.mds.zstd
```

X-gram hash token map：

```text
assets/token_maps/injection_token_map_smollm2_fineweb_alpha0_5_M32_Cap75968_mc3_mwd0_8.npz
```

建议同时复制 frequency stats，方便远端重建/审计 token map：

```text
assets/frequency_stats/smollm2_fineweb_10b_token_freqs.npy
assets/frequency_stats/smollm2_fineweb_top200_tokens.npz
```

Tokenizer / base model tokenizer，建议放在 repo 内固定目录：

```text
assets/tokenizers/SmolLM2-360M/
```

至少应包含：

```text
assets/tokenizers/SmolLM2-360M/tokenizer.json
assets/tokenizers/SmolLM2-360M/tokenizer_config.json
assets/tokenizers/SmolLM2-360M/special_tokens_map.json
```

下游 eval 数据随本地包复制：

```text
packages/olmo_in_loop_evals/src/olmo_eval/hf_datasets/
packages/olmo_in_loop_evals/src/olmo_eval/oe_eval_tasks/
```

## 3. 建议一并放进 repo 的实验产物

这些内容都建议作为私有镜像的一部分：

```text
runs/
runs_eval/
runs_speedtest/
logs/
out/
wandb/
reports/
reports/engram_variant_sweep_20260804.md
reports/downstream_comparison_20260729.md
docs/25_Engram最新Faircompare对比结果_20260730/
```

历史 checkpoint 可以保留用于 debug 和回溯，但不要再把旧 `runs/` 当成新服务器 fair compare 的结论来源。

## 4. fallback 打包命令

只有在不能用 GitHub 或需要完整工作树快照时，才在当前服务器仓库根目录执行：

```bash
tar \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='xgram.egg-info' \
  --exclude='**/__pycache__' \
  -czf xgram_remote_code_and_configs.tar.gz \
  OLMo-core packages scripts configs python_slurm tools assets data docs reports runs runs_eval runs_speedtest logs out wandb \
  pyproject.toml README.md README_zh.md LICENSE METHOD.md REPRODUCTION.md handoff.md \
  docs/25_Engram最新Faircompare对比结果_20260730 \
  docs/26_RemoteEngramServer_Codex实验指导.md \
  docs/26_RemoteEngramServer_复制清单.md

tar -czf xgram_fineweb_10b_streaming.tar.gz data/fineweb_10b_streaming
```

如果要直接复制 tokenizer snapshot：

```bash
tar -czf smollm2_360m_tokenizer_snapshot.tar.gz \
  -C assets/tokenizers \
  SmolLM2-360M
```

传到远端：

```bash
scp xgram_remote_code_and_configs.tar.gz xgram_fineweb_10b_streaming.tar.gz smollm2_360m_tokenizer_snapshot.tar.gz user@server:/path/to/work/
```

远端解压：

```bash
mkdir -p /path/to/X-gram
tar -xzf xgram_remote_code_and_configs.tar.gz -C /path/to/X-gram
tar -xzf xgram_fineweb_10b_streaming.tar.gz -C /path/to/X-gram
mkdir -p /path/to/X-gram/assets/tokenizers/SmolLM2-360M
tar -xzf smollm2_360m_tokenizer_snapshot.tar.gz -C /path/to/X-gram/assets/tokenizers/SmolLM2-360M --strip-components=1
```

## 5. 远端解压后的检查

```bash
cd /path/to/X-gram

test -f OLMo-core/src/olmo_core/nn/embedding_injection/engram.py
test -f scripts/train/olmo_train.py
test -f configs/faircompare_baseline_360m.yaml
test -f configs/faircompare_xgram_360m.yaml
test -f configs/faircompare_engram_2g3g_vpath_d512_lam05_warm240_360m.yaml
test -f assets/token_maps/injection_token_map_smollm2_fineweb_alpha0_5_M32_Cap75968_mc3_mwd0_8.npz
test -f data/fineweb_10b_streaming/index.json
test -f assets/tokenizers/SmolLM2-360M/tokenizer.json
```

然后运行：

```bash
python -m compileall \
  scripts \
  OLMo-core/src/olmo_core \
  packages/ubdataloader/src \
  packages/olmo_in_loop_evals/src
```

## 6. 远端必须改的本地路径

所有这些地方可能包含旧路径 `/home/bcjiang/X-gram` 或 `/home/bcjiang/.cache/...`：

```text
configs/*.yaml
python_slurm/*.slurm
scripts/train/olmo_train.sh
```

至少要处理：

- `data.streaming_tokenizer_model`
- `embedding_injection.engram_tokenizer_id`
- Slurm `cd` 工作目录
- Slurm `#SBATCH --output/--error`
- Slurm `PYTHONPATH`
- Slurm conda/venv activate 命令

如果不想批量 patch YAML，可以在训练时设置：

```bash
export STREAMING_TOKENIZER_MODEL=/path/to/X-gram/assets/tokenizers/SmolLM2-360M
export ENGRAM_TOKENIZER_ID=/path/to/X-gram/assets/tokenizers/SmolLM2-360M
```

但 Slurm 脚本里写死的 tokenizer 路径仍需改掉。

## 7. 复制完成后交给远端 Codex 的入口

让远端 Codex 先读：

```text
docs/26_RemoteEngramServer_Codex实验指导.md
docs/25_Engram最新Faircompare对比结果_20260730/README.md
```

远端 Codex 的第一步不是开跑，而是完成：

- 环境安装。
- tokenizer/data 路径修正。
- 创建 `remote_fullmb_*` Engram configs。
- `validate_engram_xgrammatch_config.py` 检查通过。
- `--print-resolved-config` 检查通过。

这些都通过后，再开始 Baseline / X-gram / Engram 训练。
