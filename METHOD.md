---
title: README
description: so this is some basic introduction of the website
publishDate: 2026-06-11
tags:
minutesRead:
---
# ComEmbedding methods on X-GRAM

结论：可以接入。X-GRAM 把 lookup memory 拆成三段：

1. `R_phi`: token / n-gram 到物理 memory row 的映射和查表。
2. `T_psi`: 对查到的序列做 `RMSNorm + gated ShortConv`，把 1-gram lookup 变成局部 x-gram feature。
3. `I_s`: 把融合后的 `Delta_l` 注入 attention value stream 或 inter-layer residual。

我们已有的方法主要作用在第 1 段，也就是替换或增强 X-GRAM 的 `Lookup_l^m(x_t)`。因此最稳妥的接法是：保留 X-GRAM 的 ShortConv extraction、multi-view `1/sqrt(M)` 融合、depth-aware warm gate、value/residual injection，只替换 memory view 的 lookup module。

下面先列接入方向。frequency-aware row-memory 是最符合“X-GRAM 第一步不动，只替换 dense embedding table”的方向，因此放在前面，并额外扩展两条同类 row-memory 变体；explicit n-gram view 作为辅助方向放到最后。

## 1. QRAddProductFlatResidual + reverse

### 为什么选它

这是当前 clean/fair 结果里最强的代表之一。对应 setting：

```text
QRAddProductFlatResidual-third_b4096_r136_seed42_perm_reverse
Compression: 7.17x
Acc AVG:     36.45
Norm AVG:    36.46
```

它的作用是把一个 token-indexed dense table 分解成两个 QR codebook 的加法项、一个小的乘性 interaction，以及一个低秩 token residual：

```text
token_id' = reverse(token_id)
i = token_id' % B
j = token_id' // B
e = C1[i] + C2[j] + sigmoid(beta) * C1[i] * C2[j]
e = e + W_r R[token_id]
```

这和 X-GRAM 很合适，有两种接法：一是本节这种完整 lookup replacement，直接用 token-id reverse 的 QR lookup 生成 `E_l^m`；二是沿 frequency-aware row-memory 方案，只保留 X-GRAM 的 frequency-aware routing，把 router 输出的 physical row id 做 row-reverse 后再 QR 分解。后者不替换 X-GRAM 第一步，只替换 dense physical table `B[j]`。`reverse` permutation 在我们的结果里明显有效，说明 id 到 QR bucket 的 assignment 很重要；放到 row-memory 方案时应理解为 physical row assignment，而不是 token routing assignment。

### 代码骨架

把下面这个 module 当成 X-GRAM 的一个 lookup view。`dim` 在 value injection 时取 `d_kv`，在 inter-layer residual injection 时取 hidden width `d`。

```python
import torch
from torch import nn


def rms_norm(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


class QRAddProductResidualLookup(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        codebook_size: int = 4096,
        residual_dim: int = 136,
        permutation: str = "reverse",
        init_std: float | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.codebook_size = codebook_size
        self.num_codes = (vocab_size + codebook_size - 1) // codebook_size
        self.permutation = permutation

        std = init_std or (2.0 / (vocab_size + dim)) ** 0.5
        self.codebook1 = nn.Embedding(codebook_size, dim)
        self.codebook2 = nn.Embedding(self.num_codes, dim)
        self.beta_logit = nn.Parameter(torch.tensor(-8.0))

        self.residual = nn.Embedding(vocab_size, residual_dim)
        self.residual_proj = nn.Linear(residual_dim, dim, bias=False)

        nn.init.normal_(self.codebook1.weight, mean=0.0, std=std / 2**0.5)
        nn.init.normal_(self.codebook2.weight, mean=0.0, std=std / 2**0.5)
        nn.init.normal_(self.residual.weight, mean=0.0, std=std)
        nn.init.zeros_(self.residual_proj.weight)

    def remap(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.permutation == "none":
            return input_ids
        if self.permutation == "reverse":
            return self.vocab_size - 1 - input_ids
        if self.permutation == "affine":
            return (input_ids * 1543 + 17).remainder(self.vocab_size)
        raise ValueError(f"bad permutation: {self.permutation}")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        ids = self.remap(input_ids)
        c1 = self.codebook1(ids % self.codebook_size)
        c2 = self.codebook2(ids // self.codebook_size)
        out = c1 + c2 + torch.sigmoid(self.beta_logit) * c1 * c2
        return out + self.residual_proj(self.residual(input_ids))
```

### 怎么接入 X-GRAM

接入点是 Algorithm 1 的第 3 行 `e_t <- Lookup_l^m(x_t)`。把 X-GRAM 原来的 dense `B_l^m[row]` lookup 换成：

```python
lookup = QRAddProductResidualLookup(
    vocab_size=vocab_size,
    dim=d_kv,              # value injection; residual injection 用 hidden_size
    codebook_size=4096,
    residual_dim=64,       # d_kv 较小时先用 32/64；全宽 residual path 可用 128/136
    permutation="reverse",
)

E = lookup(input_ids)      # [batch, seq, d_s]
E_hat = rmsnorm(E)
E_tilde = E + shortconv(E_hat)
Delta = view_gate(layer, step) * E_tilde
V = V + Delta              # value injection
```

推荐先跑两个配置：

```text
XGRAM-QR-Rev-1v:
  injection: value only
  views: one QRAddProductResidual view
  dim: d_kv
  B: 4096
  residual_dim: 64
  shortconv kernels: {3}

XGRAM-QR-Rev-1h2v:
  injection: 1 inter-layer residual + 2 value views
  h kernels: {3}
  v kernels: {3,5}
  B: 4096
  residual_dim: 64 for v, 128/136 for h
```

训练注意：

- lookup 参数单独 parameter group，LR 比 backbone 大，跟 X-GRAM 的 sparse-aware LR 一致。
- `beta_logit` 初始设 `-8`，避免乘性项开局过强。
- `residual_proj` 零初始化，保留稳定 warm start。
- 继续使用 X-GRAM 的 depth-aware gate 和 warmup，不要直接大幅注入。

## 2. Frequency-aware QRAddProduct reverse row memory

### 为什么选它

这是最贴 X-GRAM 的组合，也是 `QRAddProductFlatResidual + reverse` 沿 dense-row replacement 思路的版本：保留 X-GRAM 的 frequency-aware VIP+hash+alias routing，但把每个物理 row 的 dense table `B[j]` 换成我们的 QR/add-product/residual 参数化。这里的 reverse 不作用在 token id 上，而是作用在 X-GRAM router 输出的 physical row id 上。

X-GRAM 原始 retrieval 是：

```text
e(w) = sum_j c(w,j) * sigmoid(a_j) * B[j]
```

这里 `B[j]` 是 dense physical row。我们改成：

```text
B[j] := QRRow(j)
j' = S - 1 - j                 # row-reverse, S is X-GRAM physical rows
QRRow(j) = C1[j' % B] + C2[j' // B] + beta * C1[j' % B] * C2[j' // B] + W_r R[j]
```

这样不会破坏 X-GRAM 的 frequency-aware routing，也能继续使用 alias mixing、row-wise gate、ShortConv、value/residual injection。它主要解决 X-GRAM 表还是很大的问题：从 `rho * |V| * d_s` 进一步压到大约

```text
(B + ceil(S/B)) * d_s + S * r + r * d_s
```

其中 `S = rho * |V|` 是 X-GRAM physical row 数。

### 代码骨架

下面假设 X-GRAM 已经有 `router(input_ids)`，返回 `rows` 和 `weights`：

```python
rows:    [batch, seq, paths]  # physical row ids
weights: [batch, seq, paths]  # alias/hash aggregation weights
```

```python
import torch
from torch import nn


class QRRowMemory(nn.Module):
    def __init__(
        self,
        num_rows: int,
        dim: int,
        codebook_size: int = 4096,
        residual_dim: int = 64,
        row_permutation: str = "reverse",
    ):
        super().__init__()
        self.num_rows = num_rows
        self.dim = dim
        self.codebook_size = codebook_size
        self.num_codes = (num_rows + codebook_size - 1) // codebook_size
        self.row_permutation = row_permutation

        std = (2.0 / (num_rows + dim)) ** 0.5
        self.codebook1 = nn.Embedding(codebook_size, dim)
        self.codebook2 = nn.Embedding(self.num_codes, dim)
        self.beta_logit = nn.Parameter(torch.tensor(-8.0))
        self.row_gate = nn.Embedding(num_rows, 1)

        self.residual = nn.Embedding(num_rows, residual_dim)
        self.residual_proj = nn.Linear(residual_dim, dim, bias=False)

        nn.init.normal_(self.codebook1.weight, mean=0.0, std=std / 2**0.5)
        nn.init.normal_(self.codebook2.weight, mean=0.0, std=std / 2**0.5)
        nn.init.zeros_(self.row_gate.weight)
        nn.init.normal_(self.residual.weight, mean=0.0, std=std)
        nn.init.zeros_(self.residual_proj.weight)

    def remap_rows(self, rows: torch.Tensor) -> torch.Tensor:
        if self.row_permutation == "none":
            return rows
        if self.row_permutation == "reverse":
            return self.num_rows - 1 - rows
        if self.row_permutation == "affine":
            return (rows * 1543 + 17).remainder(self.num_rows)
        raise ValueError(f"bad row_permutation: {self.row_permutation}")

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        qr_rows = self.remap_rows(rows)
        c1 = self.codebook1(qr_rows % self.codebook_size)
        c2 = self.codebook2(qr_rows // self.codebook_size)
        out = c1 + c2 + torch.sigmoid(self.beta_logit) * c1 * c2
        out = out + self.residual_proj(self.residual(rows))
        return torch.sigmoid(self.row_gate(rows)) * out


class FrequencyAwareQRLookup(nn.Module):
    def __init__(
        self,
        router,
        num_rows: int,
        dim: int,
        codebook_size: int = 4096,
        residual_dim: int = 64,
        row_memory_cls=QRRowMemory,
        row_memory_kwargs: dict | None = None,
    ):
        super().__init__()
        self.router = router
        kwargs = row_memory_kwargs or {}
        self.row_memory = row_memory_cls(num_rows, dim, codebook_size, residual_dim, **kwargs)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        rows, weights = self.router(input_ids)
        values = self.row_memory(rows)
        weights = weights.to(values.dtype)
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return (values * weights.unsqueeze(-1)).sum(dim=-2) / denom
```

### 怎么接入 X-GRAM

这不是替换 X-GRAM 的 router，而是替换 router 后面的 physical table：

```python
router = XGramFrequencyRouter(
    vocab_freq=freq,
    vocab_size=vocab_size,
    rho=0.5,
    num_buckets=32,
    vip_size=200,
    dense_hash_paths=2,
    alpha=0.5,
    max_alias_paths=3,
    alias_decay=0.8,
)

lookup = FrequencyAwareQRLookup(
    router=router,
    num_rows=int(0.5 * vocab_size),
    dim=d_kv,
    codebook_size=4096,
    residual_dim=64,
    row_memory_cls=QRRowMemory,
    row_memory_kwargs={"row_permutation": "reverse"},
)

E = lookup(input_ids)
E = E + shortconv(rmsnorm(E))
V = V + gate(layer, step) * E
```

推荐先跑：

```text
XGRAM-FAQR-50%-2v:
  X-GRAM routing: rho=0.5, K_vip=200, B=32, H=2, alpha=0.5
  row memory: QR add-product residual with row_reverse
  injection: 2 value views
  v kernels: {3,5}
  residual_dim: 64

XGRAM-FAQR-50%-1h2v:
  same routing
  h row memory dim=hidden_size, residual_dim=128/136, kernel {3}
  v row memory dim=d_kv, residual_dim=64, kernels {3,5}
```

训练注意：

- X-GRAM 的 row-wise gate `a_l,j` 可以并入 `QRRowMemory.row_gate`，不要重复乘两个 gate。
- VIP rows 和 hashed/aliased rows 最好分 parameter group：hashed/aliased rows 用更大的 LR multiplier，VIP/head rows capped。
- 这个方法压缩最强；`row_reverse` 只改变 physical row 到 QR codebook 的分配，不改变 X-GRAM 的 token-to-row routing。
- 相比 plain add，这条更接近我们已有最强 setting；风险主要来自 collision noise 与 QR approximation error 叠加。建议先只在 value path 跑 `2v`，不要一开始 full mixed injection。

### 压缩率和 residual_dim 选择

`reverse` 本身不增加参数，add-product 相比 plain add 也基本只多一个 scalar `beta`。压缩率主要由 `rho`、目标维度 `d_s`、`residual_dim=r` 和 `codebook_size=B` 决定。

frequency-aware row-memory 的参数量近似为：

```text
S = rho * |V|
P_dense_xgram = S * d_s
P_dense_full  = |V| * d_s
P_qr_row      = (B + ceil(S / B)) * d_s + S * r + r * d_s + S
```

其中最后的 `+ S` 是 row-wise gate。对 SmolLM2-360M：

```text
|V| = 49152
rho = 0.5
S = 24576
B = 4096
hidden_size = 960
num_attention_heads = 15
num_key_value_heads = 5
head_dim = 64
value width before repeat_kv = 5 * 64 = 320
```

因此 value injection 如果接在 `repeat_kv` 之前，目标维度是 `d_s=320`，不是我们之前 STEM 里的 `d_ff=2560`。这就是 `r=128/136` 压缩率明显下降的原因：`d_s` 小了 8 倍，但 `S*r` 这项没有跟着变小。

默认先跑仍然沿用之前表现最好的思路，但 value path 的 residual rank 要按 320 维缩小：

```text
default value view:
  d_s = 320
  residual_dim = 64
  codebook_size = 4096
  row_permutation = reverse

default hidden/residual view:
  d_s = 960
  residual_dim = 128/136
  codebook_size = 4096
  row_permutation = reverse
```

按 `d_s=320` 的 value view 计算，单 view 的压缩率约为：

```text
residual_dim  over X-GRAM dense table  over full vocab dense table
r=16          4.62x                    9.24x
r=32          3.69x                    7.37x
r=64          2.68x                    5.37x
r=96          2.10x                    4.20x
r=128         1.73x                    3.45x
r=136         1.66x                    3.33x
```

建议含义：

- 默认主跑：`r=64`，保留足够 token/row individuality，同时压缩率还能接受。
- 压缩优先：`r=32`。
- 更激进压缩：`r=16`。
- 容量上限对照：`r=128` 或 `r=136`，主要用来判断是不是 value residual capacity 卡住了效果。

如果跑 `1h2v`，推荐：

```text
h view: d_s=960, residual_dim=128/136
v view: d_s=320, residual_dim=64 by default
v rank ablation: residual_dim=32
```


## 3. Frequency-aware QRAddNormProduct row memory

### 为什么选它

这条是 frequency-aware row-memory 的同类变体：仍然保留 X-GRAM 的 `VIP + bucket + hash + alias` routing，只把 physical dense row table `B[j]` 换成我们的 compositional row memory。

它对应之前表现很好的 seed42-aligned 结构：

```text
QRAddNormProductFlatResidual-third_b4096_r136_seed42
Compression: 7.17x
Acc AVG:     36.32
Norm AVG:    36.39
```

和 `QRAddProduct` row memory 不同，这里把乘性项先归一化再加回去：

```text
B[j] := C1[j % B] + C2[j // B] + beta * RMSNorm(C1 * C2) * target_std + W_r R[j]
```

这个版本更适合 X-GRAM 的 hash/alias 场景：当多个 token 共享 physical row 时，原始 `C1 * C2` 的尺度可能比较飘，normalized product 能把乘性信息控制在稳定范围内，再交给 X-GRAM 的 ShortConv 和 gate 选择是否使用。

### 代码骨架

```python
class QRAddNormProductRowMemory(nn.Module):
    def __init__(self, num_rows: int, dim: int, codebook_size: int = 4096, residual_dim: int = 64):
        super().__init__()
        self.num_rows = num_rows
        self.dim = dim
        self.codebook_size = codebook_size
        self.num_codes = (num_rows + codebook_size - 1) // codebook_size
        self.target_std = (2.0 / (num_rows + dim)) ** 0.5

        self.codebook1 = nn.Embedding(codebook_size, dim)
        self.codebook2 = nn.Embedding(self.num_codes, dim)
        self.beta_logit = nn.Parameter(torch.tensor(-2.0))
        self.row_gate = nn.Embedding(num_rows, 1)
        self.residual = nn.Embedding(num_rows, residual_dim)
        self.residual_proj = nn.Linear(residual_dim, dim, bias=False)

        nn.init.normal_(self.codebook1.weight, mean=0.0, std=self.target_std / 2**0.5)
        nn.init.normal_(self.codebook2.weight, mean=0.0, std=self.target_std / 2**0.5)
        nn.init.zeros_(self.row_gate.weight)
        nn.init.normal_(self.residual.weight, mean=0.0, std=self.target_std)
        nn.init.zeros_(self.residual_proj.weight)

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        c1 = self.codebook1(rows % self.codebook_size)
        c2 = self.codebook2(rows // self.codebook_size)
        product = rms_norm(c1 * c2) * self.target_std
        out = c1 + c2 + torch.sigmoid(self.beta_logit) * product
        out = out + self.residual_proj(self.residual(rows))
        return torch.sigmoid(self.row_gate(rows)) * out
```

### 怎么接入 X-GRAM

只改 frequency-aware row-memory 方案里面的 row memory class：

```python
lookup = FrequencyAwareQRLookup(
    router=router,                         # X-GRAM 原始 frequency-aware router
    num_rows=int(0.5 * vocab_size),
    dim=d_kv,
    codebook_size=4096,
    residual_dim=64,
    row_memory_cls=QRAddNormProductRowMemory,
)
```

推荐配置：

```text
XGRAM-FANormQR-50%-2v:
  routing: X-GRAM rho=0.5, K_vip=200, buckets=32, H=2, alpha=0.5
  dense row replacement: QRAddNormProductRowMemory
  injection: 2 value views
  v kernels: {3,5}
  residual_dim: 64

XGRAM-FANormQR-50%-1h2v:
  h row dim=hidden_size, residual_dim=128/136, kernel {3}
  v row dim=d_kv, residual_dim=64, kernels {3,5}
```

训练注意：

- `beta_logit=-2` 比 add-product 的 `-8` 更积极，但 product 已经 RMS-normalized，通常更稳。
- 如果 raw add-product row memory 在 X-GRAM hash collision 下尺度不稳，这条作为紧邻对照；否则主线仍优先 `QRRowMemory(row_reverse)`。
- 不要再额外乘一层 X-GRAM row-wise gate；保留一个 row gate 就够。

## 4. Frequency-aware QRAdd row memory

### 为什么选它

这条也是 frequency-aware row-memory 的同类变体，目标是做一个最稳、最简单的 row-table 替换：保留 X-GRAM routing，只把 dense `B[j]` 替换成 QR add + low-rank residual。

对应之前的 S15 clean baseline：

```text
QRAddFlatResidual-third_b4096_r128
Compression: 7.34x
Acc AVG:     35.88
Norm AVG:    36.37
```

它没有乘性项，因此表达力低于 add-product / add-norm-product，但训练风险也最低。放到 X-GRAM 中，它适合作为第一条 table-compression sanity check：如果这条都不稳，说明问题在 X-GRAM 接入或 routing/ShortConv/gate，而不是乘性项本身。

### 代码骨架

```python
class QRAddResidualRowMemory(nn.Module):
    def __init__(self, num_rows: int, dim: int, codebook_size: int = 4096, residual_dim: int = 64):
        super().__init__()
        self.num_rows = num_rows
        self.dim = dim
        self.codebook_size = codebook_size
        self.num_codes = (num_rows + codebook_size - 1) // codebook_size
        std = (2.0 / (num_rows + dim)) ** 0.5

        self.codebook1 = nn.Embedding(codebook_size, dim)
        self.codebook2 = nn.Embedding(self.num_codes, dim)
        self.row_gate = nn.Embedding(num_rows, 1)
        self.residual = nn.Embedding(num_rows, residual_dim)
        self.residual_proj = nn.Linear(residual_dim, dim, bias=False)

        nn.init.normal_(self.codebook1.weight, mean=0.0, std=std / 2**0.5)
        nn.init.normal_(self.codebook2.weight, mean=0.0, std=std / 2**0.5)
        nn.init.zeros_(self.row_gate.weight)
        nn.init.normal_(self.residual.weight, mean=0.0, std=std)
        nn.init.zeros_(self.residual_proj.weight)

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        c1 = self.codebook1(rows % self.codebook_size)
        c2 = self.codebook2(rows // self.codebook_size)
        out = c1 + c2 + self.residual_proj(self.residual(rows))
        return torch.sigmoid(self.row_gate(rows)) * out
```

### 怎么接入 X-GRAM

接法和 frequency-aware row-memory 方案完全一样，只换 row memory：

```python
lookup = FrequencyAwareQRLookup(
    router=router,
    num_rows=int(0.5 * vocab_size),
    dim=d_kv,
    codebook_size=4096,
    residual_dim=64,
    row_memory_cls=QRAddResidualRowMemory,
)
```

推荐配置：

```text
XGRAM-FAAddQR-50%-2v:
  routing: X-GRAM original frequency-aware routing
  dense row replacement: QRAddResidualRowMemory
  injection: 2 value views
  kernels: {3,5}
  residual_dim: 64
```

训练注意：

- 这条是最低风险 baseline，不一定最强。
- 可以先跑很短的 validation/perplexity smoke test，确认 X-GRAM dense row replacement 没有 shape 和 scale 问题。
- 如果它稳定但收益不够，再切到 `QRAddNormProductRowMemory` 或 raw `QRRowMemory`。

## 5. ContextMask NgramPQ as explicit x-gram view

### 为什么选它

X-GRAM 用 ShortConv 从 1-gram retrieved sequence 里提取局部 x-gram 特征；我们的 `ContextMaskNgramPQ` 可以在 lookup 阶段就显式引入 current token、bigram、skip-bigram、trigram 四类 key，再交给 X-GRAM 的 ShortConv 做二次 refinement。

已有 clean 结果中，`ContextMaskNgramPQ G32K4096 split8-12-8-4` 是 n-gram 系里相对稳的代表：

```text
ContextMaskNgramPQ-third_G32K4096_split8-12-8-4
Compression: 12.00x
Acc AVG:     35.92
Norm AVG:    36.13
```

它不如 QR reverse 的最终分数，但它和 X-GRAM 的设计互补性更强：X-GRAM 论文明确指出固定 n-gram table 容易长尾 under-train 和 slot collapse；我们的 PQ 版本把显式 n-gram table 做成分组压缩，再让 X-GRAM 的 ShortConv/gate 去缓解 collision noise 和 redundancy。

### 代码骨架

```python
import torch
import torch.nn.functional as F
from torch import nn


def hash_key(x: torch.Tensor, salt: int, modulo: int) -> torch.Tensor:
    return (x * (1103515245 + 194 * salt) + 12345 + 104729 * salt).remainder(2**31).remainder(modulo)


def key_by_split(group: int, keys: tuple[torch.Tensor, ...], split: tuple[int, ...]) -> torch.Tensor:
    end = 0
    for view, count in enumerate(split):
        end += count
        if group < end:
            return keys[view]
    return keys[-1]


class ContextMaskNgramPQLookup(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        codebook_size: int = 4096,
        groups: int = 32,
        split: tuple[int, int, int, int] = (8, 12, 8, 4),
    ):
        super().__init__()
        assert dim % groups == 0
        assert sum(split) == groups
        self.vocab_size = vocab_size
        self.dim = dim
        self.codebook_size = codebook_size
        self.groups = groups
        self.split = split
        group_dim = dim // groups
        self.codebooks = nn.ModuleList([nn.Embedding(codebook_size, group_dim) for _ in range(groups)])

        std = (2.0 / (vocab_size + dim)) ** 0.5
        for table in self.codebooks:
            nn.init.normal_(table.weight, mean=0.0, std=std)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        cur = input_ids
        prev1 = F.pad(input_ids[:, :-1], (1, 0), value=0)
        prev2 = F.pad(input_ids[:, :-2], (2, 0), value=0) if input_ids.size(1) > 1 else torch.zeros_like(input_ids)
        keys = (
            cur,
            (prev1 * 65537 + cur).remainder(2**31),
            (prev2 * 65537 + cur).remainder(2**31),
            (prev2 * 65537 * 17 + prev1 * 65537 + cur).remainder(2**31),
        )

        chunks = []
        for group, table in enumerate(self.codebooks):
            key = key_by_split(group, keys, self.split)
            row = hash_key(key, group + 1, self.codebook_size)
            chunks.append(table(row))
        return torch.cat(chunks, dim=-1)
```

### 怎么接入 X-GRAM

它应作为 X-GRAM 的一个或多个 memory view，而不是替代全部 view。建议直接接在 `T_psi` 之前：

```python
ngram_lookup = ContextMaskNgramPQLookup(
    vocab_size=vocab_size,
    dim=d_kv,
    codebook_size=4096,
    groups=32,
    split=(8, 12, 8, 4),
)

E_ngram = ngram_lookup(input_ids)
E_ngram = E_ngram + shortconv(rmsnorm(E_ngram))
Delta_v = gate_v(layer, step) * E_ngram
V = V + Delta_v
```

推荐先跑：

```text
XGRAM-CtxMaskPQ-2v:
  injection: value only
  views:
    v0 = QRAddProductResidualLookup, kernel 3
    v1 = ContextMaskNgramPQLookup, kernel 5
  dim: d_kv
  pq groups: 32
  codebook size: 4096
  split: 8,12,8,4

XGRAM-CtxMaskPQ-1h2v:
  h = QRAddProductResidualLookup, kernel 3
  v0 = ContextMaskNgramPQLookup, kernel 3
  v1 = ContextMaskNgramPQLookup, kernel 5
```

训练注意：

- 如果只用 explicit n-gram view，长尾 hash collision 会比较重；最好和 QR token view 混合。
- 保留 X-GRAM 的 `1/sqrt(M)` multi-view normalization。
- `split=(8,12,8,4)` 是已有较稳配置；如果想贴近 X-GRAM multi-scale，可试 `split=(6,12,10,4)` 或 `split=(4,14,10,4)`，但先不要扩大搜索。

## X-GRAM 接入总流程

无论用上面哪一种 lookup，X-GRAM 主体可以保持同一个 wrapper：

```python
class XGramComEmbeddingView(nn.Module):
    def __init__(self, lookup, shortconv, gate):
        super().__init__()
        self.lookup = lookup
        self.shortconv = shortconv
        self.gate = gate
        self.norm = RMSNorm(lookup.dim)

    def forward(self, input_ids, layer_idx: int, step: int):
        e = self.lookup(input_ids)
        e = e + self.shortconv(self.norm(e))
        return self.gate(layer_idx, step) * e
```

注入 value stream：

```python
q, k, v = self.q_proj(h), self.k_proj(h), self.v_proj(h)
delta = sum(view(input_ids, layer_idx, step) for view in self.xgram_v_views) / len(self.xgram_v_views) ** 0.5
v = v + delta.view_as(v)
out = attention(q, k, v)
```

注入 inter-layer residual：

```python
delta_h = sum(view(input_ids, layer_idx, step) for view in self.xgram_h_views) / len(self.xgram_h_views) ** 0.5
h = h + delta_h
h = transformer_block(h)
```

## 建议实验顺序

如果目标是贴近 X-GRAM 论文、只替换 dense physical table，优先看 frequency-aware row-memory 系列：

1. 先跑 `XGRAM-FAAddQR-50%-2v`：最低风险 row-table replacement smoke check，确认 routing、ShortConv、gate 和注入路径都稳定。
2. 再跑 `XGRAM-FAQR-50%-2v`：主力方案，也就是 `QRAddProductFlatResidual + reverse` 的 row-memory 版本；保留 X-GRAM routing，只对 physical row id 做 `row_reverse` 后 QR/add-product/residual 参数化，value view 默认 `r=64`。
3. 再跑 `XGRAM-FANormQR-50%-2v`：normalized product 的相邻对照；如果 raw product 在 hash/alias row 上尺度不稳，它应该更稳。
4. 稳定后跑 `XGRAM-FAQR-50%-1h2v`：沿主力 row memory 扩到 1 个 residual view + 2 个 value views。
5. 并行参考跑 `XGRAM-QR-Rev-1v`：完整 lookup replacement ablation，用来区分收益来自 QR reverse 本身还是来自保留 X-GRAM routing。
6. 最后跑 `XGRAM-CtxMaskPQ-2v`：explicit n-gram 辅助 view，主要验证它和 ShortConv 是否互补。

## 不建议优先接的方向

- Q/K injection：X-GRAM 论文里 Q/K 比 value/residual 更脆弱；我们的现有结果也没有支持优先做 Q/K。
- 额外 Token/Layer add-on：后续 `Tok.../Layer...` reverse 扩展在当前结果里掉到 `35.x`，不如原始 `perm_reverse`。
- 单独纯 n-gram PQ 替代所有 view：长尾 collision 会更重，最好作为辅助 view。
- factor-init 作为最终 fair claim：它可以做 oracle 或 warm-start，但会引入 teacher table / checkpoint，不适合作为 clean comparison。
