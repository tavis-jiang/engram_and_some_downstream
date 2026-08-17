from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from olmo_core.exceptions import OLMoConfigurationError


_COMEMBED_DEBUG_FINITE = os.environ.get("COMEMBED_DEBUG_FINITE", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_COMEMBED_GRAD_CLIP = float(os.environ.get("COMEMBED_GRAD_CLIP", "1.0"))


def _debug_check_finite(name: str, tensor: torch.Tensor) -> None:
    if not _COMEMBED_DEBUG_FINITE:
        return
    if torch.isfinite(tensor).all():
        return
    raise RuntimeError(f"ComEmbed non-finite tensor detected at {name}")


def _debug_register_grad_check(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if not _COMEMBED_DEBUG_FINITE or not tensor.requires_grad:
        return tensor

    def _hook(grad: torch.Tensor) -> torch.Tensor:
        _debug_check_finite(f"{name}.grad", grad)
        return grad

    tensor.register_hook(_hook)
    return tensor


def _register_tensor_grad_sanitizer(tensor: torch.Tensor) -> torch.Tensor:
    if not tensor.requires_grad or getattr(tensor, "_comembed_grad_sanitizer", False):
        return tensor
    tensor.register_hook(sanitize_comembed_grad)
    tensor._comembed_grad_sanitizer = True
    return tensor


def _debug_register_param_grad_checks(module: nn.Module, *, prefix: str) -> None:
    if not _COMEMBED_DEBUG_FINITE:
        return
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        param.register_hook(lambda grad, full_name=f"{prefix}.{name}": (_debug_check_finite(f"{full_name}.grad", grad), grad)[1])


def sanitize_comembed_grad(grad: torch.Tensor) -> torch.Tensor:
    """Keep rare attention-backward non-finite gradients from poisoning ComEmbed weights."""
    grad = torch.nan_to_num(grad, nan=0.0, posinf=_COMEMBED_GRAD_CLIP, neginf=-_COMEMBED_GRAD_CLIP)
    if _COMEMBED_GRAD_CLIP > 0:
        grad = grad.clamp(min=-_COMEMBED_GRAD_CLIP, max=_COMEMBED_GRAD_CLIP)
    return grad


def register_comembed_grad_sanitizer(module: nn.Module) -> None:
    for param in module.parameters():
        if not param.requires_grad or getattr(param, "_comembed_grad_sanitizer", False):
            continue
        param.register_hook(sanitize_comembed_grad)
        param._comembed_grad_sanitizer = True


def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def _normal_(module: nn.Embedding, *, std: float) -> None:
    if not module.weight.is_meta:
        nn.init.normal_(module.weight, mean=0.0, std=std)


def _zeros_(tensor: torch.Tensor) -> None:
    if not tensor.is_meta:
        nn.init.zeros_(tensor)


def _hash_key(x: torch.Tensor, salt: int, modulo: int) -> torch.Tensor:
    return (
        x * (1103515245 + 194 * salt) + 12345 + 104729 * salt
    ).remainder(2**31).remainder(modulo)


def _key_by_split(group: int, keys: Tuple[torch.Tensor, ...], split: Tuple[int, ...]) -> torch.Tensor:
    end = 0
    for view, count in enumerate(split):
        end += count
        if group < end:
            return keys[view]
    return keys[-1]


class QRAddProductResidualLookup(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        dim: int,
        codebook_size: int = 4096,
        residual_dim: int = 136,
        permutation: str = "reverse",
        init_std: Optional[float] = None,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.dim = int(dim)
        self.codebook_size = int(codebook_size)
        self.num_codes = (self.vocab_size + self.codebook_size - 1) // self.codebook_size
        self.residual_dim = int(residual_dim)
        self.permutation = permutation
        self.init_std = float(init_std or math.sqrt(2.0 / (self.vocab_size + self.dim)))

        self.codebook1 = nn.Embedding(self.codebook_size, self.dim, device=device, dtype=dtype)
        self.codebook2 = nn.Embedding(self.num_codes, self.dim, device=device, dtype=dtype)
        self.beta_logit = nn.Parameter(torch.tensor([-8.0], device=device, dtype=torch.float32))
        self.residual = nn.Embedding(self.vocab_size, self.residual_dim, device=device, dtype=dtype)
        self.residual_proj = nn.Linear(
            self.residual_dim,
            self.dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.reset_comembed_parameters()

    def reset_comembed_parameters(self) -> None:
        _normal_(self.codebook1, std=self.init_std / math.sqrt(2.0))
        _normal_(self.codebook2, std=self.init_std / math.sqrt(2.0))
        _normal_(self.residual, std=self.init_std)
        _zeros_(self.residual_proj.weight)
        if not self.beta_logit.is_meta:
            self.beta_logit.data.fill_(-8.0)

    def remap(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.permutation == "none":
            return input_ids
        if self.permutation == "reverse":
            return self.vocab_size - 1 - input_ids
        if self.permutation == "affine":
            return (input_ids * 1543 + 17).remainder(self.vocab_size)
        raise OLMoConfigurationError(f"Unknown ComEmbed token permutation: {self.permutation}")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        ids = self.remap(input_ids)
        c1 = self.codebook1(ids % self.codebook_size)
        c2 = self.codebook2(ids // self.codebook_size)
        out = c1 + c2 + torch.sigmoid(self.beta_logit.to(c1.dtype)) * c1 * c2
        return out + self.residual_proj(self.residual(input_ids))


class QRRowMemory(nn.Module):
    def __init__(
        self,
        *,
        num_rows: int,
        dim: int,
        codebook_size: int = 4096,
        residual_dim: int = 64,
        row_permutation: str = "reverse",
        disable_row_gate: bool = False,
        output_rmsnorm: bool = False,
        init_std: Optional[float] = None,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.num_rows = int(num_rows)
        self.dim = int(dim)
        self.codebook_size = int(codebook_size)
        self.num_codes = (self.num_rows + self.codebook_size - 1) // self.codebook_size
        self.residual_dim = int(residual_dim)
        self.row_permutation = row_permutation
        self.disable_row_gate = bool(disable_row_gate)
        self.output_rmsnorm = bool(output_rmsnorm)
        self.init_std = float(init_std or math.sqrt(2.0 / (self.num_rows + self.dim)))
        self.row_gate_init = -8.0
        self.product_beta_init = -12.0
        self.row_gate_init = -8.0
        self.product_clip = 4.0 * self.init_std

        self.codebook1 = nn.Embedding(self.codebook_size, self.dim, device=device, dtype=dtype)
        self.codebook2 = nn.Embedding(self.num_codes, self.dim, device=device, dtype=dtype)
        self.beta_logit = nn.Parameter(torch.tensor([self.product_beta_init], device=device, dtype=torch.float32))
        self.row_gate = nn.Embedding(self.num_rows, 1, device=device, dtype=dtype)
        self.residual = nn.Embedding(self.num_rows, self.residual_dim, device=device, dtype=dtype)
        self.residual_proj = nn.Linear(
            self.residual_dim,
            self.dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.reset_comembed_parameters()

    def reset_comembed_parameters(self) -> None:
        _normal_(self.codebook1, std=self.init_std / math.sqrt(2.0))
        _normal_(self.codebook2, std=self.init_std / math.sqrt(2.0))
        if not self.row_gate.weight.is_meta:
            self.row_gate.weight.data.fill_(self.row_gate_init)
        _normal_(self.residual, std=self.init_std)
        _zeros_(self.residual_proj.weight)
        if not self.beta_logit.is_meta:
            self.beta_logit.data.fill_(self.product_beta_init)

    def remap_rows(self, rows: torch.Tensor) -> torch.Tensor:
        if self.row_permutation == "none":
            return rows
        if self.row_permutation == "reverse":
            return self.num_rows - 1 - rows
        if self.row_permutation == "affine":
            return (rows * 1543 + 17).remainder(self.num_rows)
        raise OLMoConfigurationError(f"Unknown ComEmbed row permutation: {self.row_permutation}")

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        qr_rows = self.remap_rows(rows)
        c1 = self.codebook1(qr_rows % self.codebook_size)
        c2 = self.codebook2(qr_rows // self.codebook_size)
        c1 = _register_tensor_grad_sanitizer(c1)
        c2 = _register_tensor_grad_sanitizer(c2)
        c1 = _debug_register_grad_check("QRRowMemory.c1", c1)
        c2 = _debug_register_grad_check("QRRowMemory.c2", c2)
        _debug_check_finite("QRRowMemory.c1", c1)
        _debug_check_finite("QRRowMemory.c2", c2)
        product = (c1.float() * c2.float()).clamp_(-self.product_clip, self.product_clip).to(c1.dtype)
        product = _register_tensor_grad_sanitizer(product)
        product = _debug_register_grad_check("QRRowMemory.product", product)
        _debug_check_finite("QRRowMemory.product", product)
        out = c1 + c2 + torch.sigmoid(self.beta_logit.to(c1.dtype)) * product
        out = _register_tensor_grad_sanitizer(out)
        out = _debug_register_grad_check("QRRowMemory.product_out", out)
        _debug_check_finite("QRRowMemory.product_out", out)
        out = out + self.residual_proj(self.residual(rows))
        if self.output_rmsnorm:
            out = rms_norm(out) * self.init_std
        _debug_check_finite("QRRowMemory.residual_out", out)
        gated = out if self.disable_row_gate else torch.sigmoid(self.row_gate(rows)) * out
        gated = _register_tensor_grad_sanitizer(gated)
        _debug_check_finite("QRRowMemory.gated_out", gated)
        return _debug_register_grad_check("QRRowMemory.gated_out", gated)


class QRAddNormProductRowMemory(nn.Module):
    def __init__(
        self,
        *,
        num_rows: int,
        dim: int,
        codebook_size: int = 4096,
        residual_dim: int = 64,
        disable_row_gate: bool = False,
        output_rmsnorm: bool = False,
        init_std: Optional[float] = None,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.num_rows = int(num_rows)
        self.dim = int(dim)
        self.codebook_size = int(codebook_size)
        self.num_codes = (self.num_rows + self.codebook_size - 1) // self.codebook_size
        self.residual_dim = int(residual_dim)
        self.disable_row_gate = bool(disable_row_gate)
        self.output_rmsnorm = bool(output_rmsnorm)
        self.init_std = float(init_std or math.sqrt(2.0 / (self.num_rows + self.dim)))
        self.row_gate_init = -8.0

        self.codebook1 = nn.Embedding(self.codebook_size, self.dim, device=device, dtype=dtype)
        self.codebook2 = nn.Embedding(self.num_codes, self.dim, device=device, dtype=dtype)
        self.beta_logit = nn.Parameter(torch.tensor([-2.0], device=device, dtype=torch.float32))
        self.row_gate = nn.Embedding(self.num_rows, 1, device=device, dtype=dtype)
        self.residual = nn.Embedding(self.num_rows, self.residual_dim, device=device, dtype=dtype)
        self.residual_proj = nn.Linear(
            self.residual_dim,
            self.dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.reset_comembed_parameters()

    def reset_comembed_parameters(self) -> None:
        _normal_(self.codebook1, std=self.init_std / math.sqrt(2.0))
        _normal_(self.codebook2, std=self.init_std / math.sqrt(2.0))
        if not self.row_gate.weight.is_meta:
            self.row_gate.weight.data.fill_(self.row_gate_init)
        _normal_(self.residual, std=self.init_std)
        _zeros_(self.residual_proj.weight)
        if not self.beta_logit.is_meta:
            self.beta_logit.data.fill_(-2.0)

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        c1 = self.codebook1(rows % self.codebook_size)
        c2 = self.codebook2(rows // self.codebook_size)
        c1 = _register_tensor_grad_sanitizer(c1)
        c2 = _register_tensor_grad_sanitizer(c2)
        c1 = _debug_register_grad_check("QRAddNormProductRowMemory.c1", c1)
        c2 = _debug_register_grad_check("QRAddNormProductRowMemory.c2", c2)
        _debug_check_finite("QRAddNormProductRowMemory.c1", c1)
        _debug_check_finite("QRAddNormProductRowMemory.c2", c2)
        c1_norm = rms_norm(c1) * self.init_std
        c2_norm = rms_norm(c2) * self.init_std
        c1_norm = _register_tensor_grad_sanitizer(c1_norm)
        c2_norm = _register_tensor_grad_sanitizer(c2_norm)
        c1_norm = _debug_register_grad_check("QRAddNormProductRowMemory.c1_norm", c1_norm)
        c2_norm = _debug_register_grad_check("QRAddNormProductRowMemory.c2_norm", c2_norm)
        product = rms_norm(c1_norm * c2_norm) * self.init_std
        product = _register_tensor_grad_sanitizer(product)
        product = _debug_register_grad_check("QRAddNormProductRowMemory.product", product)
        _debug_check_finite("QRAddNormProductRowMemory.product", product)
        out = c1 + c2 + torch.sigmoid(self.beta_logit.to(c1.dtype)) * product
        out = _register_tensor_grad_sanitizer(out)
        out = _debug_register_grad_check("QRAddNormProductRowMemory.out", out)
        out = out + self.residual_proj(self.residual(rows))
        if self.output_rmsnorm:
            out = rms_norm(out) * self.init_std
        gated = out if self.disable_row_gate else torch.sigmoid(self.row_gate(rows)) * out
        gated = _register_tensor_grad_sanitizer(gated)
        _debug_check_finite("QRAddNormProductRowMemory.gated_out", gated)
        return _debug_register_grad_check("QRAddNormProductRowMemory.gated_out", gated)


class QRAddResidualRowMemory(nn.Module):
    def __init__(
        self,
        *,
        num_rows: int,
        dim: int,
        codebook_size: int = 4096,
        residual_dim: int = 64,
        disable_row_gate: bool = False,
        output_rmsnorm: bool = False,
        init_std: Optional[float] = None,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.num_rows = int(num_rows)
        self.dim = int(dim)
        self.codebook_size = int(codebook_size)
        self.num_codes = (self.num_rows + self.codebook_size - 1) // self.codebook_size
        self.residual_dim = int(residual_dim)
        self.disable_row_gate = bool(disable_row_gate)
        self.output_rmsnorm = bool(output_rmsnorm)
        self.init_std = float(init_std or math.sqrt(2.0 / (self.num_rows + self.dim)))
        self.row_gate_init = -8.0

        self.codebook1 = nn.Embedding(self.codebook_size, self.dim, device=device, dtype=dtype)
        self.codebook2 = nn.Embedding(self.num_codes, self.dim, device=device, dtype=dtype)
        self.row_gate = nn.Embedding(self.num_rows, 1, device=device, dtype=dtype)
        self.residual = nn.Embedding(self.num_rows, self.residual_dim, device=device, dtype=dtype)
        self.residual_proj = nn.Linear(
            self.residual_dim,
            self.dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.reset_comembed_parameters()

    def reset_comembed_parameters(self) -> None:
        _normal_(self.codebook1, std=self.init_std / math.sqrt(2.0))
        _normal_(self.codebook2, std=self.init_std / math.sqrt(2.0))
        if not self.row_gate.weight.is_meta:
            self.row_gate.weight.data.fill_(self.row_gate_init)
        _normal_(self.residual, std=self.init_std)
        _zeros_(self.residual_proj.weight)

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        c1 = self.codebook1(rows % self.codebook_size)
        c2 = self.codebook2(rows // self.codebook_size)
        c1 = _register_tensor_grad_sanitizer(c1)
        c2 = _register_tensor_grad_sanitizer(c2)
        c1 = _debug_register_grad_check("QRAddResidualRowMemory.c1", c1)
        c2 = _debug_register_grad_check("QRAddResidualRowMemory.c2", c2)
        _debug_check_finite("QRAddResidualRowMemory.c1", c1)
        _debug_check_finite("QRAddResidualRowMemory.c2", c2)
        out = c1 + c2 + self.residual_proj(self.residual(rows))
        out = _register_tensor_grad_sanitizer(out)
        out = _debug_register_grad_check("QRAddResidualRowMemory.out", out)
        _debug_check_finite("QRAddResidualRowMemory.out_pre_norm", out)
        if self.output_rmsnorm:
            out = rms_norm(out) * self.init_std
        _debug_check_finite("QRAddResidualRowMemory.out_post_norm", out)
        gated = out if self.disable_row_gate else torch.sigmoid(self.row_gate(rows)) * out
        gated = _register_tensor_grad_sanitizer(gated)
        _debug_check_finite("QRAddResidualRowMemory.gated_out", gated)
        return _debug_register_grad_check("QRAddResidualRowMemory.gated_out", gated)


class FrequencyAwareQRLookup(nn.Module):
    def __init__(
        self,
        *,
        router: nn.Module,
        dim: int,
        codebook_size: int = 4096,
        residual_dim: int = 64,
        row_memory_cls: Type[nn.Module] = QRRowMemory,
        row_memory_kwargs: Optional[Dict[str, Any]] = None,
        init_std: Optional[float] = None,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        if not hasattr(router, "total_capacity"):
            raise OLMoConfigurationError("FrequencyAwareQRLookup router must expose total_capacity")
        self.router = router
        self.dim = int(dim)
        self.row_memory = row_memory_cls(
            num_rows=int(router.total_capacity),
            dim=self.dim,
            codebook_size=codebook_size,
            residual_dim=residual_dim,
            init_std=init_std,
            device=device,
            dtype=dtype,
            **(row_memory_kwargs or {}),
        )
        _debug_register_param_grad_checks(self.row_memory, prefix=self.row_memory.__class__.__name__)
        register_comembed_grad_sanitizer(self.row_memory)

    def reset_comembed_parameters(self) -> None:
        if hasattr(self.row_memory, "reset_comembed_parameters"):
            self.row_memory.reset_comembed_parameters()

    def _reset_injection_buffers(self, *, device: torch.device | str) -> None:
        if hasattr(self.router, "_reset_injection_buffers"):
            self.router._reset_injection_buffers(device=device)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        router_out = self.router(input_ids)
        if not isinstance(router_out, tuple) or len(router_out) != 2:
            raise OLMoConfigurationError(
                "FrequencyAwareQRLookup requires a router that returns (rows, weights)"
            )
        rows, weights = router_out
        weights = _register_tensor_grad_sanitizer(weights)
        weights = _debug_register_grad_check("FrequencyAwareQRLookup.weights", weights)
        _debug_check_finite("FrequencyAwareQRLookup.weights", weights)
        values = self.row_memory(rows)
        values = _register_tensor_grad_sanitizer(values)
        values = _debug_register_grad_check("FrequencyAwareQRLookup.values", values)
        _debug_check_finite("FrequencyAwareQRLookup.values", values)
        weights = weights.to(dtype=values.dtype, device=values.device)
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        out = (values * weights.unsqueeze(-1)).sum(dim=-2) / denom
        out = _register_tensor_grad_sanitizer(out)
        _debug_check_finite("FrequencyAwareQRLookup.out", out)
        return _debug_register_grad_check("FrequencyAwareQRLookup.out", out)


class ContextMaskNgramPQLookup(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        dim: int,
        codebook_size: int = 4096,
        groups: int = 32,
        split: Tuple[int, int, int, int] = (8, 12, 8, 4),
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        if dim % groups != 0:
            raise OLMoConfigurationError(f"ComEmbed PQ dim ({dim}) must be divisible by groups ({groups})")
        if sum(split) != groups:
            raise OLMoConfigurationError(f"ComEmbed PQ split {split} must sum to groups ({groups})")
        self.vocab_size = int(vocab_size)
        self.dim = int(dim)
        self.codebook_size = int(codebook_size)
        self.groups = int(groups)
        self.split = tuple(int(x) for x in split)
        self.init_std = math.sqrt(2.0 / (self.vocab_size + self.dim))
        group_dim = self.dim // self.groups
        self.codebooks = nn.ModuleList(
            [
                nn.Embedding(self.codebook_size, group_dim, device=device, dtype=dtype)
                for _ in range(self.groups)
            ]
        )
        self.reset_comembed_parameters()

    def reset_comembed_parameters(self) -> None:
        for table in self.codebooks:
            _normal_(table, std=self.init_std)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        cur = input_ids
        prev1 = F.pad(input_ids[:, :-1], (1, 0), value=0)
        prev2 = (
            F.pad(input_ids[:, :-2], (2, 0), value=0)
            if input_ids.size(1) > 1
            else torch.zeros_like(input_ids)
        )
        keys = (
            cur,
            (prev1 * 65537 + cur).remainder(2**31),
            (prev2 * 65537 + cur).remainder(2**31),
            (prev2 * 65537 * 17 + prev1 * 65537 + cur).remainder(2**31),
        )

        chunks = []
        for group, table in enumerate(self.codebooks):
            key = _key_by_split(group, keys, self.split)
            row = _hash_key(key, group + 1, self.codebook_size)
            chunks.append(table(row))
        return torch.cat(chunks, dim=-1)
