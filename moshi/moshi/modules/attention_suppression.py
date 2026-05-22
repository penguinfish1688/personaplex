"""Pre-interruption attention suppression utilities."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import torch


TIMESTEP_KEYS = (
    "t_int",
    "interrupt_timestep",
    "interruption_timestep",
    "start_timestep",
    "query_start_timestep",
    "zero_buffer_start",
)

SECONDS_KEYS = (
    "interrupt_start",
    "interruption_start",
    "interrupt_start_sec",
    "interruption_start_sec",
    "start_seconds",
    "start_sec",
)

ID_KEYS = ("example_id", "sample_id", "id", "uid", "name")


def _validate_lambda(lambda_suppression: float) -> None:
    if not (0.0 < float(lambda_suppression) <= 1.0):
        raise ValueError(
            f"lambda_suppression must satisfy 0 < lambda <= 1, got {lambda_suppression}"
        )


def _contiguous_ranges(indices: list[int]) -> list[list[int]]:
    if not indices:
        return []
    ranges: list[list[int]] = []
    start = prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
            continue
        ranges.append([start, prev + 1])
        start = prev = idx
    ranges.append([start, prev + 1])
    return ranges


def apply_pre_interrupt_attention_suppression(
    attn_logits: torch.Tensor,
    *,
    current_timestep: int,
    interrupt_timestep: int,
    kv_cache_start_timestep: int = 0,
    k_post_interrupt: int = 10,
    n_pre_interrupt: int = 30,
    lambda_suppression: float = 0.2,
    key_positions: Optional[torch.Tensor] = None,
    stats: Optional[dict[str, Any]] = None,
    layer_idx: Optional[int] = None,
) -> torch.Tensor:
    """Add log(lambda) to pre-interruption key logits before softmax.

    Args:
        attn_logits: Attention logits shaped ``[batch, heads, q_len, kv_len]``.
        current_timestep: Absolute timestep for the first query position.
        interrupt_timestep: Absolute interruption onset timestep.
        kv_cache_start_timestep: Absolute timestep aligned to key index 0 when
            ``key_positions`` is not provided.
        key_positions: Optional absolute key positions in logit key-dimension
            order. This is required for ring-buffer caches whose slots are not
            chronological.
    """
    _validate_lambda(float(lambda_suppression))
    if float(lambda_suppression) == 1.0:
        return attn_logits
    if attn_logits.dim() != 4:
        raise ValueError(
            f"attn_logits must have shape [batch, heads, q_len, kv_len], got {tuple(attn_logits.shape)}"
        )

    q_len = int(attn_logits.shape[-2])
    kv_len = int(attn_logits.shape[-1])
    if q_len != 1:
        raise ValueError(
            "Pre-interruption attention suppression currently supports q_len == 1. "
            f"Got q_len={q_len}; pass per-query absolute timesteps before enabling chunked decode."
        )

    current_timestep = int(current_timestep)
    interrupt_timestep = int(interrupt_timestep)
    if current_timestep < interrupt_timestep or current_timestep >= interrupt_timestep + int(k_post_interrupt):
        return attn_logits

    suppress_start_abs = max(0, interrupt_timestep - int(n_pre_interrupt))
    suppress_end_abs = interrupt_timestep
    if suppress_start_abs >= suppress_end_abs:
        return attn_logits

    bias = math.log(float(lambda_suppression))
    if key_positions is not None:
        positions = key_positions.to(device=attn_logits.device)
        if positions.dim() != 1:
            positions = positions.reshape(-1)
        if int(positions.numel()) != kv_len:
            raise ValueError(
                f"key_positions length {int(positions.numel())} does not match kv_len {kv_len}"
            )
        suppress_mask = (positions >= suppress_start_abs) & (positions < suppress_end_abs)
        if not bool(suppress_mask.any().item()):
            return attn_logits
        suppressed_indices = torch.where(suppress_mask)[0].detach().cpu().tolist()
        attn_logits[..., suppress_mask] = attn_logits[..., suppress_mask] + bias
        suppressed_ranges_idx = _contiguous_ranges([int(i) for i in suppressed_indices])
    else:
        suppress_start_idx = max(0, suppress_start_abs - int(kv_cache_start_timestep))
        suppress_end_idx = min(kv_len, suppress_end_abs - int(kv_cache_start_timestep))
        if suppress_start_idx >= suppress_end_idx:
            return attn_logits
        attn_logits[..., suppress_start_idx:suppress_end_idx] = (
            attn_logits[..., suppress_start_idx:suppress_end_idx] + bias
        )
        suppressed_ranges_idx = [[int(suppress_start_idx), int(suppress_end_idx)]]

    if stats is not None:
        stats.setdefault("suppressed_query_steps", set()).add(int(current_timestep))
        stats.setdefault("layers_applied", set()).add(int(layer_idx) if layer_idx is not None else None)
        stats.setdefault("suppressed_key_range_abs", [int(suppress_start_abs), int(suppress_end_abs)])
        stats.setdefault("suppressed_key_range_idx", suppressed_ranges_idx[0])
        stats.setdefault("suppressed_key_ranges_idx", suppressed_ranges_idx)

    return attn_logits


def _numeric(value: Any, *, path: Path, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Expected numeric '{field}' in {path}, got {type(value).__name__}")
    return float(value)


def extract_interrupt_timestep(
    payload: dict[str, Any],
    *,
    path: Path,
    token_rate_hz: Optional[float] = None,
) -> int:
    """Extract interruption onset timestep from one timing JSON payload."""
    candidates: list[tuple[str, int]] = []
    for key in TIMESTEP_KEYS:
        if key in payload:
            raw = _numeric(payload[key], path=path, field=key)
            if not float(raw).is_integer():
                raise ValueError(f"Timing field '{key}' in {path} must be an integer timestep, got {raw}")
            candidates.append((key, int(raw)))

    for key in SECONDS_KEYS:
        if key in payload:
            if token_rate_hz is None:
                raise ValueError(
                    f"Timing field '{key}' in {path} is seconds; provide token_rate_hz for conversion"
                )
            seconds = _numeric(payload[key], path=path, field=key)
            candidates.append((key, int(seconds * float(token_rate_hz))))

    if not candidates:
        fields = ", ".join(TIMESTEP_KEYS + SECONDS_KEYS)
        raise KeyError(f"No interruption timing field found in {path}. Expected one of: {fields}")

    unique_values = {value for _, value in candidates}
    if len(unique_values) > 1:
        details = ", ".join(f"{key}={value}" for key, value in candidates)
        raise ValueError(f"Multiple incompatible interruption timing fields in {path}: {details}")
    return int(candidates[0][1])


def load_interrupt_timesteps(
    timing_path: str | Path,
    *,
    token_rate_hz: Optional[float] = None,
) -> dict[str, int]:
    """Load example-id to interruption timestep mapping from timing JSON."""
    path = Path(timing_path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        if any(key in payload for key in TIMESTEP_KEYS + SECONDS_KEYS):
            return {path.parent.name: extract_interrupt_timestep(payload, path=path, token_rate_hz=token_rate_hz)}
        out: dict[str, int] = {}
        for example_id, item in payload.items():
            if not isinstance(item, dict):
                raise ValueError(f"Expected dict timing payload for example '{example_id}' in {path}")
            out[str(example_id)] = extract_interrupt_timestep(item, path=path, token_rate_hz=token_rate_hz)
        if not out:
            raise ValueError(f"No timing entries found in {path}")
        return out

    if isinstance(payload, list):
        out: dict[str, int] = {}
        for idx, item in enumerate(payload):
            if not isinstance(item, dict):
                raise ValueError(f"Expected dict timing payload at index {idx} in {path}")
            ids = [str(item[key]) for key in ID_KEYS if key in item]
            if len(set(ids)) > 1:
                raise ValueError(f"Multiple incompatible example id fields at index {idx} in {path}: {ids}")
            example_id = ids[0] if ids else str(idx)
            out[example_id] = extract_interrupt_timestep(item, path=path, token_rate_hz=token_rate_hz)
        if not out:
            raise ValueError(f"No timing entries found in {path}")
        return out

    raise ValueError(f"Expected dict or list in {path}, got {type(payload).__name__}")
