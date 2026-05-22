# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Offline inference entrypoint for PersonaPlex that mirrors server.py behavior without a WebSocket server.

High-level flow:
- Load Mimi encoders/decoders, Moshi LM, and tokenizer (same as server.py)
- Warmup to initialize CUDA graphs and streaming state
- Prompt phase: load system text tokens and a voice prompt WAV (agent side)
- Streaming-like phase: feed user audio frames from a WAV file into the "input" channels,
  autoregressively sample text + agent audio channels each step, and decode audio frames
- Concatenate generated frames and write an output WAV matching the input duration

This script reuses helpers from lm.py (load_audio, _iterate_audio, encode_from_sphn) to
keep parity with voice-prompt feeding logic in the server.
"""

import argparse
import os
import tarfile
import gc
import pickle
from pathlib import Path
import json
from typing import Optional, List, Dict, Any, cast

import numpy as np
import torch
import sentencepiece
import sphn
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from .client_utils import make_log
from .models import loaders, LMGen, MimiModel
from .models.lm import load_audio as lm_load_audio
from .models.lm import _iterate_audio as lm_iterate_audio
from .models.lm import encode_from_sphn as lm_encode_from_sphn
from .models.lm import HiddenLayerOutputs, SILENCE_TOKENS
from .modules.attention_suppression import _validate_lambda

def log(level: str, msg: str):
    print(make_log(level, msg))


def log_memory(prefix: str = ""):
    """Log current GPU memory usage stats."""
    if not torch.cuda.is_available():
        return
    
    allocated = torch.cuda.memory_allocated() / 1e9  # GB
    reserved = torch.cuda.memory_reserved() / 1e9    # GB
    max_allocated = torch.cuda.max_memory_allocated() / 1e9  # GB
    
    prefix_str = f"[{prefix}] " if prefix else ""
    log("info", f"{prefix_str}GPU Memory: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB, max_allocated={max_allocated:.2f}GB")


def seed_all(seed: int):
    """Seed torch, CUDA, numpy, and Python RNG for reproducible runs.

    Matches the seeding strategy in server.py.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    import random
    import numpy as _np
    random.seed(seed)
    _np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def wrap_with_system_tags(text: str) -> str:
    """Add system tags as the model expects if they are missing.
    Example: "<system> You enjoy having a good conversation. Have a deep conversation about technology. Your name is Jane. <system>"
    """
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


def average_hidden_layers(hidden_layers_list: List[HiddenLayerOutputs]) -> HiddenLayerOutputs:
    """Average a list of HiddenLayerOutputs across steps into a single averaged output.
    
    Args:
        hidden_layers_list: List of HiddenLayerOutputs objects (one per step)
    
    Returns:
        Single HiddenLayerOutputs object with averaged tensors
    """
    if not hidden_layers_list:
        raise ValueError("Cannot average empty hidden layers list")
    
    if len(hidden_layers_list) == 1:
        return hidden_layers_list[0]
    
    num_steps = len(hidden_layers_list)
    
    # Average text transformer layers
    avg_text_transformer = None
    if hidden_layers_list[0].text_transformer is not None:
        num_text_layers = len(hidden_layers_list[0].text_transformer)
        avg_text_transformer = [
            torch.stack([h.text_transformer[i] for h in hidden_layers_list]).mean(dim=0)
            for i in range(num_text_layers)
        ]
    
    # Average depth transformer layers
    avg_depth_transformer = None
    if hidden_layers_list[0].depth_transformer is not None:
        num_codebooks = len(hidden_layers_list[0].depth_transformer)
        num_depth_layers = len(hidden_layers_list[0].depth_transformer[0])
        avg_depth_transformer = [
            [
                torch.stack([h.depth_transformer[c][l] for h in hidden_layers_list]).mean(dim=0)
                for l in range(num_depth_layers)
            ]
            for c in range(num_codebooks)
        ]

    avg_text_pre_unembed = None
    if hidden_layers_list[0].text_pre_unembed is not None:
        avg_text_pre_unembed = torch.stack(
            [h.text_pre_unembed for h in hidden_layers_list if h.text_pre_unembed is not None]
        ).mean(dim=0)
    
    return HiddenLayerOutputs(
        text_transformer=avg_text_transformer,
        text_pre_unembed=avg_text_pre_unembed,
        depth_transformer=avg_depth_transformer
    )


def _extract_text_hidden_per_layer(step_hidden: HiddenLayerOutputs) -> torch.Tensor:
    """Convert one step of text hidden layers to a dense `[L, D]` CPU tensor.

    Args:
        step_hidden: Hidden layers for one generated token.

    Returns:
        Tensor of shape `[num_text_layers, hidden_dim]` in `float32` on CPU.
    """
    if not step_hidden.text_transformer:
        raise RuntimeError("Missing text transformer hidden layers in step output.")
    layers: list[torch.Tensor] = []
    for layer_hidden in step_hidden.text_transformer:
        layer = layer_hidden.detach().cpu().float()
        if layer.dim() == 3 and layer.shape[0] == 1 and layer.shape[1] == 1:
            layer = layer[0, 0]
        elif layer.dim() != 1:
            layer = layer.reshape(-1)
        layers.append(layer)
    return torch.stack(layers, dim=0)


def _extract_text_attention_per_layer(step_hidden: HiddenLayerOutputs) -> Optional[torch.Tensor]:
    """Convert one step of text attention weights to a `[L, H, K]` CPU tensor.

    `K` is the available context length at this generation step and can vary by token.
    """
    if not step_hidden.text_attention_weights:
        return None
    per_layer: list[torch.Tensor] = []
    for layer_attn in step_hidden.text_attention_weights:
        attn = layer_attn.detach().cpu().float()
        # Expected shape from transformer: [B, H, Tq, Tk] with B=1, Tq=1 in step mode.
        if attn.dim() == 4 and attn.shape[0] == 1:
            attn = attn[0]
        if attn.dim() == 3 and attn.shape[1] == 1:
            attn = attn[:, 0, :]
        elif attn.dim() == 3 and attn.shape[0] == 1:
            attn = attn[0]
        if attn.dim() != 2:
            raise RuntimeError(f"Unexpected attention shape {tuple(attn.shape)}")
        per_layer.append(attn)
    return torch.stack(per_layer, dim=0)


def _extract_text_pre_unembed_state(step_hidden: HiddenLayerOutputs) -> torch.Tensor:
    """Convert one step of pre-unembedding text state to a dense `[D]` CPU tensor."""
    if step_hidden.text_pre_unembed is None:
        raise RuntimeError("Missing pre-unembedding text state in step output.")
    x = step_hidden.text_pre_unembed.detach().cpu().float()
    if x.dim() == 3 and x.shape[0] == 1 and x.shape[1] == 1:
        x = x[0, 0]
    elif x.dim() == 2 and x.shape[0] == 1:
        x = x[0]
    elif x.dim() != 1:
        x = x.reshape(-1)
    return x


def _extract_full_input_embedding_for_step(step_embedding: torch.Tensor) -> torch.Tensor:
    """Convert one step of full transformer input embedding to a dense `[D]` CPU tensor.

    `step_embedding` is expected to be the output of `LMGen.step(..., return_embeddings=True)`
    for a single token step, i.e. embedding of (text + autoregressive audio + user audio).
    """
    x = step_embedding.detach().cpu().float()
    if x.dim() == 3 and x.shape[0] == 1 and x.shape[1] == 1:
        x = x[0, 0]
    elif x.dim() == 2 and x.shape[0] == 1:
        x = x[0]
    elif x.dim() != 1:
        x = x.reshape(-1)
    return x


def _extract_step_token_ids(step_tokens: torch.Tensor) -> torch.Tensor:
    """Convert one-step token tensor to dense `[K]` int64 CPU ids.

    Accepts shape `[B, K, 1]` (expected in step mode) and returns the first batch item.
    """
    x = step_tokens.detach().cpu().long()
    if x.dim() == 3:
        if x.shape[0] < 1 or x.shape[2] != 1:
            raise ValueError(f"Expected step token shape [B, K, 1], got {tuple(x.shape)}")
        x = x[0, :, 0]
    elif x.dim() == 2:
        if x.shape[0] < 1:
            raise ValueError(f"Expected step token shape [B, K], got {tuple(x.shape)}")
        x = x[0]
    elif x.dim() != 1:
        raise ValueError(f"Unsupported step token shape: {tuple(x.shape)}")
    return x


def _text_transformer_offset_cpu(lm: Any) -> int:
    """Return the current absolute text-transformer stream offset."""
    layers = getattr(getattr(lm, "transformer", None), "layers", None)
    if layers is None or len(layers) == 0:
        return 0
    attn = getattr(layers[0], "self_attn", None)
    state = getattr(attn, "_streaming_state", None)
    if state is None:
        return 0
    return int(getattr(state, "offset_cpu", 0))


def _json_ready_attention_suppression_stats(stats: dict[str, Any]) -> dict[str, Any]:
    out = dict(stats)
    query_steps = sorted(int(x) for x in out.pop("suppressed_query_steps", set()))
    layers = sorted(int(x) for x in out.pop("layers_applied", set()) if x is not None)
    out["number_of_suppressed_query_steps"] = len(query_steps)
    out["suppressed_query_steps_abs"] = query_steps
    out["layers_applied"] = layers
    return out


def _extract_target_layer_text_keys_and_positions(
    target_layer_module: Any,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """Extract chronological key cache and absolute positions from a target layer.

    Returns:
      - text_keys: [1, H, K, D_h] on CPU (chronological order)
      - text_key_positions: [K] int64 absolute positions
      - text_key_cache_meta: scalar cache coverage metadata
    """
    attn = target_layer_module.self_attn
    state = getattr(attn, "_streaming_state", None)
    if state is None or getattr(state, "kv_cache", None) is None:
        raise RuntimeError("Target layer has no streaming KV cache state.")

    kv_cache = state.kv_cache
    keys = kv_cache.cache[0]  # [B, H, C, D]
    end_offset = int(kv_cache.end_offset.item())
    capacity = int(kv_cache.capacity)
    valid_len = min(end_offset, capacity)
    if valid_len <= 0:
        raise RuntimeError("Target layer KV cache is empty.")

    # Mirror RingKVCache.complete() position mapping, then reorder to chronological.
    ring_indexes = torch.arange(capacity, device=keys.device, dtype=torch.long)
    invalid = ring_indexes >= end_offset
    end_index = end_offset % capacity
    delta = ring_indexes - end_index
    ring_positions = torch.where(
        delta <= 0,
        torch.full_like(ring_indexes, end_offset) + delta,
        torch.full_like(ring_indexes, end_offset) + delta - capacity,
    )
    ring_positions = torch.where(invalid, torch.full_like(ring_positions, -1), ring_positions)

    valid_ring_indices = torch.where(ring_positions >= 0)[0]
    if valid_ring_indices.numel() <= 0:
        raise RuntimeError("Target layer KV cache has no valid key positions.")
    valid_positions = ring_positions.index_select(0, valid_ring_indices)
    _, sort_order = valid_positions.sort()
    chrono_ring_indices = valid_ring_indices.index_select(0, sort_order)

    ordered_keys = keys.index_select(2, chrono_ring_indices)
    ordered_positions = ring_positions.index_select(0, chrono_ring_indices)

    text_key_cache_meta = {
        "capacity": int(capacity),
        "end_offset": int(end_offset),
        "valid_len": int(valid_len),
        "dropped_prefix_tokens": int(max(0, end_offset - valid_len)),
    }
    return (
        ordered_keys.detach().cpu().float(),
        ordered_positions.detach().cpu().long(),
        text_key_cache_meta,
    )


def _build_hidden_payload(
    *,
    input_wav: str,
    output_wav: str,
    output_text: str,
    frame_rate_hz: float,
    text_token_ids: list[int],
    text_token_pieces: list[str],
    text_hidden_layers_per_token: list[torch.Tensor],
    text_pre_unembed_states_per_token: list[torch.Tensor],
    full_input_embeddings_per_token: list[torch.Tensor],
    input_token_ids_per_token: list[torch.Tensor],
    output_token_ids_per_token: list[torch.Tensor],
    text_attention_layers_per_token: list[Optional[torch.Tensor]],
    text_target_layer_keys: torch.Tensor,
    text_key_positions: torch.Tensor,
    text_key_cache_meta: dict[str, int],
) -> Dict[str, Any]:
    """Build the serialized `.pt` payload consumed by premature decode tools.

    Saved schema (`schema_version=6`):
        - `text_hidden_layers`: `torch.FloatTensor[T, L, D]`
          - `T`: number of generated output tokens from `input.wav` processing only.
          - `L`: number of main text transformer layers.
          - `D`: hidden size.
        - `text_attention_weights`: `list[torch.FloatTensor[L, H, K_t]]`
          - Per-token attention weights for the main text transformer only.
          - `H`: attention heads.
          - `K_t`: available key length at token `t` (can vary with causal growth).
        - `token_ids`: `torch.LongTensor[T]` generated text token ids.
        - `token_names`: `list[str]` generated token pieces (special tokens preserved).
                - `input_token_ids`: `torch.LongTensor[T, K_in]` per-step model input token ids.
                    - `K_in` is typically 17: text(1) + model-audio(8) + user-audio(8).
                - `output_token_ids`: `torch.LongTensor[T, K_out]` per-step model output token ids.
                    - `K_out` is typically 9: text(1) + model-audio(8).
        - `times`: `torch.FloatTensor[T]` token start time in seconds.
        - `token_time_ranges_sec`: `torch.FloatTensor[T, 2]` token `[start, end)`.
        - `hidden_states`: `torch.FloatTensor[T, D]` final-layer hidden states (compat key).
                - `text_pre_unembed_states`: `torch.FloatTensor[T, D]` main transformer output
                    after output norm and before text unembedding matrix.
                - `full_input_embeddings`: `torch.FloatTensor[T, D]` per-step full transformer input
                    embedding from `embed_codes` (text + autoregressive audio + user audio).
        - `frame_rate`: scalar float, default 12.5 for Moshi.
        - `text_keys`: `torch.FloatTensor[1, H, K, D_h]` final accumulated text KV-cache keys.
                - `text_key_positions`: `torch.LongTensor[K]` absolute key positions aligned to `text_keys`.
                - `text_key_cache_meta`: scalar cache metadata (`capacity`, `end_offset`, `valid_len`,
                    `dropped_prefix_tokens`).

    Token-time alignment:
        token `0` corresponds to `[0, 1/frame_rate_hz)` seconds of generated response,
        token `t` corresponds to `[t/frame_rate_hz, (t+1)/frame_rate_hz)`.
    """
    if len(text_hidden_layers_per_token) == 0:
        raise RuntimeError("Cannot save hidden payload: no generated tokens were captured.")

    hidden_tensor = torch.stack(text_hidden_layers_per_token, dim=0)  # [T, L, D]
    token_ids_tensor = torch.tensor(text_token_ids, dtype=torch.long)
    t = hidden_tensor.shape[0]
    times = torch.arange(t, dtype=torch.float32) / float(frame_rate_hz)
    token_time_ranges = torch.stack(
        [times, times + (1.0 / float(frame_rate_hz))],
        dim=1,
    )

    if len(text_pre_unembed_states_per_token) != t:
        raise RuntimeError(
            "Mismatch in payload lengths: "
            f"hidden={t}, text_pre_unembed={len(text_pre_unembed_states_per_token)}"
        )
    if len(full_input_embeddings_per_token) != t:
        raise RuntimeError(
            "Mismatch in payload lengths: "
            f"hidden={t}, full_input_embeddings={len(full_input_embeddings_per_token)}"
        )
    if len(input_token_ids_per_token) != t:
        raise RuntimeError(
            "Mismatch in payload lengths: "
            f"hidden={t}, input_token_ids={len(input_token_ids_per_token)}"
        )
    if len(output_token_ids_per_token) != t:
        raise RuntimeError(
            "Mismatch in payload lengths: "
            f"hidden={t}, output_token_ids={len(output_token_ids_per_token)}"
        )

    input_token_ids = torch.stack(input_token_ids_per_token, dim=0)
    output_token_ids = torch.stack(output_token_ids_per_token, dim=0)
    input_token_width = int(input_token_ids.shape[1])
    output_token_width = int(output_token_ids.shape[1])

    payload: Dict[str, Any] = {
        "schema_version": 6,
        "input_wav": input_wav,
        "output_wav": output_wav,
        "output_text": output_text,
        "frame_rate": float(frame_rate_hz),
        "token_ids": token_ids_tensor,
        "token_names": text_token_pieces,
        "input_token_ids": input_token_ids,
        "input_token_width": input_token_width,
        "output_token_ids": output_token_ids,
        "output_token_width": output_token_width,
        "times": times,
        "token_time_ranges_sec": token_time_ranges,
        "text_hidden_layers": hidden_tensor,
        "text_pre_unembed_states": torch.stack(text_pre_unembed_states_per_token, dim=0),
        "full_input_embeddings": torch.stack(full_input_embeddings_per_token, dim=0),
        "text_attention_weights": text_attention_layers_per_token,
        "hidden_states": hidden_tensor[:, -1, :],
        "text_keys": text_target_layer_keys,
        "text_key_positions": text_key_positions,
        "text_key_cache_meta": text_key_cache_meta,
    }
    return payload


def _summarize_scalar_statistics(values: torch.Tensor) -> Dict[str, float]:
    """Return mean and uncertainty estimates for a 1D tensor."""
    flat = values.detach().cpu().float().reshape(-1)
    n = int(flat.numel())
    if n == 0:
        raise ValueError("Cannot summarize empty values tensor")

    mean = float(flat.mean().item())
    if n == 1:
        std = 0.0
        sem = 0.0
    else:
        std = float(flat.std(unbiased=True).item())
        sem = float(std / np.sqrt(n))

    ci95_half_width = 1.96 * sem
    return {
        "n": float(n),
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_low": mean - ci95_half_width,
        "ci95_high": mean + ci95_half_width,
    }


def _compute_embed_stats_for_instance(
    *,
    lm: Any,
    input_token_ids_per_token: list[torch.Tensor],
    device: str,
) -> Dict[str, Any]:
    """Compute per-instance text/audio embedding norm stats and cosine similarity stats."""
    if len(input_token_ids_per_token) == 0:
        raise RuntimeError("No input tokens collected; cannot compute embed stats")

    input_token_ids = torch.stack(input_token_ids_per_token, dim=0).to(device)  # [T, K]
    if input_token_ids.dim() != 2 or input_token_ids.shape[1] < 2:
        raise RuntimeError(f"Unexpected token id shape for embed stats: {tuple(input_token_ids.shape)}")

    text_token_ids = input_token_ids[:, 0]
    text_emb = lm.text_emb(text_token_ids)  # [T, D]

    num_audio_codebooks = int(min(len(lm.emb), int(input_token_ids.shape[1]) - 1))
    if num_audio_codebooks <= 0:
        raise RuntimeError("No audio codebook channels available for embed stats")

    audio_emb_per_codebook: list[torch.Tensor] = []
    for cb_idx in range(num_audio_codebooks):
        cb_token_ids = input_token_ids[:, 1 + cb_idx]
        audio_emb_per_codebook.append(lm.emb[cb_idx](cb_token_ids))
    avg_audio_emb = torch.stack(audio_emb_per_codebook, dim=1).mean(dim=1)  # [T, D]

    text_norm = torch.linalg.vector_norm(text_emb, ord=2, dim=-1)
    avg_audio_norm = torch.linalg.vector_norm(avg_audio_emb, ord=2, dim=-1)
    cos_sim = torch.nn.functional.cosine_similarity(text_emb, avg_audio_emb, dim=-1, eps=1e-8)

    return {
        "num_steps": int(input_token_ids.shape[0]),
        "num_audio_codebooks_used": num_audio_codebooks,
        "text_embedding_norm": _summarize_scalar_statistics(text_norm),
        "audio_embedding_norm_avg_codebooks": _summarize_scalar_statistics(avg_audio_norm),
        "cosine_similarity_text_vs_avg_audio": _summarize_scalar_statistics(cos_sim),
    }


def _load_steering_vectors(steering_vectors_path: str) -> list[Optional[torch.Tensor]]:
    """Load per-token steering vectors from disk.

    Expected semantics:
    - `len(steering_vectors)` is number of token steps (~audio_seconds * 12.5Hz).
    - element is `None` => no steering at that token.
    - element tensor => 1D steering vector for that token.

    Supported file contents:
    - `.npy`:
      - shape `[T, D]` => list of `T` vectors.
      - shape `[D]` => single-token list.
    - `.pt` / `.pth` containing:
      - tensor `[T, D]` or `[D]`
      - list/tuple of tensors / `None`
      - dict key `steering_vectors` (preferred), `steering_vector`, or `vector`
    """
    if not os.path.exists(steering_vectors_path):
        raise FileNotFoundError(f"Steering vectors file not found: {steering_vectors_path}")

    suffix = Path(steering_vectors_path).suffix.lower()
    if suffix == ".npy":
        loaded: Any = torch.from_numpy(np.load(steering_vectors_path))
    else:
        loaded = torch.load(steering_vectors_path, map_location="cpu")

    if isinstance(loaded, dict):
        for key in ("steering_vectors", "steering_vector", "vector"):
            if key in loaded:
                loaded = loaded[key]
                break
        else:
            raise ValueError(
                "Steering .pt dict must contain one of: 'steering_vectors', 'steering_vector', 'vector'."
            )

    out: list[Optional[torch.Tensor]] = []

    if isinstance(loaded, torch.Tensor):
        t = loaded.detach().float()
        if t.dim() == 1:
            out = [t.reshape(-1)]
        elif t.dim() == 2:
            out = [t[i].reshape(-1) for i in range(t.shape[0])]
        else:
            raise ValueError(f"Tensor steering vectors must be 1D or 2D, got shape {tuple(t.shape)}")
    elif isinstance(loaded, (list, tuple)):
        for idx, elem in enumerate(loaded):
            if elem is None:
                out.append(None)
                continue
            if isinstance(elem, torch.Tensor):
                vec = elem.detach().float().reshape(-1)
            else:
                try:
                    vec = torch.as_tensor(elem, dtype=torch.float32).reshape(-1)
                except Exception as exc:
                    raise ValueError(f"Invalid steering vector at index {idx}: {type(elem)}") from exc
            if vec.numel() == 0:
                raise ValueError(f"Steering vector at index {idx} is empty")
            out.append(vec)
    else:
        raise ValueError(f"Unsupported steering vectors content type: {type(loaded)}")

    if len(out) == 0:
        raise ValueError("Loaded steering_vectors is empty")

    return out


def warmup(mimi: MimiModel, other_mimi: MimiModel, lm_gen: LMGen, device: str, frame_size: int):
    """Run a short warmup loop to initialize CUDA graphs and streaming state.

    Replicates the same warmup behavior as server.py: zeros → encode → LMGen.step → decode.
    """
    for _ in range(4):
        chunk = torch.zeros(1, 1, frame_size, dtype=torch.float32, device=device)
        codes = mimi.encode(chunk)
        _ = other_mimi.encode(chunk)
        for c in range(codes.shape[-1]):
            tokens = lm_gen.step(codes[:, :, c : c + 1])
            if tokens is None:
                continue
            # Decode agent audio channels to ensure decode graphs/states are primed
            _ = mimi.decode(tokens[:, 1:9])
            _ = other_mimi.decode(tokens[:, 1:9])
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def decode_tokens_to_pcm(mimi: MimiModel, other_mimi: MimiModel, lm_gen: LMGen, tokens: torch.Tensor) -> np.ndarray:
    """Decode a single step of model tokens to PCM using Mimi.

    tokens is shaped [B, dep_q+1, 1]; channels 1..dep_q are the agent audio codebooks.
    Returns a 1D float32 numpy array (mono) for the current frame.
    """
    with torch.no_grad():
        pcm = mimi.decode(tokens[:, 1:9])
        _ = other_mimi.decode(tokens[:, 1:9])
        pcm_np = pcm.detach().cpu().numpy()[0, 0]
        del pcm  # Explicitly free GPU memory
    return pcm_np


def _get_voice_prompt_dir(voice_prompt_dir: Optional[str], hf_repo: str) -> Optional[str]:
    """
    If voice_prompt_dir is None:
      - download voices.tgz from HF
      - extract it once
      - return extracted directory
    If voice_prompt_dir is provided:
      - just return it
    """
    if voice_prompt_dir is not None:
        return voice_prompt_dir

    log("info", "retrieving voice prompts")
    voices_tgz = hf_hub_download(hf_repo, "voices.tgz")
    voices_tgz = Path(voices_tgz)
    voices_dir = voices_tgz.parent / "voices"

    if not voices_dir.exists():
        log("info", f"extracting {voices_tgz} to {voices_dir}")
        with tarfile.open(voices_tgz, "r:gz") as tar:
            tar.extractall(path=voices_tgz.parent)

    if not voices_dir.exists():
        raise RuntimeError("voices.tgz did not contain a 'voices/' directory")

    return str(voices_dir)


def run_inference(
    input_wav: str,
    output_wav: str,
    output_text: str,
    text_prompt: str,
    voice_prompt_path: str,
    tokenizer_path: Optional[str],
    moshi_weight: Optional[str],
    mimi_weight: Optional[str],
    hf_repo: str,
    device: str,
    seed: Optional[int],
    temp_audio: float,
    temp_text: float,
    topk_audio: int,
    topk_text: int,
    greedy: bool,
    save_voice_prompt_embeddings: bool,
    cpu_offload: bool = False,
    return_hidden_layers: bool = False,
) -> Optional[HiddenLayerOutputs]:
    """Run offline inference using an input WAV as the user-side stream.

    - Loads/initializes models and tokenizer
    - Warms up execution
    - Loads system text tokens and voice prompt
    - Runs prompt phases (text + voice + silences) via LMGen.step_system_prompts
    - Streams the user WAV frames into the input channels and samples model outputs
    - Decodes and writes an output WAV of the same duration
    """
    if seed is not None and seed != -1:
        seed_all(seed)

    # Download config.json to increment download counter
    # No worries about double-counting since config.json will be cached the second time
    hf_hub_download(hf_repo, "config.json")

    # 1) Load Mimi encoders/decoders (same as server.py)
    log("info", "loading mimi")
    if mimi_weight is None:
        mimi_weight = hf_hub_download(hf_repo, loaders.MIMI_NAME)  # type: ignore
    mimi = loaders.get_mimi(mimi_weight, device)
    other_mimi = loaders.get_mimi(mimi_weight, device)
    log("info", "mimi loaded")

    # 2) Load tokenizer
    if tokenizer_path is None:
        tokenizer_path = hf_hub_download(hf_repo, loaders.TEXT_TOKENIZER_NAME)  # type: ignore
    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)  # type: ignore

    # 3) Load Moshi LM and eval mode
    log("info", "loading moshi")
    if moshi_weight is None:
        moshi_weight = hf_hub_download(hf_repo, loaders.MOSHI_NAME)  # type: ignore
    lm = loaders.get_moshi_lm(moshi_weight, device=device, cpu_offload=cpu_offload)
    lm.eval()
    log("info", "moshi loaded")

    # 4) Construct LMGen like server.py's ServerState does
    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    lm_gen = LMGen(
        lm,
        audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),  # spacer after prompts
        sample_rate=mimi.sample_rate,
        device=device,
        frame_rate=mimi.frame_rate,
        save_voice_prompt_embeddings=save_voice_prompt_embeddings,
        use_sampling=not greedy,
        temp=temp_audio,
        temp_text=temp_text,
        top_k=topk_audio,
        top_k_text=topk_text,
    )
    # Keep models in streaming mode similar to the server
    mimi.streaming_forever(1)
    other_mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)

    # 5) Warmup
    log("info", "warming up the model")
    warmup(mimi, other_mimi, lm_gen, device, frame_size)

    # 6) Prompt configuration (text + voice)
    # System text tokens (k=0) and agent voice-prompt audio (k=1..dep_q) are forced
    if voice_prompt_path.endswith('.pt'):
        # Load pre-saved voice prompt embeddings
        lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
    else:
        lm_gen.load_voice_prompt(voice_prompt_path)
    lm_gen.text_prompt_tokens = (
        text_tokenizer.encode(wrap_with_system_tags(text_prompt)) if len(text_prompt) > 0 else None
    )

    # 7) Reset streaming and run initial prompt phases
    #    - Voice prompt injection
    #    - Audio silence
    #    - Text prompt injection
    #    - Final audio silence
    mimi.reset_streaming()
    other_mimi.reset_streaming()
    lm_gen.reset_streaming()
    lm_gen.step_system_prompts(mimi)
    # Reset mimi streaming after voice prompt encoding
    mimi.reset_streaming()

    # 8) Load and iterate user audio frames for feeding into the input channels
    sample_rate = mimi.sample_rate
    user_audio = lm_load_audio(input_wav, sample_rate)  # (C, T) at model SR

    # 9) Encode user audio with Mimi (same iterator logic used for voice prompts),
    #    and step the model one frame at a time, collecting decoded PCM frames
    generated_frames: List[np.ndarray] = []
    generated_text_tokens: List[str] = []
    total_target_samples = user_audio.shape[-1]

    for user_encoded in lm_encode_from_sphn(
        mimi,
        lm_iterate_audio(
            user_audio, sample_interval_size=lm_gen._frame_size, pad=True
        ),
        max_batch=1,
    ):
        # user_encoded: [1, K, T]. Feed one step at a time (usually T==1)
        steps = user_encoded.shape[-1]
        for c in range(steps):
            step_in = user_encoded[:, :, c : c + 1]
            # Feed user-side input channels; text + agent audio are sampled
            # Also return all the hidden layers' values for persona vector analysis
            if return_hidden_layers:
                result = lm_gen.step(step_in, return_hidden_layers=True)
                tokens, hidden_layers = result  # type: ignore
                """
                assert isinstance(hidden_layers, HiddenLayerOutputs)
                log("info", f"Retrieved {len(hidden_layers.text_transformer)} text transformer hidden layers and {len(hidden_layers.depth_transformer)} codebooks, each with {len(hidden_layers.depth_transformer[0])} layers at this step.")
                log("info", f"Text transformer hidden layers shapes: {[h.shape for h in hidden_layers.text_transformer[:3]]}...")  # Show first 3
                for i, codebook_layers in enumerate(hidden_layers.depth_transformer):
                    log("info", f"Codebook {i} hidden layers ({len(codebook_layers)} layers): {[h.shape for h in codebook_layers]}")
                    if i >= 2:  # Limit output to first few codebooks
                        log("info", f"... (showing first 3 codebooks only, total: {len(hidden_layers.depth_transformer)})")
                        break
                """
        
            else:
                tokens = lm_gen.step(step_in)
            if tokens is None:
                continue
            # Decode current sampled agent frame to PCM
            pcm = decode_tokens_to_pcm(mimi, other_mimi, lm_gen, tokens)
            generated_frames.append(pcm)
            # Decode text token
            text_token = tokens[0, 0, 0].item()
            if text_token not in (0, 3):
                _text = text_tokenizer.id_to_piece(text_token)  # type: ignore
                _text = _text.replace("▁", " ")
                log("info", f"text token '{_text}'")
                generated_text_tokens.append(_text)
            else:
                text_token_map = ['EPAD', 'BOS', 'EOS', 'PAD']
                log("info", f"text token '{text_token_map[text_token]}'")
                generated_text_tokens.append(text_token_map[text_token])

    if len(generated_frames) == 0:
        log("error", "No audio frames were generated. Check input file and configuration.")
        return

    # 10) Concatenate frames and trim/pad to match input duration
    output_pcm = np.concatenate(generated_frames, axis=-1)
    if output_pcm.shape[-1] > total_target_samples:
        output_pcm = output_pcm[:total_target_samples]
    elif output_pcm.shape[-1] < total_target_samples:
        pad_len = total_target_samples - output_pcm.shape[-1]
        output_pcm = np.concatenate(
            [output_pcm, np.zeros(pad_len, dtype=output_pcm.dtype)], axis=-1
        )

    # 11) Write mono WAV at model sample rate
    sphn.write_wav(output_wav, output_pcm, sample_rate)
    log("info", f"Wrote output audio to {output_wav}")

    # 12) Write text tokens
    with open(output_text, "w") as file:
        json.dump(generated_text_tokens, file, ensure_ascii=False)
    log("info", f"Wrote output text to {output_text}")

    if return_hidden_layers:
        log("info", "Hidden layers were returned during inference.")
        return hidden_layers


def run_batch_inference(
    input_wavs: List[str],
    output_wavs: List[str],
    output_texts: List[str],
    text_prompts: List[str],
    voice_prompt_path: str,
    tokenizer_path: Optional[str],
    moshi_weight: Optional[str],
    mimi_weight: Optional[str],
    hf_repo: str,
    device: str,
    seed: Optional[int],
    temp_audio: float,
    temp_text: float,
    topk_audio: int,
    topk_text: int,
    greedy: bool,
    save_voice_prompt_embeddings: bool,
    cpu_offload: bool = False,
    return_hidden_layers: bool = False,
    save_hidden_payload: bool = False,
    output_hiddens: Optional[List[str]] = None,
    steering_vectors: Optional[list[Optional[torch.Tensor]]] = None,
    steering_layer: Optional[int] = None,
    steering_vectors_by_layer: Optional[dict[int, list[Optional[torch.Tensor]]]] = None,
    steer_attn_only: bool = False,
    payload_target_layer: Optional[int] = None,
    embed_stat: bool = False,
    force_pad_start_steps: Optional[List[Optional[int]]] = None,
    force_pad_num_steps: int = 0,
    attention_suppression_configs: Optional[List[Optional[dict[str, Any]]]] = None,
    attention_suppression_stats_path: Optional[str] = None,
) -> Optional[List[List[HiddenLayerOutputs]]]:
    """Run batch offline inference using multiple input WAVs and text prompts.
    
    Args:
        input_wavs: List of paths to input WAV files (user audio)
        output_wavs: List of paths to output WAV files to write (agent audio)
        output_texts: List of paths to output JSON files to write (agent text)
        text_prompts: List of text prompts corresponding to each input
        voice_prompt_path: Path to voice prompt file (shared across all instances)
        Other parameters: Same as run_inference
        
    Returns:
        If return_hidden_layers=True, returns list of HiddenLayerOutputs for each input
        Otherwise returns None
    """

    # Validate input lengths
    assert len(input_wavs) == len(text_prompts), f"input_wavs ({len(input_wavs)}) and text_prompts ({len(text_prompts)}) must have same length"
    assert len(input_wavs) == len(output_wavs), f"input_wavs ({len(input_wavs)}) and output_wavs ({len(output_wavs)}) must have same length"
    assert len(input_wavs) == len(output_texts), f"input_wavs ({len(input_wavs)}) and output_texts ({len(output_texts)}) must have same length"
    if output_hiddens is not None:
        assert len(input_wavs) == len(output_hiddens), (
            f"input_wavs ({len(input_wavs)}) and output_hiddens ({len(output_hiddens)}) must have same length"
        )
    
    if len(input_wavs) == 0:
        log("warning", "Empty input lists provided")
        return [] if return_hidden_layers else None

    if force_pad_start_steps is not None and len(force_pad_start_steps) != len(input_wavs):
        raise ValueError(
            "force_pad_start_steps must have the same length as input_wavs when provided"
        )
    if attention_suppression_configs is not None and len(attention_suppression_configs) != len(input_wavs):
        raise ValueError(
            "attention_suppression_configs must have the same length as input_wavs when provided"
        )
    if int(force_pad_num_steps) < 0:
        raise ValueError(f"force_pad_num_steps must be >= 0, got {force_pad_num_steps}")
    if attention_suppression_configs is not None:
        for cfg in attention_suppression_configs:
            if cfg is not None and bool(cfg.get("enabled", False)):
                _validate_lambda(float(cfg.get("lambda_suppression", 0.2)))

    has_single_steer = steering_vectors is not None
    has_multi_steer = steering_vectors_by_layer is not None and len(steering_vectors_by_layer) > 0
    if has_single_steer and has_multi_steer:
        raise ValueError("Provide either steering_vectors/steering_layer or steering_vectors_by_layer, not both")
    if has_single_steer and steering_layer is None:
        raise ValueError("steering_vectors provided but steering_layer is None")
    
    log("info", f"Starting batch inference with {len(input_wavs)} instances")
    
    if seed is not None and seed != -1:
        seed_all(seed)

    # Download config.json to increment download counter
    hf_hub_download(hf_repo, "config.json")

    # 1) Load Mimi encoders/decoders (shared across all instances)
    log("info", "loading mimi")
    if mimi_weight is None:
        mimi_weight = hf_hub_download(hf_repo, loaders.MIMI_NAME)  # type: ignore
    mimi = loaders.get_mimi(mimi_weight, device)
    other_mimi = loaders.get_mimi(mimi_weight, device)
    log("info", "mimi loaded")

    # 2) Load tokenizer (shared across all instances)
    if tokenizer_path is None:
        tokenizer_path = hf_hub_download(hf_repo, loaders.TEXT_TOKENIZER_NAME)  # type: ignore
    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)  # type: ignore

    # 3) Load Moshi LM and eval mode (shared across all instances)
    log("info", "loading moshi")
    if moshi_weight is None:
        moshi_weight = hf_hub_download(hf_repo, loaders.MOSHI_NAME)  # type: ignore
    lm = loaders.get_moshi_lm(moshi_weight, device=device, cpu_offload=cpu_offload)
    lm.eval()
    log("info", "moshi loaded")
    text_pad_token_id = int(lm.text_padding_token_id)
    if force_pad_start_steps is not None and int(force_pad_num_steps) > 0:
        log(
            "info",
            f"force-pad enabled: text_pad_token_id={text_pad_token_id}, "
            f"audio_silence_tokens={SILENCE_TOKENS.tolist()}, span={int(force_pad_num_steps)}",
        )
    if attention_suppression_configs is not None:
        enabled_count = sum(1 for cfg in attention_suppression_configs if cfg is not None and bool(cfg.get("enabled", False)))
        log("info", f"attention suppression enabled for {enabled_count}/{len(input_wavs)} instances")

    # 4) Construct LMGen (shared across all instances)
    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    lm_gen = LMGen(
        lm,
        audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
        sample_rate=mimi.sample_rate,
        device=device,
        frame_rate=mimi.frame_rate,
        save_voice_prompt_embeddings=save_voice_prompt_embeddings,
        use_sampling=not greedy,
        temp=temp_audio,
        temp_text=temp_text,
        top_k=topk_audio,
        top_k_text=topk_text,
    )
    
    # Keep models in streaming mode
    mimi.streaming_forever(1)
    other_mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)

    # 5) Warmup (once for all instances)
    log("info", "warming up the model")
    warmup(mimi, other_mimi, lm_gen, device, frame_size)

    # 6) Load voice prompt (shared across all instances)
    if voice_prompt_path.endswith('.pt'):
        lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
    else:
        lm_gen.load_voice_prompt(voice_prompt_path)

    batch_hidden_layers: List[List[HiddenLayerOutputs]] = []
    capture_hidden = return_hidden_layers or save_hidden_payload
    
    if payload_target_layer is not None:
        target_layer_idx = int(payload_target_layer)
    elif steering_layer is not None:
        target_layer_idx = int(steering_layer)
    elif has_multi_steer:
        assert steering_vectors_by_layer is not None
        target_layer_idx = int(sorted(steering_vectors_by_layer.keys())[0])
    else:
        target_layer_idx = len(lm.transformer.layers) - 1
    if target_layer_idx < 0 or target_layer_idx >= len(lm.transformer.layers):
        raise ValueError(
            f"target_layer_idx out of range: {target_layer_idx}, valid [0, {len(lm.transformer.layers)-1}]"
        )

    # 7) Process each instance
    for i, (input_wav, output_wav, output_text, text_prompt) in tqdm(enumerate(zip(input_wavs, output_wavs, output_texts, text_prompts))):
        log("info", f"Processing instance {i+1}/{len(input_wavs)}: {input_wav}")
        
        # Set text prompt for this instance
        lm_gen.text_prompt_tokens = (
            text_tokenizer.encode(wrap_with_system_tags(text_prompt)) if len(text_prompt) > 0 else None
        )

        # Reset streaming state for this instance
        mimi.reset_streaming()
        other_mimi.reset_streaming()
        lm_gen.reset_streaming()
        lm_gen.step_system_prompts(mimi)
        mimi.reset_streaming()
        prompt_offset = _text_transformer_offset_cpu(lm)
        step_attention_suppression: Optional[dict[str, Any]] = None
        attention_suppression_stats: Optional[dict[str, Any]] = None
        if attention_suppression_configs is not None:
            raw_suppression = attention_suppression_configs[i]
            if raw_suppression is not None and bool(raw_suppression.get("enabled", False)):
                attention_suppression_stats = {
                    "example_id": raw_suppression.get("example_id", Path(input_wav).parent.name),
                    "input_wav": input_wav,
                    "interrupt_timestep": int(raw_suppression["interrupt_timestep"]) + int(prompt_offset),
                    "interrupt_timestep_input_relative": int(raw_suppression["interrupt_timestep"]),
                    "prompt_offset": int(prompt_offset),
                    "k_post_interrupt": int(raw_suppression.get("k_post_interrupt", 10)),
                    "n_pre_interrupt": int(raw_suppression.get("n_pre_interrupt", 30)),
                    "lambda_suppression": float(raw_suppression.get("lambda_suppression", 0.2)),
                    "layers": raw_suppression.get("layers", [23]),
                    "suppressed_key_range_abs": [
                        max(0, int(raw_suppression["interrupt_timestep"]) + int(prompt_offset) - int(raw_suppression.get("n_pre_interrupt", 30))),
                        int(raw_suppression["interrupt_timestep"]) + int(prompt_offset),
                    ],
                }
                step_attention_suppression = dict(raw_suppression)
                step_attention_suppression["interrupt_timestep"] = int(raw_suppression["interrupt_timestep"]) + int(prompt_offset)
                step_attention_suppression["stats"] = attention_suppression_stats

        # Load and process user audio for this instance
        sample_rate = mimi.sample_rate
        user_audio = lm_load_audio(input_wav, sample_rate)

        # Process audio frames and collect outputs
        generated_frames: List[np.ndarray] = []
        generated_text_tokens: List[str] = []
        generated_text_token_ids: List[int] = []
        total_target_samples = user_audio.shape[-1]

        # Token-rate sanity check for steering vectors.
        if has_single_steer:
            expected_tokens = int(round((total_target_samples / float(sample_rate)) * float(mimi.frame_rate)))
            actual_tokens = len(steering_vectors)
            if abs(actual_tokens - expected_tokens) > 2:
                raise AssertionError(
                    f"steering_vectors length mismatch: len={actual_tokens}, expected~{expected_tokens} "
                    f"(audio_seconds={total_target_samples / float(sample_rate):.3f}, frame_rate={float(mimi.frame_rate):.3f})"
                )

            hidden_dim = int(lm.dim)
            for idx, sv in enumerate(steering_vectors):
                if sv is None:
                    continue
                if int(sv.numel()) != hidden_dim:
                    raise AssertionError(
                        f"steering_vectors[{idx}] dim mismatch: got {int(sv.numel())}, expected {hidden_dim}"
                    )
        elif has_multi_steer:
            assert steering_vectors_by_layer is not None
            expected_tokens = int(round((total_target_samples / float(sample_rate)) * float(mimi.frame_rate)))
            hidden_dim = int(lm.dim)
            for layer_idx, layer_vectors in steering_vectors_by_layer.items():
                actual_tokens = len(layer_vectors)
                if abs(actual_tokens - expected_tokens) > 2:
                    raise AssertionError(
                        f"steering_vectors_by_layer[{layer_idx}] length mismatch: len={actual_tokens}, expected~{expected_tokens} "
                        f"(audio_seconds={total_target_samples / float(sample_rate):.3f}, frame_rate={float(mimi.frame_rate):.3f})"
                    )
                for idx, sv in enumerate(layer_vectors):
                    if sv is None:
                        continue
                    if int(sv.numel()) != hidden_dim:
                        raise AssertionError(
                            f"steering_vectors_by_layer[{layer_idx}][{idx}] dim mismatch: got {int(sv.numel())}, expected {hidden_dim}"
                        )
        
        hidden_layers_list: List[HiddenLayerOutputs] = []
        text_hidden_layers_per_token: list[torch.Tensor] = []
        text_pre_unembed_states_per_token: list[torch.Tensor] = []
        full_input_embeddings_per_token: list[torch.Tensor] = []
        input_token_ids_per_token: list[torch.Tensor] = []
        output_token_ids_per_token: list[torch.Tensor] = []
        text_attention_layers_per_token: list[Optional[torch.Tensor]] = []
        need_step_input_tokens = save_hidden_payload or embed_stat
        steer_idx = 0
        force_start = None
        if force_pad_start_steps is not None:
            force_start = force_pad_start_steps[i]
            if force_start is not None:
                force_start = int(force_start)
                if force_start < 0:
                    force_start = None
        force_span = int(force_pad_num_steps)
        step_idx = 0
        for user_encoded in lm_encode_from_sphn(
            mimi,
            lm_iterate_audio(
                user_audio, sample_interval_size=lm_gen._frame_size, pad=True
            ),
            max_batch=1,
        ):
            steps = user_encoded.shape[-1]
            # Store hidden layers for each step
            for c in range(steps):
                step_in = user_encoded[:, :, c : c + 1]
                current_step_idx = step_idx
                step_idx += 1
                force_this_step = (
                    force_start is not None
                    and force_span > 0
                    and current_step_idx >= force_start
                    and current_step_idx < (force_start + force_span)
                )
                forced_moshi_tokens: Optional[torch.Tensor] = None
                forced_text_token: Optional[torch.Tensor] = None
                if force_this_step:
                    forced_moshi_tokens = torch.as_tensor(
                        SILENCE_TOKENS,
                        device=step_in.device,
                        dtype=step_in.dtype,
                    ).reshape(1, -1, 1)
                    forced_text_token = torch.full(
                        (step_in.shape[0],),
                        text_pad_token_id,
                        dtype=step_in.dtype,
                        device=step_in.device,
                    )
                step_steering_vector: Optional[torch.Tensor] = None
                step_steering_vectors_by_layer: Optional[dict[int, torch.Tensor]] = None
                if has_single_steer:
                    if steer_idx >= len(steering_vectors):
                        raise AssertionError(
                            f"Steering index out of range at step {steer_idx} with len={len(steering_vectors)}"
                        )
                    step_steering_vector = steering_vectors[steer_idx]
                elif has_multi_steer:
                    assert steering_vectors_by_layer is not None
                    step_steering_vectors_by_layer = {}
                    for layer_idx, layer_vectors in steering_vectors_by_layer.items():
                        if steer_idx >= len(layer_vectors):
                            raise AssertionError(
                                f"Steering index out of range at step {steer_idx} with len={len(layer_vectors)} for layer {layer_idx}"
                            )
                        layer_step_vec = layer_vectors[steer_idx]
                        if layer_step_vec is not None:
                            step_steering_vectors_by_layer[int(layer_idx)] = layer_step_vec
                if has_single_steer or has_multi_steer:
                    steer_idx += 1
                
                if capture_hidden:
                    step_kwargs: dict[str, Any] = {}
                    if forced_moshi_tokens is not None:
                        step_kwargs["moshi_tokens"] = forced_moshi_tokens
                    if forced_text_token is not None:
                        step_kwargs["text_token"] = forced_text_token
                    result = lm_gen.step(
                        step_in,
                        return_embeddings=save_hidden_payload,
                        return_hidden_layers=True,
                        return_attention_weights=save_hidden_payload,
                        return_step_input_tokens=need_step_input_tokens,
                        steering_vector=step_steering_vector,
                        steering_layer=steering_layer,
                        steering_vectors_by_layer=step_steering_vectors_by_layer,
                        steer_attn_only=steer_attn_only,
                        attention_suppression=step_attention_suppression,
                        **step_kwargs,
                    )
                    if save_hidden_payload:
                        tokens, step_embeddings, hidden_layers, step_input_tokens = cast(
                            tuple[torch.Tensor, torch.Tensor, HiddenLayerOutputs, torch.Tensor],
                            result,
                        )
                    elif need_step_input_tokens:
                        tokens, hidden_layers, step_input_tokens = cast(
                            tuple[torch.Tensor, HiddenLayerOutputs, torch.Tensor],
                            result,
                        )
                    else:
                        tokens, hidden_layers = cast(
                            tuple[torch.Tensor, HiddenLayerOutputs],
                            result,
                        )
                    assert isinstance(hidden_layers, HiddenLayerOutputs), "Hidden layers were requested but not captured."
                else:
                    step_kwargs: dict[str, Any] = {}
                    if forced_moshi_tokens is not None:
                        step_kwargs["moshi_tokens"] = forced_moshi_tokens
                    if forced_text_token is not None:
                        step_kwargs["text_token"] = forced_text_token
                    result = lm_gen.step(
                        step_in,
                        return_step_input_tokens=need_step_input_tokens,
                        steering_vector=step_steering_vector,
                        steering_layer=steering_layer,
                        steering_vectors_by_layer=step_steering_vectors_by_layer,
                        steer_attn_only=steer_attn_only,
                        attention_suppression=step_attention_suppression,
                        **step_kwargs,
                    )
                    if need_step_input_tokens:
                        tokens, step_input_tokens = cast(
                            tuple[torch.Tensor, torch.Tensor],
                            result,
                        )
                    else:
                        tokens = cast(torch.Tensor, result)
                    hidden_layers = None
                
                if tokens is None:
                    continue

                if capture_hidden:
                    assert hidden_layers is not None
                    if return_hidden_layers:
                        hidden_layers_list.append(hidden_layers)
                    if save_hidden_payload:
                        tokens = cast(torch.Tensor, tokens)
                        text_hidden_layers_per_token.append(_extract_text_hidden_per_layer(hidden_layers))
                        text_pre_unembed_states_per_token.append(_extract_text_pre_unembed_state(hidden_layers))
                        full_input_embeddings_per_token.append(
                            _extract_full_input_embedding_for_step(step_embeddings)
                        )
                        output_token_ids_per_token.append(_extract_step_token_ids(tokens))
                        text_attention_layers_per_token.append(_extract_text_attention_per_layer(hidden_layers))
                if need_step_input_tokens:
                    input_token_ids_per_token.append(_extract_step_token_ids(step_input_tokens))
                    
                # Decode current sampled agent frame to PCM
                pcm = decode_tokens_to_pcm(mimi, other_mimi, lm_gen, tokens)
                generated_frames.append(pcm)
                
                # Decode text token
                text_token = tokens[0, 0, 0].item()
                generated_text_token_ids.append(int(text_token))
                if text_token not in (0, 3):
                    _text = text_tokenizer.id_to_piece(text_token)  # type: ignore
                    _text = _text.replace("▁", " ")
                    generated_text_tokens.append(_text)
                else:
                    text_token_map = ['EPAD', 'BOS', 'EOS', 'PAD']
                    generated_text_tokens.append(text_token_map[text_token])

        if has_single_steer:
            # `lm_iterate_audio(..., pad=True)` can introduce a small boundary mismatch,
            # so we allow a tiny slack instead of requiring exact equality.
            unused = len(steering_vectors) - steer_idx
            if abs(unused) > 2:
                raise AssertionError(
                    f"Steering token consumption mismatch: used={steer_idx}, provided={len(steering_vectors)}, "
                    f"unused={unused}"
                )
        elif has_multi_steer:
            assert steering_vectors_by_layer is not None
            for layer_idx, layer_vectors in steering_vectors_by_layer.items():
                unused = len(layer_vectors) - steer_idx
                if abs(unused) > 2:
                    raise AssertionError(
                        f"Steering token consumption mismatch for layer {layer_idx}: used={steer_idx}, provided={len(layer_vectors)}, "
                        f"unused={unused}"
                    )

        if len(generated_frames) == 0:
            log("error", f"No audio frames were generated for instance {i+1}. Check input file: {input_wav}")
            continue

        # Concatenate frames and trim/pad to match input duration
        output_pcm = np.concatenate(generated_frames, axis=-1)
        if output_pcm.shape[-1] > total_target_samples:
            output_pcm = output_pcm[:total_target_samples]
        elif output_pcm.shape[-1] < total_target_samples:
            pad_len = total_target_samples - output_pcm.shape[-1]
            output_pcm = np.concatenate(
                [output_pcm, np.zeros(pad_len, dtype=output_pcm.dtype)], axis=-1
            )

        # Write outputs for this instance
        sphn.write_wav(output_wav, output_pcm, sample_rate)
        log("info", f"Wrote output audio to {output_wav}")

        with open(output_text, "w") as file:
            json.dump(generated_text_tokens, file, ensure_ascii=False)
        log("info", f"Wrote output text to {output_text}")

        embed_stats_payload: Optional[Dict[str, Any]] = None
        if embed_stat:
            embed_stats_payload = _compute_embed_stats_for_instance(
                lm=lm,
                input_token_ids_per_token=input_token_ids_per_token,
                device=device,
            )
            embed_stats_path = str(Path(output_text).with_suffix(".embed_stat.json"))
            with open(embed_stats_path, "w", encoding="utf-8") as f:
                json.dump(embed_stats_payload, f, ensure_ascii=False, indent=2)
            log("info", f"Wrote embed stats to {embed_stats_path}")
        
        # Store hidden layers for this instance
        if return_hidden_layers:
            assert len(hidden_layers_list) == len(generated_text_tokens), "Mismatch in hidden layers and generated token count"
            batch_hidden_layers.append(hidden_layers_list)
        if save_hidden_payload:
            output_hidden = output_hiddens[i] if output_hiddens is not None else str(Path(output_wav).with_name("output_hidden.pt"))
            Path(output_hidden).parent.mkdir(parents=True, exist_ok=True)

            target_layer = lm.transformer.layers[target_layer_idx]
            text_target_layer_keys, text_key_positions, text_key_cache_meta = _extract_target_layer_text_keys_and_positions(target_layer)
            key_pos_start = int(text_key_positions[0].item()) if text_key_positions.numel() > 0 else -1
            key_pos_end = int(text_key_positions[-1].item()) if text_key_positions.numel() > 0 else -1

            log(
                "info",
                (
                    f"Target layer {target_layer_idx} shapes: "
                    f"text_keys={tuple(text_target_layer_keys.shape)}, "
                    f"text_key_positions={tuple(text_key_positions.shape)} [{key_pos_start}..{key_pos_end}], "
                    f"text_key_cache_meta={text_key_cache_meta}"
                ),
            )

            payload = _build_hidden_payload(
                input_wav=input_wav,
                output_wav=output_wav,
                output_text=output_text,
                frame_rate_hz=float(mimi.frame_rate),
                text_token_ids=generated_text_token_ids,
                text_token_pieces=generated_text_tokens,
                text_hidden_layers_per_token=text_hidden_layers_per_token,
                text_pre_unembed_states_per_token=text_pre_unembed_states_per_token,
                full_input_embeddings_per_token=full_input_embeddings_per_token,
                input_token_ids_per_token=input_token_ids_per_token,
                output_token_ids_per_token=output_token_ids_per_token,
                text_attention_layers_per_token=text_attention_layers_per_token,
                text_target_layer_keys=text_target_layer_keys,
                text_key_positions=text_key_positions,
                text_key_cache_meta=text_key_cache_meta,
            )
            if embed_stats_payload is not None:
                payload["embed_stats"] = embed_stats_payload
            torch.save(payload, output_hidden)
            log("info", f"Wrote hidden payload to {output_hidden}")

        if attention_suppression_stats is not None:
            stats_payload = _json_ready_attention_suppression_stats(attention_suppression_stats)
            stats_out = attention_suppression_stats_path
            if stats_out is None:
                stats_out = str(Path(output_wav).parent.parent / "attention_suppression_stats.jsonl")
            Path(stats_out).parent.mkdir(parents=True, exist_ok=True)
            with open(stats_out, "a", encoding="utf-8") as f:
                f.write(json.dumps(stats_payload, ensure_ascii=False) + "\n")
            log("info", f"Wrote attention suppression stats to {stats_out}")

    for streaming_obj in (mimi, other_mimi, lm_gen):
        try:
            streaming_obj._stop_streaming()
        except Exception:
            pass
    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass

    log("info", f"Batch inference completed for {len(input_wavs)} instances")
    if return_hidden_layers and len(batch_hidden_layers) > 0:
        log("info", f"Hidden layers for {len(batch_hidden_layers)} instances with {len(batch_hidden_layers[0])} steps were returned during batch inference.")
        return batch_hidden_layers


def run_batch_inference_two_phase(
    question_wavs: List[str],
    output_wavs: List[str],
    output_texts: List[str],
    text_prompts: List[str],
    voice_prompt_path: str,
    tokenizer_path: Optional[str],
    moshi_weight: Optional[str],
    mimi_weight: Optional[str],
    hf_repo: str,
    device: str,
    seed: Optional[int],
    temp_audio: float,
    temp_text: float,
    topk_audio: int,
    topk_text: int,
    greedy: bool,
    save_voice_prompt_embeddings: bool,
    cpu_offload: bool = False,
    delay1_seconds: float = 2.5,
    max_response_seconds: float = 30.0,
    sample_rate: int = 24000,
    start_instance_idx: int = 0,
    condition: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> None:
    """Run two-phase batch inference for persona vector extraction.
    
    Phase 1: Feed delay1 + question audio to provide context (no hidden extraction)
    Phase 2: Feed empty audio, wait for response with silence detection, extract hidden layers
    
    The hidden layers are extracted ONLY during phase 2, when the model is responding.
    This is the correct approach for persona vector extraction because we want to capture
    the model's response behavior, not its processing of the question.
    
    Args:
        question_wavs: List of paths to question WAV files (user asks question)
        output_wavs: List of paths to output WAV files to write (agent audio response)
        output_texts: List of paths to output JSON files to write (agent text)
        text_prompts: List of text prompts (instructions from trait JSON files)
        voice_prompt_path: Path to voice prompt file (shared across all instances)
        delay1_seconds: Delay before question audio (default 2.5s)
        max_response_seconds: Maximum time for model to respond before stopping (default 30s)
        sample_rate: Audio sample rate (default 24000 Hz for Moshi)
        start_instance_idx: Index to start from (default 0). Useful for resuming from a checkpoint.
        Other parameters: Same as run_batch_inference
        
    Returns:
        List of List[HiddenLayerOutputs] for each input (only from phase 2)
    """
    import time
    
    # Validate input lengths
    assert len(question_wavs) == len(text_prompts), f"question_wavs ({len(question_wavs)}) and text_prompts ({len(text_prompts)}) must have same length"
    assert len(question_wavs) == len(output_wavs), f"question_wavs ({len(question_wavs)}) and output_wavs ({len(output_wavs)}) must have same length"
    assert len(question_wavs) == len(output_texts), f"question_wavs ({len(question_wavs)}) and output_texts ({len(output_texts)}) must have same length"
    
    if len(question_wavs) == 0:
        log("warning", "Empty input lists provided")
        return
    
    log("info", f"Starting two-phase batch inference with {len(question_wavs)} instances")
    if start_instance_idx > 0:
        log("info", f"Resume mode: starting from instance {start_instance_idx+1}/{len(question_wavs)}")
    log("info", f"Parameters: delay1={delay1_seconds}s, max_response={max_response_seconds}s")
    
    if seed is not None and seed != -1:
        seed_all(seed)

    # Download config.json to increment download counter
    hf_hub_download(hf_repo, "config.json")

    # 1) Load Mimi encoders/decoders (shared across all instances)
    log("info", "loading mimi")
    if mimi_weight is None:
        mimi_weight = hf_hub_download(hf_repo, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device)
    other_mimi = loaders.get_mimi(mimi_weight, device)
    log("info", "mimi loaded")

    # 2) Load tokenizer (shared across all instances)
    if tokenizer_path is None:
        tokenizer_path = hf_hub_download(hf_repo, loaders.TEXT_TOKENIZER_NAME)
    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)

    # 3) Load Moshi LM and eval mode (shared across all instances)
    log("info", "loading moshi")
    if moshi_weight is None:
        moshi_weight = hf_hub_download(hf_repo, loaders.MOSHI_NAME)
    lm = loaders.get_moshi_lm(moshi_weight, device=device, cpu_offload=cpu_offload)
    lm.eval()
    log("info", "moshi loaded")

    # 4) Construct LMGen (shared across all instances)
    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    lm_gen = LMGen(
        lm,
        audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
        sample_rate=mimi.sample_rate,
        device=device,
        frame_rate=mimi.frame_rate,
        save_voice_prompt_embeddings=save_voice_prompt_embeddings,
        use_sampling=not greedy,
        temp=temp_audio,
        temp_text=temp_text,
        top_k=topk_audio,
        top_k_text=topk_text,
    )
    
    # Keep models in streaming mode
    mimi.streaming_forever(1)
    other_mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)

    # 5) Warmup (once for all instances)
    log("info", "warming up the model")
    warmup(mimi, other_mimi, lm_gen, device, frame_size)

    # 6) Load voice prompt (shared across all instances)
    if voice_prompt_path.endswith('.pt'):
        lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
    else:
        lm_gen.load_voice_prompt(voice_prompt_path)

    # Pre-compute delay audio
    delay1_samples = int(delay1_seconds * mimi.sample_rate)
    max_response_frames = int(max_response_seconds * mimi.frame_rate)
    
    # Create silence array for delay1
    delay1_silence = np.zeros(delay1_samples, dtype=np.float32)
    
    # Encode a single silence frame for phase 2
    silence_frame = torch.zeros(1, 1, frame_size, dtype=torch.float32, device=device)

    batch_hidden_layers: List[List[HiddenLayerOutputs]] = []
    
    # Validate that condition and output_dir are provided for checkpoint saving
    if condition and output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        log("warning", "condition and output_dir not provided. Hidden layers will not be saved.")
    
    # 7) Process each instance with two phases
    for i, (question_wav, output_wav, output_text, text_prompt) in tqdm(enumerate(zip(question_wavs, output_wavs, output_texts, text_prompts)), total=len(question_wavs)):
        # Skip instances before start_instance_idx (for resume functionality)
        if i < start_instance_idx:
            log("info", f"Skipping instance {i+1}/{len(question_wavs)} (resume mode)")
            if i >= len(batch_hidden_layers):
                batch_hidden_layers.append([])
            continue
        
        log("info", f"Processing instance {i+1}/{len(question_wavs)}: {question_wav}")
        log_memory(f"Instance {i+1} start")
        instance_start_time = time.time()
        
        # Timestamps for logging
        timestamps = {
            "start": instance_start_time,
            "question_start": None,
            "question_end": None,
            "response_start": None,
            "silence_detected": None,
            "max_response_reached": None,
        }
        
        # Set text prompt for this instance (from trait JSON instruction)
        lm_gen.text_prompt_tokens = (
            text_tokenizer.encode(wrap_with_system_tags(text_prompt)) if len(text_prompt) > 0 else None
        )

        # Reset streaming state for this instance
        mimi._stop_streaming()
        other_mimi._stop_streaming()
        lm_gen._stop_streaming()

        mimi.streaming_forever(1)
        other_mimi.streaming_forever(1)
        lm_gen.streaming_forever(1)

        # Keep models in streaming mode
        mimi.reset_streaming()
        other_mimi.reset_streaming()
        lm_gen.reset_streaming()
        lm_gen.step_system_prompts(mimi)
        mimi.reset_streaming()


        # Load question audio
        question_audio = lm_load_audio(question_wav, mimi.sample_rate)
        
        # Create phase 1 audio: delay1 + question audio
        # lm_load_audio returns numpy array, so no need to call .numpy()
        if isinstance(question_audio, torch.Tensor):
            question_audio_np = question_audio.numpy().flatten()
        else:
            question_audio_np = question_audio.flatten()
        phase1_audio = np.concatenate([delay1_silence, question_audio_np])
        phase1_audio = torch.from_numpy(phase1_audio).unsqueeze(0)  # Shape: [1, T]

        # Process audio frames and collect outputs
        generated_frames: List[np.ndarray] = []
        generated_text_tokens: List[str] = []
        
        total_steps = 0
        phase1_steps = 0
        phase2_steps = 0
        hidden_layers_list: List[HiddenLayerOutputs] = []
        
        # ============ PHASE 1: Feed question audio to provide context ============
        timestamps["question_start"] = time.time()
        log("info", f"  Phase 1: Feeding question audio ({phase1_audio.shape[-1] / mimi.sample_rate:.2f}s)")
        
        for user_encoded in lm_encode_from_sphn(
            mimi,
            lm_iterate_audio(
                phase1_audio, sample_interval_size=lm_gen._frame_size, pad=True
            ),
            max_batch=1,
        ):
            steps = user_encoded.shape[-1]
            for c in range(steps):
                total_steps += 1
                phase1_steps += 1
                step_in = user_encoded[:, :, c : c + 1]
                
                # Phase 1: No hidden layer extraction, just feed context
                tokens = lm_gen.step(step_in)
                
                if tokens is None:
                    continue
                    
                # Decode current sampled agent frame to PCM
                pcm = decode_tokens_to_pcm(mimi, other_mimi, lm_gen, tokens)
                generated_frames.append(pcm)
                
                # Decode text token
                text_token = tokens[0, 0, 0].item()
                if text_token not in (0, 3):
                    _text = text_tokenizer.id_to_piece(text_token)
                    _text = _text.replace("▁", " ")
                    generated_text_tokens.append(_text)
                else:
                    text_token_map = ['EPAD', 'BOS', 'EOS', 'PAD']
                    generated_text_tokens.append(text_token_map[text_token])
        
        timestamps["question_end"] = time.time()
        log("info", f"  Phase 1 completed: {phase1_steps} steps ({timestamps['question_end'] - timestamps['question_start']:.2f}s)")
        log_memory(f"Instance {i+1} after phase 1")
        
        # ============ PHASE 2: Feed silence, wait for response, extract hidden layers ============
        
        timestamps["response_start"] = time.time()
        log("info", f"  Phase 2: Waiting for model response (max {max_response_seconds}s, silence detection enabled)")
        
        silence_detected = False
        consecutive_silence_count = 0
        SILENCE_THRESHOLD = 3  # Number of consecutive silence tokens to confirm end of response

        for frame_idx in range(max_response_frames):
            total_steps += 1
            phase2_steps += 1
            
            # Encode silence frame
            silence_encoded = mimi.encode(silence_frame)
            
            for c in range(silence_encoded.shape[-1]):
                step_in = silence_encoded[:, :, c : c + 1]
                
                # Phase 2: Extract hidden layers and check for silence
                result = lm_gen.step(step_in, return_hidden_layers=True, check_silence_token=True)
                tokens, hidden_layers, is_silence = result
                
                if tokens is None:
                    continue
                
                # Store hidden layers on CPU (move immediately to free GPU memory)
                if hidden_layers is not None:
                    # Move each tensor in hidden_layers to CPU to free GPU VRAM
                    cpu_hidden_layers = HiddenLayerOutputs(
                        text_transformer=[t.detach().cpu() for t in hidden_layers.text_transformer] if hidden_layers.text_transformer else None,
                        depth_transformer=[[t.detach().cpu() for t in layer_list] for layer_list in hidden_layers.depth_transformer] if hidden_layers.depth_transformer else None
                    )
                    hidden_layers_list.append(cpu_hidden_layers)
                    del hidden_layers  # Free GPU reference
                
                # Check for silence token
                if is_silence:
                    consecutive_silence_count += 1
                    if consecutive_silence_count >= SILENCE_THRESHOLD:
                        silence_detected = True
                        timestamps["silence_detected"] = time.time()
                        log("info", f"  Silence token detected at step {total_steps} (phase2 step {phase2_steps})")
                        break
                else:
                    consecutive_silence_count = 0
                
                # Decode current sampled agent frame to PCM
                pcm = decode_tokens_to_pcm(mimi, other_mimi, lm_gen, tokens)
                generated_frames.append(pcm)
                
                # Decode text token
                text_token = tokens[0, 0, 0].item()
                if text_token not in (0, 3):
                    _text = text_tokenizer.id_to_piece(text_token)
                    _text = _text.replace("▁", " ")
                    generated_text_tokens.append(_text)
                else:
                    text_token_map = ['EPAD', 'BOS', 'EOS', 'PAD']
                    generated_text_tokens.append(text_token_map[text_token])
                
                # Immediately free token tensor to reduce GPU memory fragmentation
                del tokens
            
            if silence_detected:
                break
            
            # Clean up silence_encoded to free GPU memory
            del silence_encoded
        
        if not silence_detected:
            timestamps["max_response_reached"] = time.time()
            log("info", f"  Max response time reached at step {total_steps}")
        
        log("info", f"  Phase 2 completed: {phase2_steps} steps, collected {len(hidden_layers_list)} hidden layer outputs")
        
        # Log timing summary
        total_time = time.time() - timestamps["start"]
        log("info", f"  Timing summary for instance {i+1}:")
        log("info", f"    Total steps: {total_steps} (phase1: {phase1_steps}, phase2: {phase2_steps})")
        log("info", f"    Total time: {total_time:.2f}s")
        if timestamps["silence_detected"]:
            log("info", f"    Response ended by silence detection at {timestamps['silence_detected'] - timestamps['response_start']:.2f}s into phase 2")

        if len(generated_frames) == 0:
            log("error", f"No audio frames were generated for instance {i+1}. Check input file: {question_wav}")
            continue

        # Concatenate frames
        output_pcm = np.concatenate(generated_frames, axis=-1)

        # Write outputs for this instance
        sphn.write_wav(output_wav, output_pcm, mimi.sample_rate)
        log("info", f"Wrote output audio to {output_wav}")

        with open(output_text, "w") as file:
            json.dump(generated_text_tokens, file, ensure_ascii=False)
        log("info", f"Wrote output text to {output_text}")
        
        # Store hidden layers for this instance (only phase 2 hidden layers)
        # Update or append to batch_hidden_layers to support resume mode with overwrites
        # Save checkpoint file for this instance (averaged across steps)
        if condition and output_dir:
            checkpoint_file = os.path.join(output_dir, f"{condition}_{i:05d}.pt")
            try:
                if len(hidden_layers_list) > 0:
                    # Average hidden layers across steps before saving
                    avg_hidden = average_hidden_layers(hidden_layers_list)
                    torch.save(avg_hidden, checkpoint_file)
                    log("info", f"Saved checkpoint to {checkpoint_file} (averaged {len(hidden_layers_list)} steps)")
                else:
                    log("warning", f"No hidden layers collected for instance {i+1}, skipping checkpoint")
            except Exception as e:
                log("error", f"Failed to save checkpoint to {checkpoint_file}: {e}")

    log("info", f"Two-phase batch inference completed for {len(question_wavs)} instances")
    if len(batch_hidden_layers) > 0 and len(batch_hidden_layers[0]) > 0:
        log("info", f"Hidden layers collected: {len(batch_hidden_layers)} instances, first has {len(batch_hidden_layers[0])} steps")


def main():
    """Parse CLI args and run offline inference."""
    parser = argparse.ArgumentParser(
        description="Offline inference from WAV input using Moshi server components."
    )
    parser.add_argument(
        "--input-wav", required=True, type=str, help="Path to input WAV file (user audio)"
    )
    parser.add_argument(
        "--output-wav", required=True, type=str, help="Path to output WAV file of agent audio to write"
    )
    parser.add_argument(
        "--output-text", required=True, type=str, help="Path to output JSON file of agent text to write"
    )
    parser.add_argument("--text-prompt", default="You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.", type=str, help="Text prompt")

    parser.add_argument(
        "--voice-prompt", required=True, type=str, help="Voice prompt filename (basename) inside --voice-prompt-dir (e.g. 'NATM1.pt')."
    )
    parser.add_argument(
        "--voice-prompt-dir",
        type=str,
        help=(
            "Directory containing voice prompt files. "
            "If omitted, voices.tgz is downloaded from HF and extracted."
            "Voice prompt filenames from -voice-prompt arg will be joined with this directory path."
        )
    )

    # Model assets
    parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file.")
    parser.add_argument("--moshi-weight", type=str, help="Path to a local checkpoint file for Moshi.")
    parser.add_argument("--mimi-weight", type=str, help="Path to a local checkpoint file for Mimi.")
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=loaders.DEFAULT_REPO,
        help="HF repo to look into (defaults to pre-trained model repo)",
    )

    # Runtime / sampling controls (mirror UI semantics)
    parser.add_argument(
        "--temp-audio", type=float, default=0.8, help="Audio sampling temperature (default: 0.8)"
    )
    parser.add_argument(
        "--temp-text", type=float, default=0.7, help="Text sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--topk-audio", type=int, default=250, help="Audio top-k sampling (default: 250)"
    )
    parser.add_argument(
        "--topk-text", type=int, default=25, help="Text top-k sampling (default: 25)"
    )
    parser.add_argument(
        "--greedy", action="store_true", help="Disable sampling (greedy decoding)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device on which to run, defaults to 'cuda'."
    )
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Offload LM model layers to CPU when GPU memory is insufficient. "
                             "Requires 'accelerate' package.")
    parser.add_argument("--seed", type=int, default=-1, help="Seed for reproducibility (-1 disables)")

    parser.add_argument("--return-hidden-layers", action="store_true",
                        help="If set, the model will return hidden layer activations at each step.")
    parser.add_argument(
        "--steering-vectors",
        type=str,
        default=None,
        help="Path to per-token steering vectors (.npy/.pt). If provided, steering is enabled token-wise.",
    )
    parser.add_argument(
        "--steering-layer",
        type=int,
        default=None,
        help="Main transformer layer index (0-31) where steering is injected. Required with --steering-vectors.",
    )
    parser.add_argument(
        "--payload-target-layer",
        type=int,
        default=None,
        help=(
            "Text transformer layer index used to extract payload fields "
            "(text_keys, text_key_positions, text_key_cache_meta). "
            "If not set, defaults to steering_layer (if provided) else last layer."
        ),
    )
    parser.add_argument(
        "--embed-stat",
        action="store_true",
        help=(
            "If set, compute per-instance embedding statistics over generation steps: "
            "text embedding norm mean, average-audio embedding norm mean (audio codebooks averaged), "
            "and cosine similarity between text embedding and averaged audio embedding; "
            "for each quantity also report uncertainty (std, sem, 95% CI)."
        ),
    )

    args = parser.parse_args()

    # If --voice-prompt-dir is omitted, voices.tgz is downloaded from HF and extracted.
    voice_prompt_dir = _get_voice_prompt_dir(
        args.voice_prompt_dir,
        args.hf_repo,
    )
    if not os.path.exists(voice_prompt_dir):
        raise FileNotFoundError(f"voice_prompt_dir does not exist: {voice_prompt_dir}")
    log("info", f"voice_prompt_dir = {voice_prompt_dir}")

    # Join basename with directory (DO NOT mutate args.voice_prompt)
    voice_prompt_path = os.path.join(voice_prompt_dir, args.voice_prompt)
    if not os.path.exists(voice_prompt_path):
        raise FileNotFoundError(
            f"Voice prompt '{args.voice_prompt}' not found in "
            f"'{voice_prompt_dir}' (resolved: {voice_prompt_path})"
        )

    # Normalize greedy flag behavior (True if present, False otherwise)
    greedy = bool(args.greedy)

    steering_vectors: Optional[list[Optional[torch.Tensor]]] = None
    if args.steering_vectors is not None:
        if args.steering_layer is None:
            parser.error("--steering-vectors requires --steering-layer")
        steering_vectors = _load_steering_vectors(args.steering_vectors)
        num_non_null = sum(1 for v in steering_vectors if v is not None)
        log(
            "info",
            f"Loaded steering_vectors: total={len(steering_vectors)}, non_null={num_non_null} from {args.steering_vectors}",
        )

    with torch.no_grad():
        run_batch_inference(
            input_wavs=[args.input_wav],
            output_wavs=[args.output_wav],
            output_texts=[args.output_text],
            text_prompts=[args.text_prompt],
            voice_prompt_path=voice_prompt_path,
            tokenizer_path=args.tokenizer,
            moshi_weight=args.moshi_weight,
            mimi_weight=args.mimi_weight,
            hf_repo=args.hf_repo,
            device=args.device,
            seed=args.seed,
            temp_audio=args.temp_audio,
            temp_text=args.temp_text,
            topk_audio=args.topk_audio,
            topk_text=args.topk_text,
            greedy=greedy,
            save_voice_prompt_embeddings=False,
            cpu_offload=args.cpu_offload,
            return_hidden_layers=args.return_hidden_layers,
            steering_vectors=steering_vectors,
            steering_layer=args.steering_layer,
            payload_target_layer=args.payload_target_layer,
            embed_stat=args.embed_stat,
        )


if __name__ == "__main__":
    main()
