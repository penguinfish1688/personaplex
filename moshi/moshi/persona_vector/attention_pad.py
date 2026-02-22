from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from .premature_decode import DecodeData, PAD_TOKEN_ID

logger = logging.getLogger(__name__)


def visualize_wav_amplitude(input_wav):
    """
    Visualize the amplitude of the input WAV file over time.
    output to same directory as input_wav with name "input_amplitude.png"
    """
    wav_path = Path(input_wav)
    waveform, sr = sf.read(str(wav_path))
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    waveform = waveform.astype(np.float32)
    times = np.arange(waveform.shape[0], dtype=np.float32) / float(sr)

    fig, ax = plt.subplots(figsize=(12, 3), dpi=150)
    ax.plot(times, waveform, linewidth=0.6, color="#3366cc")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Input waveform amplitude")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    output_path = wav_path.with_name("input_amplitude.png")
    fig.savefig(output_path)
    plt.close(fig)


def user_active_mask(input_wav, frame_rate, threshold):
    """
    Analyze the input WAV file to determine which tokens correspond to user activity.

    Args:
        input_wav (str): Path to the input WAV file.
        frame_rate (int): Frame rate of the moshi token frame (e.g., 12.5 Hz).
        threshold (float): .wav amplitude threshold to consider as user activity.

    Returns:
        List[int]: A list of boolean that correspond to user activity.
        >> mask = user_active_mask("input.wav", frame_rate=12.5, threshold=0.1)
        >> [true, true, false, ...]
        >> where each element corresponds to a token. true means the user is active during this token.
    """
    waveform, sr = sf.read(str(input_wav))
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    waveform = np.asarray(waveform, dtype=np.float32)

    samples_per_token = max(1, int(round(float(sr) / float(frame_rate))))
    total_tokens = int(np.ceil(waveform.shape[0] / samples_per_token))

    mask: list[bool] = []
    for token_idx in range(total_tokens):
        start = token_idx * samples_per_token
        end = min((token_idx + 1) * samples_per_token, waveform.shape[0])
        if end <= start:
            mask.append(False)
            continue
        segment = waveform[start:end]
        is_active = bool(np.max(np.abs(segment)) >= float(threshold))
        mask.append(is_active)
    return mask


def output_pad_mask(decode_data_list: list[DecodeData]):
    """
    If the model output is <PAD>, set true for this token. Otherwise, set false.
    Args:
        decode_data_list (list[DecodeData]): A list of DecodeData, each corresponds to a token. The list is ordered by time.

    Returns:
        List[int]: A list of boolean that correspond to whether the token is <PAD>.
        >> mask = output_pad_mask(decode_data_list)
        >> [true, true, false, ...]
        >> where each element corresponds to a token. true means the model output is <PAD> for this token.
    """
    mask: list[bool] = []
    for decode_data in decode_data_list:
        final_token_id, _, _ = decode_data.get_final_output()
        mask.append(final_token_id == PAD_TOKEN_ID)
    return mask


def _attention_to_layer_time(attention_weights: torch.Tensor) -> torch.Tensor:
    """Convert attention tensor to `[L, K]` by averaging heads when present."""
    if attention_weights.dim() == 3:
        # [L, H, K] -> [L, K]
        return attention_weights.float().mean(dim=1)
    if attention_weights.dim() == 2:
        # already [L, K]
        return attention_weights.float()
    raise ValueError(f"Unsupported attention weight shape: {tuple(attention_weights.shape)}")


def _align_mask_to_k(mask: list[bool], k: int) -> torch.Tensor:
    """Right-align a token mask to attention key length K.

    Attention keys can include pre-roll/system tokens. Right-alignment keeps the most
    recent generated-token mask positions aligned with the most recent key positions.
    """
    if k <= 0:
        return torch.zeros((0,), dtype=torch.bool)
    if len(mask) >= k:
        aligned = mask[-k:]
    else:
        aligned = [False] * (k - len(mask)) + mask
    return torch.tensor(aligned, dtype=torch.bool)


def attention_weight_masked_sum(decode_data: DecodeData, mask: list[bool]):
    """
    Compute the attention-weighted sum of the token embeddings, masking out tokens that do not correspond to user activity.

    Args:
        decode_data (DecodeData): The data structure containing token embeddings and attention weights.
        mask (List[int]): A list of boolean mask. i.e., [true, true, false, ...] where each element corresponds to a token.
    Returns:
        user_activity_aware_attention_sum (float): The attention weight sum of this token after masked.
    """
    if decode_data.attention_weights is None:
        raise ValueError("decode_data.attention_weights is None")

    layer_time = _attention_to_layer_time(decode_data.attention_weights)  # [L, K]
    k = int(layer_time.shape[-1])
    if k == 0:
        return torch.zeros((layer_time.shape[0],), dtype=torch.float32)

    mask_t = _align_mask_to_k(mask, k).to(layer_time.device)
    masked_sum = (layer_time * mask_t.unsqueeze(0).float()).sum(dim=-1)  # [L]
    return masked_sum


def pad_lookback_ratio(
    decode_data_list: list[DecodeData],
    input_wav: str,
    frame_rate: float = 12.5,
    activity_threshold: float = 0.1,
    context_prefix_tokens: Optional[int] = None,
    token_offset: int = 0,
):
    """
    Compute the user-silent pad lookback ratio of each token.

    for each token T:
        for each layer l:
            ratio =
                sum of attention weight on key i, where i is a generated token < T,
                    model output at i is <PAD>, AND user is silent at i
                -------------------------------------------------------------------
                sum of attention weight on all generated tokens < T

    The denominator is the total attention on all lookback generated tokens (i < T),
    so the ratio represents the share of lookback attention going to PAD-and-silent tokens.

    Args:
        decode_data_list (list[DecodeData]): A list of DecodeData, each corresponds to a token.
            The list is ordered by time and may be a SLICE of the full sequence.
        input_wav (str): Path to the user input wav used to estimate user activity over time.
        frame_rate (float): Token frame rate, default 12.5.
        activity_threshold (float): Absolute-amplitude threshold for active speech.
        context_prefix_tokens (Optional[int]): Fixed number of attention key positions to ignore
            as pre-input context. If None, inferred from token-0 attention length as ``K0 - 1``.
            When *decode_data_list* is a slice starting at full-sequence index ``token_offset``,
            the inferred value is ``actual_prefix + token_offset``.
        token_offset (int): The full-sequence index of the first token in *decode_data_list*.
            Needed to align ``user_active_mask`` with the correct audio segment and to correctly
            determine pad status for lookback tokens within the slice.  Default 0 (no slice).

    Returns:
        torch.FloatTensor: Ratio tensor of shape [T, L] where T is token count and L is layer count.
    """
    if len(decode_data_list) == 0:
        return torch.empty((0, 0), dtype=torch.float32)

    if decode_data_list[0].attention_weights is None:
        raise ValueError("attention weights are required to compute pad_lookback_ratio")

    first_layer_time = _attention_to_layer_time(decode_data_list[0].attention_weights)
    num_layers = int(first_layer_time.shape[0])
    first_k = int(first_layer_time.shape[1])
    assert first_k >= 1, "Token-0 attention must have at least one key position"

    inferred_prefix = first_k - 1
    if context_prefix_tokens is None:
        context_prefix_tokens = inferred_prefix
    assert context_prefix_tokens >= 0, "context_prefix_tokens must be non-negative"

    t_total = len(decode_data_list)

    # ── masks ────────────────────────────────────────────────────────────
    # pad_mask[i] refers to decode_data_list[i]  →  full-sequence token (token_offset + i).
    pad_mask = output_pad_mask(decode_data_list)

    # user_active covers the ENTIRE wav from time-0 so that we can index by
    # the full-sequence generated-token index rather than the local slice index.
    user_active_full = user_active_mask(input_wav, frame_rate=frame_rate, threshold=activity_threshold)

    # ── pre-compute a combined boolean: is full-sequence token i  (PAD ∧ user-silent)? ──
    # Only tokens inside the slice [token_offset, token_offset + t_total) have known pad status.
    max_full_gen_idx = token_offset + t_total
    combined_pad_silent: list[bool] = [False] * max_full_gen_idx
    for i in range(t_total):
        full_i = token_offset + i
        is_pad = pad_mask[i]
        is_silent = (full_i >= len(user_active_full)) or (not user_active_full[full_i])
        combined_pad_silent[full_i] = is_pad and is_silent
    combined_tensor = torch.tensor(combined_pad_silent, dtype=torch.bool)

    # ── diagnostic logging ───────────────────────────────────────────────
    n_pad = sum(pad_mask)
    n_user_active = sum(1 for flag in user_active_full[token_offset:token_offset + t_total]
                        if token_offset < len(user_active_full) and flag)
    n_user_silent = t_total - n_user_active
    n_combined = sum(combined_pad_silent)
    logger.info(
        "pad_lookback_ratio config: t_total=%d  token_offset=%d  "
        "context_prefix_tokens=%d  first_k=%d  num_layers=%d",
        t_total, token_offset, context_prefix_tokens, first_k, num_layers,
    )
    logger.info(
        "  masks: pad_count=%d/%d  user_active=%d  user_silent=%d  "
        "combined(pad∧silent)=%d  user_active_full_len=%d",
        n_pad, t_total, n_user_active, n_user_silent,
        n_combined, len(user_active_full),
    )

    ratios = torch.zeros((t_total, num_layers), dtype=torch.float32)

    for t, decode_data in enumerate(decode_data_list):
        if decode_data.attention_weights is None:
            continue

        layer_time = _attention_to_layer_time(decode_data.attention_weights)  # [L, K]
        k = int(layer_time.shape[-1])
        if k == 0:
            continue
        assert layer_time.shape[0] == num_layers, "Layer count must stay constant across tokens"

        # Dump raw attention weight stats for first few tokens (debug).
        if t < 3 or t == t_total - 1:
            attn_sum = float(layer_time.sum().item())
            attn_max = float(layer_time.max().item())
            attn_nonzero = int((layer_time > 0).sum().item())
            logger.info(
                "  [attn debug] t=%d  K=%d  attn_sum=%.6f  attn_max=%.6f  "
                "attn_nonzero=%d/%d  shape=%s",
                t, k, attn_sum, attn_max, attn_nonzero,
                layer_time.numel(), list(layer_time.shape),
            )

        # ── absolute-position mapping (supports sliding/truncated KV windows) ──
        abs_t = context_prefix_tokens + t
        start_abs = abs_t - (k - 1)
        key_abs = torch.arange(k, device=layer_time.device, dtype=torch.long) + int(start_abs)

        # gen_idx_local:  index relative to decode_data_list  (0 = first entry)
        # gen_idx_full:   index in the full generated-token sequence
        gen_idx_local = key_abs - int(context_prefix_tokens)          # range: [t-(k-1) .. t]
        gen_idx_full  = gen_idx_local + token_offset                  # full-sequence index

        # ── denominator: sum of attention on all generated lookback tokens (i < t) ──
        lookback_in_slice = (gen_idx_local >= 0) & (gen_idx_local < t)
        denominator = (layer_time * lookback_in_slice.unsqueeze(0).float()).sum(dim=-1)  # [L]

        # ── numerator: lookback keys that are (generated ∧ <t ∧ PAD ∧ user-silent) ──
        # A key qualifies for numerator when:
        #   • it maps to a generated token in the slice:  0 <= gen_idx_local < t
        #   • combined_pad_silent[gen_idx_full] is True
        in_combined_range = (gen_idx_full >= 0) & (gen_idx_full < max_full_gen_idx)

        # Clamp for safe indexing (the out-of-range slots are already excluded by masks).
        safe_full_idx = gen_idx_full.clamp(0, max(max_full_gen_idx - 1, 0))
        num_mask = lookback_in_slice & in_combined_range & combined_tensor[safe_full_idx]

        numerator = (layer_time * num_mask.unsqueeze(0).float()).sum(dim=-1)  # [L]

        ratio = torch.where(denominator > 0, numerator / denominator, torch.zeros_like(denominator))
        ratios[t] = ratio.detach().cpu().float()

        # Per-token diagnostics for first few + last token.
        if t < 3 or t == t_total - 1:
            logger.info(
                "  step t=%d: K=%d  abs_t=%d  lookback_in_slice=%d  "
                "num_positions=%d  numerator_mean=%.6f  denom_mean=%.6f  ratio_mean=%.6f",
                t, k, abs_t,
                int(lookback_in_slice.sum().item()),
                int(num_mask.sum().item()),
                float(numerator.mean().item()),
                float(denominator.mean().item()),
                float(ratio.mean().item()),
            )

    total_nonzero = int((ratios > 0).sum().item())
    logger.info(
        "pad_lookback_ratio result: shape=%s  non-zero=%d/%d  "
        "max=%.6f  mean=%.6f",
        list(ratios.shape), total_nonzero, ratios.numel(),
        float(ratios.max().item()), float(ratios.mean().item()),
    )
    return ratios


if __name__ == "__main__":
    import argparse
    # Visualize the amplitude of the input wav
    parser = argparse.ArgumentParser(description="Visualize input WAV amplitude and compute pad lookback ratios.")
    parser.add_argument("--input_wav", type=str, help="Path to the input WAV file.")
    
    args = parser.parse_args()
    visualize_wav_amplitude(args.input_wav)