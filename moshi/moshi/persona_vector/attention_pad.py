from __future__ import annotations

from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from .premature_decode import DecodeData, PAD_TOKEN_ID


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
):
    """
    Compute the user-silent pad lookback ratio of each tokens.

    for each token T:
        for each layer l:
            user_silent_pad_lookback_ratio =
                sum of attention weight on token_t, t < T, where token_t is <PAD> and user is silent
                --------------------------------------------------------------------------------------
                sum of attention weight on all token_t, t < T.

    Args:
        decode_data_list (list[DecodeData]): A list of DecodeData, each corresponds to a token. The list is ordered by time.
        input_wav (str): Path to the user input wav used to estimate user activity over time.
        frame_rate (float): Token frame rate, default 12.5.
        activity_threshold (float): Absolute-amplitude threshold for active speech.
        context_prefix_tokens (Optional[int]): Fixed number of attention key positions to ignore
            as pre-input context. If None, inferred from token-0 attention length as `K0 - 1`.

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

    pad_mask = output_pad_mask(decode_data_list)
    user_active = user_active_mask(input_wav, frame_rate=frame_rate, threshold=activity_threshold)
    if len(user_active) < t_total:
        user_active = list(user_active) + [False] * (t_total - len(user_active))
    user_silent_mask = [not bool(flag) for flag in user_active[:t_total]]

    numerator_token_mask = [pad and silent for pad, silent in zip(pad_mask, user_silent_mask)]

    ratios = torch.zeros((t_total, num_layers), dtype=torch.float32)

    for t, decode_data in enumerate(decode_data_list):
        if decode_data.attention_weights is None:
            continue

        layer_time = _attention_to_layer_time(decode_data.attention_weights)  # [L, K]
        k = int(layer_time.shape[-1])
        if k == 0:
            continue
        assert layer_time.shape[0] == num_layers, "Layer count must stay constant across tokens"

        # Expected mapping for generated token indices under fixed context prefix:
        # key index = context_prefix_tokens + token_index
        expected_min_k = context_prefix_tokens + t + 1
        assert k >= expected_min_k, (
            f"Attention key length too short at token {t}: K={k}, expected at least {expected_min_k}. "
            "This usually means context-window truncation happened; increase context or reduce analyzed range."
        )

        # Enforce fixed prefix assumption requested by user.
        # For token t, generated-token slot [0..t] should start at `context_prefix_tokens`.
        # (There may be extra history to the right only if architecture changes; we disallow it.)
        implied_prefix = k - (t + 1)
        assert implied_prefix == context_prefix_tokens, (
            f"Non-constant context prefix detected at token {t}: implied={implied_prefix}, "
            f"expected={context_prefix_tokens}."
        )

        # Full-key masks with fixed ignored context prefix.
        # We include lookback tokens [0..t-1] and exclude current token t.
        num_mask_full = torch.zeros((k,), dtype=torch.float32, device=layer_time.device)
        den_mask_full = torch.zeros((k,), dtype=torch.float32, device=layer_time.device)

        if t > 0:
            lookback_num = torch.tensor(
                numerator_token_mask[:t],
                dtype=torch.float32,
                device=layer_time.device,
            )
            num_mask_full[context_prefix_tokens : context_prefix_tokens + t] = lookback_num
            den_mask_full[context_prefix_tokens : context_prefix_tokens + t] = 1.0

        numerator = (layer_time * num_mask_full.unsqueeze(0)).sum(dim=-1)
        denominator = (layer_time * den_mask_full.unsqueeze(0)).sum(dim=-1)

        ratio = torch.where(denominator > 0, numerator / denominator, torch.zeros_like(denominator))
        ratios[t] = ratio.detach().cpu().float()

    return ratios


if __name__ == "__main__":
    import argparse
    # Visualize the amplitude of the input wav
    parser = argparse.ArgumentParser(description="Visualize input WAV amplitude and compute pad lookback ratios.")
    parser.add_argument("--input_wav", type=str, help="Path to the input WAV file.")
    
    args = parser.parse_args()
    visualize_wav_amplitude(args.input_wav)