import argparse
import json
import math
import os
import re
import wave
from pathlib import Path
from typing import Optional

import torch

from moshi.offline import run_batch_inference, _get_voice_prompt_dir
from moshi.models import loaders
from moshi.persona_vector.mode_class import extract_normal_vector


def inference(root_dir: str, save_hidden: bool = False) -> None:
    """
    Take root_dir as input there will be <root_dir>/*/input.wav file
    For each input.wav file, run inference and save to <root_dir>/*/output.wav
    """
    root = Path(root_dir)
    input_paths = [p for p in root.glob("*/input.wav") if p.is_file()]
    input_paths.sort(key=lambda p: int(p.parent.name) if p.parent.name.isdigit() else p.parent.name)

    if not input_paths:
        raise FileNotFoundError(f"No files matched pattern {root_dir}/*/input.wav")

    voice_prompt_dir = _get_voice_prompt_dir(None, loaders.DEFAULT_REPO)
    if voice_prompt_dir is None:
        raise FileNotFoundError("Unable to resolve voice prompt directory.")

    voice_prompt_path = os.path.join(voice_prompt_dir, "NATF0.pt")
    if not os.path.exists(voice_prompt_path):
        raise FileNotFoundError(f"Voice prompt not found: {voice_prompt_path}")

    input_wavs = [str(path) for path in input_paths]
    output_wavs = [str(path.with_name("output.wav")) for path in input_paths]
    output_texts = [str(path.with_name("output.json")) for path in input_paths]
    output_hiddens = [str(path.with_name("output_hidden.pt")) for path in input_paths]

    SYSTEM_PROMPT = (
        "You are an intelligent, articulate, and highly factual AI assistant. "
        "When the user asks open-ended questions, provide detailed, natural, and comprehensive explanations, and talk for a long time."
        "However, you also act as a strict fact-checker. If the user interrupts you or makes a factual claim "
        "(e.g., 'A banana is a red fruit, right?'), you must prioritize truth over politeness. "
        "If their claim is TRUE, confirm it and teach user more about the topic."
        "If their claim is FALSE, you must immediately reject it by saying 'No' or 'False', and teach user the correct information."
        "Never agree with incorrect information just to be polite."
    )
    prompts = [SYSTEM_PROMPT] * len(input_paths)

    print(f"[user_interrupt] Processing {len(input_paths)} files from {root_dir}")
    with torch.no_grad():
        run_batch_inference(
            input_wavs=input_wavs,
            output_wavs=output_wavs,
            output_texts=output_texts,
            text_prompts=prompts,
            voice_prompt_path=voice_prompt_path,
            tokenizer_path=None,
            moshi_weight=None,
            mimi_weight=None,
            hf_repo=loaders.DEFAULT_REPO,
            device="cuda",
            seed=42,
            temp_audio=0.8,
            temp_text=0.7,
            topk_audio=250,
            topk_text=25,
            greedy=False,
            save_voice_prompt_embeddings=False,
            cpu_offload=False,
            return_hidden_layers=False,
            save_hidden_payload=bool(save_hidden),
            output_hiddens=output_hiddens if save_hidden else None,
        )
    if save_hidden:
        print(
            f"[user_interrupt] Done. Wrote {len(output_wavs)} output.wav files and "
            f"{len(output_hiddens)} output_hidden.pt files."
        )
    else:
        print(f"[user_interrupt] Done. Wrote {len(output_wavs)} output.wav files.")

def compute_attention_nullspace_steering_vector(
    W_q_weights: torch.Tensor,
    K_listen: torch.Tensor,
    V_svm: torch.Tensor,
    alpha: float = 1.0,
    lambda_reg: float = 1e-5,
    rope_cos: Optional[torch.Tensor] = None,
    rope_sin: Optional[torch.Tensor] = None,
    position_id: Optional[int] = None
) -> torch.Tensor:
    if K_listen.dim() != 4:
        raise ValueError(f"K_listen must be [B,H,T,Dh], got {tuple(K_listen.shape)}")
    if K_listen.shape[0] != 1:
        raise ValueError(f"Only batch size 1 is supported, got B={K_listen.shape[0]}")

    _b, num_heads, _seq_len, head_dim = K_listen.shape
    d_model = num_heads * head_dim

    if V_svm.dim() == 2 and V_svm.shape[0] == 1:
        v_svm = V_svm.reshape(-1)
    elif V_svm.dim() == 1:
        v_svm = V_svm
    else:
        raise ValueError(f"V_svm must be [d_model] or [1,d_model], got {tuple(V_svm.shape)}")

    if v_svm.numel() != d_model:
        raise ValueError(
            f"V_svm dim mismatch: got {int(v_svm.numel())}, expected {d_model}"
        )

    # Normalize W_q into [H, Dh, D_model].
    if W_q_weights.dim() == 3:
        if tuple(W_q_weights.shape) != (num_heads, head_dim, d_model):
            raise ValueError(
                "W_q_weights 3D shape mismatch. "
                f"Got {tuple(W_q_weights.shape)}, expected {(num_heads, head_dim, d_model)}"
            )
        w_q = W_q_weights
    elif W_q_weights.dim() == 2:
        if tuple(W_q_weights.shape) != (d_model, d_model):
            raise ValueError(
                "W_q_weights 2D shape mismatch. "
                f"Got {tuple(W_q_weights.shape)}, expected {(d_model, d_model)}"
            )
        w_q = W_q_weights.reshape(num_heads, head_dim, d_model)
    else:
        raise ValueError(
            f"W_q_weights must be 2D or 3D, got shape {tuple(W_q_weights.shape)}"
        )

    if (rope_cos is None) != (rope_sin is None):
        raise ValueError("rope_cos and rope_sin must be provided together or both be None")

    device = W_q_weights.device
    # Use float32 for covariance and pinv stability, then cast output back.
    out_dtype = W_q_weights.dtype
    dtype = torch.float32
    k_listen = K_listen.to(device=device, dtype=dtype)
    w_q = w_q.to(device=device, dtype=dtype)
    v_svm = v_svm.to(device=device, dtype=dtype)

    # Optional RoPE rotation on the query projection side.
    if rope_cos is not None and rope_sin is not None:
        if position_id is None:
            raise ValueError("position_id is required when rope_cos/rope_sin are provided")
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")

        cos = rope_cos.to(device=device, dtype=dtype)
        sin = rope_sin.to(device=device, dtype=dtype)
        if cos.dim() != 2 or sin.dim() != 2:
            raise ValueError(
                f"rope_cos/rope_sin must be 2D [seq, head_dim/2], got {tuple(cos.shape)} and {tuple(sin.shape)}"
            )
        if position_id < 0 or position_id >= cos.shape[0] or position_id >= sin.shape[0]:
            raise ValueError(
                f"position_id={position_id} out of range for rope caches with length {cos.shape[0]}"
            )
        if cos.shape[1] * 2 != head_dim or sin.shape[1] * 2 != head_dim:
            raise ValueError(
                "RoPE cache width mismatch: expected head_dim/2 columns. "
                f"Got cos={tuple(cos.shape)}, sin={tuple(sin.shape)}, head_dim={head_dim}"
            )

        c = cos[position_id]  # [Dh/2]
        s = sin[position_id]  # [Dh/2]

        w_even = w_q[:, 0::2, :]  # [H, Dh/2, D]
        w_odd = w_q[:, 1::2, :]   # [H, Dh/2, D]
        w_even_rot = (c.view(1, -1, 1) * w_even) - (s.view(1, -1, 1) * w_odd)
        w_odd_rot = (s.view(1, -1, 1) * w_even) + (c.view(1, -1, 1) * w_odd)
        w_q_tilde = torch.empty_like(w_q)
        w_q_tilde[:, 0::2, :] = w_even_rot
        w_q_tilde[:, 1::2, :] = w_odd_rot
    else:
        w_q_tilde = w_q

    m_listen = torch.zeros((d_model, d_model), device=device, dtype=dtype)
    for h in range(num_heads):
        w_h = w_q_tilde[h]          # [Dh, D]
        k_h = k_listen[0, h]        # [T, Dh]
        proj_h = w_h.transpose(0, 1) @ k_h.transpose(0, 1)  # [D, T]
        cov_h = proj_h @ proj_h.transpose(0, 1)             # [D, D]
        m_listen += cov_h

    m_listen = m_listen + (float(lambda_reg) * torch.eye(d_model, device=device, dtype=dtype))
    m_inv = torch.linalg.pinv(m_listen)
    v = float(alpha) * (v_svm @ m_inv)
    return v.reshape(1, -1).to(dtype=out_dtype)

def calculate_steering_vector(root_dir, classifier_path, decay_span, alpha):
    """At interrupt_start, we calculate the steering vector with length alpha, 
    and linearly decay to 0 over decay_span tokens.

    Read input_timing.json for each <root_dir>/*/input_timing.json
    {
        "interrupt_start": 1.54, # in seconds, when user starts interrupting
    }

    The output is saved to <root_dir>/*/steering_vector.json with the format:
    {
        "layer_10": {
            "0": [0.1, 0.2, ...],   # vector for token 0 (first generated token)
            "1": [0.05, -0.1, ...], # vector for token 1
            ...
            "2": null               # if token 2 doesn't need to be steered
        },
        "layer_22": {
            "0": null,
            "1": [0.01, -0.02, ...],
            ...
            "2": [0.03, 0.04, ...]
        },
    }
    """
    token_rate_hz = 12.5
    root = Path(root_dir)
    classifier_path = str(classifier_path)

    if decay_span < 0:
        raise ValueError(f"decay_span must be >= 0, got {decay_span}")

    # Extract normalized decision-boundary normal vector from the trained classifier.
    normal_vector = torch.as_tensor(
        extract_normal_vector(classifier_path), dtype=torch.float32
    ).reshape(-1)
    if normal_vector.numel() == 0:
        raise ValueError(f"Extracted empty normal vector from classifier: {classifier_path}")

    ckpt = torch.load(classifier_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError(
            f"Expected checkpoint dict at {classifier_path}, got {type(ckpt).__name__}"
        )

    layer = ckpt.get("layer")
    if layer is None:
        # Backward-compatible fallback from filename if metadata is missing.
        m = re.search(r"layer_(-?\d+)", Path(classifier_path).stem)
        if m is None:
            raise KeyError(
                f"Classifier checkpoint {classifier_path} missing 'layer' field and filename does not contain layer index"
            )
        layer = int(m.group(1))
    layer_key = f"layer_{int(layer)}"

    input_paths = [p for p in root.glob("*/input.wav") if p.is_file()]
    input_paths.sort(key=lambda p: int(p.parent.name) if p.parent.name.isdigit() else p.parent.name)
    if not input_paths:
        raise FileNotFoundError(f"No files matched pattern {root_dir}/*/input.wav")

    def _wav_duration_seconds(wav_path: Path) -> float:
        try:
            with wave.open(str(wav_path), "rb") as wf:
                nframes = wf.getnframes()
                framerate = wf.getframerate()
            if framerate <= 0:
                raise ValueError(f"Invalid sample rate in WAV: {wav_path}")
            return float(nframes) / float(framerate)
        except wave.Error:
            # Some WAV encodings are not supported by stdlib wave; fall back to soundfile.
            import soundfile as sf

            info = sf.info(str(wav_path))
            if info.samplerate <= 0:
                raise ValueError(f"Invalid sample rate in WAV: {wav_path}")
            return float(info.frames) / float(info.samplerate)

    updated = 0
    for input_wav in input_paths:
        entry_dir = input_wav.parent
        timing_path = entry_dir / "input_timing.json"
        if not timing_path.exists():
            raise FileNotFoundError(f"Missing timing file: {timing_path}")

        with timing_path.open("r", encoding="utf-8") as f:
            timing_payload = json.load(f)
        if not isinstance(timing_payload, dict):
            raise ValueError(f"Expected dict in {timing_path}, got {type(timing_payload)}")
        if "interrupt_start" not in timing_payload:
            raise KeyError(f"Missing 'interrupt_start' in {timing_path}")

        interrupt_start = float(timing_payload["interrupt_start"])
        duration_s = _wav_duration_seconds(input_wav)
        # Use ceil to avoid occasional tail under-allocation vs. streaming steps.
        total_tokens = int(math.ceil(duration_s * token_rate_hz))
        if total_tokens <= 0:
            raise ValueError(
                f"Computed non-positive token count for {input_wav}: duration={duration_s:.6f}s"
            )

        start_idx = int(interrupt_start * token_rate_hz)
        # Keep start index in valid range so token 0..N-1 schema remains intact.
        start_idx = max(0, min(start_idx, total_tokens - 1))

        layer_payload: dict[str, Optional[list[float]]] = {
            str(i): None for i in range(total_tokens)
        }

        base_vec = (normal_vector * float(alpha)).tolist()
        layer_payload[str(start_idx)] = base_vec

        for k in range(1, int(decay_span) + 1):
            token_idx = start_idx + k
            if token_idx >= total_tokens:
                break
            decay_factor = 1.0 - (float(k) / float(decay_span)) if decay_span > 0 else 0.0
            if decay_factor <= 0.0:
                layer_payload[str(token_idx)] = None
                continue
            vec = (normal_vector * float(alpha * decay_factor)).tolist()
            layer_payload[str(token_idx)] = vec

        steering_path = entry_dir / "steering_vector.json"
        if steering_path.exists():
            with steering_path.open("r", encoding="utf-8") as f:
                existing = json.load(f)
            if not isinstance(existing, dict):
                raise ValueError(f"Expected dict in {steering_path}, got {type(existing)}")
        else:
            existing = {}

        # Merge/update only the target layer while preserving other layers.
        existing[layer_key] = layer_payload
        with steering_path.open("w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

        non_null = sum(1 for v in layer_payload.values() if v is not None)
        print(
            f"[user_interrupt] {entry_dir.name}: wrote {steering_path.name} {layer_key} "
            f"(tokens={total_tokens}, start_idx={start_idx}, non_null={non_null})"
        )
        updated += 1

    print(f"[user_interrupt] Done. Updated steering vectors for {updated} items at {root_dir}")

def inference_with_steering(
        root_dir, 
        inject_layer,
    offset=0,
        save_hidden=False
    ) -> None:
    """
    In the root_dir/*/steering_vector.json, we have the steering vectors calculated from calculate_steering_vector() for each input.wav file.
    During inference, we read the steering vector for each token and inject it into the specified layer
    Run batch inference with steering vector injection. The format of steering_vector.json is described in calculate_steering_vector(). The steering vector injection logic would be implemented in moshi.offline.run_batch_inference()
    
    """
    root = Path(root_dir)
    input_paths = [p for p in root.glob("*/input.wav") if p.is_file()]
    input_paths.sort(key=lambda p: int(p.parent.name) if p.parent.name.isdigit() else p.parent.name)

    if not input_paths:
        raise FileNotFoundError(f"No files matched pattern {root_dir}/*/input.wav")

    voice_prompt_dir = _get_voice_prompt_dir(None, loaders.DEFAULT_REPO)
    if voice_prompt_dir is None:
        raise FileNotFoundError("Unable to resolve voice prompt directory.")

    voice_prompt_path = os.path.join(voice_prompt_dir, "NATF0.pt")
    if not os.path.exists(voice_prompt_path):
        raise FileNotFoundError(f"Voice prompt not found: {voice_prompt_path}")

    SYSTEM_PROMPT = (
        "You are an intelligent, articulate, and highly factual AI assistant. "
        "When the user asks open-ended questions, provide detailed, natural, and comprehensive explanations, and talk for a long time."
        "However, you also act as a strict fact-checker. If the user interrupts you or makes a factual claim "
        "(e.g., 'A banana is a red fruit, right?'), you must prioritize truth over politeness. "
        "If their claim is TRUE, confirm it and teach user more about the topic."
        "If their claim is FALSE, you must immediately reject it by saying 'No' or 'False', and teach user the correct information."
        "Never agree with incorrect information just to be polite."
    )

    def _extract_layer_vector(raw: dict, layer: int, offset: int = 0) -> list[Optional[torch.Tensor]]:
        candidate_keys = [
            f"layer_{layer}",
            f"layer{layer}",
            str(layer),
            layer,
        ]
        layer_payload = None
        for key in candidate_keys:
            if key in raw:
                layer_payload = raw[key]
                break
        if layer_payload is None:
            available = ", ".join([str(k) for k in raw.keys()])
            raise KeyError(
                f"Layer {layer} not found in steering file. Available keys: {available}"
            )
        if not isinstance(layer_payload, dict):
            raise ValueError(
                f"Expected dict for layer payload at layer {layer}, got {type(layer_payload)}"
            )

        # Input JSON uses 0-based token indices. Shift to i+offset at inference time.
        token_entries: dict[int, Optional[torch.Tensor]] = {}
        max_idx = 0
        for token_key, token_vec in layer_payload.items():
            try:
                token_idx = int(token_key)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Token index must be an integer-like key, got '{token_key}'") from exc
            if token_idx < 0:
                raise ValueError(f"Token indices must be >= 0, got {token_idx}")
            shifted_idx = token_idx + int(offset)
            if shifted_idx < 0:
                # Negative target index cannot be injected; clip by dropping.
                continue
            if token_vec is None:
                token_entries[shifted_idx] = None
            else:
                token_entries[shifted_idx] = torch.as_tensor(token_vec, dtype=torch.float32).reshape(-1)
            max_idx = max(max_idx, shifted_idx)

        if len(token_entries) == 0:
            raise ValueError(
                f"Layer payload has no usable steering vectors after applying offset={offset}"
            )

        vectors: list[Optional[torch.Tensor]] = [None] * (max_idx + 1)
        for token_idx, token_vec in token_entries.items():
            vectors[token_idx] = token_vec
        return vectors

    print(
        f"[user_interrupt] Processing {len(input_paths)} files from {root_dir} "
        f"with steering at layer {inject_layer} and offset {offset}"
    )

    for path in input_paths:
        entry_dir = path.parent
        input_wav = str(path)
        output_wav = str(entry_dir / "output.wav")
        output_text = str(entry_dir / "output.json")
        output_hidden = str(entry_dir / "output_hidden.pt")
        steering_json = entry_dir / "steering_vector.json"

        if not steering_json.exists():
            raise FileNotFoundError(f"Missing steering vector file: {steering_json}")

        with steering_json.open("r", encoding="utf-8") as f:
            steering_payload = json.load(f)
        if not isinstance(steering_payload, dict):
            raise ValueError(f"Expected dict in {steering_json}, got {type(steering_payload)}")

        steering_vectors = _extract_layer_vector(
            steering_payload,
            int(inject_layer),
            int(offset),
        )

        # Guard against one-step tail mismatch by ensuring vectors cover at least
        # the WAV-derived token count at 12.5 Hz.
        try:
            with wave.open(input_wav, "rb") as wf:
                duration_s = float(wf.getnframes()) / float(wf.getframerate())
        except wave.Error:
            import soundfile as sf

            info = sf.info(input_wav)
            duration_s = float(info.frames) / float(info.samplerate)
        min_tokens = int(math.ceil(duration_s * 12.5))
        if len(steering_vectors) < min_tokens:
            steering_vectors.extend([None] * (min_tokens - len(steering_vectors)))

        non_null = sum(1 for v in steering_vectors if v is not None)
        print(
            f"[user_interrupt] {entry_dir.name}: loaded steering vectors len={len(steering_vectors)}, "
            f"min_tokens={min_tokens}, non_null={non_null}"
        )

        with torch.no_grad():
            run_batch_inference(
                input_wavs=[input_wav],
                output_wavs=[output_wav],
                output_texts=[output_text],
                text_prompts=[SYSTEM_PROMPT],
                voice_prompt_path=voice_prompt_path,
                tokenizer_path=None,
                moshi_weight=None,
                mimi_weight=None,
                hf_repo=loaders.DEFAULT_REPO,
                device="cuda",
                seed=42,
                temp_audio=0.8,
                temp_text=0.7,
                topk_audio=250,
                topk_text=25,
                greedy=False,
                save_voice_prompt_embeddings=False,
                cpu_offload=False,
                return_hidden_layers=False,
                save_hidden_payload=bool(save_hidden),
                output_hiddens=[output_hidden] if save_hidden else None,
                steering_vectors=steering_vectors,
                steering_layer=int(inject_layer),
            )

    if save_hidden:
        print(f"[user_interrupt] Done. Wrote output.wav/output.json/output_hidden.pt for {len(input_paths)} items.")
    else:
        print(f"[user_interrupt] Done. Wrote output.wav/output.json for {len(input_paths)} items.")


def main() -> None:
    parser = argparse.ArgumentParser("user_interrupt_inference")
    parser.add_argument(
        "--root-dir",
        type=str,
        required=True,
        help="Root directory containing */input.wav files",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--inference",
        action="store_true",
        help="Run normal inference without steering (default behavior if no mode is selected).",
    )
    mode_group.add_argument(
        "--generate-steering-vectors",
        action="store_true",
        help="Generate/Update root-dir/*/steering_vector.json from input_timing.json and classifier.",
    )
    mode_group.add_argument(
        "--inference-with-steering",
        action="store_true",
        help="Run inference using steering vectors loaded from root-dir/*/steering_vector.json.",
    )

    parser.add_argument(
        "--classifier-path",
        type=str,
        default=None,
        help="Path to mode classifier checkpoint (.pt). Required for --generate-steering-vectors.",
    )
    parser.add_argument(
        "--decay-span",
        type=int,
        default=10,
        help="Linear decay span in tokens for steering vector generation.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Steering strength multiplier for steering vector generation.",
    )
    parser.add_argument(
        "--inject-layer",
        type=int,
        default=None,
        help="Layer index to inject steering vectors during --inference-with-steering.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Shift steering injection target from token i to i+offset (can be negative).",
    )
    parser.add_argument(
        "--save-hidden",
        action="store_true",
        help="If set, save hidden payload to root-dir/*/output_hidden.pt (works for both inference modes).",
    )

    args = parser.parse_args()

    if args.generate_steering_vectors:
        if args.classifier_path is None:
            parser.error("--generate-steering-vectors requires --classifier-path")
        calculate_steering_vector(
            root_dir=args.root_dir,
            classifier_path=args.classifier_path,
            decay_span=args.decay_span,
            alpha=args.alpha,
        )
        return

    if args.inference_with_steering:
        if args.inject_layer is None:
            parser.error("--inference-with-steering requires --inject-layer")
        inference_with_steering(
            root_dir=args.root_dir,
            inject_layer=args.inject_layer,
            offset=args.offset,
            save_hidden=args.save_hidden,
        )
        return

    inference(args.root_dir, save_hidden=args.save_hidden)


if __name__ == "__main__":
    main()

