import argparse
import json
import math
import os
import re
import wave
from pathlib import Path
from typing import Any, Optional, cast
from tqdm import tqdm
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from moshi.offline import run_batch_inference, _get_voice_prompt_dir
from moshi.models import loaders
from moshi.persona_vector.mode_class import extract_normal_vector


def inference(
    root_dir: str,
    save_hidden: bool = False,
    payload_target_layer: Optional[int] = None,
) -> None:
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
            payload_target_layer=payload_target_layer,
        )
    if save_hidden:
        print(
            f"[user_interrupt] Done. Wrote {len(output_wavs)} output.wav files and "
            f"{len(output_hiddens)} output_hidden.pt files."
        )
    else:
        print(f"[user_interrupt] Done. Wrote {len(output_wavs)} output.wav files.")

def compute_attention_mapped_steering_vector(root_dir, classifier_path, decay_span, alpha):
    """Just like calculate_steering_vector()
    but instead of just using the normal vector from the SVM classifier
    use _compute_attention_mapped_steering_vector() to optimize of the 
    steering vector by mapping it. Then save the mapped vector and decay logic
    as calculate_steering_vector() does, so that it can be injected during inference.
    """
    token_rate_hz = 12.5
    root = Path(root_dir)
    classifier_path = str(classifier_path)

    if decay_span < 0:
        raise ValueError(f"decay_span must be >= 0, got {decay_span}")

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
        m = re.search(r"layer_(-?\d+)", Path(classifier_path).stem)
        if m is None:
            raise KeyError(
                f"Classifier checkpoint {classifier_path} missing 'layer' field and filename does not contain layer index"
            )
        layer = int(m.group(1))
    layer = int(layer)
    layer_key = f"layer_{layer}"

    moshi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME)
    lm = loaders.get_moshi_lm(moshi_weight, device="cpu", cpu_offload=False)
    lm.eval()

    num_layers = len(lm.transformer.layers)
    layer_idx = layer if layer >= 0 else num_layers + layer
    if layer_idx < 0 or layer_idx >= num_layers:
        raise ValueError(
            f"Layer index {layer} (resolved to {layer_idx}) out of range for model with {num_layers} layers"
        )

    target_layer = lm.transformer.layers[layer_idx]
    attn = cast(Any, target_layer.self_attn)
    w = attn.in_proj_weight.detach().cpu().float()
    embed_dim = int(attn.embed_dim)
    weights_per_step = int(getattr(attn, "weights_per_step", 0))

    if weights_per_step > 0 and w.dim() == 2 and w.shape[0] == weights_per_step * 3 * embed_dim:
        # Use step 0 for offline steering-vector generation.
        w = w.view(weights_per_step, 3 * embed_dim, embed_dim)[0]

    if w.dim() != 2 or w.shape[0] != 3 * embed_dim or w.shape[1] != embed_dim:
        raise RuntimeError(f"Unexpected in_proj_weight shape for packed QKV: {tuple(w.shape)}")

    w_q = w[:embed_dim, :].contiguous()
    w_k = w[embed_dim : 2 * embed_dim, :].contiguous()

    print("W_q shape:", w_q.shape)
    print("W_k shape:", w_k.shape)

    mu_s = ckpt.get("ave_hidden_pos")
    mu_l = ckpt.get("ave_hidden_neg")
    if mu_s is None or mu_l is None:
        raise KeyError(
            "Classifier checkpoint must contain 'ave_hidden_pos' and 'ave_hidden_neg'. "
            "Please retrain or re-save the classifier with class-average hidden vectors."
        )
    mu_s = torch.as_tensor(mu_s, dtype=torch.float32).reshape(-1)
    mu_l = torch.as_tensor(mu_l, dtype=torch.float32).reshape(-1)
    if mu_s.numel() == 0 or mu_l.numel() == 0:
        raise ValueError("'ave_hidden_pos'/'ave_hidden_neg' must be non-empty vectors")
    if mu_s.numel() != embed_dim or mu_l.numel() != embed_dim:
        raise ValueError(
            f"Classifier mean vector dim mismatch: mu_s={mu_s.numel()}, mu_l={mu_l.numel()}, expected={embed_dim}"
        )

    mapped_vector = _compute_attention_mapped_steering_vector(
        mu_s=mu_s,
        mu_l=mu_l,
        W_q_weights=w_q,
        W_k_weights=w_k,
        alpha=float(alpha),
    ).reshape(-1)
    # Print the cosine similarity between mapped_vector and mu_s - mu_l
    cos_sim = F.cosine_similarity(mapped_vector, mu_s - mu_l, dim=0).item()
    print("Cosine similarity between mapped_vector and (mu_s - mu_l):", cos_sim)

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
        total_tokens = int(math.ceil(duration_s * token_rate_hz))
        if total_tokens <= 0:
            raise ValueError(
                f"Computed non-positive token count for {input_wav}: duration={duration_s:.6f}s"
            )

        start_idx = int(interrupt_start * token_rate_hz)
        start_idx = max(0, min(start_idx, total_tokens - 1))

        layer_payload: dict[str, Optional[list[float]]] = {
            str(i): None for i in range(total_tokens)
        }

        base_vec = mapped_vector.tolist()
        layer_payload[str(start_idx)] = base_vec

        for k in range(1, int(decay_span) + 1):
            token_idx = start_idx + k
            if token_idx >= total_tokens:
                break
            decay_factor = 1.0 - (float(k) / float(decay_span)) if decay_span > 0 else 0.0
            if decay_factor <= 0.0:
                layer_payload[str(token_idx)] = None
                continue
            vec = (mapped_vector * float(decay_factor)).tolist()
            layer_payload[str(token_idx)] = vec

        steering_path = entry_dir / "steering_vector.json"
        if steering_path.exists():
            with steering_path.open("r", encoding="utf-8") as f:
                existing = json.load(f)
            if not isinstance(existing, dict):
                raise ValueError(f"Expected dict in {steering_path}, got {type(existing)}")
        else:
            existing = {}

        existing[layer_key] = layer_payload
        with steering_path.open("w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

        non_null = sum(1 for v in layer_payload.values() if v is not None)
        print(
            f"[user_interrupt] {entry_dir.name}: wrote {steering_path.name} {layer_key} "
            f"(tokens={total_tokens}, start_idx={start_idx}, non_null={non_null})"
        )
        updated += 1

    print(f"[user_interrupt] Done. Updated attention-mapped steering vectors for {updated} items at {root_dir}")

import torch
import torch.nn.functional as F
from tqdm import tqdm

def get_rope_matrix(
    n: int,
    head_dim: int,
    base: float = 10000.0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Constructs the explicit dense [head_dim, head_dim] block-diagonal RoPE rotation matrix 
    for a specific relative distance n. (Using Interleaved Format)
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    theta = n * inv_freq  # [head_dim // 2]
    
    cos_val = torch.cos(theta)
    sin_val = torch.sin(theta)
    
    # Create an empty [head_dim, head_dim] matrix
    R_n = torch.zeros((head_dim, head_dim), device=device, dtype=torch.float32)
    
    # Fill the block diagonal (Interleaved format)
    idx = torch.arange(0, head_dim, 2, device=device)
    R_n[idx, idx] = cos_val
    R_n[idx, idx + 1] = -sin_val
    R_n[idx + 1, idx] = sin_val
    R_n[idx + 1, idx + 1] = cos_val
    
    return R_n


def _compute_attention_mapped_steering_vector(
    mu_s: torch.Tensor,
    mu_l: torch.Tensor,
    W_q_weights: torch.Tensor,
    W_k_weights: torch.Tensor,
    rope_base: float = 10000.0,
    rope_context_len: int = 200,
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    Strategy 3: Unconstrained Bi-directional Gradient Superposition.
    Directly computes the gradient vector that maximizes attention to the Speaking mode 
    while minimizing attention to the Listening mode, averaged over RoPE distances.
    """
    # 1. Shape Verification & Normalization
    if W_q_weights.dim() == 2:
        d_model = W_q_weights.shape[0]
        # Moshi/Llama default heuristics: head_dim is usually 128
        head_dim = 128 
        num_heads = d_model // head_dim
        w_q = W_q_weights.reshape(num_heads, head_dim, d_model)
        w_k = W_k_weights.reshape(num_heads, head_dim, d_model)
    else:
        num_heads, head_dim, d_model = W_q_weights.shape
        w_q = W_q_weights
        w_k = W_k_weights

    device = w_q.device
    out_dtype = w_q.dtype
    dtype = torch.float32  # High precision for tensor accumulation
    
    mu_s = mu_s.reshape(-1).to(device=device, dtype=dtype)
    mu_l = mu_l.reshape(-1).to(device=device, dtype=dtype)
    w_q = w_q.to(device=device, dtype=dtype)
    w_k = w_k.to(device=device, dtype=dtype)

    # The contrastive target vector in the residual stream
    v_svm = mu_s - 0.2 * mu_l
    
    all_grads = []
    
    print(f"[Math Engine] Computing Strategy 3 Gradient Superposition over RoPE n=0 to {rope_context_len-1}...")
    
    for n in tqdm(range(rope_context_len)):
        grad_n = torch.zeros(d_model, device=device, dtype=dtype)
        
        # Calculate the gradient contribution for each Attention Head
        for i in range(num_heads):
            w_q_i = w_q[i]  # [head_dim, d_model]
            w_k_i = w_k[i]  # [head_dim, d_model]
            r_n = get_rope_matrix(n, head_dim, rope_base, device=device) # [head_dim, head_dim]
            
            # --- The elegant linear math ---
            # 1. Map target vector into Key space: W_K @ v_svm
            key_proj = torch.matmul(w_k_i, v_svm)  # shape: [head_dim]
            
            # 2. Apply RoPE rotation: R_n @ key_proj
            rotated_key = torch.matmul(r_n, key_proj) # shape: [head_dim]
            
            # 3. Pull gradient back to Residual Stream via Query weights: W_Q^T @ rotated_key
            # Note: PyTorch w_q_i is [head_dim, d_model], so w_q_i.T acts as the mapping back to d_model
            grad_head_i = torch.matmul(w_q_i.T, rotated_key) # shape: [d_model]
            
            grad_n += grad_head_i
            
        all_grads.append(grad_n)

    # Average the gradient over all RoPE distances to create a robust static vector
    v_star = torch.stack(all_grads).mean(dim=0)  # [d_model]

    # Normalize and scale by steering strength (alpha)
    v_star_norm = F.normalize(v_star, dim=0)
    v_opt = v_star_norm * float(alpha)

    print("[Math Engine] Strategy 3 optimal vector computed successfully.")
    return v_opt.reshape(1, -1).to(dtype=out_dtype)


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
        "--generate-steering-vectors-optimized",
        action="store_true",
        help="Generate/Update root-dir/*/steering_vector.json using attention-optimized steering vectors.",
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
    parser.add_argument(
        "--payload-target-layer",
        type=int,
        default=0,
        help=(
            "Text transformer layer index used to extract payload fields "
            "(text_keys, W_q, rope_cos, rope_sin). "
            "If not set, defaults to steering_layer (if provided) else last layer."
        ),
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

    if args.generate_steering_vectors_optimized:
        if args.classifier_path is None:
            parser.error("--generate-steering-vectors-optimized requires --classifier-path")
        compute_attention_mapped_steering_vector(
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

    inference(
        args.root_dir,
        save_hidden=args.save_hidden,
        payload_target_layer=args.payload_target_layer,
    )


if __name__ == "__main__":
    main()

