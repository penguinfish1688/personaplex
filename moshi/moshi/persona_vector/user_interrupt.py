import argparse
import json
import math
import os
import re
import tempfile
import wave
from pathlib import Path
from typing import Any, Optional
from tqdm import tqdm
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from moshi.offline import run_batch_inference, _get_voice_prompt_dir
from moshi.models import loaders
from moshi.models.lm import SILENCE_TOKENS
from moshi.persona_vector.mode_class import extract_normal_vector


MAIN_LAYER_MIN = 0
MAIN_LAYER_MAX = 31
JSON_LAYER_MIN = MAIN_LAYER_MIN
JSON_LAYER_MAX = MAIN_LAYER_MAX


_MODE_CLASS_HIDDEN_CACHE: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _load_existing_steering_payload(steering_path: Path) -> dict:
    """Load an existing steering JSON payload.

    If the file is malformed JSON, delete it and start fresh.
    """
    if not steering_path.exists():
        return {}

    try:
        with steering_path.open("r", encoding="utf-8") as f:
            existing = json.load(f)
    except json.JSONDecodeError as exc:
        try:
            steering_path.unlink()
        except FileNotFoundError:
            pass
        print(
            f"[user_interrupt] Warning: malformed JSON at {steering_path}. "
            f"Deleted file and recreating it. Error: {exc}"
        )
        return {}

    if not isinstance(existing, dict):
        raise ValueError(f"Expected dict in {steering_path}, got {type(existing)}")
    return existing


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Atomically write JSON to reduce risk of partial/truncated files."""
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent), delete=False) as tf:
        json.dump(payload, tf, indent=2, ensure_ascii=False)
        tf.flush()
        os.fsync(tf.fileno())
        tmp_name = tf.name
    os.replace(tmp_name, path)


def _is_valid_main_layer(layer: int) -> bool:
    return MAIN_LAYER_MIN <= int(layer) <= MAIN_LAYER_MAX


def _internal_layer_to_json_layer(layer: int) -> int:
    return int(layer)


def _json_layer_to_internal_layer(layer: int) -> Optional[int]:
    # Steering JSON schema is 0-based layer numbers (0..31).
    if _is_valid_main_layer(int(layer)):
        return int(layer)
    return None


def _discover_classifier_paths(classifier_dir: str) -> dict[int, str]:
    base = Path(classifier_dir)
    if not base.is_dir():
        raise FileNotFoundError(f"Classifier directory not found: {classifier_dir}")

    discovered: dict[int, str] = {}
    for p in base.glob("hidden_mode_classifier_layer_*.pt"):
        m = re.search(r"hidden_mode_classifier_layer_(-?\d+)\.pt$", p.name)
        if m is None:
            continue
        layer = int(m.group(1))
        if not _is_valid_main_layer(layer):
            continue
        discovered[layer] = str(p)
    if not discovered:
        raise FileNotFoundError(
            f"No classifier checkpoints found in {classifier_dir} for valid layers [{MAIN_LAYER_MIN}..{MAIN_LAYER_MAX}]. "
            "Expected files like hidden_mode_classifier_layer_<layer>.pt"
        )
    return discovered


def _resolve_requested_layers(
    requested_layers: list[int],
    available_layers: list[int],
) -> list[int]:
    if not available_layers:
        raise ValueError("No available layers to resolve")
    available_filtered = sorted(set(int(x) for x in available_layers if _is_valid_main_layer(int(x))))
    if -1 in requested_layers:
        return available_filtered
    resolved = [int(x) for x in requested_layers]
    invalid = [x for x in resolved if not _is_valid_main_layer(x)]
    if invalid:
        raise ValueError(
            f"Invalid layer(s) requested: {invalid}. Supported range is [{MAIN_LAYER_MIN}..{MAIN_LAYER_MAX}] or -1 for all."
        )
    missing = [x for x in resolved if x not in set(available_filtered)]
    if missing:
        raise FileNotFoundError(
            f"Requested layers not found: {missing}. Available layers: {available_filtered}"
        )
    # Preserve user ordering but drop duplicates.
    unique: list[int] = []
    seen: set[int] = set()
    for x in resolved:
        if x not in seen:
            unique.append(x)
            seen.add(x)
    return unique


def _extract_hidden_layer_from_payload(payload: dict[str, Any], layer: int) -> torch.Tensor:
    """Extract hidden states for a specific layer as [T, D]."""
    if "text_hidden_layers" in payload:
        hidden = payload["text_hidden_layers"]
        if not isinstance(hidden, torch.Tensor):
            hidden = torch.as_tensor(hidden)
        num_layers = int(hidden.shape[1])
        actual_layer = int(layer) if int(layer) >= 0 else num_layers + int(layer)
        if actual_layer < 0 or actual_layer >= num_layers:
            raise ValueError(
                f"Layer {layer} out of range for payload with {num_layers} layers"
            )
        return hidden[:, actual_layer, :].float()

    if "hidden_states" in payload:
        if int(layer) != -1:
            raise ValueError(
                "Payload has only 'hidden_states'; this supports layer=-1 only. "
                f"Requested layer={layer}."
            )
        hidden = payload["hidden_states"]
        if not isinstance(hidden, torch.Tensor):
            hidden = torch.as_tensor(hidden)
        return hidden.float()

    raise KeyError("Payload has neither 'text_hidden_layers' nor 'hidden_states'")


def _build_mode_labels(
    num_tokens: int,
    listening_ranges: list[list[int]],
    speaking_ranges: list[list[int]],
) -> torch.Tensor:
    """Build per-token labels: listening=0, speaking=1."""
    labels = torch.zeros(num_tokens, dtype=torch.float32)
    for start, end in listening_ranges:
        if start < 0 or end >= num_tokens:
            raise ValueError(
                f"Listening range [{start}, {end}] out of bounds for {num_tokens} tokens"
            )
        labels[start : end + 1] = 0.0
    for start, end in speaking_ranges:
        if start < 0 or end >= num_tokens:
            raise ValueError(
                f"Speaking range [{start}, {end}] out of bounds for {num_tokens} tokens"
            )
        labels[start : end + 1] = 1.0
    return labels


def _load_mode_class_hidden_sets(
    classifier_dir: str,
    layer: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load H_s/H_l token sets from mode-class dataset under classifier_dir.

    Expects entries like classifier_dir/*/input.json and corresponding
    complete/incomplete *_hidden.pt files with mode ranges in input.json.
    Returns:
      H_s: [N_s, D] speaking hidden states
      H_l: [N_l, D] listening hidden states
    """
    cache_key = (str(Path(classifier_dir).resolve()), int(layer))
    cached = _MODE_CLASS_HIDDEN_CACHE.get(cache_key)
    if cached is not None:
        return cached

    base = Path(classifier_dir)
    entries = sorted(
        [p for p in base.glob("*/input.json") if p.is_file()],
        key=lambda p: int(p.parent.name) if p.parent.name.isdigit() else p.parent.name,
    )
    if not entries:
        raise FileNotFoundError(
            f"No mode-class dataset entries found in {classifier_dir}. "
            "Expected */input.json with *_hidden.pt files."
        )

    speaking_chunks: list[torch.Tensor] = []
    listening_chunks: list[torch.Tensor] = []

    for entry_json in entries:
        entry_dir = entry_json.parent
        with entry_json.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        if not isinstance(meta, dict):
            raise ValueError(f"Expected dict in {entry_json}, got {type(meta)}")

        sample_specs = [
            ("complete_sentence_hidden.pt", "complete_modes"),
            ("incomplete_sentence_hidden.pt", "incomplete_modes"),
        ]
        for hidden_name, mode_key in sample_specs:
            hidden_path = entry_dir / hidden_name
            if not hidden_path.exists():
                raise FileNotFoundError(
                    f"Missing hidden payload: {hidden_path}. "
                    "Run mode_class --gen-dataset-hidden first."
                )
            if mode_key not in meta:
                raise KeyError(f"Missing '{mode_key}' in {entry_json}")

            payload = torch.load(hidden_path, map_location="cpu", weights_only=False)
            if not isinstance(payload, dict):
                raise TypeError(
                    f"Expected dict payload in {hidden_path}, got {type(payload).__name__}"
                )
            hidden = _extract_hidden_layer_from_payload(payload, int(layer))
            num_tokens = int(hidden.shape[0])
            mode_payload = meta[mode_key]
            if not isinstance(mode_payload, dict):
                raise ValueError(f"Expected dict for '{mode_key}' in {entry_json}")
            listening_ranges = mode_payload.get("listening", [])
            speaking_ranges = mode_payload.get("speaking", [])
            labels = _build_mode_labels(num_tokens, listening_ranges, speaking_ranges)

            speaking_mask = labels == 1
            listening_mask = labels == 0
            if int(speaking_mask.sum()) > 0:
                speaking_chunks.append(hidden[speaking_mask])
            if int(listening_mask.sum()) > 0:
                listening_chunks.append(hidden[listening_mask])

    if not speaking_chunks or not listening_chunks:
        raise RuntimeError(
            f"Insufficient labeled tokens in {classifier_dir} for layer {layer}. "
            f"speaking_chunks={len(speaking_chunks)}, listening_chunks={len(listening_chunks)}"
        )

    H_s = torch.cat(speaking_chunks, dim=0)
    H_l = torch.cat(listening_chunks, dim=0)
    _MODE_CLASS_HIDDEN_CACHE[cache_key] = (H_s, H_l)
    return H_s, H_l


def _extract_qk_weights_for_layer(lm: Any, layer: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Extract packed Q/K projection weights for a model layer as [d_model, d_model]."""
    num_layers = len(lm.transformer.layers)
    layer_idx = int(layer) if int(layer) >= 0 else num_layers + int(layer)
    if layer_idx < 0 or layer_idx >= num_layers:
        raise ValueError(
            f"Layer index {layer} (resolved to {layer_idx}) out of range for model with {num_layers} layers"
        )

    target_layer = lm.transformer.layers[layer_idx]
    attn = target_layer.self_attn
    w = attn.in_proj_weight.detach().cpu().float()
    embed_dim = int(attn.embed_dim)
    weights_per_step = int(getattr(attn, "weights_per_step", 0))

    if weights_per_step > 0 and w.dim() == 2 and w.shape[0] == weights_per_step * 3 * embed_dim:
        # Use step 0 for offline steering-vector generation.
        w = w.view(weights_per_step, 3 * embed_dim, embed_dim)[0]

    if w.dim() != 2 or w.shape[0] != 3 * embed_dim or w.shape[1] != embed_dim:
        raise RuntimeError(
            f"Unexpected in_proj_weight shape for packed QKV at layer {layer}: {tuple(w.shape)}"
        )

    w_q = w[:embed_dim, :].contiguous()
    w_k = w[embed_dim : 2 * embed_dim, :].contiguous()
    return w_q, w_k, embed_dim


def _apply_resume_index(input_paths: list[Path], resume: int, root_dir: str) -> list[Path]:
    """Return dataset slice starting from the 0-based resume index."""
    resume_idx = int(resume)
    if resume_idx < 0:
        raise ValueError(f"--resume must be >= 0 (0-based index), got {resume_idx}")
    if resume_idx >= len(input_paths):
        raise ValueError(
            f"--resume={resume_idx} out of range for dataset size {len(input_paths)} under {root_dir}"
        )
    if resume_idx > 0:
        print(
            f"[user_interrupt] Resume enabled: skipping first {resume_idx} files, "
            f"starting from index {resume_idx}"
        )
    return input_paths[resume_idx:]


def inference(
    root_dir: str,
    save_hidden: bool = False,
    payload_target_layer: Optional[int] = None,
    resume: int = 0,
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

    input_paths = _apply_resume_index(input_paths, resume, root_dir)

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

def _compute_attention_mapped_steering_vector_single_layer(root_dir, classifier_dir, layer, decay_span, alpha):
    """Generate one optimized steering vector for a target layer.

    Uses mode-class dataset hidden states (H_s/H_l) loaded from ``classifier_dir``
    and writes the per-token decay schedule into ``root_dir/*/steering_vector.json``.
    """
    token_rate_hz = 12.5
    root = Path(root_dir)
    classifier_dir = str(classifier_dir)
    layer = int(layer)

    if decay_span < 0:
        raise ValueError(f"decay_span must be >= 0, got {decay_span}")

    if not _is_valid_main_layer(layer):
        raise ValueError(f"Invalid layer {layer}; expected [{MAIN_LAYER_MIN}..{MAIN_LAYER_MAX}]")
    layer_key = f"layer_{_internal_layer_to_json_layer(layer)}"

    moshi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME)
    lm = loaders.get_moshi_lm(moshi_weight, device="cpu", cpu_offload=False)
    lm.eval()

    w_q, w_k, embed_dim = _extract_qk_weights_for_layer(lm, layer)

    H_s, H_l = _load_mode_class_hidden_sets(classifier_dir, layer)
    if int(H_s.shape[-1]) != int(H_l.shape[-1]) or int(H_s.shape[-1]) != int(embed_dim):
        raise ValueError(
            f"Hidden dim mismatch for layer {layer}: H_s={int(H_s.shape[-1])}, "
            f"H_l={int(H_l.shape[-1])}, expected={int(embed_dim)}"
        )

    mapped_vector = _compute_attention_mapped_steering_vector(
        H_s=H_s,
        H_l=H_l,
        W_q_weights=w_q,
        W_k_weights=w_k,
        alpha=float(alpha),
    ).reshape(-1)
    mu_diff = H_s.mean(dim=0) - H_l.mean(dim=0)
    cos_sim = F.cosine_similarity(mapped_vector, mu_diff, dim=0).item()
    print("Cosine similarity between mapped_vector and mean(H_s) - mean(H_l):", cos_sim)

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
        existing = _load_existing_steering_payload(steering_path)

        existing[layer_key] = layer_payload
        _atomic_write_json(steering_path, existing)

        non_null = sum(1 for v in layer_payload.values() if v is not None)
        print(
            f"[user_interrupt] {entry_dir.name}: wrote {steering_path.name} {layer_key} "
            f"(tokens={total_tokens}, start_idx={start_idx}, non_null={non_null})"
        )
        updated += 1

    print(f"[user_interrupt] Done. Updated attention-mapped steering vectors for {updated} items at {root_dir}")

def compute_attention_mapped_steering_vector(
    root_dir: str,
    classifier_dir: str,
    layers: list[int],
    decay_span: int,
    alpha: float,
) -> None:
    discovered = _discover_classifier_paths(classifier_dir)
    resolved_layers = _resolve_requested_layers(layers, list(discovered.keys()))
    for layer in resolved_layers:
        _compute_attention_mapped_steering_vector_single_layer(
            root_dir=root_dir,
            classifier_dir=classifier_dir,
            layer=layer,
            decay_span=decay_span,
            alpha=alpha,
        )

def compute_attention_mapped_steering_vector_average(
    root_dir: str,
    classifier_dir: str,
    target_layer: int,
    decay_span: int,
    alpha: float,
) -> None:
    """Compute the average of attention-mapped steering vectors across ALL available layers
    (using H_s/H_l loaded from the mode-class dataset under ``classifier_dir``),
    then save that single averaged vector (with decay schedule) under the key
    for ``target_layer`` in each root_dir/*/steering_vector.json.
    """
    token_rate_hz = 12.5
    root = Path(root_dir)

    if decay_span < 0:
        raise ValueError(f"decay_span must be >= 0, got {decay_span}")
    if not _is_valid_main_layer(target_layer):
        raise ValueError(
            f"target_layer {target_layer} must be in [{MAIN_LAYER_MIN}..{MAIN_LAYER_MAX}]"
        )

    discovered = _discover_classifier_paths(classifier_dir)
    all_layers = sorted(discovered.keys())
    print(
        f"[user_interrupt] Computing averaged optimized vector across "
        f"{len(all_layers)} layers: {all_layers}"
    )

    layer_vectors: list[torch.Tensor] = []
    layer_mean_diffs: list[tuple[int, torch.Tensor]] = []  # (layer, mean(H_s) - mean(H_l))

    moshi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME)
    lm = loaders.get_moshi_lm(moshi_weight, device="cpu", cpu_offload=False)
    lm.eval()

    for layer in all_layers:
        w_q, w_k, embed_dim = _extract_qk_weights_for_layer(lm, layer)
        H_s, H_l = _load_mode_class_hidden_sets(classifier_dir, layer)
        if int(H_s.shape[-1]) != int(H_l.shape[-1]) or int(H_s.shape[-1]) != int(embed_dim):
            raise ValueError(
                f"Hidden dim mismatch at layer {layer}: H_s={int(H_s.shape[-1])}, "
                f"H_l={int(H_l.shape[-1])}, expected={int(embed_dim)}"
            )
        layer_mean_diffs.append((layer, H_s.mean(dim=0) - H_l.mean(dim=0)))

        print(f"[user_interrupt] Computing mapped vector for layer {layer}...")
        # Use alpha=1.0; normalize and scale after averaging across layers.
        mapped = _compute_attention_mapped_steering_vector(
            H_s=H_s,
            H_l=H_l,
            W_q_weights=w_q,
            W_k_weights=w_k,
            alpha=1.0,
        ).reshape(-1)
        layer_vectors.append(mapped)

    if not layer_vectors:
        raise RuntimeError("No layer vectors computed; cannot average.")

    # Average the per-layer (unit-norm) vectors, renormalize, then apply alpha.
    avg_vector = torch.stack(layer_vectors).mean(dim=0)
    avg_vector = F.normalize(avg_vector, dim=0) * float(alpha)

    print("[user_interrupt] Cosine similarity between avg_vector and each layer's mean(H_s)-mean(H_l):")
    for layer_idx, mean_diff in layer_mean_diffs:
        cos_sim = F.cosine_similarity(avg_vector.unsqueeze(0), mean_diff.unsqueeze(0)).item()
        print(f"  layer {layer_idx:3d}: {cos_sim:.6f}")

    layer_key = f"layer_{_internal_layer_to_json_layer(target_layer)}"
    print(
        f"[user_interrupt] Averaged {len(layer_vectors)} layer vectors; "
        f"writing under key '{layer_key}'"
    )

    # Write decay schedule to each entry directory.
    input_paths = [p for p in root.glob("*/input.wav") if p.is_file()]
    input_paths.sort(
        key=lambda p: int(p.parent.name) if p.parent.name.isdigit() else p.parent.name
    )
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

        start_idx = max(0, min(int(interrupt_start * token_rate_hz), total_tokens - 1))

        layer_payload: dict[str, Optional[list[float]]] = {
            str(i): None for i in range(total_tokens)
        }
        layer_payload[str(start_idx)] = avg_vector.tolist()

        for k in range(1, int(decay_span) + 1):
            token_idx = start_idx + k
            if token_idx >= total_tokens:
                break
            decay_factor = 1.0 - (float(k) / float(decay_span)) if decay_span > 0 else 0.0
            if decay_factor <= 0.0:
                layer_payload[str(token_idx)] = None
                continue
            layer_payload[str(token_idx)] = (avg_vector * float(decay_factor)).tolist()

        steering_path = entry_dir / "steering_vector.json"
        existing = _load_existing_steering_payload(steering_path)
        existing[layer_key] = layer_payload
        _atomic_write_json(steering_path, existing)

        non_null = sum(1 for v in layer_payload.values() if v is not None)
        print(
            f"[user_interrupt] {entry_dir.name}: wrote averaged vector under {layer_key} "
            f"(tokens={total_tokens}, start_idx={start_idx}, non_null={non_null})"
        )
        updated += 1

    print(
        f"[user_interrupt] Done. Wrote averaged optimized steering vectors for "
        f"{updated} items at {root_dir}"
    )

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
    H_s: torch.Tensor,        # [N_s, d_model] - Speaking mode hidden states
    H_l: torch.Tensor,        # [N_l, d_model] - Listening mode hidden states
    W_q_weights: torch.Tensor, # [num_heads, head_dim, d_model] or [d_model, d_model]
    W_k_weights: torch.Tensor, # [num_heads, head_dim, d_model] or [d_model, d_model]
    rope_base: float = 10000.0,
    rope_context_len: int = 200,
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    Persona Vector Theory v4 (SVD Enhanced): Gradient Space Subspace Intersection.
    
    1. Maps H_s and H_l to Attention Gradient Space (D_s, D_l).
    2. Performs expanded PCA (q=5) to find the high-dimensional subspace of each mode.
    3. Uses SVD to find the true principal intersection (neutral direction 'n').
    4. Normalizes gradients such that their projection on 'n' is exactly 1.
    5. Subtracts normalized gradients to annihilate the neutral component.
    """
    original_device = W_q_weights.device
    compute_device = torch.device("cuda") if torch.cuda.is_available() else original_device
    dtype = torch.float32 # 確保幾何運算的精度
    
    # --- 1. 權重與維度處理 ---
    if W_q_weights.dim() == 2:
        d_model = W_q_weights.shape[0]
        head_dim = 128 
        num_heads = d_model // head_dim
        w_q = W_q_weights.reshape(num_heads, head_dim, d_model).to(device=compute_device, dtype=dtype)
        w_k = W_k_weights.reshape(num_heads, head_dim, d_model).to(device=compute_device, dtype=dtype)
    else:
        num_heads, head_dim, d_model = W_q_weights.shape
        w_q = W_q_weights.to(device=compute_device, dtype=dtype)
        w_k = W_k_weights.to(device=compute_device, dtype=dtype)

    H_s = H_s.to(device=compute_device, dtype=dtype)
    H_l = H_l.to(device=compute_device, dtype=dtype)

    # --- 2. 映射至梯度空間 (D_s, D_l) ---
    def get_attention_gradients(H_states):
        D = []
        combined_rope_map = torch.zeros(num_heads, d_model, d_model, device=compute_device, dtype=dtype)
        for n in tqdm(range(rope_context_len), desc="Computing RoPE Matrix"):
            # 假設 get_rope_matrix 在外部已定義
            r_n = get_rope_matrix(n, head_dim, rope_base, device=compute_device).to(dtype=dtype)
            for i in range(num_heads):
                combined_rope_map[i] += torch.matmul(w_q[i].T, torch.matmul(r_n, w_k[i]))
        combined_rope_map /= rope_context_len

        for h in H_states:
            grad = torch.zeros(d_model, device=compute_device, dtype=dtype)
            for i in range(num_heads):
                grad += torch.matmul(combined_rope_map[i], h)
            D.append(grad)
        return torch.stack(D) # [N, d_model]

    print("[Math Engine] Mapping activations to gradient space...")
    D_s = get_attention_gradients(H_s) # [N_s, d_model]
    D_l = get_attention_gradients(H_l) # [N_l, d_model]

    # --- 3. 擴張的 PCA 子空間提取 ---
    q_dim = 5 
    _, _, V_s = torch.pca_lowrank(D_s, q=q_dim) # V_s: [d_model, q_dim]
    _, _, V_l = torch.pca_lowrank(D_l, q=q_dim) # V_l: [d_model, q_dim]

    # --- 4. 使用 SVD 尋找真實的子空間交集 (True Subspace Intersection) ---
    M = torch.matmul(V_s.T, V_l) # [q_dim, q_dim]
    
    # SVD 解構：找出梯度空間中最完美重合的「中性常識軸」
    U, S, Vh = torch.linalg.svd(M)
    max_alignment = S[0].item()

    # 組合出真正的中性軸方向 n
    n_dir_s = torch.matmul(V_s, U[:, 0])      # [d_model]
    n_dir_l = torch.matmul(V_l, Vh[0, :])     # [d_model]
    
    # 確保符號一致並融合
    if torch.dot(n_dir_s, n_dir_l) < 0:
        n_dir_l = -n_dir_l
    n = F.normalize(n_dir_s + n_dir_l, dim=0)

    # 診斷：看看這根 n 軸是由哪些 PC 構成的
    cos_sim_Vs = torch.matmul(V_s.T, n)
    idx_s = torch.argmax(torch.abs(cos_sim_Vs)).item()
    print(f"[Math Engine] n aligns most with D_s PC{idx_s} (cos: {cos_sim_Vs[idx_s].item():.4f})")

    # --- 5. 分量歸一化與相減 (Normalization Annihilation) ---
    grad_s_avg = D_s.mean(dim=0) # [d_model]
    grad_l_avg = D_l.mean(dim=0) # [d_model]

    def normalize_on_n(v, n_vec):
        projection_len = torch.dot(v, n_vec)
        return v / projection_len

    # 在梯度空間中進行消融
    grad_s_norm = normalize_on_n(grad_s_avg, n)
    grad_l_norm = normalize_on_n(grad_l_avg, n)

    # 最終對消：獲得純淨的神經網路意圖向量
    v_star = grad_s_norm - grad_l_norm
    v_opt = F.normalize(v_star, dim=0) * float(alpha)

    # --- 6. 最終防呆與輸出 ---
    mu_s = H_s.mean(dim=0)
    mu_l = H_l.mean(dim=0)
    mu_diff = mu_s - mu_l
    cos_vopt_mu = F.cosine_similarity(v_opt, mu_diff, dim=0).item()
    
    if cos_vopt_mu > 0.0:
        v_opt = -v_opt
        cos_vopt_mu = F.cosine_similarity(v_opt, mu_diff, dim=0).item()
        print("[Math Engine] cos(v_opt, mu_s-mu_l) was positive; flipped v_opt sign.")

    print(f"[Math Engine] True Neutral axis alignment (SVD, q={q_dim}): {max_alignment:.4f}")
    print(f"[Math Engine] final cos(v_opt, mu_s-mu_l): {cos_vopt_mu:.6f}")

    return v_opt.reshape(1, -1).to(device=original_device, dtype=W_q_weights.dtype)

def _compute_attention_mapped_steering_vector_simple(
    H_s: torch.Tensor,        # [N_s, d_model] - Speaking mode hidden states
    H_l: torch.Tensor,        # [N_l, d_model] - Listening mode hidden states
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    Persona Vector Theory v5 (SVD Enhanced) with PCA Component Alignment Diagnostics.
    """
    original_device = H_s.device
    compute_device = torch.device("cuda") if torch.cuda.is_available() else original_device
    dtype = torch.float32 
    
    H_s = H_s.to(device=compute_device, dtype=dtype)
    H_l = H_l.to(device=compute_device, dtype=dtype)

    # --- 1. 擴張的 PCA 子空間提取 ---
    q_dim = 5 
    _, _, V_s = torch.pca_lowrank(H_s, q=q_dim) # V_s: [d_model, q_dim]
    _, _, V_l = torch.pca_lowrank(H_l, q=q_dim) # V_l: [d_model, q_dim]

    # --- 2. 使用 SVD 尋找真實的子空間交集 ---
    M = torch.matmul(V_s.T, V_l) # [q_dim, q_dim]
    U, S, Vh = torch.linalg.svd(M)
    max_alignment = S[0].item()

    n_dir_s = torch.matmul(V_s, U[:, 0])      
    n_dir_l = torch.matmul(V_l, Vh[0, :])     
    
    if torch.dot(n_dir_s, n_dir_l) < 0:
        n_dir_l = -n_dir_l
    n = F.normalize(n_dir_s + n_dir_l, dim=0)

    # =====================================================================
    # 🔍 診斷區塊：檢查 Neutral Axis (n) 與原始 PCA 主成分的相似度
    # =====================================================================
    # 因為 V_s, V_l 每行是單位向量，n 也是單位向量，所以內積即為 Cosine 相似度
    cos_sim_Vs = torch.matmul(V_s.T, n) # shape: [q_dim]
    cos_sim_Vl = torch.matmul(V_l.T, n) # shape: [q_dim]
    
    # 找出絕對值最大（最平行）的那根 PC 的 index
    idx_s = torch.argmax(torch.abs(cos_sim_Vs)).item()
    idx_l = torch.argmax(torch.abs(cos_sim_Vl)).item()
    
    # 取得實際的 Cosine 值（帶正負號）
    best_cos_s = cos_sim_Vs[idx_s].item()
    best_cos_l = cos_sim_Vl[idx_l].item()
    
    print(f"[Math Engine] True Neutral axis alignment (SVD, q={q_dim}): {max_alignment:.4f}")
    print(f"[Math Engine] n aligns most with V_s PC{idx_s} (cos: {best_cos_s:.4f})")
    print(f"[Math Engine] n aligns most with V_l PC{idx_l} (cos: {best_cos_l:.4f})")
    # =====================================================================

    # --- 3. 平均隱藏狀態計算 ---
    mu_s = H_s.mean(dim=0) 
    mu_l = H_l.mean(dim=0) 

    # --- 4. 分量歸一化與相減 (Normalization Annihilation) ---
    def normalize_on_n(v, n_vec):
        projection_len = torch.dot(v, n_vec)
        return v / projection_len

    mu_s_star = normalize_on_n(mu_s, n)
    mu_l_star = normalize_on_n(mu_l, n)

    v_star = mu_s_star - mu_l_star
    v_opt = F.normalize(v_star, dim=0) * float(alpha)

    # --- 5. 強制確保與 mu_s - mu_l 呈負相關/正交 ---
    mu_diff = mu_s - mu_l
    cos_vopt_mu = F.cosine_similarity(v_opt, mu_diff, dim=0).item()
    
    if cos_vopt_mu > 0.0:
        v_opt = -v_opt  
        cos_vopt_mu = F.cosine_similarity(v_opt, mu_diff, dim=0).item()
        print("[Math Engine] cos(v_opt, mu_s-mu_l) was positive; flipped v_opt sign.")

    print(f"[Math Engine] final cos(v_opt, mu_s-mu_l): {cos_vopt_mu:.6f}")

    return v_opt.reshape(1, -1).to(device=original_device, dtype=torch.float32)

def _calculate_steering_vector_single_layer(root_dir, classifier_path, decay_span, alpha):
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
    print(f"[Debug] normal vector have norm {torch.norm(normal_vector):.4f} and shape {normal_vector.shape}")
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
    # Keep legacy key schema for normal (non-attn-optimized) vectors.
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
        existing = _load_existing_steering_payload(steering_path)

        # Merge/update only the target layer while preserving other layers.
        existing[layer_key] = layer_payload
        _atomic_write_json(steering_path, existing)

        non_null = sum(1 for v in layer_payload.values() if v is not None)
        print(
            f"[user_interrupt] {entry_dir.name}: wrote {steering_path.name} {layer_key} "
            f"(tokens={total_tokens}, start_idx={start_idx}, non_null={non_null})"
        )
        updated += 1

    print(f"[user_interrupt] Done. Updated steering vectors for {updated} items at {root_dir}")

def calculate_steering_vector(
    root_dir: str,
    classifier_dir: str,
    layers: list[int],
    decay_span: int,
    alpha: float,
) -> None:
    discovered = _discover_classifier_paths(classifier_dir)
    resolved_layers = _resolve_requested_layers(layers, list(discovered.keys()))
    for layer in resolved_layers:
        _calculate_steering_vector_single_layer(
            root_dir=root_dir,
            classifier_path=discovered[layer],
            decay_span=decay_span,
            alpha=alpha,
        )


def _load_mean_hidden_diff_vectors(mean_hidden_diff_path: Path) -> dict[int, torch.Tensor]:
    """Load mean-hidden-diff vectors from JSON as {layer: tensor[D]}.

    Expected JSON format:
      {"0": [...], "1": [...], ..., "31": [...]}.
    """
    if not mean_hidden_diff_path.exists():
        raise FileNotFoundError(
            f"mean_hidden_diff.json not found: {mean_hidden_diff_path}"
        )

    with mean_hidden_diff_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected dict in {mean_hidden_diff_path}, got {type(payload).__name__}"
        )

    vectors: dict[int, torch.Tensor] = {}
    for key, value in payload.items():
        try:
            layer = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid layer key '{key}' in {mean_hidden_diff_path}; expected integer-like keys"
            ) from exc
        if not _is_valid_main_layer(layer):
            continue
        if not isinstance(value, list):
            raise ValueError(
                f"Layer {layer} in {mean_hidden_diff_path} must be a list, got {type(value).__name__}"
            )
        vec = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
        if vec.numel() == 0:
            raise ValueError(f"Layer {layer} vector is empty in {mean_hidden_diff_path}")
        vectors[layer] = vec

    if not vectors:
        raise RuntimeError(
            f"No valid layer vectors found in {mean_hidden_diff_path}. "
            f"Expected layer keys in [{MAIN_LAYER_MIN}..{MAIN_LAYER_MAX}]."
        )
    return vectors


def _calculate_steering_vector_mean_diff_single_layer(
    root_dir: str,
    mean_hidden_diff: torch.Tensor,
    layer: int,
    decay_span: int,
    alpha: Optional[float],
) -> None:
    """Generate steering vectors for one layer from mean-hidden-diff source.

    If ``alpha`` is provided, source vector is scaled directly as ``alpha * base``
    (no normalization). If ``alpha`` is None, use raw mean diff directly.
    Everything else follows ``_calculate_steering_vector_single_layer`` behavior.
    """
    token_rate_hz = 12.5
    root = Path(root_dir)
    layer = int(layer)

    if decay_span < 0:
        raise ValueError(f"decay_span must be >= 0, got {decay_span}")
    if not _is_valid_main_layer(layer):
        raise ValueError(f"Invalid layer {layer}; expected [{MAIN_LAYER_MIN}..{MAIN_LAYER_MAX}]")

    base = torch.as_tensor(mean_hidden_diff, dtype=torch.float32).reshape(-1)
    if base.numel() == 0:
        raise ValueError(f"Layer {layer} mean-hidden-diff vector is empty")
    base_norm = torch.norm(base).item()
    print(f"[user_interrupt] layer {layer} mean-diff base norm ||base||={base_norm:.6f}")
    if alpha is None:
        if base_norm <= 0.0:
            raise ValueError(f"Layer {layer} mean-hidden-diff vector has zero norm")
        # Raw mean-diff mode uses base vector directly.
        base_vector = base
    else:
        if base_norm <= 0.0:
            raise ValueError(f"Layer {layer} mean-hidden-diff vector has zero norm")
        # Scale base vector directly by alpha (no normalization).
        base_vector = base * float(alpha)
    layer_key = f"layer_{layer}"

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

        layer_payload[str(start_idx)] = base_vector.tolist()

        for k in range(1, int(decay_span) + 1):
            token_idx = start_idx + k
            if token_idx >= total_tokens:
                break
            decay_factor = 1.0 - (float(k) / float(decay_span)) if decay_span > 0 else 0.0
            if decay_factor <= 0.0:
                layer_payload[str(token_idx)] = None
                continue
            vec = (base_vector * float(decay_factor)).tolist()
            layer_payload[str(token_idx)] = vec

        steering_path = entry_dir / "steering_vector.json"
        existing = _load_existing_steering_payload(steering_path)
        existing[layer_key] = layer_payload
        _atomic_write_json(steering_path, existing)

        non_null = sum(1 for v in layer_payload.values() if v is not None)
        print(
            f"[user_interrupt] {entry_dir.name}: wrote {steering_path.name} {layer_key} "
            f"(tokens={total_tokens}, start_idx={start_idx}, non_null={non_null})"
        )
        updated += 1

    print(
        f"[user_interrupt] Done. Updated mean-diff steering vectors for layer {layer} "
        f"in {updated} items at {root_dir}"
    )


def calculate_steering_vector_mean_diff(
    root_dir: str,
    classifier_dir: str,
    layers: list[int],
    decay_span: int,
    alpha: Optional[float],
) -> None:
    """Generate steering vectors from classifier_dir/mean_hidden_diff.json.

    If ``alpha`` is provided, diff vectors are directly scaled as ``alpha * base``
    (no normalization). If ``alpha`` is omitted, raw mean diff is used.
    """
    mean_hidden_diff_path = Path(classifier_dir) / "mean_hidden_diff.json"
    vectors = _load_mean_hidden_diff_vectors(mean_hidden_diff_path)
    resolved_layers = _resolve_requested_layers(layers, list(vectors.keys()))

    expected_dim: Optional[int] = None
    for layer in resolved_layers:
        vec = vectors[layer]
        if expected_dim is None:
            expected_dim = int(vec.numel())
        elif int(vec.numel()) != int(expected_dim):
            raise ValueError(
                f"Vector dim mismatch in {mean_hidden_diff_path}: "
                f"layer {layer} has {int(vec.numel())}, expected {expected_dim}"
            )
        _calculate_steering_vector_mean_diff_single_layer(
            root_dir=root_dir,
            mean_hidden_diff=vec,
            layer=layer,
            decay_span=decay_span,
            alpha=alpha,
        )

def inference_with_steering(
        root_dir,
        inject_layers,
    offset=0,
        save_hidden=False,
        steer_attn_only: bool = False,
        resume: int = 0,
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

    input_paths = _apply_resume_index(input_paths, resume, root_dir)

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

    requested_layers = [int(x) for x in inject_layers]
    legacy_single_layer = (
        len(requested_layers) == 1
        and int(requested_layers[0]) != -1
        and not bool(steer_attn_only)
    )

    def _extract_layer_vector_legacy(
        raw: dict,
        layer: int,
        offset: int = 0,
    ) -> list[Optional[torch.Tensor]]:
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

        token_entries: dict[int, Optional[torch.Tensor]] = {}
        max_idx = 0
        for token_key, token_vec in layer_payload.items():
            try:
                token_idx = int(token_key)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Token index must be an integer-like key, got '{token_key}'"
                ) from exc
            if token_idx < 0:
                raise ValueError(f"Token indices must be >= 0, got {token_idx}")
            shifted_idx = token_idx + int(offset)
            if shifted_idx < 0:
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

    def _extract_layer_vectors(
        raw: dict,
        layers: list[int],
        offset: int = 0,
    ) -> dict[int, list[Optional[torch.Tensor]]]:
        if not isinstance(raw, dict):
            raise ValueError(f"Expected dict steering payload, got {type(raw)}")

        available_map: dict[int, object] = {}
        for k, v in raw.items():
            if isinstance(k, str):
                m = re.fullmatch(r"layer_(-?\d+)", k)
                if m is not None:
                    layer = _json_layer_to_internal_layer(int(m.group(1)))
                    if layer is not None:
                        available_map[layer] = v
                    continue
            try:
                layer = _json_layer_to_internal_layer(int(k))
                if layer is not None:
                    available_map[layer] = v
            except (TypeError, ValueError):
                continue

        if -1 in layers:
            resolved = sorted(available_map.keys())
        else:
            resolved = [int(x) for x in layers if _is_valid_main_layer(int(x))]

        invalid_requested = [int(x) for x in layers if int(x) != -1 and not _is_valid_main_layer(int(x))]
        if invalid_requested:
            raise ValueError(
                f"Invalid requested layer(s): {sorted(set(invalid_requested))}. Supported range is [{MAIN_LAYER_MIN}..{MAIN_LAYER_MAX}] or -1."
            )

        if not resolved:
            raise KeyError("No layer information found in steering_vector.json")

        missing = [x for x in resolved if x not in available_map]
        if missing:
            raise KeyError(
                f"Requested layer(s) {missing} missing in steering_vector.json. "
                f"Available layers: {sorted(available_map.keys())}"
            )

        out: dict[int, list[Optional[torch.Tensor]]] = {}
        for layer in resolved:
            layer_payload = available_map[layer]
            if not isinstance(layer_payload, dict):
                raise ValueError(
                    f"Expected dict for layer payload at layer {layer}, got {type(layer_payload)}"
                )

            token_entries: dict[int, Optional[torch.Tensor]] = {}
            max_idx = 0
            for token_key, token_vec in layer_payload.items():
                try:
                    token_idx = int(token_key)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Token index must be an integer-like key, got '{token_key}'"
                    ) from exc
                if token_idx < 0:
                    raise ValueError(f"Token indices must be >= 0, got {token_idx}")
                shifted_idx = token_idx + int(offset)
                if shifted_idx < 0:
                    continue
                if token_vec is None:
                    token_entries[shifted_idx] = None
                else:
                    token_entries[shifted_idx] = torch.as_tensor(token_vec, dtype=torch.float32).reshape(-1)
                max_idx = max(max_idx, shifted_idx)

            if len(token_entries) == 0:
                raise ValueError(
                    f"Layer {layer} payload has no usable steering vectors after applying offset={offset}"
                )

            vectors: list[Optional[torch.Tensor]] = [None] * (max_idx + 1)
            for token_idx, token_vec in token_entries.items():
                vectors[token_idx] = token_vec
            out[int(layer)] = vectors
        return out

    print(
        f"[user_interrupt] Processing {len(input_paths)} files from {root_dir} "
        f"with steering layers {requested_layers}, offset {offset}, "
        f"steer_attn_only={steer_attn_only}"
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

        if legacy_single_layer:
            inject_layer = int(requested_layers[0])
            steering_vectors = _extract_layer_vector_legacy(
                steering_payload,
                inject_layer,
                int(offset),
            )
        else:
            steering_vectors_by_layer = _extract_layer_vectors(
                steering_payload,
                requested_layers,
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
        if legacy_single_layer:
            if len(steering_vectors) < min_tokens:
                steering_vectors.extend([None] * (min_tokens - len(steering_vectors)))
            non_null = sum(1 for v in steering_vectors if v is not None)
            print(
                f"[user_interrupt] {entry_dir.name}: loaded steering vectors len={len(steering_vectors)}, "
                f"min_tokens={min_tokens}, non_null={non_null}"
            )
        else:
            for layer, vectors in steering_vectors_by_layer.items():
                if len(vectors) < min_tokens:
                    vectors.extend([None] * (min_tokens - len(vectors)))
                non_null = sum(1 for v in vectors if v is not None)
                print(
                    f"[user_interrupt] {entry_dir.name}: layer={layer} loaded steering vectors len={len(vectors)}, "
                    f"min_tokens={min_tokens}, non_null={non_null}"
                )

        with torch.no_grad():
            if legacy_single_layer:
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
                    steering_layer=int(requested_layers[0]),
                )
            else:
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
                    steering_vectors_by_layer=steering_vectors_by_layer,
                    steer_attn_only=bool(steer_attn_only),
                )

    if save_hidden:
        print(f"[user_interrupt] Done. Wrote output.wav/output.json/output_hidden.pt for {len(input_paths)} items.")
    else:
        print(f"[user_interrupt] Done. Wrote output.wav/output.json for {len(input_paths)} items.")

def inference_force_pad(
    root_dir: str,
    num_pad: int,
    save_hidden: bool = False,
    payload_target_layer: Optional[int] = None,
    resume: int = 0,
) -> None:
    """Run inference while force-overriding AR feedback tokens around interruption.

    For each ``root_dir/*`` item, read ``input_timing.json`` and use
    ``interrupt_start`` to compute token step index at 12.5 Hz. Starting at that
    step, for ``num_pad`` steps we replace the model's previous-step feedback
    tokens with:
    - text: PAD token (model text padding token id)
    - audio: silence tokens (Moshi fixed silence codebook ids)
    """
    if int(num_pad) <= 0:
        raise ValueError(f"num_pad must be > 0, got {num_pad}")

    token_rate_hz = 12.5
    root = Path(root_dir)
    input_paths = [p for p in root.glob("*/input.wav") if p.is_file()]
    input_paths.sort(key=lambda p: int(p.parent.name) if p.parent.name.isdigit() else p.parent.name)

    if not input_paths:
        raise FileNotFoundError(f"No files matched pattern {root_dir}/*/input.wav")

    input_paths = _apply_resume_index(input_paths, resume, root_dir)

    force_pad_start_steps: list[Optional[int]] = []
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
        force_start_idx = max(0, int(interrupt_start * token_rate_hz))
        force_pad_start_steps.append(force_start_idx)

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

    print(
        "[user_interrupt] Force-pad mode: "
        f"num_pad={int(num_pad)}, text_pad_token=3, silence_tokens={SILENCE_TOKENS.tolist()}"
    )
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
            force_pad_start_steps=force_pad_start_steps,
            force_pad_num_steps=int(num_pad),
        )

    if save_hidden:
        print(
            f"[user_interrupt] Done. Wrote {len(output_wavs)} output.wav files and "
            f"{len(output_hiddens)} output_hidden.pt files."
        )
    else:
        print(f"[user_interrupt] Done. Wrote {len(output_wavs)} output.wav files.")




def scale_steering_vectors(root_dir: str, scale: float) -> None:
    """Normalize then scale steering vectors under root_dir/*/steering_vector.json.

    For each steering JSON file:
    - find the maximum L2 norm across all non-None vectors,
    - divide every non-None vector by that max norm (so max norm becomes 1),
    - multiply by ``scale``.

    None entries remain None.
    """
    root = Path(root_dir)
    steering_paths = [p for p in root.glob("*/steering_vector.json") if p.is_file()]
    steering_paths.sort(key=lambda p: int(p.parent.name) if p.parent.name.isdigit() else p.parent.name)
    if not steering_paths:
        raise FileNotFoundError(f"No files matched pattern {root_dir}/*/steering_vector.json")

    updated = 0
    factor = float(scale)
    for steering_path in steering_paths:
        with steering_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected dict in {steering_path}, got {type(payload)}")

        # Pass 1: compute max L2 norm among all non-None vectors in this file.
        max_norm = 0.0
        for layer_key, layer_payload in payload.items():
            if not isinstance(layer_payload, dict):
                raise ValueError(
                    f"Expected dict for layer payload at key '{layer_key}' in {steering_path}, got {type(layer_payload)}"
                )
            for token_key, token_vec in layer_payload.items():
                if token_vec is None:
                    continue
                if not isinstance(token_vec, list):
                    raise ValueError(
                        f"Expected list or None for token '{token_key}' under '{layer_key}' in {steering_path}, got {type(token_vec)}"
                    )
                norm = math.sqrt(sum(float(x) * float(x) for x in token_vec))
                if norm > max_norm:
                    max_norm = norm

        # Pass 2: normalize by max_norm and then multiply by scale.
        denom = max_norm if max_norm > 0.0 else 1.0

        for layer_key, layer_payload in payload.items():
            for token_key, token_vec in layer_payload.items():
                if token_vec is None:
                    continue
                layer_payload[token_key] = [(float(x) / denom) * factor for x in token_vec]

        _atomic_write_json(steering_path, payload)

        print(
            f"[user_interrupt] {steering_path.parent.name}: normalized by max_norm={max_norm:.6g} "
            f"then scaled by {factor}"
        )
        updated += 1

    print(
        f"[user_interrupt] Done. Normalized+scaled steering vectors in {updated} files at {root_dir}; "
        f"target max norm per file = {factor}"
    )


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
        "--generate-steering-vectors-mean-diff",
        action="store_true",
        help=(
            "Generate/Update root-dir/*/steering_vector.json from classifier-dir/mean_hidden_diff.json. "
            "Each layer vector is directly scaled by --alpha (no normalization)."
        ),
    )
    mode_group.add_argument(
        "--generate-steering-vectors-optimized",
        action="store_true",
        help="Generate/Update root-dir/*/steering_vector.json using attention-optimized steering vectors.",
    )
    mode_group.add_argument(
        "--generate-steering-vectors-optimized-average",
        action="store_true",
        help=(
            "Compute the average of attention-mapped vectors across ALL available layers in "
            "--classifier-dir, then save the averaged vector under the --layer key in "
            "root-dir/*/steering_vector.json. --layer must be a single integer."
        ),
    )
    mode_group.add_argument(
        "--inference-with-steering",
        action="store_true",
        help="Run inference using steering vectors loaded from root-dir/*/steering_vector.json.",
    )
    mode_group.add_argument(
        "--inference-force-pad",
        action="store_true",
        help="Run inference without steering, but replace AR feedback with PAD/silence around interrupt_start.",
    )
    mode_group.add_argument(
        "--scale-steering-vectors",
        action="store_true",
        help="Scale all vectors in root-dir/*/steering_vector.json by --scale (None stays None).",
    )

    parser.add_argument(
        "--classifier-dir",
        type=str,
        default=None,
        help=(
            "Directory containing classifier checkpoints named "
            "hidden_mode_classifier_layer_<layer>.pt. "
            "Required for steering-vector generation modes."
        ),
    )
    parser.add_argument(
        "--classifier-path",
        type=str,
        default=None,
        help=(
            "Legacy path to a single mode-classifier checkpoint. "
            "If provided with --generate-steering-vectors, uses f6f-compatible single-layer generation logic."
        ),
    )
    parser.add_argument(
        "--layer",
        type=int,
        nargs="+",
        default=[-1],
        help=(
            "Target layer indices. Use multiple values for multi-layer operation. "
            "Use -1 to select all available layers."
        ),
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
        default=None,
        help=(
            "Steering strength multiplier. For --generate-steering-vectors-mean-diff: "
            "if provided, use alpha * mean_diff (no normalization); if omitted, use raw mean diff. For other "
            "steering generation modes, defaults to 0.05 when omitted."
        ),
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale factor for --scale-steering-vectors (1.0 keeps values unchanged).",
    )
    parser.add_argument(
        "--inject-layer",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Layer indices to inject steering vectors during --inference-with-steering. "
            "Use multiple values for multi-layer injection; use -1 for all layers found in steering_vector.json."
        ),
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Shift steering injection target from token i to i+offset (can be negative).",
    )
    parser.add_argument(
        "--resume",
        type=int,
        default=0,
        help=(
            "0-based dataset index to resume from for --inference and --inference-with-steering. "
            "Example: --resume 10 skips first 10 files and starts at index 10 (the 11th file)."
        ),
    )
    parser.add_argument(
        "--steer-attn-only",
        action="store_true",
        help=(
            "If set with steering inference, inject steering right before self-attention and "
            "subtract it before FFN so only attention path is affected."
        ),
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
    parser.add_argument(
        "--num-pad",
        type=int,
        default=0,
        help="Number of token steps to force PAD/silence after interrupt_start when using --inference-force-pad.",
    )

    args = parser.parse_args()

    if args.generate_steering_vectors:
        if args.classifier_path is not None:
            _calculate_steering_vector_single_layer(
                root_dir=args.root_dir,
                classifier_path=str(args.classifier_path),
                decay_span=args.decay_span,
                alpha=0.05 if args.alpha is None else float(args.alpha),
            )
            return

        if args.classifier_dir is None:
            parser.error("--generate-steering-vectors requires --classifier-dir or --classifier-path")
        calculate_steering_vector(
            root_dir=args.root_dir,
            classifier_dir=args.classifier_dir,
            layers=[int(x) for x in args.layer],
            decay_span=args.decay_span,
            alpha=0.05 if args.alpha is None else float(args.alpha),
        )
        return

    if args.generate_steering_vectors_mean_diff:
        if args.classifier_dir is None:
            parser.error("--generate-steering-vectors-mean-diff requires --classifier-dir")
        calculate_steering_vector_mean_diff(
            root_dir=args.root_dir,
            classifier_dir=args.classifier_dir,
            layers=[int(x) for x in args.layer],
            decay_span=args.decay_span,
            alpha=None if args.alpha is None else float(args.alpha),
        )
        return

    if args.generate_steering_vectors_optimized:
        if args.classifier_dir is None:
            parser.error("--generate-steering-vectors-optimized requires --classifier-dir")
        compute_attention_mapped_steering_vector(
            root_dir=args.root_dir,
            classifier_dir=args.classifier_dir,
            layers=[int(x) for x in args.layer],
            decay_span=args.decay_span,
            alpha=0.05 if args.alpha is None else float(args.alpha),
        )
        return

    if args.generate_steering_vectors_optimized_average:
        if args.classifier_dir is None:
            parser.error("--generate-steering-vectors-optimized-average requires --classifier-dir")
        if len(args.layer) != 1:
            parser.error(
                "--generate-steering-vectors-optimized-average requires exactly one --layer value "
                "specifying where to save the averaged vector"
            )
        compute_attention_mapped_steering_vector_average(
            root_dir=args.root_dir,
            classifier_dir=args.classifier_dir,
            target_layer=int(args.layer[0]),
            decay_span=args.decay_span,
            alpha=0.05 if args.alpha is None else float(args.alpha),
        )
        return

    if args.inference_with_steering:
        if args.inject_layer is None:
            parser.error("--inference-with-steering requires --inject-layer")
        inference_with_steering(
            root_dir=args.root_dir,
            inject_layers=[int(x) for x in args.inject_layer],
            offset=args.offset,
            save_hidden=args.save_hidden,
            steer_attn_only=bool(args.steer_attn_only),
            resume=args.resume,
        )
        return

    if args.scale_steering_vectors:
        scale_steering_vectors(
            root_dir=args.root_dir,
            scale=float(args.scale),
        )
        return

    if args.inference_force_pad:
        if int(args.num_pad) <= 0:
            parser.error("--inference-force-pad requires --num-pad > 0")
        inference_force_pad(
            root_dir=args.root_dir,
            num_pad=int(args.num_pad),
            save_hidden=args.save_hidden,
            payload_target_layer=args.payload_target_layer,
            resume=args.resume,
        )
        return

    inference(
        args.root_dir,
        save_hidden=args.save_hidden,
        payload_target_layer=args.payload_target_layer,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
