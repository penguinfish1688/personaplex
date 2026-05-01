"""False-injection robustness utilities.

This script supports two workflows:
1) Generate false-trigger steering schedules under ``root_dir/*/steering_vector.json``:
   - The steering vector is the mean-hidden-diff vector computed from a mode-class dataset
     (same source as ``user_interrupt.py --generate-steering-vectors-mean-diff``)
   - Vector is multiplied by ``alpha`` and injected at ``layer``
   - Randomly inject false signals with the requested expectation interval in seconds
     (e.g. 20 means one signal every 20s; -1 disables false signals)
2) Evaluate generated outputs (0-5 relatedness score) and save summary to:
   ``root_dir/false_injection_{layer}_{expectation}.json``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch
from openai import OpenAI

from moshi.offline import run_batch_inference, _get_voice_prompt_dir
from moshi.models import loaders


TOKEN_RATE_HZ = 12.5
MAIN_LAYER_MIN = 0
MAIN_LAYER_MAX = 31


SYSTEM_PROMPT = (
  "You are an intelligent, articulate, and highly factual AI assistant. "
  "When the user asks open-ended questions, provide detailed, natural, and comprehensive explanations, and talk for a long time."
  "However, you also act as a strict fact-checker. If the user interrupts you or makes a factual claim "
  "(e.g., 'A banana is a red fruit, right?'), you must prioritize truth over politeness. "
  "If their claim is TRUE, confirm it and teach user more about the topic."
  "If their claim is FALSE, you must immediately reject it by saying 'No' or 'False', and teach user the correct information."
  "Never agree with incorrect information just to be polite."
)


def _is_valid_main_layer(layer: int) -> bool:
  return MAIN_LAYER_MIN <= int(layer) <= MAIN_LAYER_MAX


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent), delete=False) as tf:
    json.dump(payload, tf, indent=2, ensure_ascii=False)
    tf.flush()
    os.fsync(tf.fileno())
    tmp_name = tf.name
  os.replace(tmp_name, path)


def _load_existing_steering_payload(steering_path: Path) -> dict[str, Any]:
  if not steering_path.exists():
    return {}
  try:
    with steering_path.open("r", encoding="utf-8") as f:
      payload = json.load(f)
  except json.JSONDecodeError as exc:
    raise ValueError(f"Malformed JSON at {steering_path}: {exc}") from exc

  if not isinstance(payload, dict):
    raise ValueError(f"Expected dict in {steering_path}, got {type(payload)}")
  return payload


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


def _format_expectation_tag(expectation: float) -> str:
  if expectation == -1.0:
    return "-1"
  text = f"{float(expectation):.6f}".rstrip("0").rstrip(".")
  return text or "0"


def _build_injection_mask(total_tokens: int, expectation: float, seed_key: str) -> list[bool]:
  if expectation == -1.0:
    return [False] * total_tokens
  if expectation <= 0.0:
    raise ValueError(f"expectation must be > 0 or -1, got {expectation}")

  probability = min(1.0, 1.0 / max(1.0, float(expectation) * TOKEN_RATE_HZ))
  digest = hashlib.sha256(seed_key.encode("utf-8")).digest()
  seed = int.from_bytes(digest[:8], "little", signed=False)
  generator = torch.Generator(device="cpu")
  generator.manual_seed(seed)
  return (torch.rand(total_tokens, generator=generator) < probability).tolist()


def _extract_all_layer_hidden_from_payload(payload: dict[str, Any]) -> torch.Tensor:
  if "text_hidden_layers" in payload:
    hidden = payload["text_hidden_layers"]
    if not isinstance(hidden, torch.Tensor):
      hidden = torch.as_tensor(hidden)
    if hidden.ndim != 3:
      raise ValueError(f"Expected text_hidden_layers [T,L,D], got shape {tuple(hidden.shape)}")
    return hidden.float()

  if "hidden_states" in payload:
    hidden = payload["hidden_states"]
    if not isinstance(hidden, torch.Tensor):
      hidden = torch.as_tensor(hidden)
    if hidden.ndim == 2:
      hidden = hidden.unsqueeze(1)
    if hidden.ndim != 3:
      raise ValueError(f"Expected hidden_states [T,L,D] or [T,D], got shape {tuple(hidden.shape)}")
    return hidden.float()

  raise KeyError("Payload has neither 'text_hidden_layers' nor 'hidden_states'")


def _load_mean_hidden_diff_vectors(mean_hidden_diff_path: Path) -> dict[int, torch.Tensor]:
  with mean_hidden_diff_path.open("r", encoding="utf-8") as f:
    payload = json.load(f)
  if not isinstance(payload, dict):
    raise ValueError(f"Expected dict in {mean_hidden_diff_path}, got {type(payload).__name__}")

  vectors: dict[int, torch.Tensor] = {}
  for key, value in payload.items():
    try:
      layer = int(key)
    except (TypeError, ValueError) as exc:
      raise ValueError(f"Invalid layer key '{key}' in {mean_hidden_diff_path}") from exc
    if not _is_valid_main_layer(layer):
      continue
    if not isinstance(value, list):
      raise ValueError(f"Layer {layer} vector in {mean_hidden_diff_path} must be a list")
    vec = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
    if vec.numel() == 0:
      raise ValueError(f"Layer {layer} vector is empty in {mean_hidden_diff_path}")
    vectors[layer] = vec
  if not vectors:
    raise ValueError(f"No valid layer vectors found in {mean_hidden_diff_path}")
  return vectors


def _compute_mean_hidden_diff_vectors_from_mode_class_dataset(classifier_dir: str) -> dict[int, torch.Tensor]:
  base = Path(classifier_dir)
  if not base.is_dir():
    raise FileNotFoundError(f"classifier_dir not found: {classifier_dir}")

  sample_dirs = [p for p in base.iterdir() if p.is_dir()]
  sample_dirs.sort(key=lambda p: int(p.name) if p.name.isdigit() else p.name)
  valid_dirs = [
    sd for sd in sample_dirs
    if (sd / "output_hidden.pt").is_file() and (sd / "input_timing.json").is_file()
  ]
  if not valid_dirs:
    raise FileNotFoundError(
      f"No valid mode-class samples in {classifier_dir}. Expected */output_hidden.pt and */input_timing.json"
    )

  listen_sum: Optional[torch.Tensor] = None
  speak_sum: Optional[torch.Tensor] = None
  listen_count = 0
  speak_count = 0

  for sd in valid_dirs:
    payload = torch.load(sd / "output_hidden.pt", map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
      raise TypeError(f"Expected dict payload in {sd / 'output_hidden.pt'}, got {type(payload).__name__}")
    hidden = _extract_all_layer_hidden_from_payload(payload)
    frame_rate = float(payload.get("frame_rate", TOKEN_RATE_HZ))
    if frame_rate <= 0.0:
      raise ValueError(f"Invalid frame_rate in {sd / 'output_hidden.pt'}: {frame_rate}")

    with (sd / "input_timing.json").open("r", encoding="utf-8") as f:
      timing_payload = json.load(f)
    if not isinstance(timing_payload, dict):
      raise ValueError(f"Expected dict in {sd / 'input_timing.json'}, got {type(timing_payload).__name__}")
    if "question_start" not in timing_payload or "question_end" not in timing_payload:
      raise KeyError(f"Missing question_start/question_end in {sd / 'input_timing.json'}")

    question_start = float(timing_payload["question_start"])
    question_end = float(timing_payload["question_end"])
    if question_end <= question_start:
      raise ValueError(f"Expected question_end > question_start in {sd / 'input_timing.json'}")

    times = torch.arange(int(hidden.shape[0]), dtype=torch.float32) / frame_rate
    listen_mask = (times > (question_start + 1.0)) & (times < (question_end - 1.0))
    speak_mask = (times > (question_end + 1.0)) & (times < (question_end + 11.0))

    if int(listen_mask.sum()) > 0:
      chunk = hidden[listen_mask].sum(dim=0).cpu()
      if listen_sum is None:
        listen_sum = torch.zeros_like(chunk)
      if listen_sum.shape != chunk.shape:
        raise ValueError(f"Listening hidden shape mismatch in {sd}")
      listen_sum += chunk
      listen_count += int(listen_mask.sum())

    if int(speak_mask.sum()) > 0:
      chunk = hidden[speak_mask].sum(dim=0).cpu()
      if speak_sum is None:
        speak_sum = torch.zeros_like(chunk)
      if speak_sum.shape != chunk.shape:
        raise ValueError(f"Speaking hidden shape mismatch in {sd}")
      speak_sum += chunk
      speak_count += int(speak_mask.sum())

  if listen_sum is None or speak_sum is None or listen_count <= 0 or speak_count <= 0:
    raise RuntimeError(
      f"Insufficient labeled tokens in mode-class dataset: listening={listen_count} speaking={speak_count}"
    )

  diff = (speak_sum / float(speak_count)) - (listen_sum / float(listen_count))
  vectors: dict[int, torch.Tensor] = {}
  for layer in range(int(diff.shape[0])):
    if _is_valid_main_layer(layer):
      vectors[layer] = diff[layer].reshape(-1).float()
  if not vectors:
    raise RuntimeError(f"No valid layers found from computed mean diff. Computed layers={int(diff.shape[0])}")
  return vectors


def _load_or_compute_mean_diff_vector(classifier_dir: str, layer: int) -> torch.Tensor:
  mean_hidden_diff_path = Path(classifier_dir) / "mean_hidden_diff.json"
  if mean_hidden_diff_path.exists():
    vectors = _load_mean_hidden_diff_vectors(mean_hidden_diff_path)
    print(f"[false_injection] Loaded mean-hidden-diff vectors from {mean_hidden_diff_path}")
  else:
    vectors = _compute_mean_hidden_diff_vectors_from_mode_class_dataset(classifier_dir)
    serializable = {str(k): v.tolist() for k, v in vectors.items()}
    _atomic_write_json(mean_hidden_diff_path, serializable)
    print(
      f"[false_injection] Computed mean-hidden-diff vectors from mode-class hidden states and wrote {mean_hidden_diff_path}"
    )
  if int(layer) not in vectors:
    raise FileNotFoundError(
      f"Layer {layer} mean-hidden-diff vector not found in {classifier_dir}. "
      f"Available layers: {sorted(vectors.keys())}"
    )
  return vectors[int(layer)].reshape(-1).float()


def generate_random_vector(
  root_dir: str,
  layer: int,
  expectation: float,
  alpha: float,
  decay_span: int,
  classifier_dir: Optional[str],
) -> None:
  if not _is_valid_main_layer(layer):
    raise ValueError(f"layer must be in [{MAIN_LAYER_MIN}..{MAIN_LAYER_MAX}], got {layer}")
  if expectation != -1.0 and expectation <= 0.0:
    raise ValueError(f"expectation must be > 0 or -1, got {expectation}")
  if decay_span < 0:
    raise ValueError(f"decay_span must be >= 0, got {decay_span}")
  if classifier_dir is None:
    raise ValueError("--generate-random-vector requires --classifier-dir")

  mean_diff = _load_or_compute_mean_diff_vector(classifier_dir, int(layer))
  base_vector = mean_diff * float(alpha)
  if base_vector.numel() == 0:
    raise ValueError(f"Layer {layer} mean-hidden-diff vector is empty")
  print(
    f"[false_injection] layer={layer} alpha={alpha} decay_span={decay_span} "
    f"||mean_diff||={torch.norm(mean_diff).item():.6f} ||vector||={torch.norm(base_vector).item():.6f}"
  )

  root = Path(root_dir)
  input_paths = [p for p in root.glob("*/input.wav") if p.is_file()]
  input_paths.sort(key=lambda p: int(p.parent.name) if p.parent.name.isdigit() else p.parent.name)
  if not input_paths:
    raise FileNotFoundError(f"No files matched pattern {root_dir}/*/input.wav")

  layer_key = f"layer_{int(layer)}"
  expectation_tag = _format_expectation_tag(expectation)

  updated = 0
  for input_wav in input_paths:
    duration_s = _wav_duration_seconds(input_wav)
    total_tokens = int(math.ceil(duration_s * TOKEN_RATE_HZ))
    if total_tokens <= 0:
      raise ValueError(
        f"Computed non-positive token count for {input_wav}: duration={duration_s:.6f}s"
      )

    inject_mask = _build_injection_mask(
      total_tokens=total_tokens,
      expectation=float(expectation),
      seed_key=f"{input_wav.parent.name}:{layer}:{expectation_tag}",
    )
    layer_payload: dict[str, Optional[list[float]]] = {
      str(tok_idx): None for tok_idx in range(total_tokens)
    }
    injected_count = 0
    for tok_idx in range(total_tokens):
      if inject_mask[tok_idx]:
        layer_payload[str(tok_idx)] = base_vector.tolist()
        injected_count += 1
        for k in range(1, int(decay_span) + 1):
          next_idx = tok_idx + k
          if next_idx >= total_tokens:
            break
          decay_factor = 1.0 - (float(k) / float(decay_span)) if decay_span > 0 else 0.0
          if decay_factor > 0.0:
            layer_payload[str(next_idx)] = (base_vector * decay_factor).tolist()

    steering_path = input_wav.parent / "steering_vector.json"
    existing = _load_existing_steering_payload(steering_path)
    existing[layer_key] = layer_payload
    _atomic_write_json(steering_path, existing)

    print(
      f"[false_injection] {input_wav.parent.name}: wrote {steering_path.name} {layer_key} "
      f"(tokens={total_tokens}, triggers={injected_count}, non_null={sum(v is not None for v in layer_payload.values())}, "
      f"expectation={expectation_tag}s)"
    )
    updated += 1

  print(
    f"[false_injection] Done. Updated random steering vectors for {updated} items at {root_dir}"
  )


def _extract_layer_vector_legacy(raw: dict[str, Any], layer: int) -> list[Optional[torch.Tensor]]:
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
    raise KeyError(f"Layer {layer} not found in steering file. Available keys: {available}")
  if not isinstance(layer_payload, dict):
    raise ValueError(f"Expected dict for layer payload at layer {layer}, got {type(layer_payload)}")

  token_entries: dict[int, Optional[torch.Tensor]] = {}
  max_idx = 0
  for token_key, token_vec in layer_payload.items():
    try:
      token_idx = int(token_key)
    except (TypeError, ValueError) as exc:
      raise ValueError(f"Token index must be an integer-like key, got '{token_key}'") from exc
    if token_idx < 0:
      raise ValueError(f"Token indices must be >= 0, got {token_idx}")
    if token_vec is None:
      token_entries[token_idx] = None
    else:
      token_entries[token_idx] = torch.as_tensor(token_vec, dtype=torch.float32).reshape(-1)
    max_idx = max(max_idx, token_idx)

  if len(token_entries) == 0:
    raise ValueError("Layer payload has no usable steering vectors")

  vectors: list[Optional[torch.Tensor]] = [None] * (max_idx + 1)
  for token_idx, token_vec in token_entries.items():
    vectors[token_idx] = token_vec
  return vectors


def inference_with_steering(root_dir: str, layer: int, expectation: float) -> None:
  if not _is_valid_main_layer(layer):
    raise ValueError(f"layer must be in [{MAIN_LAYER_MIN}..{MAIN_LAYER_MAX}], got {layer}")

  root = Path(root_dir)
  input_paths = [p for p in root.glob("*/input.wav") if p.is_file()]
  input_paths.sort(key=lambda p: int(p.parent.name) if p.parent.name.isdigit() else p.parent.name)
  if not input_paths:
    raise FileNotFoundError(f"No files matched pattern {root_dir}/*/input.wav")

  # Use the same live steering file as normal steering inference. Each scan
  # parameter overwrites this file before inference; aggregate eval files carry
  # the parameterized names.
  for p in input_paths:
    steering_json = p.parent / "steering_vector.json"
    if not steering_json.exists():
      raise FileNotFoundError(
        f"Missing required steering file for inference: {steering_json}. "
        "Run --generate-random-vector with matching --layer/--expectation first."
      )

  voice_prompt_dir = _get_voice_prompt_dir(None, loaders.DEFAULT_REPO)
  if voice_prompt_dir is None:
    raise FileNotFoundError("Unable to resolve voice prompt directory.")
  voice_prompt_path = os.path.join(voice_prompt_dir, "NATF0.pt")
  if not os.path.exists(voice_prompt_path):
    raise FileNotFoundError(f"Voice prompt not found: {voice_prompt_path}")

  for path in input_paths:
    entry_dir = path.parent
    input_wav = str(path)
    output_wav = str(entry_dir / "output.wav")
    output_text = str(entry_dir / "output.json")
    steering_json = entry_dir / "steering_vector.json"

    with steering_json.open("r", encoding="utf-8") as f:
      steering_payload = json.load(f)
    if not isinstance(steering_payload, dict):
      raise ValueError(f"Expected dict in {steering_json}, got {type(steering_payload)}")

    steering_vectors = _extract_layer_vector_legacy(steering_payload, int(layer))

    try:
      with wave.open(input_wav, "rb") as wf:
        duration_s = float(wf.getnframes()) / float(wf.getframerate())
    except wave.Error:
      import soundfile as sf

      info = sf.info(input_wav)
      duration_s = float(info.frames) / float(info.samplerate)
    min_tokens = int(math.ceil(duration_s * TOKEN_RATE_HZ))
    if len(steering_vectors) < min_tokens:
      steering_vectors.extend([None] * (min_tokens - len(steering_vectors)))

    non_null = sum(1 for v in steering_vectors if v is not None)
    print(
      f"[false_injection] {entry_dir.name}: using {steering_json.name}, "
      f"len={len(steering_vectors)}, min_tokens={min_tokens}, non_null={non_null}"
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
        save_hidden_payload=False,
        output_hiddens=None,
        steering_vectors=steering_vectors,
        steering_layer=int(layer),
      )
    print(
      f"[false_injection] {entry_dir.name}: wrote {Path(output_wav).name} and {Path(output_text).name}"
    )

  print(
    f"[false_injection] Done. Wrote output.wav/output.json for {len(input_paths)} items at {root_dir}"
  )


def _build_openai_client() -> OpenAI:
  api_key = os.getenv("OPENAI_API_KEY", "")
  if not api_key:
    raise ValueError("OPENAI_API_KEY is not set. Please export OPENAI_API_KEY before evaluation.")
  return OpenAI(api_key=api_key)


def _load_output_text(output_json_path: Path) -> str:
  if not output_json_path.exists():
    return ""
  with output_json_path.open("r", encoding="utf-8") as f:
    payload = json.load(f)
  if not isinstance(payload, dict):
    return ""
  text = payload.get("text", "")
  return str(text).strip()


def _parse_rating(raw_text: str) -> int:
  m = re.search(r"\b([0-5])\b", raw_text)
  if m is None:
    raise ValueError(f"Cannot parse score 0-5 from judge output: {raw_text}")
  return int(m.group(1))


def _evaluate_single_text(client: OpenAI, question: str, answer: str, model: str) -> tuple[int, str]:
  system_msg = (
    "You are evaluating answer quality for one fixed user question. "
    "Rate relatedness on a 0-5 integer scale only:\n"
    "0 totally unrelated\n"
    "1 not related\n"
    "2 slightly related\n"
    "3 related\n"
    "4 highly related\n"
    "5 perfectly related"
  )
  user_msg = (
    f"Question:\n{question}\n\n"
    f"Answer:\n{answer}\n\n"
    "Return only one integer from 0 to 5."
  )

  resp = client.chat.completions.create(
    model=model,
    messages=[
      {"role": "system", "content": system_msg},
      {"role": "user", "content": user_msg},
    ],
    temperature=0.0,
    seed=0,
  )
  text = (resp.choices[0].message.content or "").strip()
  score = _parse_rating(text)
  return score, text


def evaluate_results(
  root_dir: str,
  layer: int,
  expectation: float,
  question: str,
  judge_model: str,
) -> Path:
  if not question.strip():
    raise ValueError("--question must be a non-empty string for --evaluate-results")

  root = Path(root_dir)
  expectation_tag = _format_expectation_tag(expectation)
  output_wavs = [p for p in root.glob("*/output.wav") if p.is_file()]
  output_wavs.sort(key=lambda p: int(p.parent.name) if p.parent.name.isdigit() else p.parent.name)
  if not output_wavs:
    raise FileNotFoundError(
      f"No files matched pattern {root_dir}/*/output.wav"
    )

  client = _build_openai_client()

  results: list[dict[str, Any]] = []
  scores: list[int] = []
  for output_wav in output_wavs:
    sample_dir = output_wav.parent
    output_json = sample_dir / "output.json"
    answer_text = _load_output_text(output_json)

    if not answer_text:
      print(
        f"[false_injection] Warning: empty or missing output text at {output_json}; "
        "treat as score 0"
      )
      score = 0
      judge_raw = ""
    else:
      score, judge_raw = _evaluate_single_text(
        client=client,
        question=question,
        answer=answer_text,
        model=judge_model,
      )

    try:
      duration_s = _wav_duration_seconds(output_wav)
    except Exception:
      duration_s = None

    entry = {
      "sample_id": sample_dir.name,
      "output_wav": str(output_wav),
      "output_json": str(output_json),
      "duration_sec": duration_s,
      "answer": answer_text,
      "score": int(score),
      "judge_raw": judge_raw,
    }
    results.append(entry)
    scores.append(int(score))
    print(f"[false_injection] {sample_dir.name}: score={score}")

  avg_score = float(sum(scores) / len(scores)) if scores else 0.0
  output_payload: dict[str, Any] = {
    "question": question,
    "layer": int(layer),
    "expectation": float(expectation),
    "judge_model": judge_model,
    "num_samples": len(results),
    "average_score": avg_score,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "results": results,
  }

  out_path = root / f"false_injection_{int(layer)}_{expectation_tag}.json"
  _atomic_write_json(out_path, output_payload)
  print(
    f"[false_injection] Done. Saved evaluation to {out_path} "
    f"(num_samples={len(results)}, average_score={avg_score:.4f})"
  )
  return out_path


def main() -> None:
  parser = argparse.ArgumentParser("false_injection")
  parser.add_argument(
    "--root-dir",
    type=str,
    required=True,
    help="Root directory containing */input.wav and generated false-injection outputs.",
  )

  mode_group = parser.add_mutually_exclusive_group(required=True)
  mode_group.add_argument(
    "--generate-random-vector",
    action="store_true",
    help="Generate root-dir/*/steering_vector.json with random expectation-based false signals.",
  )
  mode_group.add_argument(
    "--evaluate-results",
    action="store_true",
    help="Evaluate root-dir/*/output.wav responses on 0-5 relatedness and save summary JSON.",
  )
  mode_group.add_argument(
    "--inference-with-steering",
    action="store_true",
    help="Run inference using root-dir/*/steering_vector.json and write root-dir/*/output.wav/json.",
  )

  parser.add_argument("--layer", type=int, required=True, help="Injection layer (0..31)")
  parser.add_argument(
    "--expectation",
    type=float,
    required=True,
    help="Expected interval seconds between false signals; -1 disables false signals.",
  )
  parser.add_argument(
    "--alpha",
    type=float,
    default=0.05,
    help="Multiplier for the layer mean-hidden-diff vector.",
  )
  parser.add_argument(
    "--decay-span",
    type=int,
    default=1,
    help="Linear decay span in tokens for each false trigger; 1 gives one active token.",
  )

  parser.add_argument(
    "--classifier-dir",
    type=str,
    default=None,
    help="Mode-class dataset directory containing mean_hidden_diff.json or */output_hidden.pt and */input_timing.json.",
  )

  parser.add_argument(
    "--question",
    type=str,
    default="",
    help="Single question used for 0-5 relatedness evaluation (required with --evaluate-results)",
  )
  parser.add_argument(
    "--judge-model",
    type=str,
    default="gpt-4o-mini",
    help="OpenAI model used for evaluation",
  )

  args = parser.parse_args()

  if args.generate_random_vector:
    generate_random_vector(
      root_dir=args.root_dir,
      layer=int(args.layer),
      expectation=float(args.expectation),
      alpha=float(args.alpha),
      decay_span=int(args.decay_span),
      classifier_dir=args.classifier_dir,
    )
    return

  if args.inference_with_steering:
    inference_with_steering(
      root_dir=args.root_dir,
      layer=int(args.layer),
      expectation=float(args.expectation),
    )
    return

  evaluate_results(
    root_dir=args.root_dir,
    layer=int(args.layer),
    expectation=float(args.expectation),
    question=str(args.question),
    judge_model=str(args.judge_model),
  )


if __name__ == "__main__":
  main()
