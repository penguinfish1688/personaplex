"""False-injection robustness utilities.

This script supports two workflows:
1) Generate random steering vectors under ``root_dir/*/steering_vector.json``:
   - Normal vector is extracted from SVM classifier (same source as user_interrupt.py)
   - Vector is multiplied by ``alpha``
   - For each token, inject with probability ``prob``
2) Evaluate generated outputs (0-5 relatedness score) and save summary to:
   ``root_dir/false_injection_{layer}_{prob}.json``
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import tempfile
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch
from openai import OpenAI

from moshi.persona_vector.mode_class import extract_normal_vector


TOKEN_RATE_HZ = 12.5
MAIN_LAYER_MIN = 0
MAIN_LAYER_MAX = 31


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


def _discover_classifier_path(classifier_dir: str, layer: int) -> Path:
  p = Path(classifier_dir) / f"hidden_mode_classifier_layer_{int(layer)}.pt"
  if not p.is_file():
    raise FileNotFoundError(
      f"Cannot find classifier for layer {layer}: {p}. "
      "Expected hidden_mode_classifier_layer_<layer>.pt"
    )
  return p


def generate_random_vector(
  root_dir: str,
  layer: int,
  prob: float,
  alpha: float,
  classifier_dir: Optional[str],
  classifier_path: Optional[str],
  seed: int,
) -> None:
  if not _is_valid_main_layer(layer):
    raise ValueError(f"layer must be in [{MAIN_LAYER_MIN}..{MAIN_LAYER_MAX}], got {layer}")
  if prob < 0.0 or prob > 1.0:
    raise ValueError(f"prob must be in [0, 1], got {prob}")

  if classifier_path is None:
    if classifier_dir is None:
      raise ValueError("--generate-random-vector requires --classifier-dir or --classifier-path")
    classifier_path = str(_discover_classifier_path(classifier_dir, layer))

  normal_vector = torch.as_tensor(
    extract_normal_vector(str(classifier_path)), dtype=torch.float32
  ).reshape(-1)
  if normal_vector.numel() == 0:
    raise ValueError(f"Extracted empty normal vector from classifier: {classifier_path}")
  steering_vec = (normal_vector * float(alpha)).tolist()

  root = Path(root_dir)
  input_paths = [p for p in root.glob("*/input.wav") if p.is_file()]
  input_paths.sort(key=lambda p: int(p.parent.name) if p.parent.name.isdigit() else p.parent.name)
  if not input_paths:
    raise FileNotFoundError(f"No files matched pattern {root_dir}/*/input.wav")

  rng = random.Random(int(seed))
  layer_key = f"layer_{int(layer)}"

  updated = 0
  for input_wav in input_paths:
    duration_s = _wav_duration_seconds(input_wav)
    total_tokens = int(math.ceil(duration_s * TOKEN_RATE_HZ))
    if total_tokens <= 0:
      raise ValueError(
        f"Computed non-positive token count for {input_wav}: duration={duration_s:.6f}s"
      )

    layer_payload: dict[str, Optional[list[float]]] = {}
    injected_count = 0
    for tok_idx in range(total_tokens):
      if rng.random() < float(prob):
        layer_payload[str(tok_idx)] = steering_vec
        injected_count += 1
      else:
        layer_payload[str(tok_idx)] = None

    steering_path = input_wav.parent / "steering_vector.json"
    existing = _load_existing_steering_payload(steering_path)
    existing[layer_key] = layer_payload
    _atomic_write_json(steering_path, existing)

    print(
      f"[false_injection] {input_wav.parent.name}: wrote {steering_path.name} {layer_key} "
      f"(tokens={total_tokens}, injected={injected_count}, prob={prob})"
    )
    updated += 1

  print(
    f"[false_injection] Done. Updated random steering vectors for {updated} items at {root_dir}"
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
  prob: float,
  question: str,
  judge_model: str,
) -> Path:
  if not question.strip():
    raise ValueError("--question must be a non-empty string for --evaluate-results")

  root = Path(root_dir)
  output_wavs = [p for p in root.glob("*/output.wav") if p.is_file()]
  output_wavs.sort(key=lambda p: int(p.parent.name) if p.parent.name.isdigit() else p.parent.name)
  if not output_wavs:
    raise FileNotFoundError(f"No files matched pattern {root_dir}/*/output.wav")

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
    "prob": float(prob),
    "judge_model": judge_model,
    "num_samples": len(results),
    "average_score": avg_score,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "results": results,
  }

  out_path = root / f"false_injection_{int(layer)}_{prob}.json"
  _atomic_write_json(out_path, output_payload)
  print(
    f"[false_injection] Done. Saved evaluation to {out_path} "
    f"(num_samples={len(results)}, average_score={avg_score:.4f})"
  )
  return out_path


def main() -> None:
  parser = argparse.ArgumentParser("false_injection")
  parser.add_argument("--root-dir", type=str, required=True, help="Root directory containing */input.wav and */output.wav")

  mode_group = parser.add_mutually_exclusive_group(required=True)
  mode_group.add_argument(
    "--generate-random-vector",
    action="store_true",
    help="Generate root-dir/*/steering_vector.json with random per-token false injection.",
  )
  mode_group.add_argument(
    "--evaluate-results",
    action="store_true",
    help="Evaluate root-dir/*/output.wav responses on 0-5 relatedness and save summary JSON.",
  )

  parser.add_argument("--layer", type=int, required=True, help="Injection layer (0..31)")
  parser.add_argument("--prob", type=float, required=True, help="Per-token injection probability in [0, 1]")
  parser.add_argument("--alpha", type=float, default=0.05, help="Multiplier for extracted SVM normal vector")
  parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible injection masks")

  parser.add_argument(
    "--classifier-dir",
    type=str,
    default=None,
    help="Directory containing hidden_mode_classifier_layer_<layer>.pt",
  )
  parser.add_argument(
    "--classifier-path",
    type=str,
    default=None,
    help="Optional explicit classifier checkpoint path (overrides --classifier-dir)",
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
      prob=float(args.prob),
      alpha=float(args.alpha),
      classifier_dir=args.classifier_dir,
      classifier_path=args.classifier_path,
      seed=int(args.seed),
    )
    return

  evaluate_results(
    root_dir=args.root_dir,
    layer=int(args.layer),
    prob=float(args.prob),
    question=str(args.question),
    judge_model=str(args.judge_model),
  )


if __name__ == "__main__":
  main()