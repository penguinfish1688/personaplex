"""Generate mode-class labels and token/audio visualizations.

This tool implements two steps for a dataset root containing sample dirs
with inference outputs (at minimum: ``output_hidden.pt`` and audio files).

Step 1: Generate ``input.json``
  - Classify token indices into T_listen / T_speak using thresholds over a
	middle-layer range.
  - Writes:
	  {
		"input": <text>,
		"modes": {
		  "listening": [[start, end], ...],
		  "speaking": [[start, end], ...]
		}
	  }

Step 2: Visualize classification and audio activity
  - Saves ``token_dist.png`` per sample:
	x-axis = token index
	lines = abs(input.wav), abs(output.wav) in token space
	background = green(listening), red(speaking), white(neither)

Usage examples:
  python -m moshi.persona_vector.mode_class_label generate-input \
	  --root-dir /path/to/root --theta-listen -2.0 --theta-speak -2.0 \
	  --layer-start 10 --layer-end 20

  python -m moshi.persona_vector.mode_class_label visualize-token-dist \
	  --root-dir /path/to/root
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from moshi.models import loaders


def _load_hidden_payload(path: str) -> Dict[str, Any]:
	if not os.path.exists(path):
		raise FileNotFoundError(f"Hidden payload not found: {path}")
	data = torch.load(path, map_location="cpu", weights_only=False)
	if not isinstance(data, dict):
		raise TypeError(
			f"Expected dict payload from {path}, got {type(data).__name__}."
		)
	return data


def _discover_hidden_path(sample_dir: Path) -> Path:
	candidates = [sample_dir / "output_hidden.pt", sample_dir / "output_hidden"]
	found = next((p for p in candidates if p.is_file()), None)
	if found is None:
		raise FileNotFoundError(
			f"Missing output_hidden(.pt) in sample dir: {sample_dir}"
		)
	return found


def _discover_sample_dirs(root_dir: Path) -> List[Path]:
	if not root_dir.is_dir():
		raise FileNotFoundError(f"Root directory not found: {root_dir}")
	out: List[Path] = []
	for child in sorted(root_dir.iterdir()):
		if not child.is_dir():
			continue
		try:
			_discover_hidden_path(child)
			out.append(child)
		except FileNotFoundError:
			continue
	if not out:
		raise FileNotFoundError(
			f"No sample dirs with output_hidden(.pt) found under {root_dir}"
		)
	return out


def _ranges_from_mask(mask: np.ndarray) -> List[List[int]]:
	if mask.ndim != 1:
		raise ValueError(f"Expected 1D mask, got shape={mask.shape}")
	ranges: List[List[int]] = []
	start: Optional[int] = None
	for idx, flag in enumerate(mask.tolist()):
		if flag and start is None:
			start = idx
		elif (not flag) and start is not None:
			ranges.append([start, idx - 1])
			start = None
	if start is not None:
		ranges.append([start, int(mask.shape[0]) - 1])
	return ranges


def _extract_input_text(sample_dir: Path) -> str:
	input_json = sample_dir / "input.json"
	if input_json.is_file():
		try:
			with input_json.open("r", encoding="utf-8") as f:
				data = json.load(f)
			if isinstance(data, dict) and isinstance(data.get("input"), str):
				return data["input"]
		except Exception:
			pass

	output_json = sample_dir / "output.json"
	if output_json.is_file():
		try:
			with output_json.open("r", encoding="utf-8") as f:
				data = json.load(f)
			if isinstance(data, dict) and isinstance(data.get("text"), str):
				return data["text"]
		except Exception:
			pass

	return ""


def _validate_layer_range(layer_start: int, layer_end: int, num_layers: int) -> List[int]:
	if layer_start < 0 or layer_end < 0:
		raise ValueError("layer_start and layer_end must be >= 0")
	if layer_start > layer_end:
		raise ValueError("layer_start must be <= layer_end")
	if layer_end >= num_layers:
		raise ValueError(
			f"Layer range [{layer_start}, {layer_end}] exceeds available layers [0, {num_layers - 1}]"
		)
	return list(range(layer_start, layer_end + 1))


def _compute_ll_lines_for_layer(
	payload: Dict[str, Any],
	lm: Any,
	layer: int,
) -> Tuple[np.ndarray, np.ndarray]:
	required = ["text_hidden_layers", "input_token_ids", "output_token_ids"]
	for key in required:
		if key not in payload:
			raise KeyError(f"Payload missing '{key}'")

	hidden = payload["text_hidden_layers"].float()  # [T, L, D]
	input_token_ids = payload["input_token_ids"].long()  # [T, K_in]
	output_token_ids = payload["output_token_ids"].long()  # [T, K_out]

	if hidden.ndim != 3:
		raise ValueError(f"Expected text_hidden_layers [T,L,D], got {tuple(hidden.shape)}")
	if input_token_ids.ndim != 2 or output_token_ids.ndim != 2:
		raise ValueError("Expected input/output token IDs with shape [T, K]")

	t_total, num_layers, _ = hidden.shape
	if t_total < 2:
		raise ValueError("Need at least 2 token steps for shifted listen LL.")
	if layer < 0 or layer >= num_layers:
		raise ValueError(f"Layer {layer} out of range [0, {num_layers - 1}]")
	if input_token_ids.shape[1] < 10:
		raise ValueError("input_token_ids width < 10; expected user audio cb0 at index 9")
	if output_token_ids.shape[1] < 2:
		raise ValueError("output_token_ids width < 2; expected model audio cb0 at index 1")

	h_l = hidden[:, layer, :]  # [T, D]
	text_tokens = output_token_ids[:, 0]  # [T]
	user_audio_target = input_token_ids[:, 9]  # [T]
	model_audio_target = output_token_ids[:, 1]  # [T]
	user_text_target = input_token_ids[:, 0]  # [T]
	model_text_target = output_token_ids[:, 0]  # [T]

	device = lm.device
	lm_dtype = next(lm.parameters()).dtype

	with torch.no_grad():
		x = h_l.to(device=device, dtype=lm_dtype)[:, None, :]  # [T,1,D]
		if getattr(lm, "out_norm", None) is not None:
			x = lm.out_norm(x)

		text_logits = lm.text_linear(x)[:, 0, :].float()  # [T, V_text]
		dep_in = text_tokens.to(device=device, dtype=torch.long)[:, None, None]  # [T,1,1]
		with lm.depformer.streaming(t_total):
			logits0 = lm.forward_depformer(0, dep_in, x)  # [T,1,1,V_audio]
		logits0 = logits0[:, 0, 0, :].float()  # [T, V_audio]

		def _ce(logits_2d: torch.Tensor, target_1d: torch.Tensor) -> torch.Tensor:
			probs = torch.softmax(logits_2d, dim=-1)
			return F.nll_loss(
				torch.log(probs.clamp_min(1e-12)),
				target_1d,
				reduction="none",
			)

		user_audio_ce = _ce(
			logits0[:-1],
			user_audio_target[1:].to(device=device, dtype=torch.long),
		)
		model_audio_ce = _ce(
			logits0[:-1],
			model_audio_target[:-1].to(device=device, dtype=torch.long),
		)
		user_text_ce = _ce(
			text_logits[:-1],
			user_text_target[1:].to(device=device, dtype=torch.long),
		)
		model_text_ce = _ce(
			text_logits[:-1],
			model_text_target[:-1].to(device=device, dtype=torch.long),
		)

		ce_user = 0.5 * (user_audio_ce + user_text_ce)  # line1 in mode_class.py
		ce_model = 0.5 * (model_audio_ce + model_text_ce)  # line2 in mode_class.py

	# LL is defined as negative CE/NLL.
	ll_listen = (-ce_user.detach().cpu().float()).numpy()
	ll_speak = (-ce_model.detach().cpu().float()).numpy()
	return ll_listen, ll_speak


def _load_or_compute_ll_for_layers(
	sample_dir: Path,
	payload: Dict[str, Any],
	layer_ids: Sequence[int],
	lm: Any,
) -> Tuple[np.ndarray, np.ndarray]:
	ll_listen_layers: List[np.ndarray] = []
	ll_speak_layers: List[np.ndarray] = []

	for layer in layer_ids:
		ce_json = sample_dir / f"in_out_ce_{layer}.json"
		if ce_json.is_file():
			with ce_json.open("r", encoding="utf-8") as f:
				ce_data = json.load(f)
			if not isinstance(ce_data, dict):
				raise TypeError(f"Invalid CE json format: {ce_json}")

			l1_ce = np.asarray(
				ce_data.get("line1_user_multimodal_ce", []),
				dtype=np.float32,
			)
			l2_ce = np.asarray(
				ce_data.get("line2_model_multimodal_ce", []),
				dtype=np.float32,
			)
			if l1_ce.ndim != 1 or l2_ce.ndim != 1 or l1_ce.size == 0 or l2_ce.size == 0:
				raise ValueError(f"Malformed CE arrays in {ce_json}")
			n = min(l1_ce.shape[0], l2_ce.shape[0])
			ll_listen = -l1_ce[:n]
			ll_speak = -l2_ce[:n]
		else:
			ll_listen, ll_speak = _compute_ll_lines_for_layer(payload, lm, int(layer))

		ll_listen_layers.append(ll_listen.astype(np.float32, copy=False))
		ll_speak_layers.append(ll_speak.astype(np.float32, copy=False))

	min_len = min(arr.shape[0] for arr in ll_listen_layers + ll_speak_layers)
	ll_listen_stack = np.stack([a[:min_len] for a in ll_listen_layers], axis=0)
	ll_speak_stack = np.stack([a[:min_len] for a in ll_speak_layers], axis=0)
	return ll_listen_stack, ll_speak_stack


def generate_input_jsons(
	root_dir: str,
	theta_listen: float,
	theta_speak: float,
	layer_start: int,
	layer_end: int,
	*,
	hf_repo: str = loaders.DEFAULT_REPO,
	moshi_weight: Optional[str] = None,
	device: str = "cuda",
) -> None:
	root = Path(root_dir)
	sample_dirs = _discover_sample_dirs(root)

	first_payload = _load_hidden_payload(str(_discover_hidden_path(sample_dirs[0])))
	if "text_hidden_layers" not in first_payload:
		raise KeyError("Hidden payload missing text_hidden_layers")
	num_layers = int(first_payload["text_hidden_layers"].shape[1])
	layer_ids = _validate_layer_range(layer_start, layer_end, num_layers)

	if moshi_weight is None:
		moshi_weight = hf_hub_download(hf_repo, loaders.MOSHI_NAME)  # type: ignore
	lm = loaders.get_moshi_lm(moshi_weight, device=device, cpu_offload=False)
	lm.eval()

	print(
		"[mode_class_label] Generating input.json using LL thresholds "
		f"(theta_listen={theta_listen}, theta_speak={theta_speak}, "
		f"layers={layer_start}..{layer_end})"
	)

	ok = 0
	for sample_dir in sample_dirs:
		try:
			hidden_path = _discover_hidden_path(sample_dir)
			payload = _load_hidden_payload(str(hidden_path))
			ll_listen, ll_speak = _load_or_compute_ll_for_layers(
				sample_dir,
				payload,
				layer_ids,
				lm,
			)  # [L, Tm1], [L, Tm1]

			# From paper formalization:
			# T_speak:   LL_speak >= theta_speak AND LL_listen < theta_listen for all middle layers
			# T_listen:  LL_listen >= theta_listen AND LL_speak < theta_speak for all middle layers
			speak_mask = np.all(ll_speak >= theta_speak, axis=0) & np.all(
				ll_listen < theta_listen, axis=0
			)
			listen_mask = np.all(ll_listen >= theta_listen, axis=0) & np.all(
				ll_speak < theta_speak, axis=0
			)

			out = {
				"input": _extract_input_text(sample_dir),
				"modes": {
					"listening": _ranges_from_mask(listen_mask),
					"speaking": _ranges_from_mask(speak_mask),
				},
			}

			out_path = sample_dir / "input.json"
			with out_path.open("w", encoding="utf-8") as f:
				json.dump(out, f, indent=2, ensure_ascii=False)

			ok += 1
			print(
				f"  [OK] {sample_dir.name}: tokens={listen_mask.shape[0]}, "
				f"listen={int(listen_mask.sum())}, speak={int(speak_mask.sum())}"
			)
		except Exception as exc:
			print(f"  [SKIP] {sample_dir.name}: {exc}")

	print(f"[mode_class_label] Done. Wrote input.json for {ok}/{len(sample_dirs)} samples.")


def _load_mono_wav(path: Path) -> Tuple[np.ndarray, int]:
	import sphn

	if not path.is_file():
		raise FileNotFoundError(f"WAV file not found: {path}")
	pcm, sr = sphn.read(str(path))
	wav = np.asarray(pcm)
	if wav.ndim == 2:
		wav = wav[0]
	elif wav.ndim != 1:
		wav = wav.reshape(-1)
	return wav.astype(np.float32), int(sr)


def _token_abs_amplitude(wav: np.ndarray, sr: int, frame_rate_hz: float, t_total: int) -> np.ndarray:
	if t_total <= 0:
		return np.zeros((0,), dtype=np.float32)
	samples_per_token = max(1, int(round(float(sr) / float(frame_rate_hz))))
	out = np.zeros((t_total,), dtype=np.float32)
	abs_wav = np.abs(wav)
	for t in range(t_total):
		s = t * samples_per_token
		e = min((t + 1) * samples_per_token, abs_wav.shape[0])
		if s >= abs_wav.shape[0] or e <= s:
			out[t] = np.nan
		else:
			out[t] = float(abs_wav[s:e].mean())
	return out


def _ranges_to_mask(ranges: Sequence[Sequence[int]], t_total: int) -> np.ndarray:
	mask = np.zeros((t_total,), dtype=bool)
	for pair in ranges:
		if len(pair) != 2:
			continue
		start = int(pair[0])
		end = int(pair[1])
		if end < 0 or start >= t_total:
			continue
		s = max(0, start)
		e = min(t_total - 1, end)
		if e >= s:
			mask[s : e + 1] = True
	return mask


def visualize_token_distribution(root_dir: str) -> None:
	import matplotlib.pyplot as plt

	root = Path(root_dir)
	sample_dirs = _discover_sample_dirs(root)

	print(f"[mode_class_label] Generating token_dist.png under {root}")
	ok = 0
	for sample_dir in sample_dirs:
		try:
			hidden_path = _discover_hidden_path(sample_dir)
			payload = _load_hidden_payload(str(hidden_path))

			if "text_hidden_layers" in payload:
				t_total = int(payload["text_hidden_layers"].shape[0])
			elif "hidden_states" in payload:
				t_total = int(payload["hidden_states"].shape[0])
			else:
				raise KeyError("Hidden payload has neither text_hidden_layers nor hidden_states")

			frame_rate_hz = float(payload.get("frame_rate", 12.5))
			input_wav_path = Path(str(payload.get("input_wav", sample_dir / "input.wav")))
			output_wav_path = Path(str(payload.get("output_wav", sample_dir / "output.wav")))

			input_wav, input_sr = _load_mono_wav(input_wav_path)
			output_wav, output_sr = _load_mono_wav(output_wav_path)

			in_amp = _token_abs_amplitude(input_wav, input_sr, frame_rate_hz, t_total)
			out_amp = _token_abs_amplitude(output_wav, output_sr, frame_rate_hz, t_total)

			input_json = sample_dir / "input.json"
			if not input_json.is_file():
				raise FileNotFoundError(
					f"Missing input.json in {sample_dir}; run generate-input first"
				)
			with input_json.open("r", encoding="utf-8") as f:
				data = json.load(f)
			if not isinstance(data, dict) or "modes" not in data:
				raise ValueError(f"Invalid input.json format in {sample_dir}")
			modes = data["modes"]
			if not isinstance(modes, dict):
				raise ValueError(f"input.json modes must be an object in {sample_dir}")

			listen_mask = _ranges_to_mask(modes.get("listening", []), t_total)
			speak_mask = _ranges_to_mask(modes.get("speaking", []), t_total)

			xs = np.arange(t_total, dtype=np.int32)
			fig_w = max(11.0, min(26.0, t_total * 0.06))
			fig, ax = plt.subplots(figsize=(fig_w, 4.8), dpi=170)

			# Token classification background.
			for t in range(t_total):
				if listen_mask[t]:
					ax.axvspan(t - 0.5, t + 0.5, color="#5cb85c", alpha=0.22, linewidth=0)
				elif speak_mask[t]:
					ax.axvspan(t - 0.5, t + 0.5, color="#d9534f", alpha=0.22, linewidth=0)

			ax.plot(xs, in_amp, color="#2ca02c", linewidth=1.1, alpha=0.95, label="|input.wav|")
			ax.plot(xs, out_amp, color="#1f77b4", linewidth=1.1, alpha=0.95, label="|output.wav|")

			ax.set_xlim(-0.5, t_total - 0.5)
			ax.set_xlabel("Token id")
			ax.set_ylabel("Absolute amplitude")
			ax.set_title(
				"Token classification and aligned audio activity "
				f"(listen={int(listen_mask.sum())}, speak={int(speak_mask.sum())}, T={t_total})"
			)
			ax.grid(True, axis="x", linestyle=":", linewidth=0.6, alpha=0.65)
			ax.legend(loc="upper right", fontsize=8)

			out_png = sample_dir / "token_dist.png"
			fig.tight_layout()
			fig.savefig(out_png, bbox_inches="tight")
			plt.close(fig)

			ok += 1
			print(f"  [OK] {sample_dir.name}: saved {out_png.name}")
		except Exception as exc:
			print(f"  [SKIP] {sample_dir.name}: {exc}")

	print(f"[mode_class_label] Done. Generated {ok}/{len(sample_dirs)} token_dist.png files.")


def main() -> None:
	ap = argparse.ArgumentParser(
		prog="mode_class_label",
		description="Generate input.json labels from LL thresholds and visualize token/audio distribution.",
	)
	sub = ap.add_subparsers(dest="cmd", required=True)

	gen = sub.add_parser(
		"generate-input",
		help="Generate input.json with modes.listening and modes.speaking",
	)
	gen.add_argument("--root-dir", type=str, required=True)
	gen.add_argument("--theta-listen", type=float, required=True)
	gen.add_argument("--theta-speak", type=float, required=True)
	gen.add_argument("--layer-start", type=int, required=True)
	gen.add_argument("--layer-end", type=int, required=True)
	gen.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO)
	gen.add_argument("--moshi-weight", type=str, default=None)
	gen.add_argument("--device", type=str, default="cuda")

	vis = sub.add_parser(
		"visualize-token-dist",
		help="Generate token_dist.png for each sample dir from input.json + audio",
	)
	vis.add_argument("--root-dir", type=str, required=True)

	args = ap.parse_args()

	if args.cmd == "generate-input":
		generate_input_jsons(
			root_dir=args.root_dir,
			theta_listen=args.theta_listen,
			theta_speak=args.theta_speak,
			layer_start=args.layer_start,
			layer_end=args.layer_end,
			hf_repo=args.hf_repo,
			moshi_weight=args.moshi_weight,
			device=args.device,
		)
	elif args.cmd == "visualize-token-dist":
		visualize_token_distribution(root_dir=args.root_dir)
	else:
		raise ValueError(f"Unsupported command: {args.cmd}")


if __name__ == "__main__":
	main()
