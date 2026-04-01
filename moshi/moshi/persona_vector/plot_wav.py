"""Plot anchored average absolute amplitude for input/output WAV files.

Given a dataset root directory containing entries like ``root/*`` with:
- ``input.wav``
- ``output.wav``
- ``input_timing.json``

This script builds two plots (one per anchor timing), where each plot contains:
- x-axis: relative time in seconds (anchor at x=0)
- y-axis: mean absolute amplitude across all samples
- two curves: input.wav and output.wav

Default anchor keys are ``question_start`` and ``interrupt_start``.
"""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ANCHORS = ["question_start", "interrupt_start"]


def _load_wav_mono(path: Path) -> tuple[np.ndarray, int]:
	"""Load WAV as mono float32 in [-1, 1] and return (audio, sample_rate)."""
	try:
		import soundfile as sf

		audio, sr = sf.read(str(path), always_2d=False, dtype="float32")
		if isinstance(audio, np.ndarray) and audio.ndim == 2:
			audio = audio.mean(axis=1)
		return np.asarray(audio, dtype=np.float32), int(sr)
	except Exception:
		with wave.open(str(path), "rb") as wf:
			sr = int(wf.getframerate())
			nchan = int(wf.getnchannels())
			sampwidth = int(wf.getsampwidth())
			frames = wf.readframes(wf.getnframes())

		if sampwidth == 2:
			arr = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
		elif sampwidth == 4:
			arr = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
		else:
			raise ValueError(f"Unsupported sample width {sampwidth} bytes for {path}")

		if nchan > 1:
			arr = arr.reshape(-1, nchan).mean(axis=1)
		return arr.astype(np.float32), sr


def _resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
	if src_sr == dst_sr:
		return audio
	if audio.size == 0:
		return audio.astype(np.float32)
	src_t = np.arange(audio.shape[0], dtype=np.float64) / float(src_sr)
	dst_n = int(round((audio.shape[0] / float(src_sr)) * float(dst_sr)))
	dst_n = max(dst_n, 1)
	dst_t = np.arange(dst_n, dtype=np.float64) / float(dst_sr)
	out = np.interp(dst_t, src_t, audio.astype(np.float64))
	return out.astype(np.float32)


def _extract_centered_abs_segment(
	audio: np.ndarray,
	sr: int,
	anchor_s: float,
	window_s: float,
) -> np.ndarray:
	"""Return fixed-length centered abs(audio) segment with NaN padding."""
	total = int(round(2.0 * window_s * sr)) + 1
	center = int(round(anchor_s * sr))
	start = center - total // 2
	end = start + total

	out = np.full((total,), np.nan, dtype=np.float32)
	src_l = max(0, start)
	src_r = min(audio.shape[0], end)
	if src_r <= src_l:
		return out
	dst_l = src_l - start
	dst_r = dst_l + (src_r - src_l)
	out[dst_l:dst_r] = np.abs(audio[src_l:src_r])
	return out


def _collect_entry_dirs(root_dir: Path) -> list[Path]:
	entries = [p for p in root_dir.glob("*") if p.is_dir()]
	entries.sort(key=lambda p: int(p.name) if p.name.isdigit() else p.name)
	valid = []
	for d in entries:
		if (d / "input.wav").is_file() and (d / "output.wav").is_file() and (d / "input_timing.json").is_file():
			valid.append(d)
	return valid


def plot_wav(
	root_dir: str,
	anchors: list[str],
	window_s: float,
	max_hz: float,
) -> None:
	root = Path(root_dir)
	if not root.is_dir():
		raise FileNotFoundError(f"Root directory not found: {root}")

	entry_dirs = _collect_entry_dirs(root)
	if not entry_dirs:
		raise FileNotFoundError(
			f"No valid entries found under {root}. Need */input.wav, */output.wav, */input_timing.json"
		)

	# Use first entry's input sample rate as canonical rate for averaging.
	first_in, target_sr = _load_wav_mono(entry_dirs[0] / "input.wav")
	del first_in

	for anchor in anchors:
		input_segments: list[np.ndarray] = []
		output_segments: list[np.ndarray] = []

		for d in entry_dirs:
			timing_path = d / "input_timing.json"
			with timing_path.open("r", encoding="utf-8") as f:
				timing = json.load(f)
			if not isinstance(timing, dict) or anchor not in timing:
				continue

			anchor_s = float(timing[anchor])

			in_audio, in_sr = _load_wav_mono(d / "input.wav")
			out_audio, out_sr = _load_wav_mono(d / "output.wav")

			if in_sr != target_sr:
				in_audio = _resample_linear(in_audio, in_sr, target_sr)
			if out_sr != target_sr:
				out_audio = _resample_linear(out_audio, out_sr, target_sr)

			input_segments.append(_extract_centered_abs_segment(in_audio, target_sr, anchor_s, window_s))
			output_segments.append(_extract_centered_abs_segment(out_audio, target_sr, anchor_s, window_s))

		if not input_segments or not output_segments:
			print(f"[plot_wav] Skip anchor '{anchor}': no valid samples with this key.")
			continue

		in_mean = np.nanmean(np.stack(input_segments, axis=0), axis=0)
		out_mean = np.nanmean(np.stack(output_segments, axis=0), axis=0)

		t = (np.arange(in_mean.shape[0], dtype=np.float32) - (in_mean.shape[0] // 2)) / float(target_sr)

		# Downsample plotting points for readability/performance.
		stride = max(1, int(round(target_sr / max_hz)))
		t_plot = t[::stride]
		in_plot = in_mean[::stride]
		out_plot = out_mean[::stride]

		plt.figure(figsize=(10, 5))
		plt.plot(t_plot, in_plot, label="input.wav", linewidth=1.8)
		plt.plot(t_plot, out_plot, label="output.wav", linewidth=1.8)
		plt.axvline(0.0, color="black", linestyle="--", linewidth=1.0, label=f"anchor: {anchor}")
		plt.xlabel("Time (s), centered at anchor")
		plt.ylabel("Mean absolute amplitude")
		plt.title(f"Anchored Mean |Amplitude| around {anchor}")
		plt.legend()
		plt.grid(True, alpha=0.25)
		plt.tight_layout()

		out_png = root / f"plot_wav_{anchor}.png"
		plt.savefig(out_png, dpi=160)
		plt.close()
		print(f"[plot_wav] Wrote {out_png}")


def main() -> None:
	ap = argparse.ArgumentParser("plot_wav")
	ap.add_argument("--root-dir", type=str, required=True, help="Dataset root containing */input.wav and */output.wav")
	ap.add_argument(
		"--anchors",
		type=str,
		nargs="+",
		default=DEFAULT_ANCHORS,
		help="Anchor timing keys in input_timing.json (default: question_start interrupt_start)",
	)
	ap.add_argument(
		"--window-s",
		type=float,
		default=5.0,
		help="Half window size around anchor in seconds (plot range is [-window_s, +window_s]).",
	)
	ap.add_argument(
		"--max-hz",
		type=float,
		default=200.0,
		help="Max plotting point density in Hz after downsampling.",
	)
	args = ap.parse_args()

	if float(args.window_s) <= 0.0:
		raise ValueError(f"--window-s must be > 0, got {args.window_s}")
	if float(args.max_hz) <= 0.0:
		raise ValueError(f"--max-hz must be > 0, got {args.max_hz}")

	plot_wav(
		root_dir=args.root_dir,
		anchors=[str(x) for x in args.anchors],
		window_s=float(args.window_s),
		max_hz=float(args.max_hz),
	)


if __name__ == "__main__":
	main()