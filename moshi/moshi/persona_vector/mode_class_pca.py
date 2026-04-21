"""Plot PCA distributions of mode-class hidden vectors.

Given a dataset root with entries like ``root/*`` containing:
- ``input.json`` with ``modes.listening`` / ``modes.speaking`` ranges
- hidden payload (prefers ``output_hidden.pt``, also supports
  ``complete_sentence_hidden.pt`` and ``incomplete_sentence_hidden.pt``)

This script gathers labeled vectors and, for each layer, computes PCA over
combined listening+speaking vectors. It then plots token dots with:
- x-axis: PCA value (PC1 projection)
- y-axis: layer index
- color: speaking=red, listening=green

Output:
- ``<rootdir>/mode_class_pca.png`` (or ``--output`` path)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch


def _collect_sample_dirs(root: Path) -> list[Path]:
	if not root.is_dir():
		raise FileNotFoundError(f"Root directory not found: {root}")
	dirs = [p for p in root.iterdir() if p.is_dir() and (p / "input.json").is_file()]
	dirs.sort(key=lambda p: (0, int(p.name)) if p.name.isdigit() else (1, p.name))
	if not dirs:
		raise FileNotFoundError(f"No valid sample dirs under {root}. Need */input.json")
	return dirs


def _find_hidden_path(sample_dir: Path) -> Path | None:
	candidates = [
		sample_dir / "output_hidden.pt",
		sample_dir / "output_hidden",
		sample_dir / "complete_sentence_hidden.pt",
		sample_dir / "incomplete_sentence_hidden.pt",
	]
	for p in candidates:
		if p.is_file():
			return p
	return None


def _ranges_to_indices(ranges: object, t_total: int) -> np.ndarray:
	mask = np.zeros((t_total,), dtype=bool)
	if not isinstance(ranges, list):
		return np.zeros((0,), dtype=np.int64)
	for pair in ranges:
		if not isinstance(pair, list) or len(pair) != 2:
			continue
		s = int(pair[0])
		e = int(pair[1])
		if e < 0 or s >= t_total:
			continue
		s = max(0, s)
		e = min(t_total - 1, e)
		if e >= s:
			mask[s : e + 1] = True
	return np.flatnonzero(mask)


def _load_hidden_tld(path: Path) -> torch.Tensor:
	payload = torch.load(str(path), map_location="cpu", weights_only=False)
	if not isinstance(payload, dict):
		raise TypeError(f"Expected dict payload in {path}, got {type(payload).__name__}")

	if "text_hidden_layers" in payload:
		hidden = payload["text_hidden_layers"]
		if not isinstance(hidden, torch.Tensor):
			hidden = torch.as_tensor(hidden)
		if hidden.ndim != 3:
			raise ValueError(f"Expected text_hidden_layers [T,L,D], got {tuple(hidden.shape)} in {path}")
		return hidden.float()

	if "hidden_states" in payload:
		hidden = payload["hidden_states"]
		if not isinstance(hidden, torch.Tensor):
			hidden = torch.as_tensor(hidden)
		if hidden.ndim == 2:
			hidden = hidden.unsqueeze(1)
		if hidden.ndim != 3:
			raise ValueError(f"Expected hidden_states [T,L,D] or [T,D], got {tuple(hidden.shape)} in {path}")
		return hidden.float()

	raise KeyError(f"Missing hidden fields in {path}. Need text_hidden_layers or hidden_states")


def _pca_pc1_projection(x_nd: np.ndarray) -> np.ndarray:
	"""Return 1D PC1 projections using SVD on centered data."""
	if x_nd.ndim != 2:
		raise ValueError(f"Expected 2D array [N,D], got shape={x_nd.shape}")
	if x_nd.shape[0] < 2:
		return np.zeros((x_nd.shape[0],), dtype=np.float32)
	x = x_nd.astype(np.float64, copy=False)
	x = x - x.mean(axis=0, keepdims=True)
	_, _, vt = np.linalg.svd(x, full_matrices=False)
	pc1 = vt[0]
	proj = x @ pc1
	return proj.astype(np.float32)


def plot_mode_class_pca(root_dir: str, output: str | None, max_points_per_class: int) -> Path:
	root = Path(root_dir)
	sample_dirs = _collect_sample_dirs(root)

	# Gather vectors per layer separately for each class.
	speak_by_layer: dict[int, list[np.ndarray]] = {}
	listen_by_layer: dict[int, list[np.ndarray]] = {}
	n_layers_ref: int | None = None

	for sd in sample_dirs:
		hidden_path = _find_hidden_path(sd)
		if hidden_path is None:
			continue
		with (sd / "input.json").open("r", encoding="utf-8") as f:
			data = json.load(f)
		if not isinstance(data, dict) or "modes" not in data or not isinstance(data["modes"], dict):
			continue

		hidden = _load_hidden_tld(hidden_path)  # [T,L,D]
		t_total, n_layers, _ = hidden.shape
		if n_layers_ref is None:
			n_layers_ref = int(n_layers)
		elif int(n_layers) != int(n_layers_ref):
			raise ValueError(f"Layer count mismatch at {sd}: got {n_layers}, expected {n_layers_ref}")

		modes = data["modes"]
		listen_idx = _ranges_to_indices(modes.get("listening", []), int(t_total))
		speak_idx = _ranges_to_indices(modes.get("speaking", []), int(t_total))

		if listen_idx.size > 0:
			h_listen = hidden[torch.from_numpy(listen_idx)]  # [N,L,D]
			for l in range(int(n_layers)):
				listen_by_layer.setdefault(l, []).append(h_listen[:, l, :].numpy())

		if speak_idx.size > 0:
			h_speak = hidden[torch.from_numpy(speak_idx)]  # [N,L,D]
			for l in range(int(n_layers)):
				speak_by_layer.setdefault(l, []).append(h_speak[:, l, :].numpy())

	if n_layers_ref is None:
		raise RuntimeError("No usable hidden payloads found for PCA plotting")

	rng = np.random.default_rng(42)
	fig, ax = plt.subplots(figsize=(12.5, 7.0), dpi=180)

	plotted_any = False
	for layer in range(int(n_layers_ref)):
		listen_chunks = listen_by_layer.get(layer, [])
		speak_chunks = speak_by_layer.get(layer, [])
		if not listen_chunks or not speak_chunks:
			continue

		x_listen = np.concatenate(listen_chunks, axis=0)
		x_speak = np.concatenate(speak_chunks, axis=0)

		# Balanced subsampling for readability/perf.
		if max_points_per_class > 0:
			if x_listen.shape[0] > max_points_per_class:
				idx = rng.choice(x_listen.shape[0], size=max_points_per_class, replace=False)
				x_listen = x_listen[idx]
			if x_speak.shape[0] > max_points_per_class:
				idx = rng.choice(x_speak.shape[0], size=max_points_per_class, replace=False)
				x_speak = x_speak[idx]

		x_all = np.concatenate([x_listen, x_speak], axis=0)
		proj_all = _pca_pc1_projection(x_all)
		n_l = x_listen.shape[0]
		proj_l = proj_all[:n_l]
		proj_s = proj_all[n_l:]

		# Small vertical jitter to reduce overplotting while preserving layer bands.
		y_l = np.full_like(proj_l, fill_value=float(layer), dtype=np.float32) + rng.normal(0.0, 0.06, size=proj_l.shape[0])
		y_s = np.full_like(proj_s, fill_value=float(layer), dtype=np.float32) + rng.normal(0.0, 0.06, size=proj_s.shape[0])

		ax.scatter(proj_l, y_l, s=6, c="#2ca02c", alpha=0.35, linewidths=0.0, label="listening" if layer == 0 else "_nolegend_")
		ax.scatter(proj_s, y_s, s=6, c="#d62728", alpha=0.35, linewidths=0.0, label="speaking" if layer == 0 else "_nolegend_")
		plotted_any = True

	if not plotted_any:
		raise RuntimeError("No layers had both listening and speaking labeled vectors")

	ax.set_xlabel("PCA Dimension (PC1 projection)")
	ax.set_ylabel("Layer")
	ax.set_title("Mode-Class PCA Distribution by Layer")
	ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.5)
	ax.legend(loc="upper right")
	ax.set_ylim(-0.75, float(n_layers_ref) - 0.25)

	out_path = Path(output) if output is not None else (root / "mode_class_pca.png")
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.tight_layout()
	fig.savefig(out_path, bbox_inches="tight")
	plt.close(fig)
	return out_path


def main() -> None:
	ap = argparse.ArgumentParser("mode_class_pca")
	ap.add_argument("--root-dir", type=str, required=True, help="Root dir containing */input.json and hidden payloads")
	ap.add_argument("--output", type=str, default=None, help="Output PNG path (default: <root-dir>/mode_class_pca.png)")
	ap.add_argument(
		"--max-points-per-class",
		type=int,
		default=2500,
		help="Maximum points per class per layer for plotting (<=0 means no cap)",
	)
	args = ap.parse_args()

	out = plot_mode_class_pca(
		root_dir=str(args.root_dir),
		output=None if args.output is None else str(args.output),
		max_points_per_class=int(args.max_points_per_class),
	)
	print(f"[mode_class_pca] Wrote {out}")


if __name__ == "__main__":
	main()