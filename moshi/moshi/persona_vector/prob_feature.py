"""Token probability feature extraction (Quevedo et al., 2024).

Computes five per-step features over a sliding window of generated tokens:
  mtp       – Minimum Token Probability
  avgtp     – Average Token Probability
  Mpd       – Maximum Probability Deviation
  mps       – Minimum Probability Spread
  entropy   – Shannon Entropy at the current step

CLI usage (offline, from saved ``output_hidden.pt``):

    python -m moshi.persona_vector.prob_feature \
        --root-dir /path/to/data/candor_turn_taking_mini \
        --hf-repo nvidia/personaplex-7b-v1 \
        --device cpu

For every ``<root-dir>/*/output_hidden.pt``, saves
``<root-dir>/*/prob_feature.json`` with per-token features.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    import sentencepiece

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature Extractor
# ---------------------------------------------------------------------------

SPECIAL_TOKEN_MAP: Dict[int, str] = {0: "EPAD", 1: "BOS", 2: "EOS", 3: "PAD"}
PAD_TOKEN_ID = 3


@dataclass
class _StepRecord:
    """Intermediate per-step values stored in the sliding window."""

    p_t: float
    p_v_star: float
    p_v_minus: float
    step_entropy: float


class TokenProbabilityFeatureExtractor:
    """Sliding-window feature extractor for token probability analysis.

    Maintains a deque of the last *window_size* generation steps and exposes
    five aggregate features derived from the probability distribution at each
    step (Quevedo et al., 2024).

    Parameters
    ----------
    window_size : int
        Maximum number of steps kept in the sliding window (default ``100``).
    """

    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        self._window: deque[_StepRecord] = deque(maxlen=window_size)

    # ---- step update -------------------------------------------------------

    def update(self, probs: torch.Tensor, generated_token_id: int) -> None:
        """Record one generation step.

        Parameters
        ----------
        probs : torch.Tensor
            1-D softmax probability distribution of shape ``[vocab_size]``.
        generated_token_id : int
            Token ID actually sampled / selected at this step.
        """
        p_t = probs[generated_token_id].item()
        p_v_star = probs.max().item()
        p_v_minus = probs.min().item()
        step_entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        self._window.append(
            _StepRecord(
                p_t=p_t,
                p_v_star=p_v_star,
                p_v_minus=p_v_minus,
                step_entropy=step_entropy,
            )
        )

    # ---- aggregate features ------------------------------------------------

    def get_features(self) -> Optional[Dict[str, float]]:
        """Return five features (four windowed aggregates + current-step entropy).

        Returns ``None`` when no steps have been recorded yet.
        """
        if len(self._window) == 0:
            return None

        min_pt = float("inf")
        sum_pt = 0.0
        max_dev = float("-inf")
        min_spread = float("inf")

        for rec in self._window:
            if rec.p_t < min_pt:
                min_pt = rec.p_t
            sum_pt += rec.p_t
            dev = rec.p_v_star - rec.p_t
            if dev > max_dev:
                max_dev = dev
            spread = rec.p_v_star - rec.p_v_minus
            if spread < min_spread:
                min_spread = spread

        n = len(self._window)
        latest = self._window[-1]
        return {
            "mtp": min_pt,
            "avgtp": sum_pt / n,
            "Mpd": max_dev,
            "mps": min_spread,
            "entropy": latest.step_entropy,
        }

    def reset(self) -> None:
        """Clear the sliding window."""
        self._window.clear()


# ---------------------------------------------------------------------------
# Offline: hidden -> logits -> probs -> per-token features
# ---------------------------------------------------------------------------


@dataclass
class DecoderProjection:
    """Minimal projection: out_norm (optional) + text_linear."""

    out_norm: Optional[torch.nn.Module]
    text_linear: torch.nn.Module


def _load_decoder_projection(
    *,
    hf_repo: str,
    moshi_weight: Optional[str],
    device: str,
) -> DecoderProjection:
    from huggingface_hub import hf_hub_download
    from moshi.models import loaders

    if moshi_weight is None:
        moshi_weight = hf_hub_download(hf_repo, loaders.MOSHI_NAME)
    lm = loaders.get_moshi_lm(moshi_weight, device=device, cpu_offload=(device == "cpu"))
    lm.eval()
    return DecoderProjection(out_norm=lm.out_norm, text_linear=lm.text_linear)


def _load_tokenizer(
    hf_repo: str, tokenizer_path: Optional[str]
) -> "sentencepiece.SentencePieceProcessor":
    import sentencepiece
    from huggingface_hub import hf_hub_download
    from moshi.models import loaders

    if tokenizer_path is None:
        tokenizer_path = hf_hub_download(hf_repo, loaders.TEXT_TOKENIZER_NAME)
    return sentencepiece.SentencePieceProcessor(tokenizer_path)  # type: ignore


def _id_to_piece(
    token_id: int,
    tokenizer: "sentencepiece.SentencePieceProcessor",
) -> str:
    if token_id in SPECIAL_TOKEN_MAP:
        return SPECIAL_TOKEN_MAP[token_id]
    piece = tokenizer.id_to_piece(token_id)  # type: ignore
    return piece.replace("▁", " ") if piece else f"<{token_id}>"


def hidden_to_probs(
    hidden: torch.Tensor,
    projection: DecoderProjection,
    layer: int = -1,
) -> torch.Tensor:
    """Project hidden states to softmax probabilities.

    Parameters
    ----------
    hidden : torch.Tensor
        Shape ``[T, L, D]`` – per-token, per-layer hidden states.
    projection : DecoderProjection
        Output norm + text linear head from the LM.
    layer : int
        Which layer to use (default ``-1`` = last layer).

    Returns
    -------
    torch.Tensor
        Softmax probabilities with shape ``[T, vocab_size]``.
    """
    x = hidden[:, layer, :]  # [T, D]
    w = projection.text_linear.weight
    x = x.to(device=w.device, dtype=w.dtype).unsqueeze(0)  # type: ignore[arg-type]  # [1, T, D]
    with torch.no_grad():
        if projection.out_norm is not None:
            x = projection.out_norm(x)
        logits = projection.text_linear(x)  # [1, T, V]
    return F.softmax(logits.squeeze(0).float(), dim=-1).cpu()  # [T, V]


def hidden_to_pad_probs_all_layers(
    hidden: torch.Tensor,
    projection: DecoderProjection,
) -> torch.Tensor:
    """Return PAD-token probability at every layer for every token.

    Parameters
    ----------
    hidden : torch.Tensor
        Shape ``[T, L, D]``.

    Returns
    -------
    torch.Tensor
        Shape ``[T, L]`` – ``prob[t, l] = softmax(project(h[t,l]))[PAD]``.
    """
    T, L, D = hidden.shape
    w = projection.text_linear.weight
    # Flatten to [T*L, D], project, then reshape
    x = hidden.reshape(T * L, D)
    x = x.to(device=w.device, dtype=w.dtype).unsqueeze(0)  # type: ignore[arg-type]  # [1, T*L, D]
    with torch.no_grad():
        if projection.out_norm is not None:
            x = projection.out_norm(x)
        logits = projection.text_linear(x)  # [1, T*L, V]
    probs = F.softmax(logits.squeeze(0).float(), dim=-1)  # [T*L, V]
    pad_probs = probs[:, PAD_TOKEN_ID].reshape(T, L)  # [T, L]
    return pad_probs.cpu()


def _glob_hidden_paths(root: Path) -> List[Path]:
    """Glob ``*/output_hidden.pt`` under *root*, sorted numerically."""
    paths = list(root.glob("*/output_hidden.pt"))

    def _key(p: Path):
        try:
            return (0, int(p.parent.name))
        except ValueError:
            return (1, p.parent.name)

    return sorted(paths, key=_key)


def process_one(
    hidden_path: Path,
    projection: DecoderProjection,
    tokenizer: "sentencepiece.SentencePieceProcessor",
    layer: int = -1,
    window_size: int = 100,
) -> List[Dict[str, Any]]:
    """Extract per-token probability features for a single sample.

    Returns a list of dicts, one per token.
    """
    payload = torch.load(hidden_path, map_location="cpu", weights_only=True)
    hidden = payload["text_hidden_layers"]  # [T, L, D]
    probs = hidden_to_probs(hidden, projection, layer=layer)  # [T, V]
    pad_probs = hidden_to_pad_probs_all_layers(hidden, projection)  # [T, L]
    T = probs.shape[0]

    # greedy token ids from the chosen layer
    greedy_ids = probs.argmax(dim=-1).tolist()  # list[int]

    extractor = TokenProbabilityFeatureExtractor(window_size=window_size)
    results: List[Dict[str, Any]] = []
    for t in range(T):
        token_id = greedy_ids[t]
        extractor.update(probs[t], token_id)
        feats = extractor.get_features()
        assert feats is not None
        results.append(
            {
                "step": t,
                "token_id": token_id,
                "token": _id_to_piece(token_id, tokenizer),
                **feats,
                "pad_prob_per_layer": pad_probs[t].tolist(),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

FEATURE_KEYS = ["mtp", "avgtp", "Mpd", "mps", "entropy"]
FEATURE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def plot_prob_features(json_path: str, output_path: Optional[str] = None) -> None:
    """Read *prob_feature.json* and plot all features with decoded token labels.

    Parameters
    ----------
    json_path : str
        Path to the saved ``prob_feature.json``.
    output_path : str, optional
        Destination figure path.  Defaults to sibling ``prob_feature.png``.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    with open(json_path, "r") as f:
        records: List[Dict[str, Any]] = json.load(f)

    if not records:
        logger.warning("Empty prob_feature.json, skipping plot: %s", json_path)
        return

    steps = [r["step"] for r in records]
    tokens = [r["token"] for r in records]
    num_tokens = len(steps)

    # Check if PAD-prob-per-layer data is present
    has_pad = "pad_prob_per_layer" in records[0]
    num_layers = len(records[0]["pad_prob_per_layer"]) if has_pad else 0
    n_subplots = len(FEATURE_KEYS) + (1 if has_pad else 0)

    fig_w = max(12.0, num_tokens * 0.18)
    fig, axes = plt.subplots(
        n_subplots, 1,
        figsize=(fig_w, 2.2 * n_subplots),
        sharex=True, dpi=150,
        gridspec_kw={"height_ratios": [1] * len(FEATURE_KEYS) + ([2] if has_pad else [])},
    )
    if n_subplots == 1:
        axes = [axes]  # type: ignore[list-item]

    x = np.arange(num_tokens)

    def _safe_text(t: str) -> str:
        return t.replace("\n", " ").replace("$", "\\$")

    for ax, key, color in zip(axes[:len(FEATURE_KEYS)], FEATURE_KEYS, FEATURE_COLORS):
        vals = [r[key] for r in records]
        ax.plot(x, vals, linewidth=0.8, color=color)
        ax.fill_between(x, vals, alpha=0.15, color=color)
        ax.set_ylabel(key, fontsize=9)
        ax.grid(axis="y", linewidth=0.3, alpha=0.5)
        ax.margins(x=0.002)

    # PAD probability heatmap across layers
    if has_pad:
        from matplotlib.colors import LogNorm

        pad_ax = axes[-1]
        pad_grid = np.array(
            [r["pad_prob_per_layer"] for r in records], dtype=np.float32,
        ).T  # [L, T]

        positive = pad_grid[pad_grid > 0]
        vmin = max(float(np.min(positive)), 1e-8) if positive.size > 0 else 1e-8
        vmax = max(float(np.max(pad_grid)), vmin)

        im = pad_ax.imshow(
            np.clip(pad_grid, vmin, vmax),
            aspect="auto",
            cmap="YlOrRd",
            origin="upper",
            norm=LogNorm(vmin=vmin, vmax=vmax),
            extent=[-0.5, num_tokens - 0.5, num_layers - 0.5, -0.5],
        )
        fig.colorbar(im, ax=pad_ax, pad=0.01).set_label("P(<PAD>) [log]", fontsize=8)
        pad_ax.set_ylabel("Layer", fontsize=9)
        pad_ax.set_yticks(np.arange(0, num_layers, max(1, num_layers // 8)))
        pad_ax.set_yticklabels(
            [str(i + 1) for i in range(0, num_layers, max(1, num_layers // 8))],
            fontsize=7,
        )

    # Token labels on the bottom x-axis
    bottom_ax = axes[-1]
    bottom_ax.set_xticks(x)
    tick_labels = [_safe_text(t) for t in tokens]
    try:
        bottom_ax.set_xticklabels(
            tick_labels, rotation=90, fontsize=5, ha="center", parse_math=False,
        )
    except TypeError:
        bottom_ax.set_xticklabels(
            tick_labels, rotation=90, fontsize=5, ha="center",
        )
    bottom_ax.set_xlabel("Token step", fontsize=9)

    fig.suptitle(
        Path(json_path).parent.name + " – probability features",
        fontsize=10,
        y=1.0,
    )
    fig.tight_layout()

    if output_path is None:
        output_path = str(Path(json_path).with_suffix(".png"))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s  %(levelname)s  %(message)s",
    )
    from moshi.models import loaders

    ap = argparse.ArgumentParser(
        "prob_feature",
        description="Extract per-token probability features from output_hidden.pt files.",
    )
    ap.add_argument(
        "--root-dir",
        type=str,
        required=True,
        help="Root dir containing <n>/output_hidden.pt",
    )
    ap.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Transformer layer index to decode from (default: -1 = last layer)",
    )
    ap.add_argument(
        "--window-size",
        type=int,
        default=100,
        help="Sliding window size for feature aggregation (default: 100)",
    )
    ap.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO)
    ap.add_argument("--tokenizer", type=str, default=None)
    ap.add_argument("--moshi-weight", type=str, default=None)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    root = Path(args.root_dir)
    hidden_paths = _glob_hidden_paths(root)
    if not hidden_paths:
        raise FileNotFoundError(f"No output_hidden.pt found under {root}/*/")

    logger.info("Loading decoder projection and tokenizer …")
    projection = _load_decoder_projection(
        hf_repo=args.hf_repo,
        moshi_weight=args.moshi_weight,
        device=args.device,
    )
    tokenizer = _load_tokenizer(args.hf_repo, args.tokenizer)

    logger.info("Processing %d samples (layer=%d, window=%d) …", len(hidden_paths), args.layer, args.window_size)
    for hp in hidden_paths:
        out_path = hp.with_name("prob_feature.json")
        results = process_one(
            hp,
            projection=projection,
            tokenizer=tokenizer,
            layer=args.layer,
            window_size=args.window_size,
        )
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("Wrote %s (%d tokens)", out_path, len(results))

        fig_path = out_path.with_suffix(".png")
        plot_prob_features(str(out_path), str(fig_path))
        logger.info("Wrote %s", fig_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
