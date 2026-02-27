"""Linear classifier for listening mode vs. speaking mode.

Determines whether Moshi is in *listening* mode (outputting PAD tokens while
the user speaks) or *speaking* mode (generating real text) based on
per-token hidden representations extracted during offline inference.

Classes:
    HiddenExtractor  – wraps ``run_batch_inference`` to generate hidden
                       payloads (``*_hidden.pt``) for the mode-class dataset.
    HiddenModeClassifier – trains, loads, and runs a ``nn.Linear(D, 1)``
                           binary classifier on the extracted hiddens.

CLI (``python -m moshi.persona_vector.mode_class``):
    --gen-dataset-hidden <dataset_path>
    --gen-sentence-hidden <wav_path> --output <path>
    --train-mode-classifier <dataset_path> --output <dir> [--layer L]
    --predict-mode <hidden.pt> --model <model.pt> --output <out.json>
    --plot-prediction <prediction.json> --hidden <hidden.pt> --output <out.png>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from moshi.offline import run_batch_inference, _get_voice_prompt_dir
from moshi.models import loaders


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_hidden_payload(path: str) -> Dict[str, Any]:
    """Load a hidden payload ``.pt`` file and return its dict."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Hidden payload not found: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise TypeError(
            f"Expected dict payload from {path}, got {type(data).__name__}. "
            "This file may not be a hidden payload."
        )
    return data


def _extract_layer(payload: Dict[str, Any], layer: int) -> torch.Tensor:
    """Extract a single layer's hidden states from a payload.

    Supports two payload formats:
      1. Schema v1 (``text_hidden_layers: [T, L, D]``)
      2. Simple  (``hidden_states: [T, D]``) – only ``layer == -1`` supported.

    Returns:
        ``[T, D]`` float tensor.
    """
    if "text_hidden_layers" in payload:
        hidden = payload["text_hidden_layers"]  # [T, L, D]
        num_layers = hidden.shape[1]
        actual_layer = layer if layer >= 0 else num_layers + layer
        if actual_layer < 0 or actual_layer >= num_layers:
            raise ValueError(
                f"Layer {layer} out of range for hidden with {num_layers} layers."
            )
        return hidden[:, actual_layer, :].float()
    elif "hidden_states" in payload:
        if layer != -1:
            raise ValueError(
                "Simple payload format only has the final layer (layer=-1). "
                f"Requested layer={layer}."
            )
        return payload["hidden_states"].float()  # [T, D]
    else:
        raise KeyError(
            "Payload has neither 'text_hidden_layers' nor 'hidden_states'."
        )


def _build_labels(
    num_tokens: int,
    listening_ranges: List[List[int]],
    speaking_ranges: List[List[int]],
) -> torch.Tensor:
    """Build a per-token label tensor from inclusive ``[start, end]`` ranges.

    0 = listening, 1 = speaking.  Tokens not covered by any range default to
    listening (0).
    """
    labels = torch.zeros(num_tokens, dtype=torch.float32)
    for start, end in listening_ranges:
        if start < 0 or end >= num_tokens:
            raise ValueError(
                f"Listening range [{start}, {end}] out of bounds "
                f"for {num_tokens} tokens."
            )
        labels[start : end + 1] = 0.0
    for start, end in speaking_ranges:
        if start < 0 or end >= num_tokens:
            raise ValueError(
                f"Speaking range [{start}, {end}] out of bounds "
                f"for {num_tokens} tokens."
            )
        labels[start : end + 1] = 1.0
    return labels


def _derive_output_paths(hidden_path: str) -> tuple[str, str]:
    """Derive output wav / text paths from a hidden payload path.

    ``complete_sentence_hidden.pt`` → ``complete_sentence_output.wav``,
    ``complete_sentence_output.json``.
    """
    base = hidden_path.replace("_hidden.pt", "")
    return base + "_output.wav", base + "_output.json"


# ---------------------------------------------------------------------------
# HiddenExtractor
# ---------------------------------------------------------------------------


class HiddenExtractor:
    """Generate hidden-state payloads via Moshi offline inference.

    Thin wrapper around :func:`run_batch_inference` with
    ``save_hidden_payload=True``.

    Args:
        device: CUDA device string (default ``"cuda"``).
        hf_repo: HuggingFace repo for model weights.
        voice_prompt: Voice prompt filename (e.g. ``"NATF0.pt"``).
        voice_prompt_dir: Optional directory containing voice prompts.
        text_prompt: System text prompt.
        tokenizer_path: Path to sentencepiece tokenizer.
        moshi_weight: Path to Moshi LM weights.
        mimi_weight: Path to Mimi codec weights.
        seed: Random seed.
    """

    def __init__(
        self,
        *,
        device: str = "cuda",
        hf_repo: str = loaders.DEFAULT_REPO,
        voice_prompt: str = "NATF0.pt",
        voice_prompt_dir: Optional[str] = None,
        text_prompt: str = "You are a helpful and friendly assistant.",
        tokenizer_path: Optional[str] = None,
        moshi_weight: Optional[str] = None,
        mimi_weight: Optional[str] = None,
        seed: int = 42,
    ):
        self.device = device
        self.hf_repo = hf_repo
        self.text_prompt = text_prompt
        self.tokenizer_path = tokenizer_path
        self.moshi_weight = moshi_weight
        self.mimi_weight = mimi_weight
        self.seed = seed

        # Resolve voice prompt path
        vp_dir = _get_voice_prompt_dir(voice_prompt_dir, hf_repo)
        if vp_dir is None:
            raise FileNotFoundError("Unable to resolve voice prompt directory.")
        self.voice_prompt_path = os.path.join(vp_dir, voice_prompt)
        if not os.path.exists(self.voice_prompt_path):
            raise FileNotFoundError(
                f"Voice prompt not found: {self.voice_prompt_path}"
            )

    def generate_batch(
        self,
        input_wavs: List[str],
        output_hidden_paths: List[str],
    ) -> None:
        """Run batch inference and save hidden payloads.

        For each ``input_wav``, a hidden payload is saved to the
        corresponding entry in ``output_hidden_paths``.  Intermediate
        output wav / text files are written alongside the hidden file.
        """
        assert len(input_wavs) == len(output_hidden_paths), (
            f"input_wavs ({len(input_wavs)}) and output_hidden_paths "
            f"({len(output_hidden_paths)}) must have the same length"
        )
        if not input_wavs:
            print("[mode_class] Nothing to process.")
            return

        out_wavs: List[str] = []
        out_texts: List[str] = []
        for h in output_hidden_paths:
            wav, txt = _derive_output_paths(h)
            out_wavs.append(wav)
            out_texts.append(txt)

        prompts = [self.text_prompt] * len(input_wavs)

        with torch.no_grad():
            run_batch_inference(
                input_wavs=input_wavs,
                output_wavs=out_wavs,
                output_texts=out_texts,
                text_prompts=prompts,
                voice_prompt_path=self.voice_prompt_path,
                tokenizer_path=self.tokenizer_path,
                moshi_weight=self.moshi_weight,
                mimi_weight=self.mimi_weight,
                hf_repo=self.hf_repo,
                device=self.device,
                seed=self.seed,
                temp_audio=0.8,
                temp_text=0.7,
                topk_audio=250,
                topk_text=25,
                greedy=False,
                save_voice_prompt_embeddings=False,
                cpu_offload=False,
                return_hidden_layers=False,
                save_hidden_payload=True,
                output_hiddens=output_hidden_paths,
            )

    def generate(self, input_wav: str, output_hidden_path: str) -> None:
        """Generate a single hidden payload for one WAV file."""
        self.generate_batch([input_wav], [output_hidden_path])

    def class_mode_dataset(self, dataset_path: str) -> None:
        """Generate hidden payloads for every entry in a mode-class dataset.

        Expects ``dataset_path/<id>/complete_sentence.wav`` and
        ``dataset_path/<id>/incomplete_sentence.wav`` to exist (produced by
        TTS).  Outputs ``complete_sentence_hidden.pt`` and
        ``incomplete_sentence_hidden.pt`` next to each WAV.

        Already-existing hidden files are skipped.
        """
        pattern = os.path.join(dataset_path, "*", "input.json")
        entries = sorted(
            glob(pattern),
            key=lambda p: int(os.path.basename(os.path.dirname(p))),
        )
        if not entries:
            raise FileNotFoundError(
                f"No input.json found under {dataset_path}/*/"
            )

        input_wavs: List[str] = []
        output_hiddens: List[str] = []

        for entry_json in entries:
            entry_dir = os.path.dirname(entry_json)
            for prefix in ("complete_sentence", "incomplete_sentence"):
                wav = os.path.join(entry_dir, f"{prefix}.wav")
                hidden = os.path.join(entry_dir, f"{prefix}_hidden.pt")
                if not os.path.exists(wav):
                    raise FileNotFoundError(
                        f"Expected WAV not found: {wav}. "
                        "Run TTS (--mode-class) first."
                    )
                if os.path.exists(hidden):
                    print(f"[SKIP] {hidden} already exists")
                    continue
                input_wavs.append(wav)
                output_hiddens.append(hidden)

        if not input_wavs:
            print(
                "[mode_class] All hiddens already exist, nothing to generate."
            )
            return

        print(f"[mode_class] Generating {len(input_wavs)} hidden payloads …")
        self.generate_batch(input_wavs, output_hiddens)
        print("[mode_class] Done.")


# ---------------------------------------------------------------------------
# HiddenModeClassifier
# ---------------------------------------------------------------------------


class HiddenModeClassifier:
    """Binary linear classifier: listening (0) vs. speaking (1).

    Operates on per-token hidden representations from a specified
    transformer layer.
    """

    def __init__(self) -> None:
        self.model: Optional[nn.Linear] = None
        self.layer: int = -1
        self.hidden_dim: int = 0

    # ---- training -----------------------------------------------------------

    def train(
        self,
        dataset_path: str,
        output_dir: str,
        layer: int = -1,
        *,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 256,
    ) -> None:
        """Train the linear classifier on mode-class dataset hidden states.

        For each entry under ``dataset_path/<id>/``:

        * ``complete_sentence_hidden.pt`` is labeled using the
          ``complete_modes.listening`` / ``complete_modes.speaking`` ranges
          from ``input.json``.
        * ``incomplete_sentence_hidden.pt`` is labeled using the
          ``incomplete_modes.listening`` / ``incomplete_modes.speaking``
          ranges from ``input.json``.

        Saves the trained model to
        ``output_dir/hidden_mode_classifier_layer_{layer}.pt``.
        """
        pattern = os.path.join(dataset_path, "*", "input.json")
        entries = sorted(
            glob(pattern),
            key=lambda p: int(os.path.basename(os.path.dirname(p))),
        )
        if not entries:
            raise FileNotFoundError(
                f"No input.json found under {dataset_path}/*/"
            )

        all_hiddens: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []

        for entry_json in entries:
            entry_dir = os.path.dirname(entry_json)
            entry_id = os.path.basename(entry_dir)

            with open(entry_json, "r", encoding="utf-8") as f:
                meta = json.load(f)

            # ---- complete_sentence_hidden --------------------------------
            cs_path = os.path.join(entry_dir, "complete_sentence_hidden.pt")
            if not os.path.exists(cs_path):
                raise FileNotFoundError(
                    f"Missing {cs_path}. "
                    "Run --gen-dataset-hidden first."
                )

            if "complete_modes" not in meta:
                raise KeyError(
                    f"input.json for entry {entry_id} is missing "
                    "'complete_modes' label ranges."
                )
            cs_modes = meta["complete_modes"]

            cs_payload = _load_hidden_payload(cs_path)
            cs_hidden = _extract_layer(cs_payload, layer)  # [T, D]
            T_cs = cs_hidden.shape[0]
            cs_labels = _build_labels(
                T_cs, cs_modes["listening"], cs_modes["speaking"]
            )
            all_hiddens.append(cs_hidden)
            all_labels.append(cs_labels)

            # ---- incomplete_sentence_hidden ------------------------------
            is_path = os.path.join(
                entry_dir, "incomplete_sentence_hidden.pt"
            )
            if not os.path.exists(is_path):
                raise FileNotFoundError(
                    f"Missing {is_path}. "
                    "Run --gen-dataset-hidden first."
                )

            if "incomplete_modes" not in meta:
                raise KeyError(
                    f"input.json for entry {entry_id} is missing "
                    "'incomplete_modes' label ranges."
                )
            is_modes = meta["incomplete_modes"]

            is_payload = _load_hidden_payload(is_path)
            is_hidden = _extract_layer(is_payload, layer)  # [T, D]
            T_is = is_hidden.shape[0]
            is_labels = _build_labels(
                T_is, is_modes["listening"], is_modes["speaking"]
            )
            all_hiddens.append(is_hidden)
            all_labels.append(is_labels)

        # Aggregate all tokens
        X = torch.cat(all_hiddens, dim=0)  # [N, D]
        y = torch.cat(all_labels, dim=0)  # [N]
        D = X.shape[1]

        n_listen = int((y == 0).sum())
        n_speak = int((y == 1).sum())
        print(
            f"[train] Collected {X.shape[0]} tokens, dim={D}, layer={layer}"
        )
        print(f"[train] Listening: {n_listen}, Speaking: {n_speak}")

        # Build model
        model = nn.Linear(D, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        # Train/val split (80/20)
        N = X.shape[0]
        perm_all = torch.randperm(N)
        n_train = int(N * 0.8)
        train_idx = perm_all[:n_train]
        val_idx = perm_all[n_train:]
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        print(
            f"[train] Split: {X_train.shape[0]} train, "
            f"{X_val.shape[0]} val"
        )

        N_train = X_train.shape[0]
        model.train()
        for epoch in range(1, epochs + 1):
            perm = torch.randperm(N_train)
            epoch_loss = 0.0
            num_batches = 0
            for i in range(0, N_train, batch_size):
                idx = perm[i : i + batch_size]
                xb = X_train[idx]
                yb = y_train[idx]

                logits = model(xb).squeeze(-1)
                loss = criterion(logits, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            with torch.no_grad():
                train_preds = (
                    torch.sigmoid(model(X_train).squeeze(-1)) > 0.5
                ).float()
                train_acc = (train_preds == y_train).float().mean().item()
                val_preds = (
                    torch.sigmoid(model(X_val).squeeze(-1)) > 0.5
                ).float()
                val_acc = (val_preds == y_val).float().mean().item()
                val_logits = model(X_val).squeeze(-1)
                val_loss = criterion(val_logits, y_val).item()
            print(
                f"  epoch {epoch:3d}/{epochs}  "
                f"train_loss={avg_loss:.4f}  train_acc={train_acc:.4f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            )

        # Persist
        self.model = model
        self.layer = layer
        self.hidden_dim = D

        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(
            output_dir, f"hidden_mode_classifier_layer_{layer}.pt"
        )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "layer": layer,
                "hidden_dim": D,
            },
            save_path,
        )
        print(f"[train] Saved classifier to {save_path}")

    # ---- loading ------------------------------------------------------------

    def load(self, model_path: str) -> None:
        """Load a trained classifier from disk."""
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        self.layer = ckpt["layer"]
        self.hidden_dim = ckpt["hidden_dim"]
        self.model = nn.Linear(self.hidden_dim, 1)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        print(
            f"[load] Loaded classifier (layer={self.layer}, "
            f"dim={self.hidden_dim}) from {model_path}"
        )

    # ---- prediction ---------------------------------------------------------

    def predict(self, input_hidden_path: str, output_path: str) -> None:
        """Predict per-token mode for a hidden payload and write JSON.

        Output format::

            {
                "0": {"mode": "listening", "confidence": 0.95},
                "1": {"mode": "speaking", "confidence": 0.87},
                ...
            }
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load() first.")

        payload = _load_hidden_payload(input_hidden_path)
        hidden = _extract_layer(payload, self.layer)  # [T, D]

        self.model.eval()
        with torch.no_grad():
            logits = self.model(hidden).squeeze(-1)  # [T]
            probs = torch.sigmoid(logits)  # [T]

        result: Dict[str, Dict[str, Any]] = {}
        for t in range(hidden.shape[0]):
            p = probs[t].item()
            mode = "speaking" if p > 0.5 else "listening"
            confidence = p if mode == "speaking" else 1.0 - p
            result[str(t)] = {
                "mode": mode,
                "confidence": round(confidence, 4),
            }

        os.makedirs(
            os.path.dirname(os.path.abspath(output_path)), exist_ok=True
        )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(
            f"[predict] Saved predictions "
            f"({hidden.shape[0]} tokens) to {output_path}"
        )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_safe_text(text: str) -> str:
    """Sanitize text for matplotlib rendering."""
    return text.replace("\n", " ").replace("$", "\\$")


def plot_prediction(
    prediction_path: str,
    hidden_path: str,
    output_path: str,
) -> None:
    """Plot mode prediction results.

    * **Y-axis**: continuous score in ``[0, 1]`` where 0 = listening and
      1 = speaking.  Derived from the predicted class and confidence:
      ``score = confidence`` if speaking else ``1 - confidence``.
    * **X-axis**: token index.  Each tick is labeled with the decoded
      token name from the hidden payload (``token_names``).
    * **User transcript lane**: if a sibling ``input.json`` exists next
      to *hidden_path*, its ``complete_sentence`` / ``incomplete_sentence``
      is shown as a text band below the plot.

    Args:
        prediction_path: JSON file produced by ``HiddenModeClassifier.predict``.
        hidden_path: The ``*_hidden.pt`` file used for prediction (supplies
            ``token_names``).
        output_path: Output image path (PNG).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle

    # ---- load prediction JSON ------------------------------------------------
    with open(prediction_path, "r", encoding="utf-8") as f:
        predictions: Dict[str, Dict[str, Any]] = json.load(f)

    num_tokens = len(predictions)
    if num_tokens == 0:
        raise ValueError("Prediction file is empty.")

    scores = np.zeros(num_tokens, dtype=np.float32)
    for t in range(num_tokens):
        entry = predictions[str(t)]
        conf = float(entry["confidence"])
        if entry["mode"] == "speaking":
            scores[t] = conf
        else:
            scores[t] = 1.0 - conf

    # ---- load token names from hidden payload --------------------------------
    payload = _load_hidden_payload(hidden_path)
    token_names: List[str] = payload.get("token_names", [])
    if len(token_names) < num_tokens:
        # Pad with index strings if payload has fewer names.
        token_names.extend(
            [str(i) for i in range(len(token_names), num_tokens)]
        )
    token_names = token_names[:num_tokens]

    # ---- user transcript from sibling transcript JSON -------------------------
    # For mode_class files like complete_sentence_hidden.pt -> complete_sentence.json
    # For other datasets like output_hidden.pt -> try output.json, fall back to input.json
    # Expected format: {"text": "...", "chunks": [{"text": "word", "timestamp": [start, end]}, ...]}
    hidden_p = Path(hidden_path)
    stem = hidden_p.stem  # e.g. "complete_sentence_hidden"
    # Strip "_hidden" suffix to get the sentence prefix
    transcript_prefix = stem.replace("_hidden", "")  # "complete_sentence"

    # Candidate transcript files: derived name first, then input.json as fallback
    candidates = [hidden_p.parent / f"{transcript_prefix}.json"]
    if transcript_prefix != "input":
        candidates.append(hidden_p.parent / "input.json")

    frame_rate_hz = float(payload.get("frame_rate", 12.5))
    transcript_spans: list[tuple[float, float, str]] = []

    for transcript_json in candidates:
        if not transcript_json.exists():
            continue
        with open(transcript_json, "r", encoding="utf-8") as f:
            transcript_data = json.load(f)
        if not isinstance(transcript_data, dict) or "chunks" not in transcript_data:
            continue
        # Found a valid transcript file
        chunks = transcript_data["chunks"]
        for chunk in chunks:
            word = str(chunk.get("text", "")).strip()
            ts = chunk.get("timestamp", None)
            if not word or not isinstance(ts, list) or len(ts) != 2:
                continue
            start_sec = float(ts[0])
            end_sec = float(ts[1])
            if end_sec <= start_sec:
                continue
            transcript_spans.append(
                (start_sec * frame_rate_hz, end_sec * frame_rate_hz, word)
            )
        break  # use first valid candidate

    # ---- figure layout -------------------------------------------------------
    fig_w = max(10.0, num_tokens * 0.45)
    fig_h = 5.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)

    xs = np.arange(num_tokens)

    # Color each bar by predicted mode (blue = listening, orange = speaking).
    colors = [
        "#e67e22" if s > 0.5 else "#3498db" for s in scores
    ]
    ax.bar(xs, scores, color=colors, width=0.8, edgecolor="none", alpha=0.85)

    # Horizontal reference line at 0.5 threshold.
    ax.axhline(0.5, color="#888888", linewidth=0.8, linestyle="--")

    # Token name labels on x-axis (rotated).
    safe_names = [_plot_safe_text(n) for n in token_names]
    ax.set_xticks(xs)
    try:
        ax.set_xticklabels(
            safe_names,
            rotation=90,
            fontsize=5,
            ha="center",
            parse_math=False,
        )
    except TypeError:
        # Older matplotlib without parse_math.
        ax.set_xticklabels(
            safe_names, rotation=90, fontsize=5, ha="center"
        )

    ax.set_xlim(-0.5, num_tokens - 0.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("P(speaking)")
    ax.set_xlabel("Token index")
    ax.set_title("Mode prediction (listening=0, speaking=1)")

    # ---- user transcript lane ------------------------------------------------
    if transcript_spans:
        y_base = -0.18  # below the x-axis in data coords
        lane_h = 0.08
        for start_tok, end_tok, word in transcript_spans:
            if end_tok <= -0.5 or start_tok >= num_tokens - 0.5:
                continue
            draw_start = max(start_tok, -0.5)
            draw_end = min(end_tok, num_tokens - 0.5)
            if draw_end <= draw_start:
                continue
            rect = Rectangle(
                (draw_start, y_base),
                draw_end - draw_start,
                lane_h,
                facecolor="#f3f3f3",
                edgecolor="#888888",
                linewidth=0.5,
                alpha=0.9,
                clip_on=False,
            )
            ax.add_patch(rect)
            center_x = 0.5 * (draw_start + draw_end)
            safe_word = _plot_safe_text(word)
            try:
                ax.text(
                    center_x,
                    y_base + lane_h / 2,
                    safe_word,
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="black",
                    clip_on=False,
                    parse_math=False,
                )
            except TypeError:
                ax.text(
                    center_x,
                    y_base + lane_h / 2,
                    safe_word,
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="black",
                    clip_on=False,
                )

        # "User" label.
        ax.text(
            -1.5,
            y_base + lane_h / 2,
            "User",
            ha="right",
            va="center",
            fontsize=7,
            color="black",
            clip_on=False,
        )

    fig.tight_layout()
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved prediction plot ({num_tokens} tokens) to {output_path}")


def plot_prediction_dataset(
    root_dir: str,
    model_path: str,
) -> None:
    """Predict + plot for every ``*_hidden.pt`` under ``root_dir/*/``.

    For each hidden file:
    1. Run ``HiddenModeClassifier.predict`` → save prediction JSON next to it.
    2. Run ``plot_prediction`` → save PNG next to it.

    Args:
        root_dir: Dataset root (e.g. ``data/mode_class_mini``).
        model_path: Path to trained classifier ``.pt``.
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Root directory not found: {root}")

    classifier = HiddenModeClassifier()
    classifier.load(model_path)

    # Gather all *_hidden.pt files
    hidden_files = sorted(root.glob("*/*_hidden.pt"))
    if not hidden_files:
        raise FileNotFoundError(f"No *_hidden.pt files found under {root}/*/")

    print(f"[plot-dataset] Found {len(hidden_files)} hidden files under {root}")

    for hp in hidden_files:
        stem = hp.stem  # e.g. "complete_sentence_hidden"
        pred_json = hp.with_name(f"{stem}_prediction.json")
        plot_png = hp.with_name(f"{stem}_mode_prediction.png")

        print(f"\n--- {hp.parent.name}/{hp.name} ---")

        # Predict
        try:
            classifier.predict(str(hp), str(pred_json))
        except Exception as exc:
            print(f"  [SKIP] predict failed: {exc}")
            continue

        # Plot
        try:
            plot_prediction(str(pred_json), str(hp), str(plot_png))
        except Exception as exc:
            print(f"  [SKIP] plot failed: {exc}")
            continue

    print(f"\n[plot-dataset] Done. Processed {len(hidden_files)} files.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="mode_class",
        description="Linear classifier for Moshi listening/speaking mode.",
    )

    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--gen-dataset-hidden",
        type=str,
        metavar="DATASET",
        help="Generate hidden payloads for a mode-class dataset directory.",
    )
    group.add_argument(
        "--gen-sentence-hidden",
        type=str,
        metavar="WAV",
        help="Generate a hidden payload for a single WAV file.",
    )
    group.add_argument(
        "--train-mode-classifier",
        type=str,
        metavar="DATASET",
        help="Train the mode classifier on a mode-class dataset.",
    )
    group.add_argument(
        "--predict-mode",
        type=str,
        metavar="HIDDEN_PT",
        help="Predict mode for a single hidden payload file.",
    )
    group.add_argument(
        "--plot-prediction",
        type=str,
        metavar="PRED_JSON",
        help="Plot prediction results from a JSON file.",
    )
    group.add_argument(
        "--plot-prediction-dataset",
        type=str,
        metavar="ROOT_DIR",
        help="Predict + plot for all *_hidden.pt under ROOT_DIR/*/.",
    )

    # Shared inference options (used by --gen-*)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO)
    ap.add_argument("--voice-prompt", type=str, default="NATF0.pt")
    ap.add_argument("--voice-prompt-dir", type=str, default=None)
    ap.add_argument(
        "--text-prompt",
        type=str,
        default="You are a helpful and friendly assistant.",
    )
    ap.add_argument("--tokenizer", type=str, default=None)
    ap.add_argument("--moshi-weight", type=str, default=None)
    ap.add_argument("--mimi-weight", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)

    # Training / prediction options
    ap.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (dir for train, file for others).",
    )
    ap.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained classifier (for --predict-mode).",
    )
    ap.add_argument(
        "--hidden",
        type=str,
        default=None,
        help="Path to hidden payload .pt (for --plot-prediction).",
    )
    ap.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Transformer layer index (default: -1, last layer).",
    )
    ap.add_argument(
        "--epochs", type=int, default=50, help="Training epochs (default: 50)."
    )
    ap.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)."
    )

    args = ap.parse_args()

    # ---- dispatch -----------------------------------------------------------

    if args.gen_dataset_hidden:
        extractor = HiddenExtractor(
            device=args.device,
            hf_repo=args.hf_repo,
            voice_prompt=args.voice_prompt,
            voice_prompt_dir=args.voice_prompt_dir,
            text_prompt=args.text_prompt,
            tokenizer_path=args.tokenizer,
            moshi_weight=args.moshi_weight,
            mimi_weight=args.mimi_weight,
            seed=args.seed,
        )
        extractor.class_mode_dataset(args.gen_dataset_hidden)

    elif args.gen_sentence_hidden:
        if not args.output:
            ap.error("--gen-sentence-hidden requires --output")
        extractor = HiddenExtractor(
            device=args.device,
            hf_repo=args.hf_repo,
            voice_prompt=args.voice_prompt,
            voice_prompt_dir=args.voice_prompt_dir,
            text_prompt=args.text_prompt,
            tokenizer_path=args.tokenizer,
            moshi_weight=args.moshi_weight,
            mimi_weight=args.mimi_weight,
            seed=args.seed,
        )
        extractor.generate(args.gen_sentence_hidden, args.output)

    elif args.train_mode_classifier:
        if not args.output:
            ap.error("--train-mode-classifier requires --output")
        classifier = HiddenModeClassifier()
        classifier.train(
            dataset_path=args.train_mode_classifier,
            output_dir=args.output,
            layer=args.layer,
            epochs=args.epochs,
            lr=args.lr,
        )

    elif args.predict_mode:
        if not args.model:
            ap.error("--predict-mode requires --model")
        if not args.output:
            ap.error("--predict-mode requires --output")
        classifier = HiddenModeClassifier()
        classifier.load(args.model)
        classifier.predict(args.predict_mode, args.output)

    elif args.plot_prediction:
        if not args.hidden:
            ap.error("--plot-prediction requires --hidden")
        if not args.output:
            ap.error("--plot-prediction requires --output")
        plot_prediction(args.plot_prediction, args.hidden, args.output)

    elif args.plot_prediction_dataset:
        if not args.model:
            ap.error("--plot-prediction-dataset requires --model")
        plot_prediction_dataset(
            root_dir=args.plot_prediction_dataset,
            model_path=args.model,
        )


if __name__ == "__main__":
    main()
