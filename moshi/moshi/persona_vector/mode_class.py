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
    --save-mean-hidden-diff <root_dir>
    --predict-mode <hidden.pt> --model <model.pt> --output <out.json>
    --plot-prediction <prediction.json> --hidden <hidden.pt> --output <out.png>
    --plot-attention-heatmap-dataset <root_dir> [--layer L]
    --plot-logit-lens-dataset <root_dir> [--layer L]
    --plot-logit-lens-turn-taking-from-saved <root_dir>
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
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

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


def _build_mode_masks(
    num_tokens: int,
    listening_ranges: List[List[int]],
    speaking_ranges: List[List[int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build explicit listening/speaking boolean masks from ranges.

    Unlike ``_build_labels``, tokens outside both sets are left unlabeled and
    excluded from downstream aggregation.
    """
    listening_mask = torch.zeros(num_tokens, dtype=torch.bool)
    speaking_mask = torch.zeros(num_tokens, dtype=torch.bool)

    for start, end in listening_ranges:
        if start < 0 or end >= num_tokens:
            raise ValueError(
                f"Listening range [{start}, {end}] out of bounds "
                f"for {num_tokens} tokens."
            )
        listening_mask[start : end + 1] = True

    for start, end in speaking_ranges:
        if start < 0 or end >= num_tokens:
            raise ValueError(
                f"Speaking range [{start}, {end}] out of bounds "
                f"for {num_tokens} tokens."
            )
        speaking_mask[start : end + 1] = True

    return listening_mask, speaking_mask


def _derive_output_paths(hidden_path: str) -> tuple[str, str]:
    """Derive output wav / text paths from a hidden payload path.

    ``output_hidden.pt`` → ``output.wav``, ``output.json``.

    For legacy names, falls back to ``<prefix>_output.wav/json`` where
    ``prefix`` is the hidden filename without ``_hidden.pt``.
    """
    if hidden_path.endswith("output_hidden.pt"):
        root = hidden_path[: -len("output_hidden.pt")]
        return root + "output.wav", root + "output.json"

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

        Expects ``dataset_path/<id>/input.wav`` to exist (produced by TTS).
        Outputs ``output_hidden.pt`` next to the WAV.

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
            wav = os.path.join(entry_dir, "input.wav")
            hidden = os.path.join(entry_dir, "output_hidden.pt")
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

                * ``output_hidden.pt`` is labeled using
                    ``modes.listening`` / ``modes.speaking`` ranges from ``input.json``.

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

            # ---- output_hidden -------------------------------------------
            output_hidden_path = os.path.join(entry_dir, "output_hidden.pt")
            if not os.path.exists(output_hidden_path):
                raise FileNotFoundError(
                    f"Missing {output_hidden_path}. "
                    "Run --gen-dataset-hidden first."
                )

            if "modes" not in meta:
                raise KeyError(
                    f"input.json for entry {entry_id} is missing "
                    "'modes' label ranges."
                )
            modes = meta["modes"]

            output_payload = _load_hidden_payload(output_hidden_path)
            output_hidden = _extract_layer(output_payload, layer)  # [T, D]
            T_output = output_hidden.shape[0]
            output_labels = _build_labels(
                T_output, modes["listening"], modes["speaking"]
            )
            all_hiddens.append(output_hidden)
            all_labels.append(output_labels)

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

        # Class-conditional feature means over the labeled token set.
        pos_mask = y == 1
        neg_mask = y == 0
        ave_hidden_pos = (
            X[pos_mask].mean(dim=0).detach().cpu().float()
            if int(pos_mask.sum()) > 0
            else torch.zeros(D, dtype=torch.float32)
        )
        ave_hidden_neg = (
            X[neg_mask].mean(dim=0).detach().cpu().float()
            if int(neg_mask.sum()) > 0
            else torch.zeros(D, dtype=torch.float32)
        )

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
                "ave_hidden_pos": ave_hidden_pos,
                "ave_hidden_neg": ave_hidden_neg,
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


def extract_normal_vector(mode_path):
    """
    Eetract decision boundary normal vector from a trained linear classifier checkpoint.
    return normal vector as a numpy array of shape (D,)
    """
    if not os.path.exists(mode_path):
        raise FileNotFoundError(f"Mode classifier checkpoint not found: {mode_path}")

    ckpt = torch.load(mode_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError(
            f"Expected checkpoint dict at {mode_path}, got {type(ckpt).__name__}."
        )

    state_dict = ckpt.get("state_dict")
    if not isinstance(state_dict, dict):
        raise KeyError(
            f"Checkpoint {mode_path} does not contain a valid 'state_dict'."
        )

    weight = state_dict.get("weight")
    if weight is None:
        raise KeyError(
            f"Checkpoint {mode_path} does not contain 'weight' in state_dict."
        )

    if not isinstance(weight, torch.Tensor):
        raise TypeError(
            f"Expected state_dict['weight'] to be a tensor, got {type(weight).__name__}."
        )

    # nn.Linear(D, 1) has weight shape [1, D]; flatten to decision-boundary normal [D].
    normal = weight.detach().cpu().float().reshape(-1)
    if normal.numel() == 0:
        raise ValueError(f"Extracted empty normal vector from checkpoint: {mode_path}")

    return normal.numpy()


def save_mean_hidden_diff(root_dir: str, output_path: Optional[str] = None) -> str:
    """Compute per-layer speaking/listening mean-hidden difference.

    Reads ``root_dir/*/input.json`` and matching hidden payload files:
            - ``output_hidden.pt`` with ``modes`` labels

    For each layer ``l``, computes:
      ``mean_speaking(l) - mean_listening(l)``

    Saves JSON to ``root_dir/mean_hidden_diff.json`` by default with schema:
      ``{"0": [...], "1": [...], ..., "31": [...]}``

    Returns:
        Path to the saved JSON file.
    """
    pattern = os.path.join(root_dir, "*", "input.json")
    entries = sorted(
        glob(pattern),
        key=lambda p: int(os.path.basename(os.path.dirname(p))),
    )
    if not entries:
        raise FileNotFoundError(f"No input.json found under {root_dir}/*/")

    speaking_sum: Optional[torch.Tensor] = None  # [L, D]
    listening_sum: Optional[torch.Tensor] = None  # [L, D]
    speaking_count: int = 0
    listening_count: int = 0

    for entry_json in entries:
        entry_dir = os.path.dirname(entry_json)
        with open(entry_json, "r", encoding="utf-8") as f:
            meta = json.load(f)

        sample_specs = [("output_hidden.pt", "modes")]

        for hidden_name, mode_key in sample_specs:
            hidden_path = os.path.join(entry_dir, hidden_name)
            if not os.path.exists(hidden_path):
                raise FileNotFoundError(
                    f"Missing {hidden_path}. Run --gen-dataset-hidden first."
                )
            if mode_key not in meta:
                raise KeyError(
                    f"input.json under {entry_dir} is missing '{mode_key}'."
                )

            modes = meta[mode_key]
            payload = _load_hidden_payload(hidden_path)
            if "text_hidden_layers" not in payload:
                raise KeyError(
                    f"{hidden_path} has no 'text_hidden_layers'. "
                    "This function expects multi-layer hidden payloads."
                )

            hidden = payload["text_hidden_layers"].float()  # [T, L, D]
            if hidden.ndim != 3:
                raise ValueError(
                    f"Expected [T, L, D] in {hidden_path}, got {tuple(hidden.shape)}"
                )

            num_tokens = int(hidden.shape[0])
            listening_mask, speaking_mask = _build_mode_masks(
                num_tokens,
                modes["listening"],
                modes["speaking"],
            )

            if int(speaking_mask.sum()) > 0:
                speak_chunk = hidden[speaking_mask].sum(dim=0).detach().cpu()  # [L, D]
                if speaking_sum is None:
                    speaking_sum = torch.zeros_like(speak_chunk)
                if speaking_sum.shape != speak_chunk.shape:
                    raise ValueError(
                        f"Layer/dim mismatch in {hidden_path}: "
                        f"expected {tuple(speaking_sum.shape)}, got {tuple(speak_chunk.shape)}"
                    )
                speaking_sum += speak_chunk
                speaking_count += int(speaking_mask.sum())

            if int(listening_mask.sum()) > 0:
                listen_chunk = hidden[listening_mask].sum(dim=0).detach().cpu()  # [L, D]
                if listening_sum is None:
                    listening_sum = torch.zeros_like(listen_chunk)
                if listening_sum.shape != listen_chunk.shape:
                    raise ValueError(
                        f"Layer/dim mismatch in {hidden_path}: "
                        f"expected {tuple(listening_sum.shape)}, got {tuple(listen_chunk.shape)}"
                    )
                listening_sum += listen_chunk
                listening_count += int(listening_mask.sum())

    if speaking_sum is None or listening_sum is None:
        raise RuntimeError("Insufficient labels: no speaking/listening tokens were collected.")
    if speaking_count <= 0 or listening_count <= 0:
        raise RuntimeError(
            f"Invalid token counts: speaking={speaking_count}, listening={listening_count}."
        )

    speaking_mean = speaking_sum / float(speaking_count)  # [L, D]
    listening_mean = listening_sum / float(listening_count)  # [L, D]
    diff = speaking_mean - listening_mean  # [L, D]

    out = output_path or os.path.join(root_dir, "mean_hidden_diff.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    diff_dict: Dict[str, List[float]] = {
        str(layer_idx): diff[layer_idx].tolist()
        for layer_idx in range(diff.shape[0])
    }

    with open(out, "w", encoding="utf-8") as f:
        json.dump(diff_dict, f, indent=2, ensure_ascii=False)

    print(
        f"[mean-hidden-diff] Saved {diff.shape[0]} layers to {out} "
        f"(speaking_tokens={speaking_count}, listening_tokens={listening_count})"
    )
    return out

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
            to *hidden_path*, its ``input`` field is shown as a text band below
            the plot.

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
    # For mode_class files like output_hidden.pt -> output.json
    # For other datasets like output_hidden.pt -> try output.json, fall back to input.json
    # Expected format: {"text": "...", "chunks": [{"text": "word", "timestamp": [start, end]}, ...]}
    hidden_p = Path(hidden_path)
    stem = hidden_p.stem  # e.g. "output_hidden"
    # Strip "_hidden" suffix to get the transcript prefix
    transcript_prefix = stem.replace("_hidden", "")  # "output"

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

    # X-axis labels explicitly include 0-based token index.
    safe_names = [_plot_safe_text(n) for n in token_names]
    xtick_labels = [f"{i}:{safe_names[i]}" for i in range(num_tokens)]
    ax.set_xticks(xs)
    try:
        ax.set_xticklabels(
            xtick_labels,
            rotation=90,
            fontsize=5,
            ha="center",
            parse_math=False,
        )
    except TypeError:
        # Older matplotlib without parse_math.
        ax.set_xticklabels(
            xtick_labels, rotation=90, fontsize=5, ha="center"
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

def plot_hidden_self_similarity(
    hidden_path: str,
    output_path: str,
    *,
    layer: int = -1,
    window: int = 5,
) -> None:
    """Plot a token-token hidden-state self-similarity heatmap.

    Steps:
    1. Load hidden activations from ``hidden_path``.
    2. Convert to a 2D matrix ``X`` of shape ``[T, D]`` (auto-handle batch dim).
    3. Smooth token activations with a sliding window ``[i-window, i+window]``.
    4. Remove anisotropy by centering over time.
    5. L2-normalize each token vector.
    6. Compute cosine self-similarity matrix ``M = X_norm @ X_norm.T``.
    7. Save a high-resolution seaborn heatmap.

    Args:
        hidden_path: Path to ``.pt`` file containing either a tensor or a
            payload dict with ``text_hidden_layers`` / ``hidden_states``.
        output_path: Output PNG path.
        layer: Layer index for ``text_hidden_layers`` (default ``-1``).
        window: Half-window size for sliding mean smoothing (default ``5``).
    """
    import importlib
    import matplotlib.pyplot as plt
    import numpy as np

    sns = importlib.import_module("seaborn")

    if window < 0:
        raise ValueError(f"window must be >= 0, got {window}")

    if not os.path.exists(hidden_path):
        raise FileNotFoundError(f"Hidden tensor/payload not found: {hidden_path}")

    raw = torch.load(hidden_path, map_location="cpu", weights_only=False)

    if isinstance(raw, dict):
        if "text_hidden_layers" in raw:
            hidden = raw["text_hidden_layers"]
            if hidden.ndim != 3:
                raise ValueError(
                    "Expected 'text_hidden_layers' shape [T, L, D], "
                    f"got {tuple(hidden.shape)}"
                )
            n_layers = hidden.shape[1]
            use_layer = layer if layer >= 0 else n_layers + layer
            if use_layer < 0 or use_layer >= n_layers:
                raise ValueError(
                    f"Layer {layer} out of range for {n_layers} layers."
                )
            X = hidden[:, use_layer, :].float()  # [T, D]
        elif "hidden_states" in raw:
            X = raw["hidden_states"].float()
        else:
            keys = ", ".join(sorted(raw.keys()))
            raise KeyError(
                "Unsupported payload dict. Expected one of "
                f"'text_hidden_layers'/'hidden_states', got keys: [{keys}]"
            )
    elif isinstance(raw, torch.Tensor):
        X = raw.float()
    else:
        raise TypeError(
            f"Unsupported .pt content type: {type(raw).__name__}. "
            "Expected tensor or dict payload."
        )

    if X.ndim == 3:
        # [B, T, D] -> first sample
        X = X[0]
    if X.ndim != 2:
        raise ValueError(
            f"Expected hidden shape [T, D] or [B, T, D], got {tuple(X.shape)}"
        )

    T, D = X.shape
    if T == 0 or D == 0:
        raise ValueError(f"Empty hidden matrix shape: {tuple(X.shape)}")

    # Sliding-window smoothing: X_smooth[i] = mean(X[j]) for j in [i-window, i+window].
    if window > 0:
        X_np = X.numpy()
        X_smooth_np = np.empty_like(X_np)
        for i in range(T):
            start = max(0, i - window)
            end = min(T, i + window + 1)
            X_smooth_np[i] = X_np[start:end].mean(axis=0)
        X_smooth = torch.from_numpy(X_smooth_np)
    else:
        X_smooth = X

    # Isotropy removal (centering over token/time dimension).
    mu = X_smooth.mean(dim=0, keepdim=True)
    X_centered = X_smooth - mu

    # Row-wise L2 normalization for cosine similarity via dot product.
    X_norm = X_centered / X_centered.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)

    # Self-similarity matrix [T, T].
    M = X_norm @ X_norm.T
    M_np = M.numpy()

    sns.set_theme(context="paper", style="white", font="serif")
    fig_size = max(6.0, min(14.0, T / 30.0))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=300)
    sns.heatmap(
        M_np,
        cmap="RdBu_r",
        center=0.0,
        square=True,
        linewidths=0.0,
        cbar_kws={"label": "Cosine similarity"},
        ax=ax,
    )
    ax.set_title("Hidden-State Self-Similarity")
    ax.set_xlabel("Token index")
    ax.set_ylabel("Token index")

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(
        "[plot] Saved self-similarity heatmap "
        f"(T={T}, D={D}, layer={layer}, window={window}) to {output_path}"
    )

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
        stem = hp.stem  # e.g. "output_hidden"
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


def _load_mono_wav(path: Path) -> tuple[Any, int]:
    """Load mono waveform from WAV path and return ``(samples[T], sample_rate)``."""
    import numpy as np
    import sphn

    if not path.exists():
        raise FileNotFoundError(f"WAV file not found: {path}")
    pcm, sr = sphn.read(str(path))
    wav = np.asarray(pcm)
    if wav.ndim == 2:
        wav = wav[0]
    elif wav.ndim != 1:
        wav = wav.reshape(-1)
    return wav.astype(np.float32), int(sr)


def plot_attention_heatmap(
    hidden_path: str,
    output_path: str,
    *,
    layer: int = -1,
) -> None:
    """Plot attention heatmap (top) and aligned user/model waveforms (bottom).

        Top subplot:
            - 2D attention-logit heatmap for selected layer from
                ``text_attention_weights``.
      - X-axis: key token position/time.
      - Y-axis: query token position.

    Bottom subplot:
      - input.wav (user) and output.wav (model) amplitudes over time.

    The figure uses shared x-axis time in seconds so key-axis and audio are aligned.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    import numpy as np

    payload = _load_hidden_payload(hidden_path)
    frame_rate_hz = float(payload.get("frame_rate", 12.5))

    attn_steps = payload.get("text_attention_weights", None)
    if not isinstance(attn_steps, list) or len(attn_steps) == 0:
        raise KeyError(
            "Payload is missing non-empty 'text_attention_weights'. "
            "Re-run inference with attention capture enabled."
        )

    first_attn = next((a for a in attn_steps if isinstance(a, torch.Tensor)), None)
    if first_attn is None:
        raise ValueError("All attention entries are None; cannot build heatmap.")
    if first_attn.ndim != 3:
        raise ValueError(
            "Expected per-step attention shape [L, H, K], "
            f"got {tuple(first_attn.shape)}"
        )

    num_layers = first_attn.shape[0]
    actual_layer = layer if layer >= 0 else num_layers + layer
    if actual_layer < 0 or actual_layer >= num_layers:
        raise ValueError(
            f"Layer {layer} out of range for attention with {num_layers} layers."
        )

    # Build a dense generated-token attention matrix [T, T].
    # We keep only the most recent generated-key window so the heatmap has a
    # causal lower-triangular layout in generated-token coordinates.
    T = len(attn_steps)
    attn_matrix = np.full((T, T), np.nan, dtype=np.float32)
    for t, entry in enumerate(attn_steps):
        if entry is None:
            continue
        if not isinstance(entry, torch.Tensor) or entry.ndim != 3:
            raise ValueError(
                f"Unexpected attention entry at step {t}: {type(entry).__name__}, "
                f"shape={tuple(entry.shape) if hasattr(entry, 'shape') else 'N/A'}"
            )
        if entry.shape[0] != num_layers:
            raise ValueError(
                f"Layer count mismatch at step {t}: got {entry.shape[0]}, expected {num_layers}"
            )
        # Mean over heads -> [K_t], then convert weight p to logit log(p/(1-p)).
        vec = entry[actual_layer].float().mean(dim=0).detach().cpu().numpy()
        vec = np.clip(vec, 1e-6, 1.0 - 1e-6)
        vec = np.log(vec / (1.0 - vec)).astype(np.float32)
        if vec.ndim != 1:
            raise ValueError(f"Expected [K_t] after head-mean at step {t}, got {vec.shape}")

        # Place most recent keys into [0..t] span (causal generated-token view).
        use_len = min(vec.shape[0], t + 1)
        if use_len <= 0:
            continue
        attn_matrix[t, t - use_len + 1 : t + 1] = vec[-use_len:]

    hidden_p = Path(hidden_path)
    input_wav = Path(str(payload.get("input_wav", hidden_p.with_name("input.wav"))))
    output_wav = Path(str(payload.get("output_wav", hidden_p.with_name("output.wav"))))

    in_wav, in_sr = _load_mono_wav(input_wav)
    out_wav, out_sr = _load_mono_wav(output_wav)
    in_times = np.arange(in_wav.shape[0], dtype=np.float32) / float(in_sr)
    out_times = np.arange(out_wav.shape[0], dtype=np.float32) / float(out_sr)

    token_end = float(T) / frame_rate_hz
    # Strict alignment: waveform subplot uses the exact same x-domain as the
    # attention heatmap key axis.
    max_t = token_end

    fig_w = max(11.0, min(18.0, max_t * 2.0))
    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(fig_w, 6.5),
        dpi=180,
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.0]},
    )

    # Map key-axis [0..T] to seconds for direct alignment with waveform axis.
    finite_vals = attn_matrix[np.isfinite(attn_matrix)]
    if finite_vals.size > 0:
        lo = float(np.percentile(finite_vals, 5.0))
        hi = float(np.percentile(finite_vals, 95.0))
        if hi <= lo:
            max_abs = float(np.max(np.abs(finite_vals))) if finite_vals.size > 0 else 1.0
            lo, hi = -max_abs, max_abs
        norm = TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi)
    else:
        norm = None

    img = ax_top.imshow(
        attn_matrix,
        cmap="coolwarm",
        norm=norm,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        extent=[0.0, token_end, -0.5, T - 0.5],
    )
    # Keep identical subplot box widths: place colorbar in an inset axis so it
    # does not shrink the top subplot relative to the waveform subplot.
    cax = ax_top.inset_axes([1.01, 0.0, 0.018, 1.0])
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label("Attention logit")
    ax_top.set_ylabel("Query token position")
    ax_top.set_title(
        f"Attention Logit Heatmap + Audio Timeline (layer={layer}, steps={T})"
    )
    ax_top.grid(False)

    ax_bot.plot(
        in_times,
        np.abs(in_wav),
        color="#2ca02c",
        linewidth=0.7,
        alpha=0.9,
        label="|input.wav| (user)",
    )
    ax_bot.plot(
        out_times,
        np.abs(out_wav),
        color="#9467bd",
        linewidth=0.7,
        alpha=0.85,
        label="|output.wav| (model)",
    )
    ax_bot.set_ylabel("Absolute amplitude")
    ax_bot.set_xlabel("Time (seconds)  [aligned with key-axis above]")
    ax_bot.grid(True, axis="x", linestyle=":", linewidth=0.7, alpha=0.65)
    ax_bot.legend(loc="upper right", fontsize=8)
    ax_bot.set_xlim(0.0, max_t)

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_p, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved attention heatmap plot to {output_path}")

def plot_attention_heatmap_at_turn_taking(root_dir, span=20, layer=-1):
    """Plot average turn-taking aligned attention windows over ``root_dir/*/``.

    Reads ``input_timing.json`` from each sample directory and aligns around:
      - ``question_start``
      - ``interrupt_start``

    For each anchor, creates one output figure with:
      - top: average attention-logit heatmap over window ``[t-span, t+span]`` tokens
      - bottom: average aligned user/model waveform over the same relative-time window
    """
    import matplotlib.pyplot as plt
    import numpy as np

    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Root directory not found: {root}")
    if span < 1:
        raise ValueError(f"span must be >= 1, got {span}")

    sample_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not sample_dirs:
        raise FileNotFoundError(f"No sample directories found under {root}")

    anchors = [
        "question_start",
        "question_end",
        "interrupt_start",
        "interrupt_end",
    ]

    def _extract_attention_matrix(payload: Dict[str, Any], layer: int = -1) -> np.ndarray:
        attn_steps = payload.get("text_attention_weights", None)
        if not isinstance(attn_steps, list) or len(attn_steps) == 0:
            raise KeyError("Missing non-empty text_attention_weights in payload")
        first_attn = next((a for a in attn_steps if isinstance(a, torch.Tensor)), None)
        if first_attn is None or first_attn.ndim != 3:
            raise ValueError("Invalid attention tensor shape in payload")
        num_layers = first_attn.shape[0]
        use_layer = layer if layer >= 0 else num_layers + layer
        if use_layer < 0 or use_layer >= num_layers:
            raise ValueError(f"Layer {layer} out of range for {num_layers} attention layers")

        T = len(attn_steps)
        mat = np.full((T, T), np.nan, dtype=np.float32)
        for t, entry in enumerate(attn_steps):
            if entry is None:
                continue
            if not isinstance(entry, torch.Tensor) or entry.ndim != 3:
                continue
            vec = entry[use_layer].float().mean(dim=0).detach().cpu().numpy()
            vec = np.clip(vec, 1e-6, 1.0 - 1e-6)
            vec = np.log(vec / (1.0 - vec)).astype(np.float32)
            use_len = min(vec.shape[0], t + 1)
            if use_len > 0:
                mat[t, t - use_len + 1 : t + 1] = vec[-use_len:]
        return mat

    def _extract_centered_square(mat: np.ndarray, center: int, half: int) -> np.ndarray:
        side = 2 * half + 1
        out = np.full((side, side), np.nan, dtype=np.float32)
        for i in range(side):
            src_i = center - half + i
            if src_i < 0 or src_i >= mat.shape[0]:
                continue
            for j in range(side):
                src_j = center - half + j
                if src_j < 0 or src_j >= mat.shape[1]:
                    continue
                out[i, j] = mat[src_i, src_j]
        return out

    # Gather aligned windows per anchor.
    per_anchor_heatmaps: dict[str, list[np.ndarray]] = {k: [] for k in anchors}
    per_anchor_user_wavs: dict[str, list[np.ndarray]] = {k: [] for k in anchors}
    per_anchor_model_wavs: dict[str, list[np.ndarray]] = {k: [] for k in anchors}
    per_anchor_frame_rate: dict[str, list[float]] = {k: [] for k in anchors}

    for sample_dir in sample_dirs:
        timing_path = sample_dir / "input_timing.json"
        hidden_path = sample_dir / "output_hidden.pt"
        if not hidden_path.exists():
            hidden_path = sample_dir / "output_hidden"
        if not timing_path.exists() or not hidden_path.exists():
            continue

        try:
            with timing_path.open("r", encoding="utf-8") as f:
                timing = json.load(f)
            if not isinstance(timing, dict):
                continue

            payload = _load_hidden_payload(str(hidden_path))
            frame_rate_hz = float(payload.get("frame_rate", 12.5))
            attn_mat = _extract_attention_matrix(payload, layer=layer)

            hidden_p = Path(str(hidden_path))
            input_wav = Path(str(payload.get("input_wav", hidden_p.with_name("input.wav"))))
            output_wav = Path(str(payload.get("output_wav", hidden_p.with_name("output.wav"))))
            in_wav, in_sr = _load_mono_wav(input_wav)
            out_wav, out_sr = _load_mono_wav(output_wav)
            in_t = np.arange(in_wav.shape[0], dtype=np.float32) / float(in_sr)
            out_t = np.arange(out_wav.shape[0], dtype=np.float32) / float(out_sr)

            window_sec = float(span) / frame_rate_hz
            rel_grid = np.linspace(-window_sec, window_sec, 2 * span + 1, dtype=np.float32)

            for anchor in anchors:
                if anchor not in timing:
                    continue
                anchor_sec = float(timing[anchor])
                center_tok = int(round(anchor_sec * frame_rate_hz))

                # Attention aligned around anchor token index.
                local = _extract_centered_square(attn_mat, center_tok, span)
                per_anchor_heatmaps[anchor].append(local)
                per_anchor_frame_rate[anchor].append(frame_rate_hz)

                # Waveforms aligned around anchor time (sampled to token-grid length).
                in_local = np.interp(
                    anchor_sec + rel_grid,
                    in_t,
                    in_wav,
                    left=np.nan,
                    right=np.nan,
                )
                out_local = np.interp(
                    anchor_sec + rel_grid,
                    out_t,
                    out_wav,
                    left=np.nan,
                    right=np.nan,
                )
                per_anchor_user_wavs[anchor].append(in_local.astype(np.float32))
                per_anchor_model_wavs[anchor].append(out_local.astype(np.float32))
        except Exception:
            continue

    # Plot one figure per anchor.
    for anchor in anchors:
        if len(per_anchor_heatmaps[anchor]) == 0:
            print(f"[plot-turn] No valid samples for {anchor}; skipping")
            continue

        import numpy as np
        heat_stack = np.stack(per_anchor_heatmaps[anchor], axis=0)
        if not np.isfinite(heat_stack).any():
            print(f"[plot-turn] No finite attention values for {anchor}; skipping")
            continue
        heat = np.nanmean(heat_stack, axis=0)

        user_stack = np.stack(per_anchor_user_wavs[anchor], axis=0)
        model_stack = np.stack(per_anchor_model_wavs[anchor], axis=0)
        # Average absolute amplitude to avoid sign cancellation across samples.
        avg_user = np.nanmean(np.abs(user_stack), axis=0)
        avg_model = np.nanmean(np.abs(model_stack), axis=0)

        frame_rate_hz = float(np.nanmedian(np.asarray(per_anchor_frame_rate[anchor], dtype=np.float32)))
        window_sec = float(span) / frame_rate_hz
        rel_sec = np.linspace(-window_sec, window_sec, heat.shape[0], dtype=np.float32)

        fig_w = max(10.5, min(16.0, 10.0 + 0.05 * heat.shape[0]))
        fig, (ax_top, ax_bot) = plt.subplots(
            2,
            1,
            figsize=(fig_w, 6.6),
            dpi=180,
            sharex=True,
            gridspec_kw={"height_ratios": [2.3, 1.0]},
        )

        finite_vals = heat[np.isfinite(heat)]
        heat_display = heat.copy()
        imshow_kwargs: dict[str, Any] = {
            "cmap": "coolwarm",
            "vmin": 5.0,
            "vmax": 95.0,
        }
        if finite_vals.size > 0:
            p5 = float(np.percentile(finite_vals, 5.0))
            p95 = float(np.percentile(finite_vals, 95.0))
            if p95 <= p5:
                p95 = p5 + 1e-6

            scale = 90.0 / (p95 - p5)
            finite_mask = np.isfinite(heat_display)
            heat_display[finite_mask] = (
                (heat_display[finite_mask] - p5) * scale
            ) + 5.0
            heat_display[finite_mask] = np.clip(heat_display[finite_mask], 5.0, 95.0)

        img = ax_top.imshow(
            heat_display,
            aspect="auto",
            interpolation="nearest",
            origin="lower",
            extent=[-window_sec, window_sec, -window_sec, window_sec],
            **imshow_kwargs,
        )
        cax = ax_top.inset_axes([1.01, 0.0, 0.018, 1.0])
        cbar = fig.colorbar(img, cax=cax)
        cbar.set_label("Attention logit (P5->5, P95->95)")
        ax_top.set_ylabel("Query offset (s)")
        ax_top.set_title(
            f"Average Attention Around {anchor} (layer={layer}, span={span}, n={len(per_anchor_heatmaps[anchor])})"
        )

        ax_bot.plot(rel_sec, avg_user, color="#2ca02c", linewidth=0.9, alpha=0.95, label="|input.wav| avg")
        ax_bot.plot(rel_sec, avg_model, color="#9467bd", linewidth=0.9, alpha=0.9, label="|output.wav| avg")
        ax_bot.axvline(0.0, color="#444444", linestyle="--", linewidth=0.8, alpha=0.8)
        ax_bot.set_ylabel("Absolute amplitude")
        ax_bot.set_xlabel(f"Relative time to {anchor} (s)")
        ax_bot.grid(True, axis="x", linestyle=":", linewidth=0.7, alpha=0.65)
        ax_bot.legend(loc="upper right", fontsize=8)
        ax_bot.set_xlim(-window_sec, window_sec)

        out_path = root / f"attention_heatmap_turn_taking_{anchor}.png"
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot-turn] Saved {out_path}")

def plot_attention_heatmap_dataset(
    root_dir: str,
    *,
    layer: int = -1,
) -> None:
    """Recursively find all ``output_hidden.pt`` files and plot attention+audio."""
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Root directory not found: {root}")

    hidden_files = sorted(p for p in root.rglob("output_hidden.pt") if p.is_file())
    if not hidden_files:
        raise FileNotFoundError(f"No output_hidden.pt files found under {root}")

    print(f"[plot-attn] Found {len(hidden_files)} output_hidden.pt files under {root}")
    ok = 0
    for hp in hidden_files:
        out_png = hp.with_name(f"attention_heatmap_layer_{layer}.png")
        print(f"\n--- {hp} ---")
        try:
            plot_attention_heatmap(str(hp), str(out_png), layer=layer)
            ok += 1
        except Exception as exc:
            print(f"  [SKIP] attention plot failed: {exc}")

    print(f"\n[plot-attn] Done. Generated {ok}/{len(hidden_files)} plots.")

def plot_attention_by_subseqent_token_heatmap(
    root_dir: str,
    span: int = 50,
    window: int = 50,
):
    """
    Plot attention heatmaps for each subsequence token.

    The plot
    - X-axis: tokens. start from the token at interrupt_start as indicated in input_timing.json, to that token + span
    - Y-axis: layers of transformer
    For the value of each cell, let's say token t at layer l, it should be the average attention weight of all the tokens AFTER token t,
    which is the average attention weight of t+1, t+2, ... to the end of the sequence, attending to token t at layer l. This way we can see 
    how the attention to a specific token evolves as more tokens are generated after it.
    
    """
    import matplotlib.pyplot as plt
    from matplotlib import ticker as mticker
    import numpy as np

    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Root directory not found: {root}")
    if span < 1:
        raise ValueError(f"span must be >= 1, got {span}")
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    sample_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not sample_dirs:
        raise FileNotFoundError(f"No sample directories found under {root}")

    def _jsonable_2d(arr: np.ndarray) -> List[List[Optional[float]]]:
        out: List[List[Optional[float]]] = []
        for row in arr:
            out.append([
                (float(v) if np.isfinite(v) else None) for v in row
            ])
        return out

    def _build_attn_prob_matrix(payload: Dict[str, Any], layer_idx: int) -> np.ndarray:
        attn_steps = payload.get("text_attention_weights", None)
        if not isinstance(attn_steps, list) or len(attn_steps) == 0:
            raise KeyError("Missing non-empty text_attention_weights in payload")

        T = len(attn_steps)
        mat = np.full((T, T), np.nan, dtype=np.float32)
        for q, entry in enumerate(attn_steps):
            if entry is None:
                continue
            if not isinstance(entry, torch.Tensor) or entry.ndim != 3:
                continue

            # entry[layer_idx]: [H, K_q], mean over heads -> [K_q]
            vec = entry[layer_idx].float().mean(dim=0).detach().cpu().numpy().astype(np.float32)
            if vec.ndim != 1:
                continue

            # Align only generated-token keys (causal window) into columns [0..q]
            use_len = min(vec.shape[0], q + 1)
            if use_len > 0:
                mat[q, q - use_len + 1 : q + 1] = vec[-use_len:]
        return mat

    # Collect per-sample anchored matrices for plotting: [L, span+1]
    sample_maps: List[np.ndarray] = []
    num_layers_ref: Optional[int] = None

    for sd in sample_dirs:
        timing_path = sd / "input_timing.json"
        hidden_path = sd / "output_hidden.pt"
        if not hidden_path.exists():
            hidden_path = sd / "output_hidden"
        if not timing_path.is_file() or not hidden_path.is_file():
            continue

        try:
            with timing_path.open("r", encoding="utf-8") as f:
                timing = json.load(f)
            if not isinstance(timing, dict) or "interrupt_start" not in timing:
                continue

            payload = _load_hidden_payload(str(hidden_path))
            frame_rate_hz = float(payload.get("frame_rate", 12.5))
            anchor_tok = int(round(float(timing["interrupt_start"]) * frame_rate_hz))
            if anchor_tok < 0:
                continue

            attn_steps = payload.get("text_attention_weights", None)
            if not isinstance(attn_steps, list) or len(attn_steps) == 0:
                continue
            first_attn = next((a for a in attn_steps if isinstance(a, torch.Tensor)), None)
            if first_attn is None or first_attn.ndim != 3:
                continue

            L = int(first_attn.shape[0])
            if num_layers_ref is None:
                num_layers_ref = L
            elif num_layers_ref != L:
                # Keep a consistent layer dimension across samples.
                continue

            T = len(attn_steps)
            # Full conversation values for this sample: [L, T].
            local_all_tokens = np.full((L, T), np.nan, dtype=np.float32)
            for l in range(L):
                mat = _build_attn_prob_matrix(payload, l)  # [T, T]
                for t in range(T):
                    if t >= T - 1:
                        continue
                    # Average of future queries in a bounded horizon:
                    # (t+1 .. min(t+window, T-1)) attending to key token t.
                    end_q = min(T - 1, t + window)
                    if end_q < t + 1:
                        continue
                    vals = mat[t + 1 : end_q + 1, t]
                    finite = vals[np.isfinite(vals)]
                    if finite.size > 0:
                        local_all_tokens[l, t] = float(np.mean(finite))

            # Keep plotting behavior anchored at interrupt_start.
            local = np.full((L, span + 1), np.nan, dtype=np.float32)
            for off in range(span + 1):
                t = anchor_tok + off
                if 0 <= t < T:
                    local[:, off] = local_all_tokens[:, t]

            per_sample_json = {
                "anchor": "interrupt_start",
                "span": int(span),
                "future_query_window": int(window),
                "frame_rate_hz": float(frame_rate_hz),
                "anchor_seconds": float(timing["interrupt_start"]),
                "anchor_token_index": int(anchor_tok),
                "num_layers": int(L),
                "num_steps": int(T),
                "token_indices": list(range(T)),
                "avg_future_attention_weight_by_layer_all_tokens": _jsonable_2d(local_all_tokens),
                "token_offsets_from_interrupt_start": list(range(span + 1)),
                "avg_future_attention_weight_by_layer_anchor_window": _jsonable_2d(local),
            }
            per_sample_out = sd / "attention_subseq_token.json"
            with per_sample_out.open("w", encoding="utf-8") as f:
                json.dump(per_sample_json, f, indent=2, ensure_ascii=False)

            sample_maps.append(local)
        except Exception:
            continue

    if not sample_maps:
        raise FileNotFoundError(
            "No valid samples found with input_timing.json (interrupt_start) and output_hidden(.pt)."
        )

    heat = np.nanmean(np.stack(sample_maps, axis=0), axis=0)  # [L, span+1]
    L = int(heat.shape[0])

    finite = heat[np.isfinite(heat)]
    if finite.size > 0:
        vmin = float(np.percentile(finite, 5.0))
        vmax = float(np.percentile(finite, 95.0))
        if vmax <= vmin:
            vmax = vmin + 1e-6
    else:
        vmin, vmax = 0.0, 1.0

    x = np.arange(0, span + 1, dtype=np.int32)
    fig, ax = plt.subplots(figsize=(11.0, 6.2), dpi=180)
    img = ax.imshow(
        heat,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        extent=(float(x[0]), float(x[-1]), -0.5, float(L) - 0.5),
    )
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label("Avg future attention weight to token t (vmin=P5, vmax=P95)")

    ax.set_title(
        f"Attention to Subsequent Tokens (anchor=interrupt_start, span={span}, n={len(sample_maps)})"
    )
    ax.set_xlabel("Token offset from interrupt_start")
    ax.set_ylabel("Layer")
    ax.set_yticks(np.arange(0, L, 1))
    ax.set_ylim(-0.5, float(L) - 0.5)

    out_png = root / "attention_by_subsequent_token_heatmap.png"
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot-attn-subseq] Saved {out_png}")

def plot_logit_lens_step_n(
    hidden_path: str,
    output_path: str,
    *,
    lm: Any,
    layer: int = -1,
    ma_window: int = 5,
    ce_json_path: Optional[str] = None,
    save_plot: bool = True,
) -> None:
    """Plot step-n premature-decode CE losses and aligned waveforms.

    Top subplot:
            - User Multi-modal CE (shifted): logits from step n vs user targets at step n+1
                i.e. (CE_audio_user_shift + CE_text_user_shift) / 2
    - Model Multi-modal CE: (CE_audio_model + CE_text_model) / 2

    Bottom subplot:
      - input.wav and output.wav amplitudes over physical time.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch.nn.functional as F

    payload = _load_hidden_payload(hidden_path)
    frame_rate_hz = float(payload.get("frame_rate", 12.5))

    required = ["text_hidden_layers", "input_token_ids", "output_token_ids"]
    for key in required:
        if key not in payload:
            raise KeyError(
                f"Payload is missing '{key}'. Re-run inference with latest hidden payload saving."
            )

    hidden = payload["text_hidden_layers"].float()  # [T, L, D]
    input_token_ids = payload["input_token_ids"].long()  # [T, K_in]
    output_token_ids = payload["output_token_ids"].long()  # [T, K_out]

    if hidden.ndim != 3:
        raise ValueError(f"Expected text_hidden_layers [T, L, D], got {tuple(hidden.shape)}")
    if input_token_ids.ndim != 2:
        raise ValueError(f"Expected input_token_ids [T, K_in], got {tuple(input_token_ids.shape)}")
    if output_token_ids.ndim != 2:
        raise ValueError(f"Expected output_token_ids [T, K_out], got {tuple(output_token_ids.shape)}")

    T, num_layers, d_hidden = hidden.shape
    if T < 2:
        raise ValueError("Need at least 2 token steps to compute user n+1 shifted CE.")
    if input_token_ids.shape[0] != T or output_token_ids.shape[0] != T:
        raise ValueError(
            "Token-length mismatch across hidden/token-id tensors: "
            f"hidden={T}, input_token_ids={input_token_ids.shape[0]}, output_token_ids={output_token_ids.shape[0]}"
        )

    actual_layer = layer if layer >= 0 else num_layers + layer
    if actual_layer < 0 or actual_layer >= num_layers:
        raise ValueError(f"Layer {layer} out of range for hidden with {num_layers} layers.")

    # Expected token layout:
    # input_token_ids: [text(0), model_audio(1..8), user_audio(9..16)]
    # output_token_ids: [text(0), model_audio(1..8)]
    if input_token_ids.shape[1] < 10:
        raise ValueError(
            f"input_token_ids width too small ({input_token_ids.shape[1]}), expected >=10 for user audio cb0 at index 9."
        )
    if output_token_ids.shape[1] < 2:
        raise ValueError(
            f"output_token_ids width too small ({output_token_ids.shape[1]}), expected >=2 for model audio cb0 at index 1."
        )

    h_l = hidden[:, actual_layer, :]  # [T, D]
    text_tokens = output_token_ids[:, 0]  # [T], used to condition depformer audio decode.
    user_audio_target = input_token_ids[:, 9]  # first user-audio codebook
    model_audio_target = output_token_ids[:, 1]  # first model-audio codebook
    user_text_target = input_token_ids[:, 0]
    model_text_target = output_token_ids[:, 0]

    device = lm.device
    lm_dtype = next(lm.parameters()).dtype
    with torch.no_grad():
        x = h_l.to(device=device, dtype=lm_dtype)[:, None, :]  # [T, 1, D]
        if getattr(lm, "out_norm", None) is not None:
            x = lm.out_norm(x)

        # Premature text logits directly from the text decode head.
        text_logits = lm.text_linear(x)[:, 0, :].float()  # [T, text_card(+pad)]

        dep_in = text_tokens.to(device=device, dtype=torch.long)[:, None, None]  # [T,1,1]
        # First audio codebook decode head logits.
        with lm.depformer.streaming(T):
            logits0 = lm.forward_depformer(0, dep_in, x)  # [T, 1, 1, card]
        logits0 = logits0[:, 0, 0, :].float()  # [T, card]

        def _ce_from_probs(logits_2d: torch.Tensor, target_1d: torch.Tensor) -> torch.Tensor:
            probs = torch.softmax(logits_2d, dim=-1)
            return F.nll_loss(
                torch.log(probs.clamp_min(1e-12)),
                target_1d,
                reduction="none",
            )

        # User-focus CE is shifted by +1 target step: compare probs(n) with user_target(n+1).
        user_audio_ce = _ce_from_probs(
            logits0[:-1],
            user_audio_target[1:].to(device=device, dtype=torch.long),
        )

        # Model-focus CE remains step-aligned with n on the same valid plotted range [0..T-2].
        model_audio_ce = _ce_from_probs(
            logits0[:-1],
            model_audio_target[:-1].to(device=device, dtype=torch.long),
        )

        user_text_ce = _ce_from_probs(
            text_logits[:-1],
            user_text_target[1:].to(device=device, dtype=torch.long),
        )
        model_text_ce = _ce_from_probs(
            text_logits[:-1],
            model_text_target[:-1].to(device=device, dtype=torch.long),
        )

        ce_user = 0.5 * (user_audio_ce + user_text_ce)
        ce_model = 0.5 * (model_audio_ce + model_text_ce)

    ce_user_s = ce_user.detach().cpu().float()
    ce_model_s = ce_model.detach().cpu().float()
    ratio = ce_user_s / ce_model_s.clamp_min(1e-6)

    if ce_json_path is not None:
        ce_payload = {
            "line1_user_multimodal_ce": ce_user_s.tolist(),
            "line2_model_multimodal_ce": ce_model_s.tolist(),
            "ratio_line1_over_line2": ratio.tolist(),
            "line1_shift": "n_to_n_plus_1",
            "line2_shift": "n_to_n",
            "moving_average_window": 1,
            "smoothing": "none",
            "layer": int(layer),
            "num_points": int(ce_user_s.shape[0]),
        }
        ce_out = Path(ce_json_path)
        ce_out.parent.mkdir(parents=True, exist_ok=True)
        with open(ce_out, "w", encoding="utf-8") as f:
            json.dump(ce_payload, f, indent=2, ensure_ascii=False)

    if not save_plot:
        return

    times = payload.get("times", None)
    if isinstance(times, torch.Tensor) and times.ndim == 1 and times.shape[0] == T:
        x_sec = times.detach().cpu().float().numpy()[:-1]
    else:
        x_sec = (np.arange(T - 1, dtype=np.float32) / float(frame_rate_hz)).astype(np.float32)

    hidden_p = Path(hidden_path)
    input_wav = Path(str(payload.get("input_wav", hidden_p.with_name("input.wav"))))
    output_wav = Path(str(payload.get("output_wav", hidden_p.with_name("output.wav"))))

    in_wav, in_sr = _load_mono_wav(input_wav)
    out_wav, out_sr = _load_mono_wav(output_wav)
    in_times = np.arange(in_wav.shape[0], dtype=np.float32) / float(in_sr)
    out_times = np.arange(out_wav.shape[0], dtype=np.float32) / float(out_sr)

    token_end = float(T) / frame_rate_hz
    max_t = token_end

    fig_w = max(11.0, min(18.0, max_t * 2.0))
    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(fig_w, 6.8),
        dpi=180,
        sharex=True,
        gridspec_kw={"height_ratios": [1.8, 1.0]},
    )

    ratio_np = ratio.numpy()
    ax_top.plot(
        x_sec,
        ratio_np,
        color="#1f77b4",
        linewidth=1.2,
        label="CE ratio: line1/line2 (raw)",
    )
    ax_top.set_ylabel("CE ratio")
    ax_top.set_title(
        f"Logit Lens CE Ratio line1/line2 (layer={layer}, tokens={T})"
    )
    y_all = ratio_np[np.isfinite(ratio_np)]
    if y_all.size > 0:
        y_lo = float(np.percentile(y_all, 1.0))
        y_hi = float(np.percentile(y_all, 99.0))
        if y_hi <= y_lo:
            y_mid = float(y_all.mean())
            y_lo, y_hi = y_mid - 0.05, y_mid + 0.05
        pad = max(0.01, 0.12 * (y_hi - y_lo))
        ax_top.set_ylim(y_lo - pad, y_hi + pad)
    ax_top.grid(True, axis="x", linestyle=":", linewidth=0.7, alpha=0.65)
    ax_top.legend(loc="upper right", fontsize=8)

    ax_bot.plot(
        in_times,
        in_wav,
        color="#2ca02c",
        linewidth=0.7,
        alpha=0.9,
        label="input.wav (user)",
    )
    ax_bot.plot(
        out_times,
        out_wav,
        color="#9467bd",
        linewidth=0.7,
        alpha=0.85,
        label="output.wav (model)",
    )
    ax_bot.set_ylabel("Amplitude")
    ax_bot.set_xlabel("Time (seconds)")
    ax_bot.grid(True, axis="x", linestyle=":", linewidth=0.7, alpha=0.65)
    ax_bot.legend(loc="upper right", fontsize=8)
    # Crucial alignment: same physical-time x-range for both subplots.
    ax_bot.set_xlim(0.0, max_t)

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_p, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved logit-lens CE plot to {output_path}")

def plot_logit_lens_dataset(
    root_dir: str,
    *,
    layer: int = -1,
    hf_repo: str = loaders.DEFAULT_REPO,
    moshi_weight: Optional[str] = None,
    device: str = "cuda",
    ma_window: int = 5,
) -> None:
    """Find ``root_dir/*/output_hidden(.pt)`` and plot logit-lens CE for each.

    If ``layer == -1``, generates plots/CE JSON for all available layers.
    Otherwise only the specified layer is processed.
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Root directory not found: {root}")

    hidden_files: list[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        cands = [child / "output_hidden.pt", child / "output_hidden"]
        found = next((p for p in cands if p.is_file()), None)
        if found is not None:
            hidden_files.append(found)

    if not hidden_files:
        raise FileNotFoundError(
            f"No output_hidden(.pt) files found under {root}/*/"
        )

    if moshi_weight is None:
        moshi_weight = hf_hub_download(hf_repo, loaders.MOSHI_NAME)  # type: ignore
    lm = loaders.get_moshi_lm(moshi_weight, device=device, cpu_offload=False)
    lm.eval()

    print(
        f"[plot-logit-lens] Found {len(hidden_files)} output_hidden(.pt) files under {root}/*/"
    )
    ok = 0
    total_jobs = 0
    for hp in hidden_files:
        print(f"\n--- {hp} ---")
        try:
            payload = _load_hidden_payload(str(hp))
            if layer == -1:
                if "text_hidden_layers" in payload:
                    hidden = payload["text_hidden_layers"]
                    if hidden.ndim != 3:
                        raise ValueError(
                            f"Expected text_hidden_layers [T,L,D], got {tuple(hidden.shape)}"
                        )
                    layer_list = list(range(int(hidden.shape[1])))
                elif "hidden_states" in payload:
                    layer_list = [-1]
                else:
                    raise KeyError(
                        "Payload has neither 'text_hidden_layers' nor 'hidden_states'."
                    )
            else:
                layer_list = [layer]

            total_jobs += len(layer_list)
            for use_layer in layer_list:
                out_png = hp.with_name(f"logit_lens_ce_layer_{use_layer}.png")
                out_json = hp.with_name(f"in_out_ce_{use_layer}.json")
                plot_logit_lens_step_n(
                    str(hp),
                    str(out_png),
                    lm=lm,
                    layer=use_layer,
                    ma_window=ma_window,
                    ce_json_path=str(out_json),
                    save_plot=False,
                )
                ok += 1
        except Exception as exc:
            print(f"  [SKIP] logit-lens plot failed: {exc}")

    print(f"\n[plot-logit-lens] Done. Generated {ok}/{total_jobs} plots.")


def plot_logit_lens_turn_taking_from_saved(
    root_dirs: List[str],
    *,
    span: int = 50,
) -> None:
    """Average saved logit-lens CE traces around turn-taking anchors.

    Reads ``<root>/*/in_out_ce_*.json`` and ``<root>/*/input_timing.json`` for
    each root in ``root_dirs``. For each layer and anchor
    (``question_start``, ``interrupt_start``), aligns each sample by anchor time
    and averages a token window ``[t-span, t+span]``.

    If multiple roots are provided, their averaged curves are overlaid on the
    same graph (one line per root). Outputs are saved under the first root.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if not root_dirs:
        raise ValueError("root_dirs must contain at least one directory")
    if span < 1:
        raise ValueError(f"span must be >= 1, got {span}")

    anchors = ["question_start", "interrupt_start"]

    def _extract_centered_1d(arr: np.ndarray, center: int, half: int) -> np.ndarray:
        out = np.full((2 * half + 1,), np.nan, dtype=np.float32)
        start = center - half
        end = center + half
        src_l = max(0, start)
        src_r = min(arr.shape[0] - 1, end)
        if src_r < src_l:
            return out
        dst_l = src_l - start
        dst_r = dst_l + (src_r - src_l)
        out[dst_l : dst_r + 1] = arr[src_l : src_r + 1]
        return out

    def _discover_ce_files(sample_dir: Path) -> Dict[int, Path]:
        layer_files: Dict[int, Path] = {}
        for p in sorted(sample_dir.glob("in_out_ce_*.json")):
            stem = p.stem
            suffix = stem.replace("in_out_ce_", "", 1)
            if suffix.lstrip("-").isdigit():
                layer_files[int(suffix)] = p
        legacy = sample_dir / "in_out_ce.json"
        if legacy.is_file() and -1 not in layer_files:
            layer_files[-1] = legacy
        return layer_files

    def _collect_per_root(
        root: Path,
        sample_ids: List[str],
    ) -> Dict[int, Dict[str, Dict[str, Any]]]:
        if not sample_ids:
            return {}

        per_layer: Dict[int, Dict[str, Dict[str, list[np.ndarray]]]] = {}

        for sample_id in sample_ids:
            sample_dir = root / sample_id
            timing_path = sample_dir / "input_timing.json"
            ce_files = _discover_ce_files(sample_dir)
            if not timing_path.exists() or not ce_files:
                print(
                    f"[plot-logit-turn][WARN] Missing required files in {sample_dir}; skipping"
                )
                continue

            try:
                with timing_path.open("r", encoding="utf-8") as f:
                    timing = json.load(f)
                if not isinstance(timing, dict):
                    continue

                frame_rate_hz = 12.5
                input_wav_path = sample_dir / "input.wav"
                hidden_cands = [sample_dir / "output_hidden.pt", sample_dir / "output_hidden"]
                hidden_found = next((p for p in hidden_cands if p.is_file()), None)
                if hidden_found is not None:
                    try:
                        payload = _load_hidden_payload(str(hidden_found))
                        frame_rate_hz = float(payload.get("frame_rate", frame_rate_hz))
                        input_wav_path = Path(str(payload.get("input_wav", input_wav_path)))
                    except Exception:
                        pass

                input_wav = None
                input_times = None
                try:
                    wav, sr = _load_mono_wav(input_wav_path)
                    input_wav = np.abs(wav)
                    input_times = np.arange(input_wav.shape[0], dtype=np.float32) / float(sr)
                except Exception:
                    print(
                        f"[plot-logit-turn][WARN] Unable to load input wav for {sample_dir}; "
                        "bottom amplitude plot will skip this sample"
                    )

                for layer_val, ce_path in ce_files.items():
                    with ce_path.open("r", encoding="utf-8") as f:
                        ce_data = json.load(f)
                    if not isinstance(ce_data, dict):
                        continue

                    line1 = np.asarray(
                        ce_data.get("line1_user_multimodal_ce", []), dtype=np.float32
                    )
                    line2 = np.asarray(
                        ce_data.get("line2_model_multimodal_ce", []), dtype=np.float32
                    )
                    ratio = np.asarray(
                        ce_data.get("ratio_line1_over_line2", []), dtype=np.float32
                    )
                    if line1.ndim != 1 or line2.ndim != 1:
                        continue
                    if line1.shape[0] == 0 or line2.shape[0] == 0:
                        continue

                    n = min(line1.shape[0], line2.shape[0])
                    line1 = line1[:n]
                    line2 = line2[:n]
                    if ratio.ndim == 1 and ratio.shape[0] >= n:
                        ratio = ratio[:n]
                    else:
                        ratio = line1 / np.clip(line2, 1e-6, None)

                    bucket = per_layer.setdefault(
                        int(layer_val),
                        {
                            "line1": {k: [] for k in anchors},
                            "line2": {k: [] for k in anchors},
                            "ratio": {k: [] for k in anchors},
                            "amp": {k: [] for k in anchors},
                        },
                    )

                    for anchor in anchors:
                        if anchor not in timing:
                            continue
                        anchor_sec = float(timing[anchor])
                        center_tok = int(round(anchor_sec * frame_rate_hz))
                        if center_tok < 0:
                            continue

                        bucket["line1"][anchor].append(
                            _extract_centered_1d(line1, center_tok, span)
                        )
                        bucket["line2"][anchor].append(
                            _extract_centered_1d(line2, center_tok, span)
                        )
                        bucket["ratio"][anchor].append(
                            _extract_centered_1d(ratio, center_tok, span)
                        )

                        if input_wav is not None and input_times is not None:
                            window_sec = float(span) / frame_rate_hz
                            rel_grid = np.linspace(
                                -window_sec,
                                window_sec,
                                2 * span + 1,
                                dtype=np.float32,
                            )
                            input_local = np.interp(
                                anchor_sec + rel_grid,
                                input_times,
                                input_wav,
                                left=np.nan,
                                right=np.nan,
                            ).astype(np.float32)
                            bucket["amp"][anchor].append(input_local)
            except Exception:
                continue

        out: Dict[int, Dict[str, Dict[str, Any]]] = {}
        for layer_val, bucket in per_layer.items():
            layer_out: Dict[str, Dict[str, Any]] = {}
            for anchor in anchors:
                if len(bucket["ratio"][anchor]) == 0:
                    continue
                ratio_stack = np.stack(bucket["ratio"][anchor], axis=0)
                if not np.isfinite(ratio_stack).any():
                    continue

                layer_out[anchor] = {
                    "avg_line1": np.nanmean(
                        np.stack(bucket["line1"][anchor], axis=0), axis=0
                    ),
                    "avg_line2": np.nanmean(
                        np.stack(bucket["line2"][anchor], axis=0), axis=0
                    ),
                    "avg_ratio": np.nanmean(ratio_stack, axis=0),
                    "num_samples": int(len(bucket["ratio"][anchor])),
                    "avg_input_amp": (
                        np.nanmean(np.stack(bucket["amp"][anchor], axis=0), axis=0)
                        if len(bucket["amp"][anchor]) > 0
                        else np.full((2 * span + 1,), np.nan, dtype=np.float32)
                    ),
                    "num_samples_input_amp": int(len(bucket["amp"][anchor])),
                }
            if layer_out:
                out[int(layer_val)] = layer_out
        return out

    parsed_roots: List[Path] = []
    for rd in root_dirs:
        rp = Path(rd)
        if not rp.is_dir():
            raise FileNotFoundError(f"Root directory not found: {rp}")
        parsed_roots.append(rp)

    # Build per-root valid sample id sets, then keep only the intersection.
    # This ensures all overlaid curves are computed from the same shared samples.
    root_valid_ids: Dict[Path, set[str]] = {}
    for rp in parsed_roots:
        sample_dirs = sorted([p for p in rp.iterdir() if p.is_dir()])
        if not sample_dirs:
            root_valid_ids[rp] = set()
            print(f"[plot-logit-turn][WARN] No subdirectories found under {rp}")
            continue

        valid_ids: set[str] = set()
        missing: list[str] = []
        for sd in sample_dirs:
            has_timing = (sd / "input_timing.json").is_file()
            has_ce = (sd / "in_out_ce.json").is_file() or any(
                sd.glob("in_out_ce_*.json")
            )
            if has_timing and has_ce:
                valid_ids.add(sd.name)
            else:
                missing_parts: list[str] = []
                if not has_timing:
                    missing_parts.append("input_timing.json")
                if not has_ce:
                    missing_parts.append("in_out_ce_*.json")
                missing.append(f"{sd.name} ({'+'.join(missing_parts)})")

        root_valid_ids[rp] = valid_ids
        if missing:
            preview = ", ".join(missing[:8])
            if len(missing) > 8:
                preview += ", ..."
            print(
                f"[plot-logit-turn][WARN] {rp}: {len(missing)} subdirs missing required files: {preview}"
            )

    common_ids: set[str] = set.intersection(*root_valid_ids.values()) if root_valid_ids else set()
    if not common_ids:
        raise FileNotFoundError(
            "No shared valid subdirectories across provided roots. "
            "Need subdirs that exist with both input_timing.json and in_out_ce.json in every dataset root."
        )

    # Warn about subdirs excluded because they are not present/valid in all roots.
    union_ids: set[str] = set.union(*root_valid_ids.values()) if root_valid_ids else set()
    excluded_ids = sorted(union_ids - common_ids)
    if excluded_ids:
        preview = ", ".join(excluded_ids[:10])
        if len(excluded_ids) > 10:
            preview += ", ..."
        print(
            f"[plot-logit-turn][WARN] Excluding {len(excluded_ids)} non-shared subdirs; using only common valid subset ({len(common_ids)}): {preview}"
        )
    else:
        print(
            f"[plot-logit-turn] Using {len(common_ids)} shared valid subdirs across all provided roots."
        )

    shared_ids = sorted(common_ids)

    # Build unique labels in case root basenames collide.
    used_labels: set[str] = set()
    labeled_roots: List[tuple[str, Path]] = []
    for idx, rp in enumerate(parsed_roots, start=1):
        base = rp.name
        label = base if base not in used_labels else f"{base}_{idx}"
        used_labels.add(label)
        labeled_roots.append((label, rp))

    per_root: Dict[str, Dict[int, Dict[str, Dict[str, Any]]]] = {}
    for label, rp in labeled_roots:
        root_data = _collect_per_root(rp, shared_ids)
        if root_data:
            per_root[label] = root_data

    if not per_root:
        raise FileNotFoundError(
            "No valid aligned samples found across provided root directories. "
            "Expected each root to contain subdirs with input_timing.json and in_out_ce.json."
        )

    rel_tok = np.arange(-span, span + 1, dtype=np.int32)
    save_root = parsed_roots[0]
    all_layers = sorted({lv for root_data in per_root.values() for lv in root_data.keys()})
    for layer_val in all_layers:
        for anchor in anchors:
            datasets_for_anchor = [
                (name, data[layer_val][anchor])
                for name, data in per_root.items()
                if layer_val in data and anchor in data[layer_val]
            ]
            if not datasets_for_anchor:
                print(
                    f"[plot-logit-turn] No valid samples for layer={layer_val}, anchor={anchor}; skipping"
                )
                continue

            fig, (ax_top, ax_bot) = plt.subplots(
                2,
                1,
                figsize=(11.0, 6.2),
                dpi=180,
                sharex=True,
                gridspec_kw={"height_ratios": [1.9, 1.0]},
            )
            combined_ratio_values: list[np.ndarray] = []
            combined_amp_values: list[np.ndarray] = []
            merged_json: Dict[str, Any] = {
                "layer": int(layer_val),
                "anchor": anchor,
                "window_tokens": int(span),
                "relative_token_index": rel_tok.tolist(),
                "datasets": {},
            }
            for ds_name, ds in datasets_for_anchor:
                avg_line1 = ds["avg_line1"]
                avg_line2 = ds["avg_line2"]
                avg_ratio = ds["avg_ratio"]
                n_samples = int(ds["num_samples"])
                avg_input_amp = ds["avg_input_amp"]
                n_input = int(ds["num_samples_input_amp"])

                ax_top.plot(
                    rel_tok,
                    avg_ratio,
                    linewidth=1.6,
                    label=f"{ds_name} ratio (n={n_samples})",
                )

                ax_bot.plot(
                    rel_tok,
                    avg_input_amp,
                    linewidth=1.2,
                    label=f"{ds_name} (n={n_input})",
                )

                finite_ratio = avg_ratio[np.isfinite(avg_ratio)]
                if finite_ratio.size > 0:
                    combined_ratio_values.append(finite_ratio)
                finite_amp = avg_input_amp[np.isfinite(avg_input_amp)]
                if finite_amp.size > 0:
                    combined_amp_values.append(finite_amp)

                merged_json["datasets"][ds_name] = {
                    "num_samples": n_samples,
                    "avg_line1_user_multimodal_ce": avg_line1.tolist(),
                    "avg_line2_model_multimodal_ce": avg_line2.tolist(),
                    "avg_ratio_line1_over_line2": (
                        (avg_line1 / np.clip(avg_line2, 1e-6, None)).tolist()
                    ),
                    "num_samples_input_audio": n_input,
                    "avg_input_audio_abs_amplitude": avg_input_amp.tolist(),
                }

            ax_top.axvline(0, color="#444444", linestyle="--", linewidth=0.9, alpha=0.8)
            ax_top.set_ylabel("CE ratio")
            ax_top.set_title(
                f"Average Logit-Lens CE Ratio Around {anchor} (layer={layer_val}, window=+/-{span})"
            )
            ax_top.grid(True, axis="x", linestyle=":", linewidth=0.7, alpha=0.65)
            ax_top.legend(loc="upper right", fontsize=8)

            ax_bot.axvline(0, color="#444444", linestyle="--", linewidth=0.9, alpha=0.8)
            ax_bot.set_ylabel("Input abs amp")
            ax_bot.set_xlabel(f"Relative token index to {anchor}")
            ax_bot.grid(True, axis="x", linestyle=":", linewidth=0.7, alpha=0.65)
            ax_bot.legend(loc="upper right", fontsize=8)

            y_all = (
                np.concatenate(combined_ratio_values)
                if combined_ratio_values
                else np.asarray([], dtype=np.float32)
            )
            if y_all.size > 0:
                y_lo = float(np.percentile(y_all, 1.0))
                y_hi = float(np.percentile(y_all, 99.0))
                if y_hi <= y_lo:
                    y_mid = float(np.nanmean(y_all))
                    y_lo, y_hi = y_mid - 0.05, y_mid + 0.05
                pad = max(0.01, 0.12 * (y_hi - y_lo))
                ax_top.set_ylim(y_lo - pad, y_hi + pad)

            a_all = (
                np.concatenate(combined_amp_values)
                if combined_amp_values
                else np.asarray([], dtype=np.float32)
            )
            if a_all.size > 0:
                a_lo = float(np.percentile(a_all, 1.0))
                a_hi = float(np.percentile(a_all, 99.0))
                if a_hi <= a_lo:
                    a_mid = float(np.nanmean(a_all))
                    a_lo, a_hi = max(0.0, a_mid - 0.05), a_mid + 0.05
                pad = max(0.005, 0.10 * (a_hi - a_lo))
                ax_bot.set_ylim(max(0.0, a_lo - pad), a_hi + pad)

            ax_bot.set_xlim(int(rel_tok[0]), int(rel_tok[-1]))

            suffix = "multi" if len(parsed_roots) > 1 else parsed_roots[0].name
            out_png = save_root / (
                f"logit_lens_turn_taking_layer_{layer_val}_{anchor}_{suffix}.png"
            )
            fig.tight_layout()
            fig.savefig(out_png, bbox_inches="tight")
            plt.close(fig)

            out_json = save_root / (
                f"logit_lens_turn_taking_layer_{layer_val}_{anchor}_{suffix}.json"
            )
            with out_json.open("w", encoding="utf-8") as f:
                json.dump(merged_json, f, indent=2, ensure_ascii=False)
            print(f"[plot-logit-turn] Saved {out_png}")
            print(f"[plot-logit-turn] Saved {out_json}")


def logit_lens_heatmap(root_dir) -> None:
    """
    For root dir look for
    logit_lens_turn_taking_layer_{layer}_interrupt_start_user_interrupt_ce.json
    and logit_lens_turn_taking_layer_{layer}_question_start_user_question_ce.json
    for each layer 0-31 (raise error if not all 32 layers found).

    There should be four plot two for question_start and two for interrupt_start
    for each CE type (user_interrupt_ce and user_question_ce).
    One heatmap is for listening mode CE and the other is for speaking mode CE (i'm not sure which is which (line1 or line2)) remember to show in the plot
    For each plot, the y axis should be the layer number (0-31) and the x axis should be the relative token index (-span to +span).

    The color scale for for each heatmap should be consistent across all layers
    decide the scale based on the 5th and 95th percentile to be 95% saturated of blue and 95% staturated for red
    the percentlie is calcuted from  10th to 20th layers as endpoints layers has some outliers.

    Note: input JSON stores CE/NLL values, but this plot visualizes log-likelihood
    by negating those values before rendering.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Root directory not found: {root}")

    anchors = ["question_start", "interrupt_start"]
    layers = list(range(32))
    frame_rate_hz = 12.5
    span = 35

    def _extract_centered_1d(arr: np.ndarray, center: int, half: int) -> np.ndarray:
        out = np.full((2 * half + 1,), np.nan, dtype=np.float32)
        start = center - half
        end = center + half
        src_l = max(0, start)
        src_r = min(arr.shape[0] - 1, end)
        if src_r < src_l:
            return out
        dst_l = src_l - start
        dst_r = dst_l + (src_r - src_l)
        out[dst_l : dst_r + 1] = arr[src_l : src_r + 1]
        return out

    sample_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not sample_dirs:
        raise FileNotFoundError(f"No sample directories found under {root}")

    # line1/line2 windows gathered by anchor and layer.
    bucket: Dict[str, Dict[str, Dict[int, List[np.ndarray]]]] = {
        "line1": {a: {lv: [] for lv in layers} for a in anchors},
        "line2": {a: {lv: [] for lv in layers} for a in anchors},
    }

    used_samples = 0
    for sd in sample_dirs:
        timing_path = sd / "input_timing.json"
        if not timing_path.is_file():
            continue

        try:
            with timing_path.open("r", encoding="utf-8") as f:
                timing = json.load(f)
            if not isinstance(timing, dict):
                continue

            # Require both anchors for a consistent 4-plot output.
            if any(a not in timing for a in anchors):
                continue

            # Must have all 32 per-layer CE JSONs in this sample directory.
            ce_paths = {lv: sd / f"in_out_ce_{lv}.json" for lv in layers}
            if any(not p.is_file() for p in ce_paths.values()):
                continue

            per_layer_line1: Dict[int, np.ndarray] = {}
            per_layer_line2: Dict[int, np.ndarray] = {}
            ok = True
            for lv, p in ce_paths.items():
                with p.open("r", encoding="utf-8") as f:
                    ce = json.load(f)
                if not isinstance(ce, dict):
                    ok = False
                    break
                l1 = np.asarray(ce.get("line1_user_multimodal_ce", []), dtype=np.float32)
                l2 = np.asarray(ce.get("line2_model_multimodal_ce", []), dtype=np.float32)
                if l1.ndim != 1 or l2.ndim != 1 or l1.size == 0 or l2.size == 0:
                    ok = False
                    break
                n = min(l1.shape[0], l2.shape[0])
                per_layer_line1[lv] = l1[:n]
                per_layer_line2[lv] = l2[:n]

            if not ok:
                continue

            for anchor in anchors:
                center_tok = int(round(float(timing[anchor]) * frame_rate_hz))
                if center_tok < 0:
                    continue
                for lv in layers:
                    bucket["line1"][anchor][lv].append(
                        _extract_centered_1d(per_layer_line1[lv], center_tok, span)
                    )
                    bucket["line2"][anchor][lv].append(
                        _extract_centered_1d(per_layer_line2[lv], center_tok, span)
                    )

            used_samples += 1
        except Exception:
            continue

    if used_samples == 0:
        raise FileNotFoundError(
            "No valid samples found. Need root_dir/*/ with input_timing.json and in_out_ce_0..31.json."
        )

    # Ensure all 32 layers were gathered for each anchor/line pair.
    for which in ("line1", "line2"):
        for anchor in anchors:
            missing_layers = [lv for lv in layers if len(bucket[which][anchor][lv]) == 0]
            if missing_layers:
                raise ValueError(
                    f"Missing aligned data for {which}, {anchor}, layers={missing_layers}. "
                    "Need all 32 layers (0..31)."
                )

    def _build_heat(which: str, anchor: str) -> np.ndarray:
        rows: List[np.ndarray] = []
        for lv in layers:
            stack = np.stack(bucket[which][anchor][lv], axis=0)
            rows.append(np.nanmean(stack, axis=0))
        return np.stack(rows, axis=0).astype(np.float32)

    def _get_scale_bounds(mat: np.ndarray) -> tuple[float, float]:
        # Use middle layers (10..20) to reduce endpoint outlier impact.
        vals = mat[10:21, :]
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            finite = mat[np.isfinite(mat)]
        if finite.size == 0:
            return 0.0, 1.0
        p5 = float(np.percentile(finite, 5.0))
        p95 = float(np.percentile(finite, 95.0))
        if p95 <= p5:
            p95 = p5 + 1e-6
        return p5, p95

    base = plt.get_cmap("coolwarm")
    cmap = LinearSegmentedColormap.from_list(
        "coolwarm_soft_sat",
        base(np.linspace(0.025, 0.975, 256)),
    )

    rel_tok = np.arange(-span, span + 1, dtype=np.int32)
    generated = 0
    for anchor in anchors:
        for which in ("line1", "line2"):
            # Saved JSON is CE/NLL; visualize log-likelihood by negating it.
            mat_nll = _build_heat(which=which, anchor=anchor)
            mat = -mat_nll
            vmin, vmax = _get_scale_bounds(mat)

            fig, ax = plt.subplots(figsize=(11.0, 6.5), dpi=180)
            img = ax.imshow(
                mat,
                aspect="auto",
                interpolation="nearest",
                origin="lower",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                extent=(float(rel_tok[0]), float(rel_tok[-1]), -0.5, 31.5),
            )
            cbar = fig.colorbar(img, ax=ax)
            cbar.set_label("Log-likelihood value (negated from CE/NLL, P5/P95 from layers 10-20)")

            line_desc = (
                "line1 = user multimodal log-likelihood (from negated CE; likely listening-focus)"
                if which == "line1"
                else "line2 = model multimodal log-likelihood (from negated CE; likely speaking-focus)"
            )
            ax.set_title(f"Logit-Lens Heatmap | {anchor} | {which}\n{line_desc}")
            ax.set_xlabel("Relative token index")
            ax.set_ylabel("Layer")
            ax.set_yticks(np.arange(0, 32, 1))
            ax.set_ylim(-0.5, 31.5)

            out_png = root / f"logit_lens_heatmap_{anchor}_{which}.png"
            fig.tight_layout()
            fig.savefig(out_png, bbox_inches="tight")
            plt.close(fig)
            print(f"[plot-logit-heatmap] Saved {out_png}")
            generated += 1

    print(
        f"[plot-logit-heatmap] Done. Generated {generated} heatmaps from {used_samples} samples."
    )

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
        "--save-mean-hidden-diff",
        type=str,
        metavar="ROOT_DIR",
        help=(
            "Compute per-layer (mean speaking hidden - mean listening hidden) "
            "from ROOT_DIR/*/input.json labels and save ROOT_DIR/mean_hidden_diff.json."
        ),
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
    group.add_argument(
        "--plot-hidden-self-similarity",
        type=str,
        metavar="HIDDEN_PT",
        help="Plot token-token hidden self-similarity heatmap for a hidden .pt file.",
    )
    group.add_argument(
        "--plot-attention-heatmap-dataset",
        type=str,
        metavar="ROOT_DIR",
        help="Recursively plot output_hidden attention heatmap + audio waveform for all output_hidden.pt under ROOT_DIR.",
    )
    group.add_argument(
        "--plot-logit-lens-dataset",
        type=str,
        metavar="ROOT_DIR",
        help="Plot step-n logit-lens CE + aligned waveforms for ROOT_DIR/*/output_hidden(.pt).",
    )
    group.add_argument(
        "--plot-attention-heatmap-turn-taking",
        type=str,
        metavar="ROOT_DIR",
        help="Average attention windows aligned by question_start/interrupt_start from ROOT_DIR/*/input_timing.json.",
    )
    group.add_argument(
        "--plot-logit-lens-turn-taking-from-saved",
        type=str,
        nargs="+",
        metavar="ROOT_DIR",
        help="Average saved in_out_ce.json traces aligned by question_start/interrupt_start. Accepts one or more ROOT_DIR values and overlays them.",
    )
    group.add_argument(
        "--plot-logit-lens-heatmap",
        type=str,
        metavar="ROOT_DIR",
        help=(
            "Build layer-vs-token heatmaps from saved "
            "logit_lens_turn_taking_layer_<layer>_<anchor>_<ce_type>.json files "
            "for layers 0..31."
        ),
    )
    group.add_argument(
        "--plot-attention-by-subseqent-token-heatmap",
        type=str,
        metavar="ROOT_DIR",
        help=(
            "Plot layer-vs-token-offset heatmap of future-token attention to each token, "
            "anchored at interrupt_start from ROOT_DIR/*/input_timing.json and output_hidden(.pt)."
        ),
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
    ap.add_argument(
        "--window",
        type=int,
        default=5,
        help="Half-window size for hidden smoothing (default: 5).",
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

    elif args.save_mean_hidden_diff:
        save_mean_hidden_diff(
            root_dir=args.save_mean_hidden_diff,
            output_path=args.output,
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

    elif args.plot_hidden_self_similarity:
        if not args.output:
            ap.error("--plot-hidden-self-similarity requires --output")
        plot_hidden_self_similarity(
            hidden_path=args.plot_hidden_self_similarity,
            output_path=args.output,
            layer=args.layer,
            window=args.window,
        )

    elif args.plot_attention_heatmap_dataset:
        plot_attention_heatmap_dataset(
            root_dir=args.plot_attention_heatmap_dataset,
            layer=args.layer,
        )

    elif args.plot_logit_lens_dataset:
        plot_logit_lens_dataset(
            root_dir=args.plot_logit_lens_dataset,
            layer=args.layer,
            hf_repo=args.hf_repo,
            moshi_weight=args.moshi_weight,
            device=args.device,
            ma_window=args.window,
        )

    elif args.plot_attention_heatmap_turn_taking:
        plot_attention_heatmap_at_turn_taking(
            root_dir=args.plot_attention_heatmap_turn_taking,
            span=max(1, int(args.window)),
            layer=args.layer,
        )

    elif args.plot_logit_lens_turn_taking_from_saved:
        plot_logit_lens_turn_taking_from_saved(
            root_dirs=args.plot_logit_lens_turn_taking_from_saved,
            span=35,
        )

    elif args.plot_logit_lens_heatmap:
        logit_lens_heatmap(args.plot_logit_lens_heatmap)

    elif args.plot_attention_by_subseqent_token_heatmap:
        plot_attention_by_subseqent_token_heatmap(
            root_dir=args.plot_attention_by_subseqent_token_heatmap,
            span=max(1, int(args.window)),
        )


if __name__ == "__main__":
    main()
