#!/usr/bin/env python3

import argparse
import csv
import gc
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional, Any, cast

import numpy as np
import sentencepiece
import sphn
import torch
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from moshi.models import loaders, LMGen, MimiModel
from moshi.models.lm import load_audio as lm_load_audio
from moshi.models.lm import _iterate_audio as lm_iterate_audio
from moshi.models.lm import encode_from_sphn as lm_encode_from_sphn
from moshi.models.lm import HiddenLayerOutputs
from moshi.offline import warmup, decode_tokens_to_pcm, wrap_with_system_tags, seed_all, _get_voice_prompt_dir


SILENCE_TOKEN_ID = 3


def _resolve_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


@lru_cache(maxsize=2)
def _load_text_embedding_matrix(
    tokenizer_path: Optional[str],
    moshi_weight: Optional[str],
    hf_repo: str,
) -> torch.Tensor:
    del tokenizer_path
    if moshi_weight is None:
        moshi_weight = hf_hub_download(hf_repo, loaders.MOSHI_NAME)
    lm = loaders.get_moshi_lm(moshi_weight, device="cpu", cpu_offload=False)
    lm.eval()
    matrix = lm.text_emb.weight.detach().cpu().float().clone()
    del lm
    gc.collect()
    return matrix


def get_embedding_by_token_id(
    token_id: int,
    tokenizer_path: Optional[str] = None,
    moshi_weight: Optional[str] = None,
    hf_repo: str = loaders.DEFAULT_REPO,
) -> torch.Tensor:
    embedding_matrix = _load_text_embedding_matrix(tokenizer_path, moshi_weight, hf_repo)
    if token_id < 0 or token_id >= embedding_matrix.shape[0]:
        raise ValueError(f"token_id {token_id} is out of range [0, {embedding_matrix.shape[0] - 1}]")
    return embedding_matrix[token_id]


def gamma_average(
    tokenizer_path: Optional[str] = None,
    moshi_weight: Optional[str] = None,
    hf_repo: str = loaders.DEFAULT_REPO,
    exclude_silence_token: bool = True,
) -> torch.Tensor:
    """Calculate the average embedding for moshi's vocabulary.

    Args:
        exclude_silence_token: if True, omit the silence token (SILENCE_TOKEN_ID)
            from the average. Defaults to True.
    """
    embedding_matrix = _load_text_embedding_matrix(tokenizer_path, moshi_weight, hf_repo)
    if exclude_silence_token:
        vocab_size = embedding_matrix.shape[0]
        if 0 <= SILENCE_TOKEN_ID < vocab_size:
            mask = torch.ones(vocab_size, dtype=torch.bool)
            mask[SILENCE_TOKEN_ID] = False
            filtered = embedding_matrix[mask]
            if filtered.numel() == 0:
                return embedding_matrix.mean(dim=0)
            return filtered.mean(dim=0)
    return embedding_matrix.mean(dim=0)


def gamma_covariance(
    tokenizer_path: Optional[str] = None,
    moshi_weight: Optional[str] = None,
    hf_repo: str = loaders.DEFAULT_REPO,
) -> torch.Tensor:
    """Calculate the covariance matrix for moshi's vocabulary embeddings."""
    embedding_matrix = _load_text_embedding_matrix(tokenizer_path, moshi_weight, hf_repo)
    mean = embedding_matrix.mean(dim=0, keepdim=True)
    centered = embedding_matrix - mean
    n = centered.shape[0]
    denom = max(n - 1, 1)
    cov = centered.T @ centered / float(denom)
    return cov


def gamma_silence_bar(
    tokenizer_path: Optional[str] = None,
    moshi_weight: Optional[str] = None,
    hf_repo: str = loaders.DEFAULT_REPO,
) -> torch.Tensor:
    silence_embedding = get_embedding_by_token_id(
        SILENCE_TOKEN_ID,
        tokenizer_path=tokenizer_path,
        moshi_weight=moshi_weight,
        hf_repo=hf_repo,
    )
    return silence_embedding - gamma_average(
        tokenizer_path=tokenizer_path,
        moshi_weight=moshi_weight,
        hf_repo=hf_repo,
    )


def lambda_silence_bar(
    tokenizer_path: Optional[str] = None,
    moshi_weight: Optional[str] = None,
    hf_repo: str = loaders.DEFAULT_REPO,
    ridge: float = 1e-8,
) -> torch.Tensor:
    cov = gamma_covariance(
        tokenizer_path=tokenizer_path,
        moshi_weight=moshi_weight,
        hf_repo=hf_repo,
    )
    gsb = gamma_silence_bar(
        tokenizer_path=tokenizer_path,
        moshi_weight=moshi_weight,
        hf_repo=hf_repo,
    )
    eye = torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
    return torch.linalg.solve(cov + ridge * eye, gsb)


def calculate_projection(v: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate the scalar projection of vector v onto the target vector."""
    target_norm = torch.linalg.norm(target)
    if target_norm.item() == 0:
        return 0.0
    return torch.dot(v, target / target_norm).item()


def _load_text_tokenizer(tokenizer_path: Optional[str], hf_repo: str) -> sentencepiece.SentencePieceProcessor:
    if tokenizer_path is None:
        tokenizer_path = hf_hub_download(hf_repo, loaders.TEXT_TOKENIZER_NAME)
    tokenizer = sentencepiece.SentencePieceProcessor()
    tokenizer.Load(tokenizer_path)
    return tokenizer


def _token_name(text_tokenizer: sentencepiece.SentencePieceProcessor, token_id: int) -> str:
    if token_id in (0, 1, 2, 3):
        text_token_map = ["EPAD", "BOS", "EOS", "PAD"]
        return text_token_map[token_id]
    piece = text_tokenizer.id_to_piece(token_id)  # type: ignore
    return piece.replace("▁", " ")


def _glob_sorted_by_parent_numeric(root_dir: Path, pattern: str):
    """Return list of Path objects matching pattern under root_dir/*/..., sorted
    so numeric parent directory names come first in ascending numeric order,
    followed by non-numeric names in lexicographic order."""
    paths = list(root_dir.glob(pattern))

    def _key(p: Path):
        parent = p.parent.name
        try:
            return (0, int(parent))
        except Exception:
            return (1, parent)

    return [p for p in sorted(paths, key=_key)]


def _load_turn_taking_start_map(csv_path: str) -> dict[str, int]:
    """Load mapping from folder name to turn-taking start step index.

    CSV format: first column is folder/file number, second column is step index.
    Rows with second column -1 are ignored.
    """
    mapping: dict[str, int] = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            folder_raw = row[0].strip()
            step_raw = row[1].strip()
            if folder_raw == "":
                continue
            try:
                step_idx = int(float(step_raw))
            except ValueError:
                continue
            if step_idx == -1:
                continue
            mapping[folder_raw] = step_idx
    return mapping


def plot_projection_around_turn_taking(
    root_dir: str,
    turn_taking_start_csv: str,
    tokenizer_path: Optional[str],
    moshi_weight: Optional[str],
    hf_repo: str,
):
    """Plot mean projection (with stderr) around turn-taking start for lambda/gamma.

    For each folder in root_dir/* with output_hidden.pt, uses center index from CSV and
    aggregates projections on offsets [-10, +10].
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for --plot-projection") from exc

    root = Path(root_dir)
    hidden_paths = _glob_sorted_by_parent_numeric(root, "*/output_hidden.pt")
    if len(hidden_paths) == 0:
        raise FileNotFoundError(f"No files matched pattern {root}/*/output_hidden.pt")

    turn_map = _load_turn_taking_start_map(turn_taking_start_csv)
    if len(turn_map) == 0:
        raise ValueError(f"No valid entries found in turn-taking CSV: {turn_taking_start_csv}")

    lambda_dir = lambda_silence_bar(
        tokenizer_path=tokenizer_path,
        moshi_weight=moshi_weight,
        hf_repo=hf_repo,
    ).detach().cpu().float().reshape(-1)
    gamma_dir = gamma_silence_bar(
        tokenizer_path=tokenizer_path,
        moshi_weight=moshi_weight,
        hf_repo=hf_repo,
    ).detach().cpu().float().reshape(-1)

    offsets = np.arange(-10, 11, dtype=int)
    lambda_samples: dict[int, list[float]] = {int(o): [] for o in offsets.tolist()}
    gamma_samples: dict[int, list[float]] = {int(o): [] for o in offsets.tolist()}

    valid_file_count = 0
    frame_rates: list[float] = []

    for hidden_path in tqdm(hidden_paths, desc="plot-projection", unit="file"):
        folder_name = hidden_path.parent.name
        if folder_name not in turn_map:
            continue

        center_idx = turn_map[folder_name]
        payload = torch.load(str(hidden_path), map_location="cpu")
        hidden_states = payload["hidden_states"]
        frame_rate = float(payload.get("frame_rate", 12.5))
        frame_rates.append(frame_rate)

        valid_file_count += 1
        total_steps = hidden_states.shape[0]
        for offset in offsets.tolist():
            token_idx = center_idx + int(offset)
            if token_idx < 0 or token_idx >= total_steps:
                continue
            hidden_vec = hidden_states[token_idx].detach().cpu().float().reshape(-1)
            lambda_samples[int(offset)].append(calculate_projection(hidden_vec, lambda_dir))
            gamma_samples[int(offset)].append(calculate_projection(hidden_vec, gamma_dir))

    if valid_file_count == 0:
        raise ValueError("No matching files between root_dir/*/output_hidden.pt and CSV first column")

    ref_frame_rate = float(np.mean(frame_rates)) if len(frame_rates) > 0 else 12.5
    x_seconds = offsets.astype(np.float32) / ref_frame_rate

    lambda_mean = []
    lambda_sem = []
    gamma_mean = []
    gamma_sem = []
    for offset in offsets.tolist():
        lam = np.asarray(lambda_samples[int(offset)], dtype=np.float32)
        gam = np.asarray(gamma_samples[int(offset)], dtype=np.float32)

        if lam.size == 0:
            lambda_mean.append(np.nan)
            lambda_sem.append(np.nan)
        else:
            lambda_mean.append(float(lam.mean()))
            lambda_sem.append(float(lam.std(ddof=0) / np.sqrt(valid_file_count)))

        if gam.size == 0:
            gamma_mean.append(np.nan)
            gamma_sem.append(np.nan)
        else:
            gamma_mean.append(float(gam.mean()))
            gamma_sem.append(float(gam.std(ddof=0) / np.sqrt(valid_file_count)))

    lambda_mean_arr = np.asarray(lambda_mean, dtype=np.float32)
    lambda_sem_arr = np.asarray(lambda_sem, dtype=np.float32)
    gamma_mean_arr = np.asarray(gamma_mean, dtype=np.float32)
    gamma_sem_arr = np.asarray(gamma_sem, dtype=np.float32)

    plt.figure(figsize=(8, 5))
    plt.plot(x_seconds, lambda_mean_arr, marker="o", label="projection on lambda_silence_bar")
    plt.fill_between(
        x_seconds,
        lambda_mean_arr - lambda_sem_arr,
        lambda_mean_arr + lambda_sem_arr,
        alpha=0.2,
    )

    plt.plot(x_seconds, gamma_mean_arr, marker="o", label="projection on gamma_silence_bar")
    plt.fill_between(
        x_seconds,
        gamma_mean_arr - gamma_sem_arr,
        gamma_mean_arr + gamma_sem_arr,
        alpha=0.2,
    )

    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Time relative to turn-taking start (s)")
    plt.ylabel("Projection value")
    plt.title(f"Projection around turn-taking start (n={valid_file_count})")
    plt.legend()
    plt.tight_layout()

    save_path = root / "projection_around_turn_taking.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Wrote projection plot to {save_path}")


def _select_hidden_since_time(hidden_payload: dict, time_sec: float, n: int):
    hidden_states = hidden_payload["hidden_states"]
    token_ids = hidden_payload["token_ids"]
    token_names = hidden_payload.get("token_names", None)
    times = hidden_payload.get("times", None)

    if times is None:
        frame_rate = float(hidden_payload.get("frame_rate", 12.5))
        times = torch.arange(hidden_states.shape[0], dtype=torch.float32) / frame_rate
    else:
        times = torch.as_tensor(times, dtype=torch.float32)

    mask = times >= float(time_sec)
    indices = torch.where(mask)[0]
    if indices.numel() == 0:
        return (
            torch.empty((0, hidden_states.shape[-1])),
            torch.empty((0,), dtype=torch.long),
            [],
            torch.empty((0,)),
            torch.empty((0,), dtype=torch.long),
        )
    if n > 0:
        indices = indices[:n]

    selected_hidden = hidden_states[indices]
    selected_ids = token_ids[indices]
    selected_times = times[indices]
    selected_step_indices = indices.to(dtype=torch.long)
    if token_names is None:
        selected_names = [str(int(tid)) for tid in selected_ids.tolist()]
    else:
        selected_names = [str(token_names[i]) for i in indices.tolist()]
    return selected_hidden, selected_ids, selected_names, selected_times, selected_step_indices


def inference(
    input_wav: str | list[str],
    output_wav: Optional[str],
    output_hidden: Optional[str],
    output_text: Optional[str],
    text_prompt: str,
    voice_prompt: str,
    voice_prompt_dir: Optional[str],
    tokenizer_path: Optional[str],
    moshi_weight: Optional[str],
    mimi_weight: Optional[str],
    hf_repo: str,
    device: str,
    seed: Optional[int],
    temp_audio: float,
    temp_text: float,
    topk_audio: int,
    topk_text: int,
    greedy: bool,
    save_voice_prompt_embeddings: bool,
    cpu_offload: bool = False,
):
    """Inference one or many input wavs and save wav/text/hidden outputs."""
    device = _resolve_device(device)
    if seed is not None and seed != -1:
        seed_all(seed)

    hf_hub_download(hf_repo, "config.json")

    if mimi_weight is None:
        mimi_weight = hf_hub_download(hf_repo, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device)
    other_mimi = loaders.get_mimi(mimi_weight, device)

    text_tokenizer = _load_text_tokenizer(tokenizer_path, hf_repo)

    if moshi_weight is None:
        moshi_weight = hf_hub_download(hf_repo, loaders.MOSHI_NAME)
    lm = loaders.get_moshi_lm(moshi_weight, device=device, cpu_offload=cpu_offload)
    lm.eval()

    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    lm_gen = LMGen(
        lm,
        audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
        sample_rate=mimi.sample_rate,
        device=device,
        frame_rate=int(mimi.frame_rate),
        save_voice_prompt_embeddings=save_voice_prompt_embeddings,
        use_sampling=not greedy,
        temp=temp_audio,
        temp_text=temp_text,
        top_k=topk_audio,
        top_k_text=topk_text,
    )

    mimi.streaming_forever(1)
    other_mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)
    warmup(mimi, other_mimi, lm_gen, device, frame_size)

    vp_dir = _get_voice_prompt_dir(voice_prompt_dir, hf_repo)
    if vp_dir is None:
        raise FileNotFoundError("Unable to resolve voice prompt directory.")
    voice_prompt_path = os.path.join(vp_dir, voice_prompt)
    if not os.path.exists(voice_prompt_path):
        raise FileNotFoundError(f"Voice prompt not found: {voice_prompt_path}")

    if voice_prompt_path.endswith(".pt"):
        lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
    else:
        lm_gen.load_voice_prompt(voice_prompt_path)
    tokenizer_any = cast(Any, text_tokenizer)
    lm_gen.text_prompt_tokens = (
        tokenizer_any.EncodeAsIds(wrap_with_system_tags(text_prompt)) if len(text_prompt) > 0 else None
    )

    if isinstance(input_wav, str):
        input_wavs = [input_wav]
    else:
        input_wavs = [str(x) for x in input_wav]
    if len(input_wavs) == 0:
        raise ValueError("No input wav files were provided.")

    file_iterator = tqdm(input_wavs, desc="inference", unit="file") if len(input_wavs) > 1 else input_wavs
    for input_wav_one in file_iterator:
        input_path = Path(input_wav_one)
        if len(input_wavs) == 1 and output_wav is not None and output_hidden is not None and output_text is not None:
            output_wav_one = output_wav
            output_hidden_one = output_hidden
            output_text_one = output_text
        else:
            output_dir = input_path.parent
            output_wav_one = str(output_dir / "output.wav")
            output_hidden_one = str(output_dir / "output_hidden.pt")
            output_text_one = str(output_dir / "output.json")

        mimi.reset_streaming()
        other_mimi.reset_streaming()
        lm_gen.reset_streaming()
        lm_gen.step_system_prompts(mimi)
        mimi.reset_streaming()

        sample_rate = mimi.sample_rate
        user_audio = lm_load_audio(input_wav_one, sample_rate)

        generated_frames: list[np.ndarray] = []
        generated_text_tokens: list[str] = []
        hidden_vectors: list[torch.Tensor] = []
        token_ids: list[int] = []
        token_names: list[str] = []
        times: list[float] = []

        total_target_samples = user_audio.shape[-1]
        frame_rate = float(mimi.frame_rate)
        step_idx = 0

        for user_encoded in lm_encode_from_sphn(
            mimi,
            lm_iterate_audio(user_audio, sample_interval_size=frame_size, pad=True),
            max_batch=1,
        ):
            steps = user_encoded.shape[-1]
            for c in range(steps):
                step_in = user_encoded[:, :, c : c + 1]
                result = lm_gen.step(step_in, return_hidden_layers=True)
                tokens, hidden_layers = cast(tuple[torch.Tensor | None, HiddenLayerOutputs | None], result)
                if tokens is None:
                    step_idx += 1
                    continue

                pcm = decode_tokens_to_pcm(mimi, other_mimi, lm_gen, tokens)
                generated_frames.append(pcm)

                text_token_id = int(tokens[0, 0, 0].item())
                text_name = _token_name(text_tokenizer, text_token_id)
                generated_text_tokens.append(text_name)
                token_ids.append(text_token_id)
                token_names.append(text_name)

                if hidden_layers is None or hidden_layers.text_transformer is None:
                    raise RuntimeError("Hidden layers were requested but not returned.")
                last_text_hidden = hidden_layers.text_transformer[-1]
                hidden_vectors.append(last_text_hidden.detach().cpu().float().reshape(-1))
                times.append(step_idx / frame_rate)
                step_idx += 1

        if len(generated_frames) == 0:
            raise RuntimeError(f"No audio frames were generated for {input_wav_one}. Check input/configuration.")

        output_pcm = np.concatenate(generated_frames, axis=-1)
        if output_pcm.shape[-1] > total_target_samples:
            output_pcm = output_pcm[:total_target_samples]
        elif output_pcm.shape[-1] < total_target_samples:
            pad_len = total_target_samples - output_pcm.shape[-1]
            output_pcm = np.concatenate([output_pcm, np.zeros(pad_len, dtype=output_pcm.dtype)], axis=-1)

        Path(output_wav_one).parent.mkdir(parents=True, exist_ok=True)
        Path(output_text_one).parent.mkdir(parents=True, exist_ok=True)
        Path(output_hidden_one).parent.mkdir(parents=True, exist_ok=True)

        sphn.write_wav(output_wav_one, output_pcm, sample_rate)
        with open(output_text_one, "w") as file:
            json.dump(generated_text_tokens, file, ensure_ascii=False)

        hidden_payload = {
            "hidden_states": torch.stack(hidden_vectors, dim=0),
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "token_names": token_names,
            "times": torch.tensor(times, dtype=torch.float32),
            "frame_rate": frame_rate,
            "sample_rate": int(sample_rate),
            "input_wav": str(input_wav_one),
            "output_wav": str(output_wav_one),
            "output_text": str(output_text_one),
            "text_prompt": text_prompt,
        }
        torch.save(hidden_payload, output_hidden_one)

        print(f"Wrote output audio to {output_wav_one}")
        print(f"Wrote output text to {output_text_one}")
        print(f"Wrote output hidden payload to {output_hidden_one}")


def hidden_since_time(output_hidden: str, time: float, n: int, show_token_name: bool = False) -> torch.Tensor:
    """Return hidden states of first n tokens since the given time (seconds)."""
    payload = torch.load(output_hidden, map_location="cpu")
    hidden, token_ids, token_names, times, step_indices = _select_hidden_since_time(payload, time, n)
    if show_token_name:
        for step_idx, t, tid, tname in zip(step_indices.tolist(), times.tolist(), token_ids.tolist(), token_names):
            print(f"token_idx={step_idx} time={t:.3f}s token_id={tid} token_name={tname}")
    return hidden


def projection_since_time(
    direction: torch.Tensor,
    output_hidden: str,
    time: float,
    n: int,
    show_results: bool = True,
) -> torch.Tensor:
    """Project hidden states in output_hidden onto direction for first n tokens since time."""
    payload = torch.load(output_hidden, map_location="cpu")
    hidden, token_ids, token_names, times, step_indices = _select_hidden_since_time(payload, time, n)
    if hidden.numel() == 0:
        return torch.empty((0,), dtype=torch.float32)

    direction = direction.detach().cpu().float().reshape(-1)
    direction_norm = torch.linalg.norm(direction)
    if direction_norm.item() == 0:
        raise ValueError("Projection direction has zero norm.")
    direction_unit = direction / direction_norm

    projections = []
    cosine_hidden_projection = []
    residual_norms = []
    for row in hidden:
        hidden_row = row.float().reshape(-1)
        scalar_projection = calculate_projection(hidden_row, direction)
        projection_vector = scalar_projection * direction_unit

        hidden_norm = torch.linalg.norm(hidden_row)
        projection_norm = torch.linalg.norm(projection_vector)
        if hidden_norm.item() == 0 or projection_norm.item() == 0:
            cosine_value = 0.0
        else:
            cosine_value = torch.dot(hidden_row, projection_vector).item() / (hidden_norm.item() * projection_norm.item())

        residual_norm = torch.linalg.norm(hidden_row - projection_vector).item()

        projections.append(scalar_projection)
        cosine_hidden_projection.append(cosine_value)
        residual_norms.append(residual_norm)

    proj_tensor = torch.tensor(projections, dtype=torch.float32)

    if show_results:
        for step_idx, t, tid, tname, value, cosine_value, residual_norm in zip(
            step_indices.tolist(),
            times.tolist(),
            token_ids.tolist(),
            token_names,
            proj_tensor.tolist(),
            cosine_hidden_projection,
            residual_norms,
        ):
            print(
                f"token_idx={step_idx} time={t:.3f}s token_id={tid} token_name={tname} projection={value:.6f} "
                f"cos_hidden_projection={cosine_value:.6f} hidden_minus_projection_norm={residual_norm:.6f}"
            )
    return proj_tensor


def main():
    """
    --inference INPUT_WAV : Run inference and save output wav/text/hidden in INPUT_WAV directory.
    --inferece-batch ROOT_DIR : Run inference for ROOT_DIR/*/input.wav and write outputs beside each input.
    --show-token TIME : show token names since time (seconds).
    --show-token-batch ROOT_DIR : show token names for ROOT_DIR/*/output_hidden.pt since time.
    --projection TIME : show projection on selected direction since time (seconds).
    --projection-batch ROOT_DIR : show projection for ROOT_DIR/*/output_hidden.pt since time.
    """
    ap = argparse.ArgumentParser("causal_inner_prod")
    ap.add_argument("--inference", type=str, metavar="INPUT_WAV", default=None)
    ap.add_argument(
        "--inferece-batch",
        "--inference-batch",
        dest="inferece_batch",
        type=str,
        metavar="ROOT_DIR",
        default=None,
        help="Batch mode: find ROOT_DIR/*/input.wav and write output.wav/output.json/output_hidden.pt per folder",
    )
    ap.add_argument("--output-hidden", type=str, default=None, help="Hidden payload .pt path for analysis-only mode")
    ap.add_argument("--show-token", type=float, default=None, dest="show_token", help="Time in seconds")
    ap.add_argument(
        "--show-token-batch",
        type=str,
        default=None,
        dest="show_token_batch",
        metavar="ROOT_DIR",
        help="Batch mode: show token names for ROOT_DIR/*/output_hidden.pt",
    )
    ap.add_argument("--projection", type=float, default=None, dest="projection", help="Time in seconds")
    ap.add_argument(
        "--projection-batch",
        type=str,
        default=None,
        dest="projection_batch",
        metavar="ROOT_DIR",
        help="Batch mode: show projection for ROOT_DIR/*/output_hidden.pt",
    )
    ap.add_argument("--n", type=int, default=20, help="Number of first tokens to show since given time (0 means all)")

    ap.add_argument("--voice-prompt", type=str, default="NATF0.pt")
    ap.add_argument("--voice-prompt-dir", type=str, default=None)
    ap.add_argument("--text-prompt", type=str, default="You are a helpful and friendly assistant.")
    ap.add_argument("--tokenizer", type=str, default=None)
    ap.add_argument("--moshi-weight", type=str, default=None)
    ap.add_argument("--mimi-weight", type=str, default=None)
    ap.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temp-audio", type=float, default=0.8)
    ap.add_argument("--temp-text", type=float, default=0.7)
    ap.add_argument("--topk-audio", type=int, default=250)
    ap.add_argument("--topk-text", type=int, default=25)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--save-voice-prompt-embeddings", action="store_true")
    ap.add_argument("--cpu-offload", action="store_true")
    ap.add_argument("--no-causal", action="store_true", help="Use gamma_silence_bar instead of lambda_silence_bar for projection")
    ap.add_argument(
        "--plot-projection",
        type=str,
        default=None,
        metavar="ROOT_DIR",
        help="Plot average projection around turn-taking start for ROOT_DIR/*/output_hidden.pt",
    )
    ap.add_argument(
        "--turn-taking-start",
        type=str,
        default=None,
        metavar="CSV_PATH",
        help="CSV with (folder_name, center_step_index); rows with -1 are ignored",
    )
    args = ap.parse_args()

    generated_hidden_path = None
    if args.inference is not None and args.inferece_batch is not None:
        raise ValueError("Use only one of --inference or --inferece-batch")

    if args.inference is not None:
        input_wav = args.inference
        input_path = Path(input_wav)
        output_dir = input_path.parent
        output_wav = str(output_dir / "output.wav")
        output_hidden = str(output_dir / "output_hidden.pt")
        output_text = str(output_dir / "output.json")
        inference(
            input_wav=input_wav,
            output_wav=output_wav,
            output_hidden=output_hidden,
            output_text=output_text,
            text_prompt=args.text_prompt,
            voice_prompt=args.voice_prompt,
            voice_prompt_dir=args.voice_prompt_dir,
            tokenizer_path=args.tokenizer,
            moshi_weight=args.moshi_weight,
            mimi_weight=args.mimi_weight,
            hf_repo=args.hf_repo,
            device=args.device,
            seed=args.seed,
            temp_audio=args.temp_audio,
            temp_text=args.temp_text,
            topk_audio=args.topk_audio,
            topk_text=args.topk_text,
            greedy=args.greedy,
            save_voice_prompt_embeddings=args.save_voice_prompt_embeddings,
            cpu_offload=args.cpu_offload,
        )
        generated_hidden_path = output_hidden

    if args.inferece_batch is not None:
        root_dir = Path(args.inferece_batch)
        input_wavs = [str(p) for p in _glob_sorted_by_parent_numeric(root_dir, "*/input.wav")]
        if len(input_wavs) == 0:
            raise FileNotFoundError(f"No files matched pattern {root_dir}/*/input.wav")
        inference(
            input_wav=input_wavs,
            output_wav=None,
            output_hidden=None,
            output_text=None,
            text_prompt=args.text_prompt,
            voice_prompt=args.voice_prompt,
            voice_prompt_dir=args.voice_prompt_dir,
            tokenizer_path=args.tokenizer,
            moshi_weight=args.moshi_weight,
            mimi_weight=args.mimi_weight,
            hf_repo=args.hf_repo,
            device=args.device,
            seed=args.seed,
            temp_audio=args.temp_audio,
            temp_text=args.temp_text,
            topk_audio=args.topk_audio,
            topk_text=args.topk_text,
            greedy=args.greedy,
            save_voice_prompt_embeddings=args.save_voice_prompt_embeddings,
            cpu_offload=args.cpu_offload,
        )

    hidden_path = args.output_hidden or generated_hidden_path
    if args.show_token_batch is not None and args.show_token is None:
        raise ValueError("--show-token-batch requires --show-token TIME")
    if args.projection_batch is not None and args.projection is None:
        raise ValueError("--projection-batch requires --projection TIME")

    if args.show_token is not None and args.show_token_batch is None:
        if hidden_path is None:
            raise ValueError("--show-token requires --output-hidden or --inference")
        hidden_since_time(hidden_path, args.show_token, args.n, show_token_name=True)

    if args.show_token_batch is not None:
        root_dir = Path(args.show_token_batch)
        hidden_paths = [str(p) for p in _glob_sorted_by_parent_numeric(root_dir, "*/output_hidden.pt")]
        if len(hidden_paths) == 0:
            raise FileNotFoundError(f"No files matched pattern {root_dir}/*/output_hidden.pt")
        for hp in tqdm(hidden_paths, desc="show-token", unit="file"):
            print(f"=== {hp} ===")
            hidden_since_time(hp, args.show_token, args.n, show_token_name=True)

    if args.projection is not None and args.projection_batch is None:
        if hidden_path is None:
            raise ValueError("--projection requires --output-hidden or --inference")
        if args.no_causal:
            # Directly calculate projection on embedding
            direction = gamma_silence_bar(
                tokenizer_path=args.tokenizer,
                moshi_weight=args.moshi_weight,
                hf_repo=args.hf_repo,
            )
        else:
            direction = lambda_silence_bar(
                tokenizer_path=args.tokenizer,
                moshi_weight=args.moshi_weight,
                hf_repo=args.hf_repo,
            )
        
        print("Cosine between gamma_silence_bar and lambda_silence_bar:", torch.nn.functional.cosine_similarity(
            gamma_silence_bar(
                tokenizer_path=args.tokenizer,
                moshi_weight=args.moshi_weight,
                hf_repo=args.hf_repo,
            ).reshape(1, -1),
            direction.reshape(1, -1),
        ).item())

        projection_since_time(direction, hidden_path, args.projection, args.n, show_results=True)

    if args.projection_batch is not None:
        root_dir = Path(args.projection_batch)
        hidden_paths = [str(p) for p in _glob_sorted_by_parent_numeric(root_dir, "*/output_hidden.pt")]
        if len(hidden_paths) == 0:
            raise FileNotFoundError(f"No files matched pattern {root_dir}/*/output_hidden.pt")
        if args.no_causal:
            direction = gamma_silence_bar(
                tokenizer_path=args.tokenizer,
                moshi_weight=args.moshi_weight,
                hf_repo=args.hf_repo,
            )
        else:
            direction = lambda_silence_bar(
                tokenizer_path=args.tokenizer,
                moshi_weight=args.moshi_weight,
                hf_repo=args.hf_repo,
            )

        print("Cosine between gamma_silence_bar and lambda_silence_bar:", torch.nn.functional.cosine_similarity(
            gamma_silence_bar(
                tokenizer_path=args.tokenizer,
                moshi_weight=args.moshi_weight,
                hf_repo=args.hf_repo,
            ).reshape(1, -1),
            direction.reshape(1, -1),
        ).item())

        for hp in tqdm(hidden_paths, desc="projection", unit="file"):
            print(f"=== {hp} ===")
            projection_since_time(direction, hp, args.projection, args.n, show_results=True)

    if args.plot_projection is not None:
        """Prompt for the plot
        /home/penguinfish/personaplex/personaplex/tmp/turn_taking_start.csv
        Plot the projection on lambda_silence_bar and gamma_silence_bar. y axis will be the projection value
        and x axis will be the time in seconds. The middle of x axis will be the time step index indicate at turn_taking_start.csv
        The first column of .csv is the file number under <root-dir>/THE NUMBER/input.wav, and the second column is the time step index
        Take +-5 step from the middle time step index and plot the projection values for those steps. The value is the average projection of
        <root-dir>/*/output_hidden.pt for the tokens at those steps. And if the second column of .csv is -1, then ignore that. The plot should
        have two lines, one for projection on lambda_silence_bar and one for projection on gamma_silence_bar. There should be uncertainty for each
        point on each step, i.e. std/sqrt(n) where n is the number of files under <root-dir>. Usage: --plot-projection <root-dir> --turn-taking-start <path-to-csv>
        """
        if args.turn_taking_start is None:
            raise ValueError("--plot-projection requires --turn-taking-start CSV_PATH")
        plot_projection_around_turn_taking(
            root_dir=args.plot_projection,
            turn_taking_start_csv=args.turn_taking_start,
            tokenizer_path=args.tokenizer,
            moshi_weight=args.moshi_weight,
            hf_repo=args.hf_repo,
        )


if __name__ == "__main__":
    # python3 -m moshi.persona_vector.causal_inner_prod --inference /home/penguinfish/personaplex/personaplex/tmp/causal_inner_prod_out/turn-taking-example/input.wav 
    # cd /home/penguinfish/personaplex/personaplex/moshi && python3 -m moshi.persona_vector.causal_inner_prod --show_token_name 5.0 --output-hidden /home/penguinfish/personaplex/personaplex/tmp/causal_inner_prod_out/turn-taking-example/output_hidden.pt --n 15
    # python3 -m moshi.persona_vector.causal_inner_prod --projection_on_lsb 5.0 --output-hidden /home/penguinfish/personaplex/personaplex/tmp/causal_inner_prod_out/turn-taking-example/output_hidden.pt --n 15
    main()


