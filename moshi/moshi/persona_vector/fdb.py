#!/usr/bin/env python3
"""Full-Duplex-Bench offline inference.

Usage:
    python -m moshi.persona_vector.fdb /path/to/data --voice-prompt NATF0.pt
    python -m moshi.persona_vector.fdb /path/to/data --text-prompt "Be helpful." --device cuda

Finds all ``input.wav`` files under ``<path>/**/input.wav`` and writes
``output.wav`` + ``output.json`` next to each one.
"""

import argparse
import os
from glob import glob
from pathlib import Path
from typing import List

import torch

from moshi.offline import run_batch_inference, _get_voice_prompt_dir
from moshi.models import loaders


def _find_inputs(root: str) -> List[Path]:
    """Return sorted list of input.wav files under root/**/input.wav."""
    return sorted(Path(p) for p in glob(os.path.join(root, "**", "input.wav"), recursive=True))


def main():
    ap = argparse.ArgumentParser("fdb_offline_inference")
    ap.add_argument("path", type=str, help="Root directory to search for input.wav files")
    ap.add_argument("--voice-prompt", type=str, default="NATF0.pt")
    ap.add_argument("--text-prompt", type=str, default="You are a helpful and friendly assistant.")
    ap.add_argument("--voice-prompt-dir", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO)
    ap.add_argument("--tokenizer", type=str, default=None)
    ap.add_argument("--moshi-weight", type=str, default=None)
    ap.add_argument("--mimi-weight", type=str, default=None)
    ap.add_argument("--cpu-offload", action="store_true")
    ap.add_argument("--temp-audio", type=float, default=0.8)
    ap.add_argument("--temp-text", type=float, default=0.7)
    ap.add_argument("--topk-audio", type=int, default=250)
    ap.add_argument("--topk-text", type=int, default=25)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true", help="Re-process files that already have output.wav")
    ap.add_argument("--save-hidden", action="store_true", help="Save per-token hidden/attention payload to output_hidden.pt")
    args = ap.parse_args()

    # Voice prompt
    vp_dir = _get_voice_prompt_dir(args.voice_prompt_dir, args.hf_repo)
    vp_path = os.path.join(vp_dir, args.voice_prompt)
    if not os.path.exists(vp_path):
        raise FileNotFoundError(f"Voice prompt not found: {vp_path}")

    # Find inputs
    inputs = _find_inputs(args.path)
    print(f"[FDB] Found {len(inputs)} input.wav files under {args.path}")
    if not inputs:
        return

    # Build lists, skip existing unless --overwrite
    in_wavs, out_wavs, out_txts, out_hiddens, prompts = [], [], [], [], []
    for inp in inputs:
        out_wav = inp.with_name("output.wav")
        out_txt = inp.with_name("output.json")
        out_hidden = inp.with_name("output_hidden.pt")
        if not args.overwrite and out_wav.exists():
            print(f"[SKIP] {out_wav}")
            continue
        in_wavs.append(str(inp))
        out_wavs.append(str(out_wav))
        out_txts.append(str(out_txt))
        out_hiddens.append(str(out_hidden))
        prompts.append(args.text_prompt)

    if not in_wavs:
        print("[FDB] Nothing to process (all outputs exist, use --overwrite to redo).")
        return

    print(f"[FDB] Processing {len(in_wavs)} files…")
    with torch.no_grad():
        run_batch_inference(
            input_wavs=in_wavs,
            output_wavs=out_wavs,
            output_texts=out_txts,
            text_prompts=prompts,
            voice_prompt_path=vp_path,
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
            greedy=bool(args.greedy),
            save_voice_prompt_embeddings=False,
            cpu_offload=args.cpu_offload,
            return_hidden_layers=False,
            save_hidden_payload=bool(args.save_hidden),
            output_hiddens=out_hiddens if args.save_hidden else None,
        )
    print(f"[FDB] Done. Processed {len(in_wavs)} files.")


if __name__ == "__main__":
    main()
