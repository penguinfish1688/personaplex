import argparse
import os
from pathlib import Path

import torch

from moshi.offline import run_batch_inference, _get_voice_prompt_dir
from moshi.models import loaders


def inference(root_dir: str):
    """
    Take root_dir as input there will be <root_dir>/*/input.wav file
    For each input.wav file, run inference and save to <root_dir>/*/output.wav
    """
    root = Path(root_dir)
    input_paths = [p for p in root.glob("*/input.wav") if p.is_file()]
    input_paths.sort(key=lambda p: int(p.parent.name) if p.parent.name.isdigit() else p.parent.name)

    if not input_paths:
        raise FileNotFoundError(f"No files matched pattern {root_dir}/*/input.wav")

    voice_prompt_dir = _get_voice_prompt_dir(None, loaders.DEFAULT_REPO)
    if voice_prompt_dir is None:
        raise FileNotFoundError("Unable to resolve voice prompt directory.")

    voice_prompt_path = os.path.join(voice_prompt_dir, "NATF0.pt")
    if not os.path.exists(voice_prompt_path):
        raise FileNotFoundError(f"Voice prompt not found: {voice_prompt_path}")

    input_wavs = [str(path) for path in input_paths]
    output_wavs = [str(path.with_name("output.wav")) for path in input_paths]
    output_texts = [str(path.with_name("output.json")) for path in input_paths]
    prompts = ["You are a helpful and friendly assistant."] * len(input_paths)

    print(f"[user_interrupt] Processing {len(input_paths)} files from {root_dir}")
    with torch.no_grad():
        run_batch_inference(
            input_wavs=input_wavs,
            output_wavs=output_wavs,
            output_texts=output_texts,
            text_prompts=prompts,
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
        )
    print(f"[user_interrupt] Done. Wrote {len(output_wavs)} output.wav files.")

def inference_with_steering():
    """Don't implement this yet."""
    pass


def main() -> None:
    parser = argparse.ArgumentParser("user_interrupt_inference")
    parser.add_argument(
        "--root-dir",
        type=str,
        required=True,
        help="Root directory containing */input.wav files",
    )
    args = parser.parse_args()
    inference(args.root_dir)


if __name__ == "__main__":
    main()

