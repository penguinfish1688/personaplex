"""Persona vector extraction using Moshi offline batch inference.

This script runs trait prompts (positive, negative, neutral) through the
offline batch inference pipeline and computes mean persona vectors per layer.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from moshi.offline import run_batch_inference, _get_voice_prompt_dir
from moshi.models import loaders
from moshi.models.lm import HiddenLayerOutputs


def _load_trait_file(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    with open(path, "r") as f:
        data = json.load(f)
    instructions = data.get("instruction", [])
    questions = data.get("questions", [])
    if not questions:
         # Fallback if questions are not in the file, use generic scenarios or specific format
         # Check if 'scenario' key exists (as seen in some json files)
         questions = data.get("scenario", [])
    return instructions, questions


def _build_prompt(instruction: str, question: str) -> str:
    return f"{instruction}\n\nUser: {question}"


def _repeat_path(path: str, count: int) -> List[str]:
    return [path for _ in range(count)]


def _make_output_paths(base_dir: str, prefix: str, count: int) -> Tuple[List[str], List[str]]:
    wav_dir = os.path.join(base_dir, prefix, "audio")
    txt_dir = os.path.join(base_dir, prefix, "text")
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)
    output_wavs = [os.path.join(wav_dir, f"{prefix}_{i:05d}.wav") for i in range(count)]
    output_texts = [os.path.join(txt_dir, f"{prefix}_{i:05d}.json") for i in range(count)]
    return output_wavs, output_texts


def _mean_hidden_layers(step_hidden: List[HiddenLayerOutputs]) -> Dict[str, List[torch.Tensor] | List[List[torch.Tensor]]]:
    """Compute mean hidden layer representations across steps for one prompt."""
    if len(step_hidden) == 0:
        raise ValueError("No hidden layers provided for a prompt.")

    text_layers = len(step_hidden[0].text_transformer)
    depth_codebooks = len(step_hidden[0].depth_transformer)
    depth_layers = len(step_hidden[0].depth_transformer[0])

    text_sum = [torch.zeros_like(step_hidden[0].text_transformer[i], dtype=torch.float32).cpu() for i in range(text_layers)]
    depth_sum = [
        [torch.zeros_like(step_hidden[0].depth_transformer[c][l], dtype=torch.float32).cpu() for l in range(depth_layers)]
        for c in range(depth_codebooks)
    ]

    for step in step_hidden:
        for i in range(text_layers):
            text_sum[i] += step.text_transformer[i].detach().cpu().float()
        for c in range(depth_codebooks):
            for l in range(depth_layers):
                depth_sum[c][l] += step.depth_transformer[c][l].detach().cpu().float()

    steps = float(len(step_hidden))
    text_mean = [t / steps for t in text_sum]
    depth_mean = [[d / steps for d in depth_sum[c]] for c in range(depth_codebooks)]

    return {"text": text_mean, "depth": depth_mean}


def _mean_across_prompts(per_prompt_means: List[Dict[str, List[torch.Tensor] | List[List[torch.Tensor]]]]) -> Dict[str, List[torch.Tensor] | List[List[torch.Tensor]]]:
    if len(per_prompt_means) == 0:
        raise ValueError("No prompt means provided.")

    text_layers = len(per_prompt_means[0]["text"])
    depth_codebooks = len(per_prompt_means[0]["depth"])
    depth_layers = len(per_prompt_means[0]["depth"][0])

    text_sum = [torch.zeros_like(per_prompt_means[0]["text"][i]) for i in range(text_layers)]
    depth_sum = [
        [torch.zeros_like(per_prompt_means[0]["depth"][c][l]) for l in range(depth_layers)]
        for c in range(depth_codebooks)
    ]

    for prompt_mean in per_prompt_means:
        for i in range(text_layers):
            text_sum[i] += prompt_mean["text"][i]
        for c in range(depth_codebooks):
            for l in range(depth_layers):
                depth_sum[c][l] += prompt_mean["depth"][c][l]

    count = float(len(per_prompt_means))
    text_mean = [t / count for t in text_sum]
    depth_mean = [[d / count for d in depth_sum[c]] for c in range(depth_codebooks)]

    return {"text": text_mean, "depth": depth_mean}


def _compute_diff(a: Dict[str, List[torch.Tensor] | List[List[torch.Tensor]]], b: Dict[str, List[torch.Tensor] | List[List[torch.Tensor]]]) -> Dict[str, List[torch.Tensor] | List[List[torch.Tensor]]]:
    text_diff = [a["text"][i] - b["text"][i] for i in range(len(a["text"]))]
    depth_diff = [
        [a["depth"][c][l] - b["depth"][c][l] for l in range(len(a["depth"][c]))]
        for c in range(len(a["depth"]))
    ]
    return {"text": text_diff, "depth": depth_diff}


def _write_summary(path: str, trait: str, persona_vector: Dict[str, List[torch.Tensor] | List[List[torch.Tensor]]]):
    lines = []
    lines.append(f"Trait: {trait}")
    lines.append(f"Text layers: {len(persona_vector['text'])}")
    for i, vec in enumerate(persona_vector["text"]):
        norm = torch.linalg.vector_norm(vec).item()
        lines.append(f"  Text layer {i:02d}: norm={norm:.4f}")

    lines.append(f"Depth codebooks: {len(persona_vector['depth'])}")
    for c in range(min(3, len(persona_vector["depth"]))):
        lines.append(f"  Codebook {c:02d} layers: {len(persona_vector['depth'][c])}")
        for l in range(min(3, len(persona_vector["depth"][c]))):
            norm = torch.linalg.vector_norm(persona_vector["depth"][c][l]).item()
            lines.append(f"    Depth layer {l:02d}: norm={norm:.4f}")
        if len(persona_vector["depth"][c]) > 3:
            lines.append("    ...")
    if len(persona_vector["depth"]) > 3:
        lines.append("  ...")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def run_persona_vector_extraction(
    trait_dir: str,
    traits: List[str],
    input_wav: str,
    output_dir: str,
    voice_prompt: str,
    voice_prompt_dir: str | None,
    tokenizer_path: str | None,
    moshi_weight: str | None,
    mimi_weight: str | None,
    hf_repo: str,
    device: str,
    seed: int | None,
    temp_audio: float,
    temp_text: float,
    topk_audio: int,
    topk_text: int,
    greedy: bool,
    cpu_offload: bool,
    neutral_instruction: str,
):
    # Resolve voice prompt path
    voice_prompt_dir_resolved = _get_voice_prompt_dir(voice_prompt_dir, hf_repo)
    if not os.path.exists(voice_prompt_dir_resolved):
        raise FileNotFoundError(f"voice_prompt_dir does not exist: {voice_prompt_dir_resolved}")
    voice_prompt_path = os.path.join(voice_prompt_dir_resolved, voice_prompt)
    if not os.path.exists(voice_prompt_path):
        raise FileNotFoundError(f"Voice prompt not found: {voice_prompt_path}")

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for trait in traits:
        trait_path = os.path.join(trait_dir, f"{trait}.json")
        if not os.path.exists(trait_path):
            print(f"Skipping trait {trait}: file not found at {trait_path}")
            continue
        
        print(f"Processing trait: {trait}")

        instructions, questions = _load_trait_file(trait_path)
        if len(instructions) == 0 or len(questions) == 0:
            print(f"Skipping trait {trait}: no instructions or questions found")
            continue

        pos_prompts = []
        neg_prompts = []
        neutral_prompts = []

        for q in questions:
            # Handle case where q is not a string (though it should be)
            if isinstance(q, dict):
                q_text = str(q)
            else:
                q_text = str(q)

            for inst in instructions:
                pos_inst = inst.get("pos", "")
                neg_inst = inst.get("neg", "")
                if not pos_inst or not neg_inst:
                    continue
                
                pos_prompts.append(_build_prompt(pos_inst, q_text))
                neg_prompts.append(_build_prompt(neg_inst, q_text))
                neutral_prompts.append(_build_prompt(neutral_instruction, q_text))

        print(f"  Generated {len(pos_prompts)} prompts per Condition (Pos/Neg/Neu)")
        
        if len(pos_prompts) == 0:
            continue

        trait_out_dir = os.path.join(output_dir, trait)
        os.makedirs(trait_out_dir, exist_ok=True)

        pos_wavs, pos_texts = _make_output_paths(trait_out_dir, "pos", len(pos_prompts))
        neg_wavs, neg_texts = _make_output_paths(trait_out_dir, "neg", len(neg_prompts))
        neu_wavs, neu_texts = _make_output_paths(trait_out_dir, "neutral", len(neutral_prompts))

        input_wavs = _repeat_path(input_wav, len(pos_prompts))
        
        print("  Running inference for Positive prompts...")
        pos_hidden = run_batch_inference(
            input_wavs=input_wavs,
            output_wavs=pos_wavs,
            output_texts=pos_texts,
            text_prompts=pos_prompts,
            voice_prompt_path=voice_prompt_path,
            tokenizer_path=tokenizer_path,
            moshi_weight=moshi_weight,
            mimi_weight=mimi_weight,
            hf_repo=hf_repo,
            device=device,
            seed=seed,
            temp_audio=temp_audio,
            temp_text=temp_text,
            topk_audio=topk_audio,
            topk_text=topk_text,
            greedy=greedy,
            save_voice_prompt_embeddings=False,
            cpu_offload=cpu_offload,
            return_hidden_layers=True,
        )

        print("  Running inference for Negative prompts...")
        neg_hidden = run_batch_inference(
            input_wavs=input_wavs,
            output_wavs=neg_wavs,
            output_texts=neg_texts,
            text_prompts=neg_prompts,
            voice_prompt_path=voice_prompt_path,
            tokenizer_path=tokenizer_path,
            moshi_weight=moshi_weight,
            mimi_weight=mimi_weight,
            hf_repo=hf_repo,
            device=device,
            seed=seed,
            temp_audio=temp_audio,
            temp_text=temp_text,
            topk_audio=topk_audio,
            topk_text=topk_text,
            greedy=greedy,
            save_voice_prompt_embeddings=False,
            cpu_offload=cpu_offload,
            return_hidden_layers=True,
        )
        """
        print("  Running inference for Neutral prompts...")
        neu_hidden = run_batch_inference(
            input_wavs=input_wavs,
            output_wavs=neu_wavs,
            output_texts=neu_texts,
            text_prompts=neutral_prompts,
            voice_prompt_path=voice_prompt_path,
            tokenizer_path=tokenizer_path,
            moshi_weight=moshi_weight,
            mimi_weight=mimi_weight,
            hf_repo=hf_repo,
            device=device,
            seed=seed,
            temp_audio=temp_audio,
            temp_text=temp_text,
            topk_audio=topk_audio,
            topk_text=topk_text,
            greedy=greedy,
            save_voice_prompt_embeddings=False,
            cpu_offload=cpu_offload,
            return_hidden_layers=True,
        )
        """

        if pos_hidden is None or neg_hidden is None: # or neu_hidden is None:
            raise RuntimeError("Hidden layers were not returned from batch inference.")

        # Compute mean vectors per prompt
        print("  Computing means...")
        pos_prompt_means = [_mean_hidden_layers(steps) for steps in pos_hidden]
        neg_prompt_means = [_mean_hidden_layers(steps) for steps in neg_hidden]
        # neu_prompt_means = [_mean_hidden_layers(steps) for steps in neu_hidden]

        # Compute mean vectors across prompts
        pos_mean = _mean_across_prompts(pos_prompt_means)
        neg_mean = _mean_across_prompts(neg_prompt_means)
        # neu_mean = _mean_across_prompts(neu_prompt_means)


        # Persona vectors are defined as difference between pos and neg
        persona_vector = _compute_diff(pos_mean, neg_mean)

        # Save full vectors
        torch.save(
            {
                "persona_vector": persona_vector,
            },
            os.path.join(trait_out_dir, "mean_persona_vectors.pt"),
        )

        # Write summary
        _write_summary(
            os.path.join(trait_out_dir, "summary.txt"),
            trait,
            persona_vector,
        )
        print(f"  Done. Results saved to {trait_out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Compute persona vectors from trait prompts using Moshi offline inference")
    parser.add_argument("--trait_dir", type=str, default=os.path.join(os.path.dirname(__file__), "data_generation", "trait_data_extract"))
    parser.add_argument("--traits", type=str, nargs="+", default=["all"], help="Trait names to process (e.g., apathetic evil). Use 'all' for all traits.")
    parser.add_argument("--input_wav", type=str, required=True, help="Path to input WAV file used for all prompts")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write outputs")

    parser.add_argument("--voice_prompt", required=True, type=str, help="Voice prompt filename (basename) inside --voice-prompt-dir (e.g. 'NATM1.pt').")
    parser.add_argument("--voice_prompt_dir", type=str, help="Directory containing voice prompt files.")

    parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file.")
    parser.add_argument("--moshi_weight", type=str, help="Path to a local checkpoint file for Moshi.")
    parser.add_argument("--mimi_weight", type=str, help="Path to a local checkpoint file for Mimi.")
    parser.add_argument("--hf_repo", type=str, default=loaders.DEFAULT_REPO, help="HF repo to look into (defaults to pre-trained model repo)")

    parser.add_argument("--temp_audio", type=float, default=0.8)
    parser.add_argument("--temp_text", type=float, default=0.7)
    parser.add_argument("--topk_audio", type=int, default=250)
    parser.add_argument("--topk_text", type=int, default=25)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--seed", type=int, default=-1)

    parser.add_argument("--neutral_instruction", type=str, default="Respond helpfully and neutrally.")

    args = parser.parse_args()

    trait_dir = args.trait_dir
    # Check if trait dir exists, if not warn
    if not os.path.exists(trait_dir):
        print(f"Warning: Trait directory {trait_dir} does not exist. Using current directory.")
        trait_dir = "."

    all_traits = [p.stem for p in Path(trait_dir).glob("*.json")]
    if args.traits == ["all"]:
        traits = all_traits
    else:
        traits = args.traits

    if not traits:
        print("No traits found or specified.")
        return

    run_persona_vector_extraction(
        trait_dir=trait_dir,
        traits=traits,
        input_wav=args.input_wav,
        output_dir=args.output_dir,
        voice_prompt=args.voice_prompt,
        voice_prompt_dir=args.voice_prompt_dir,
        tokenizer_path=args.tokenizer,
        moshi_weight=args.moshi_weight,
        mimi_weight=args.mimi_weight,
        hf_repo=args.hf_repo,
        device=args.device,
        seed=None if args.seed == -1 else args.seed,
        temp_audio=args.temp_audio,
        temp_text=args.temp_text,
        topk_audio=args.topk_audio,
        topk_text=args.topk_text,
        greedy=bool(args.greedy),
        cpu_offload=bool(args.cpu_offload),
        neutral_instruction=args.neutral_instruction,
    )


if __name__ == "__main__":
    main()
