"""Persona vector extraction using Moshi offline batch inference.

This script runs trait prompts (positive, negative, neutral) through the
offline batch inference pipeline and computes mean persona vectors per layer.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from tqdm import tqdm

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


def _flatten_hidden_outputs(hidden_outputs: HiddenLayerOutputs) -> HiddenLayerOutputs:
    """Flatten all tensors in HiddenLayerOutputs to 1D vectors."""
    flat_text = [t.flatten() for t in hidden_outputs.text_transformer]
    flat_depth = [[d.flatten() for d in codebook] for codebook in hidden_outputs.depth_transformer]
    return HiddenLayerOutputs(text_transformer=flat_text, depth_transformer=flat_depth)


def _mean_hidden_layers(step_hidden: List[HiddenLayerOutputs]) -> Dict[str, List[torch.Tensor] | List[List[torch.Tensor]]]:
    """Compute mean hidden layer representations across steps for one prompt.
    Assumes tensors are already flattened to 1D.
    """
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


def _compute_stats_across_prompts(per_prompt_diffs: List[Dict[str, List[torch.Tensor] | List[List[torch.Tensor]]]], persona_vector: Dict[str, List[torch.Tensor] | List[List[torch.Tensor]]]) -> Tuple[Dict[str, List[Dict[str, float]] | List[List[Dict[str, float]]]], Dict[str, Any]]:
    """Compute mean and std of norms and cosine similarities across prompts for each layer.
       Also returns raw cosine values for histogram plotting.
       Assumes all tensors are already flattened to 1D.
    """
    if not per_prompt_diffs:
        return {}, {}
    
    raw_cosines = {"text": [], "depth": []}
    
    # Text layers
    text_layers_count = len(per_prompt_diffs[0]["text"])
    text_stats = []
    for i in range(text_layers_count):
        # Stack vectors from all prompts for this layer: [N, D]
        vecs = torch.stack([d["text"][i] for d in per_prompt_diffs]).float()
        avg_vec = persona_vector["text"][i].float().unsqueeze(0) # [1, D]
        
        norms = torch.linalg.vector_norm(vecs, dim=1) # [N]
        cosines = torch.nn.functional.cosine_similarity(vecs, avg_vec, dim=1) # [N]
        
        raw_cosines["text"].append(cosines.cpu())

        text_stats.append({
            "mean": norms.mean().item(),
            "std": norms.std().item(),
            "mean_cos": cosines.mean().item(),
            "std_cos": cosines.std().item()
        })
        
    # Depth layers
    depth_codebooks = len(per_prompt_diffs[0]["depth"])
    depth_layers_count = len(per_prompt_diffs[0]["depth"][0])
    depth_stats = []
    for c in range(depth_codebooks):
        codebook_stats = []
        codebook_cosines = []
        for l in range(depth_layers_count):
            vecs = torch.stack([d["depth"][c][l] for d in per_prompt_diffs]).float()
            avg_vec = persona_vector["depth"][c][l].float().unsqueeze(0) # [1, D]
            
            norms = torch.linalg.vector_norm(vecs, dim=1)
            cosines = torch.nn.functional.cosine_similarity(vecs, avg_vec, dim=1)
            
            codebook_cosines.append(cosines.cpu())

            codebook_stats.append({
                "mean": norms.mean().item(),
                "std": norms.std().item(),
                "mean_cos": cosines.mean().item(),
                "std_cos": cosines.std().item()
            })
        depth_stats.append(codebook_stats)
        raw_cosines["depth"].append(codebook_cosines)
        
    return {"text": text_stats, "depth": depth_stats}, raw_cosines


def _write_summary(path: str, trait: str, persona_vector: Dict[str, List[torch.Tensor] | List[List[torch.Tensor]]], stats: Dict[str, Any] = None):
    lines = []
    lines.append(f"Trait: {trait}")
    
    # Text summary
    lines.append(f"Text layers: {len(persona_vector['text'])}")
    for i, vec in enumerate(persona_vector["text"]):
        norm = torch.linalg.vector_norm(vec).item()
        line = f"  Text layer {i:02d}: norm={norm:.4f}"
        if stats:
             s = stats["text"][i]
             line += f", mean_norm={s['mean']:.4f}, std_norm={s['std']:.4f}"
             line += f", mean_cos={s['mean_cos']:.4f}, std_cos={s['std_cos']:.4f}"
        lines.append(line)

    # Depth summary
    lines.append(f"Depth codebooks: {len(persona_vector['depth'])}")
    for c in range(min(3, len(persona_vector["depth"]))):
        lines.append(f"  Codebook {c:02d} layers: {len(persona_vector['depth'][c])}")
        for l in range(min(3, len(persona_vector["depth"][c]))):
            norm = torch.linalg.vector_norm(persona_vector["depth"][c][l]).item()
            line = f"    Depth layer {l:02d}: norm={norm:.4f}"
            if stats:
                s = stats["depth"][c][l]
                line += f", mean_norm={s['mean']:.4f}, std_norm={s['std']:.4f}"
                line += f", mean_cos={s['mean_cos']:.4f}, std_cos={s['std_cos']:.4f}"
            lines.append(line)
        if len(persona_vector["depth"][c]) > 3:
            lines.append("    ...")
    if len(persona_vector["depth"]) > 3:
        lines.append("  ...")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def run_text_persona_vector_extraction(
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

        for q in tqdm(questions):
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
                # neutral_prompts.append(_build_prompt(neutral_instruction, q_text))

        print(f"  Generated {len(pos_prompts)} prompts per Condition (Pos/Neg)")
        
        if len(pos_prompts) == 0:
            continue

        trait_out_dir = os.path.join(output_dir, trait)
        os.makedirs(trait_out_dir, exist_ok=True)

        pos_wavs, pos_texts = _make_output_paths(trait_out_dir, "pos", len(pos_prompts))
        neg_wavs, neg_texts = _make_output_paths(trait_out_dir, "neg", len(neg_prompts))
        # neu_wavs, neu_texts = _make_output_paths(trait_out_dir, "neutral", len(neutral_prompts))

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

        # Flatten all hidden layer tensors immediately
        print("  Flattening hidden layers...")
        pos_hidden_flat = [[_flatten_hidden_outputs(step) for step in prompt_steps] for prompt_steps in pos_hidden]
        neg_hidden_flat = [[_flatten_hidden_outputs(step) for step in prompt_steps] for prompt_steps in neg_hidden]

        # Compute mean vectors per prompt
        print("  Computing means...")
        pos_prompt_means = [_mean_hidden_layers(steps) for steps in pos_hidden_flat]
        neg_prompt_means = [_mean_hidden_layers(steps) for steps in neg_hidden_flat]
        # neu_prompt_means = [_mean_hidden_layers(steps) for steps in neu_hidden]
        
        # Calculate differences per prompt (needed for statistics)
        prompt_diffs = [_compute_diff(p, n) for p, n in zip(pos_prompt_means, neg_prompt_means)]
        
        # Compute mean vectors across prompts
        pos_mean = _mean_across_prompts(pos_prompt_means)
        neg_mean = _mean_across_prompts(neg_prompt_means)
        
        # Persona vectors are defined as difference between pos and neg
        # Effectively the same as averaging prompt_diffs
        persona_vector = _compute_diff(pos_mean, neg_mean)
        
        # Compute stats (consistency across prompts)
        stats, raw_cosines = _compute_stats_across_prompts(prompt_diffs, persona_vector)

        # Save full vectors (all flattened)
        torch.save(
            {
                "persona_vector": persona_vector,
                "stats": stats,
                "raw_cosines": raw_cosines, # Saving raw cosine values for histogram plotting
                "pos_mean": pos_mean,       # Saving per-trait mean vectors
                "neg_mean": neg_mean,
                "pos_hidden": pos_prompt_means, # Saving all individual prompt mean vectors (list of dicts) - flattened
                "neg_hidden": neg_prompt_means  # Flattened
            },
            os.path.join(trait_out_dir, "mean_persona_vectors.pt"),
        )

        # Write summary
        _write_summary(
            os.path.join(trait_out_dir, "summary.txt"),
            trait,
            persona_vector,
            stats
        )
        print(f"  Done. Results saved to {trait_out_dir}")


def run_audio_persona_vector_extraction(
    trait_dir: str,
    trait_audio_dir: str,
    traits: List[str],
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
    audio_delay: float = 2.5,
    sample_rate: int = 24000,
):
    """
    Extract persona vectors using audio prompts instead of text prompts.
    
    1. For the input_wavs, use the audio_delay (2.5 second by default) + instruction + question (concat) 
       for each trait as the audio input
    2. For the text prompts, use "You are an assistant that obeys user's instruction."
    3. The rest is the same as run_text_persona_vector_extraction
    
    Args:
        trait_dir: Directory containing trait JSON files (text format)
        trait_audio_dir: Directory containing generated audio files (from tts.py)
        traits: List of trait names to process
        output_dir: Output directory for persona vectors
        voice_prompt: Voice prompt filename
        voice_prompt_dir: Directory containing voice prompt files
        audio_delay: Delay in seconds to prepend to audio (default 2.5s)
        sample_rate: Audio sample rate (default 24000 Hz for Moshi)
        ... (other args same as run_text_persona_vector_extraction)
    """
    import soundfile as sf
    import numpy as np
    
    # Fixed text prompt for all audio-based extraction
    FIXED_TEXT_PROMPT = "You are an assistant that obeys user's instruction."
    
    # Resolve voice prompt path
    voice_prompt_dir_resolved = _get_voice_prompt_dir(voice_prompt_dir, hf_repo)
    if not os.path.exists(voice_prompt_dir_resolved):
        raise FileNotFoundError(f"voice_prompt_dir does not exist: {voice_prompt_dir_resolved}")
    voice_prompt_path = os.path.join(voice_prompt_dir_resolved, voice_prompt)
    if not os.path.exists(voice_prompt_path):
        raise FileNotFoundError(f"Voice prompt not found: {voice_prompt_path}")

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create silence for audio delay
    delay_samples = int(audio_delay * sample_rate)
    silence = np.zeros(delay_samples, dtype=np.float32)

    def _concat_audio_files(delay: np.ndarray, *audio_paths: str, output_path: str) -> str:
        """Concatenate delay silence with multiple audio files."""
        audio_parts = [delay]
        for path in audio_paths:
            if os.path.exists(path):
                audio, sr = sf.read(path, dtype='float32')
                if sr != sample_rate:
                    # Resample if needed (simple approach - for production use librosa)
                    import warnings
                    warnings.warn(f"Audio sample rate {sr} != {sample_rate}, may cause issues")
                audio_parts.append(audio)
            else:
                raise FileNotFoundError(f"Audio file not found: {path}")
        
        concatenated = np.concatenate(audio_parts)
        sf.write(output_path, concatenated, sample_rate)
        return output_path

    for trait in traits:
        trait_path = os.path.join(trait_dir, f"{trait}.json")
        trait_audio_json = os.path.join(trait_audio_dir, trait, f"{trait}.json")
        
        if not os.path.exists(trait_path):
            print(f"Skipping trait {trait}: text file not found at {trait_path}")
            continue
        if not os.path.exists(trait_audio_json):
            print(f"Skipping trait {trait}: audio json not found at {trait_audio_json}")
            continue
        
        print(f"Processing trait: {trait}")

        # Load text trait file for structure
        instructions, questions = _load_trait_file(trait_path)
        if len(instructions) == 0 or len(questions) == 0:
            print(f"Skipping trait {trait}: no instructions or questions found")
            continue
        
        # Load audio paths
        with open(trait_audio_json, 'r') as f:
            audio_data = json.load(f)
        
        instruction_audios = audio_data.get("instruction", [])
        question_audios = audio_data.get("questions", [])
        
        if len(instruction_audios) == 0 or len(question_audios) == 0:
            print(f"Skipping trait {trait}: no audio files found")
            continue

        trait_out_dir = os.path.join(output_dir, trait)
        concat_audio_dir = os.path.join(trait_out_dir, "concat_audio")
        os.makedirs(trait_out_dir, exist_ok=True)
        os.makedirs(concat_audio_dir, exist_ok=True)

        # Build concatenated audio files: delay + instruction + question
        pos_input_wavs = []
        neg_input_wavs = []
        pos_text_prompts = []
        neg_text_prompts = []
        
        print(f"  Creating concatenated audio files...")
        concat_idx = 0
        for q_idx, q_audio_path in enumerate(tqdm(question_audios, desc="  Questions")):
            for inst_idx, inst_audio in enumerate(instruction_audios):
                pos_inst_path = inst_audio.get("pos", "")
                neg_inst_path = inst_audio.get("neg", "")
                
                if not pos_inst_path or not neg_inst_path:
                    continue
                
                # Create positive concatenated audio: delay + pos_instruction + question
                pos_concat_path = os.path.join(concat_audio_dir, f"pos_{concat_idx:05d}.wav")
                _concat_audio_files(silence, pos_inst_path, q_audio_path, output_path=pos_concat_path)
                pos_input_wavs.append(pos_concat_path)
                pos_text_prompts.append(FIXED_TEXT_PROMPT)
                
                # Create negative concatenated audio: delay + neg_instruction + question
                neg_concat_path = os.path.join(concat_audio_dir, f"neg_{concat_idx:05d}.wav")
                _concat_audio_files(silence, neg_inst_path, q_audio_path, output_path=neg_concat_path)
                neg_input_wavs.append(neg_concat_path)
                neg_text_prompts.append(FIXED_TEXT_PROMPT)
                
                concat_idx += 1

        print(f"  Generated {len(pos_input_wavs)} concatenated audio files per condition (Pos/Neg)")
        
        if len(pos_input_wavs) == 0:
            continue

        pos_wavs, pos_texts = _make_output_paths(trait_out_dir, "pos", len(pos_input_wavs))
        neg_wavs, neg_texts = _make_output_paths(trait_out_dir, "neg", len(neg_input_wavs))

        print("  Running inference for Positive prompts...")
        pos_hidden = run_batch_inference(
            input_wavs=pos_input_wavs,
            output_wavs=pos_wavs,
            output_texts=pos_texts,
            text_prompts=pos_text_prompts,
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
            input_wavs=neg_input_wavs,
            output_wavs=neg_wavs,
            output_texts=neg_texts,
            text_prompts=neg_text_prompts,
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

        if pos_hidden is None or neg_hidden is None:
            raise RuntimeError("Hidden layers were not returned from batch inference.")

        # Flatten all hidden layer tensors immediately
        print("  Flattening hidden layers...")
        pos_hidden_flat = [[_flatten_hidden_outputs(step) for step in prompt_steps] for prompt_steps in pos_hidden]
        neg_hidden_flat = [[_flatten_hidden_outputs(step) for step in prompt_steps] for prompt_steps in neg_hidden]

        # Compute mean vectors per prompt
        print("  Computing means...")
        pos_prompt_means = [_mean_hidden_layers(steps) for steps in pos_hidden_flat]
        neg_prompt_means = [_mean_hidden_layers(steps) for steps in neg_hidden_flat]
        
        # Calculate differences per prompt (needed for statistics)
        prompt_diffs = [_compute_diff(p, n) for p, n in zip(pos_prompt_means, neg_prompt_means)]
        
        # Compute mean vectors across prompts
        pos_mean = _mean_across_prompts(pos_prompt_means)
        neg_mean = _mean_across_prompts(neg_prompt_means)
        
        # Persona vectors are defined as difference between pos and neg
        persona_vector = _compute_diff(pos_mean, neg_mean)
        
        # Compute stats (consistency across prompts)
        stats, raw_cosines = _compute_stats_across_prompts(prompt_diffs, persona_vector)

        # Save full vectors (all flattened)
        torch.save(
            {
                "persona_vector": persona_vector,
                "stats": stats,
                "raw_cosines": raw_cosines,
                "pos_mean": pos_mean,
                "neg_mean": neg_mean,
                "pos_hidden": pos_prompt_means,
                "neg_hidden": neg_prompt_means,
                "extraction_mode": "audio",
                "audio_delay": audio_delay,
            },
            os.path.join(trait_out_dir, "mean_persona_vectors.pt"),
        )

        # Write summary
        _write_summary(
            os.path.join(trait_out_dir, "summary.txt"),
            trait,
            persona_vector,
            stats
        )
        print(f"  Done. Results saved to {trait_out_dir}")


def main():
    # Hardcoded paths relative to this script's location
    # Script is at: personaplex/moshi/moshi/persona_vector/gen_vector.py
    # Go up 3 levels to reach personaplex/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    
    TRAIT_DIR = os.path.join(project_root, "data_generation", "trait_data_extract")
    TRAIT_AUDIO_DIR = os.path.join(project_root, "data_generation", "trait_data_extract_audio")
    EMPTY_WAV = os.path.join(project_root, "assets", "empty.wav")
    
    parser = argparse.ArgumentParser(description="Compute persona vectors from trait prompts using Moshi offline inference")
    parser.add_argument("--mode", type=str, choices=["text", "audio"], default="text", help="Extraction mode: 'text' (text prompts) or 'audio' (audio prompts)")
    parser.add_argument("--traits", type=str, nargs="+", default=["all"], help="Trait names to process (e.g., apathetic evil). Use 'all' for all traits.")
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
    parser.add_argument("--audio_delay", type=float, default=2.5, help="Silence delay in seconds before audio (for audio mode)")
    parser.add_argument("--sample_rate", type=int, default=24000, help="Audio sample rate (default 24000 Hz for Moshi)")

    args = parser.parse_args()

    # Check if trait dir exists
    if not os.path.exists(TRAIT_DIR):
        print(f"Error: Trait directory {TRAIT_DIR} does not exist.")
        return

    all_traits = [p.stem for p in Path(TRAIT_DIR).glob("*.json")]
    if args.traits == ["all"]:
        traits = all_traits
    else:
        # Filter out empty strings from trait names
        traits = [t for t in args.traits if t.strip()]

    if not traits:
        print("No traits found or specified.")
        print(f"Available traits in {TRAIT_DIR}:")
        for trait in all_traits:
            print(f"  - {trait}")
        return

    if args.mode == "text":
        # Check empty.wav exists
        if not os.path.exists(EMPTY_WAV):
            print(f"Error: Empty WAV file not found at {EMPTY_WAV}")
            print("Please create an empty/silent WAV file at this location.")
            return
        
        run_text_persona_vector_extraction(
            trait_dir=TRAIT_DIR,
            traits=traits,
            input_wav=EMPTY_WAV,
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
    elif args.mode == "audio":
        if not os.path.exists(TRAIT_AUDIO_DIR):
            print(f"Error: Trait audio directory {TRAIT_AUDIO_DIR} does not exist.")
            print("Run tts.py with --trait to generate audio files first.")
            return
        
        run_audio_persona_vector_extraction(
            trait_dir=TRAIT_DIR,
            trait_audio_dir=TRAIT_AUDIO_DIR,
            traits=traits,
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
            audio_delay=args.audio_delay,
            sample_rate=args.sample_rate,
        )




if __name__ == "__main__":
    main()
