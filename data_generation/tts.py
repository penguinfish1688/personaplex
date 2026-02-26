import os
import glob
import json
import torch
import argparse
import numpy as np

try:
    from dia2 import Dia2, GenerationConfig, SamplingConfig
except ImportError:
    print("ERROR: dia2 library not found!")
    print("Install it with: pip install git+https://github.com/nari-labs/dia2.git")
    raise

# Dia2 default sample rate
DIA2_SAMPLE_RATE = 44100


class TTS:
    def __init__(self, model_id: str = "nari-labs/Dia2-2B", dtype: str = "bfloat16"):
        """
        Initialize the Dia2 TTS model.

        Args:
            model_id: HuggingFace model ID (nari-labs/Dia2-1B or nari-labs/Dia2-2B)
            dtype: Model dtype (bfloat16, float16, or float32)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Dia2 model: {model_id} on {self.device}...")

        try:
            self.model = Dia2.from_repo(model_id, device=self.device, dtype=dtype)
            self.config = GenerationConfig(
                cfg_scale=2.0,
                audio=SamplingConfig(temperature=0.8, top_k=50),
                use_cuda_graph=True if self.device == "cuda" else False,
            )
        except Exception as e:
            print(f"\nError loading model {model_id}: {e}")
            raise

    def synthesize(self, prompt: str, output_wav_path: str):
        """
        Synthesize speech from a text prompt and save to a WAV file.

        Args:
            prompt (str): The text prompt to synthesize. Should include [S1], [S2] tags for dialogue.
            output_wav_path (str): The path to save the output audio file.
        """
        print(f"Synthesizing prompt: {prompt[:50]}...")

        os.makedirs(os.path.dirname(os.path.abspath(output_wav_path)), exist_ok=True)

        # Generate audio using Dia2
        result = self.model.generate(prompt, config=self.config, output_wav=output_wav_path, verbose=True)

        print(f"Saved audio to {output_wav_path}")


def _append_silence(wav_path: str, silence_seconds: float, sample_rate: int = DIA2_SAMPLE_RATE) -> None:
    """Append silence to an existing WAV file in-place using scipy."""
    import scipy.io.wavfile as wavfile

    sr, audio = wavfile.read(wav_path)
    silence = np.zeros(int(silence_seconds * sr), dtype=audio.dtype)
    combined = np.concatenate([audio, silence])
    wavfile.write(wav_path, sr, combined)
    print(f"  Appended {silence_seconds}s silence to {wav_path}")


def mode_class_dataset_tts(dataset_path: str, answer_time: float = 10.0) -> None:
    """
    Generate TTS audio files for each entry in a mode-class dataset.

    For every ``dataset_path/<id>/input.json`` that contains
    ``complete_sentence`` and ``incomplete_sentence``, produce:

    * ``complete_sentence.wav``   – TTS of the complete sentence + *answer_time*
      seconds of appended silence (giving the model time to respond).
    * ``incomplete_sentence.wav`` – TTS of the incomplete sentence (no extra
      silence).

    Args:
        dataset_path: Root directory (e.g. ``Full-Duplex-Bench/data/mode_class``).
        answer_time:  Seconds of silence appended after the complete sentence.
    """
    tts = TTS()

    pattern = os.path.join(dataset_path, "*", "input.json")
    input_files = sorted(glob.glob(pattern), key=lambda p: int(os.path.basename(os.path.dirname(p))))

    if not input_files:
        print(f"No input.json files found under {dataset_path}/*/")
        return

    print(f"Found {len(input_files)} entries in {dataset_path}")

    for input_json_path in input_files:
        entry_dir = os.path.dirname(input_json_path)
        entry_id = os.path.basename(entry_dir)

        with open(input_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        complete_text = data["complete_sentence"]
        incomplete_text = data["incomplete_sentence"]

        print(f"\n--- Entry {entry_id} ---")

        # 1) Synthesize complete_sentence.wav
        complete_wav = os.path.join(entry_dir, "complete_sentence.wav")
        tts.synthesize(f"[S1]{complete_text}", complete_wav)
        # Append silence so the model has time to answer
        if answer_time > 0:
            _append_silence(complete_wav, answer_time)

        # 2) Synthesize incomplete_sentence.wav
        incomplete_wav = os.path.join(entry_dir, "incomplete_sentence.wav")
        tts.synthesize(f"[S1]{incomplete_text}", incomplete_wav)

    print(f"\nDone. Generated audio for {len(input_files)} entries.")


def trait_tts(trait: str, type: str = "extract"):
    """
    Generate TTS audio files for a trait's instructions and questions.
    
    Example function:
    trait_tts("evil", "extract")
    This will read the instruction and questions from data_generation/trait_data_extract/evil.json
    and generate TTS audio files for each question under data_generation/trait_data_extract_audio/evil/
    trait_data_extract_audio/evil/ will contain:
    audio/
        instruction_0_pos.wav
        instruction_0_neg.wav
        instruction_1_pos.wav
        instruction_1_neg.wav
        ...
        question_0.wav
        question_1.wav
        ...
        eval_prompt.wav
    evil.json, but with the questions and instruction replaced with the corresponding audio file paths
    
    Args:
        trait: The trait name (e.g., "evil", "optimistic")
        type: Type of data ("extract" or other)
    """
    import json
    
    # Get the script's directory and project root
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Determine paths based on type (relative to project root)
    input_json_path = os.path.join(script_dir, f"trait_data_{type}", f"{trait}.json")
    output_dir = os.path.join(script_dir, f"trait_data_{type}_audio", trait)
    audio_dir = os.path.join(output_dir, "audio")
    
    # Create output directories
    os.makedirs(audio_dir, exist_ok=True)
    
    # Load the trait JSON file
    with open(input_json_path, 'r') as f:
        trait_data = json.load(f)
    
    # Initialize TTS
    tts = TTS()
    
    # Track generated files for the output JSON
    output_data = {
        "instruction": [],
        "questions": [],
        "eval_prompt": None
    }
    
    # Process instructions (pos/neg pairs)
    print(f"\nGenerating instruction audio files for trait '{trait}'...")
    for i, instruction_pair in enumerate(trait_data["instruction"]):
        # Generate positive instruction audio
        pos_text = f"[S1]{instruction_pair['pos']}"
        pos_audio_path = os.path.join(audio_dir, f"instruction_{i}_pos.wav")
        tts.synthesize(pos_text, pos_audio_path)
        
        # Generate negative instruction audio
        neg_text = f"[S1]{instruction_pair['neg']}"
        neg_audio_path = os.path.join(audio_dir, f"instruction_{i}_neg.wav")
        tts.synthesize(neg_text, neg_audio_path)
        
        output_data["instruction"].append({
            "pos": pos_audio_path,
            "neg": neg_audio_path
        })
    
    # Process questions
    print(f"\nGenerating question audio files for trait '{trait}'...")
    for i, question in enumerate(trait_data["questions"]):
        question_text = f"[S1]{question}"
        question_audio_path = os.path.join(audio_dir, f"question_{i}.wav")
        tts.synthesize(question_text, question_audio_path)
        output_data["questions"].append(question_audio_path)
    
    # Process eval_prompt
    print(f"\nGenerating eval_prompt audio file for trait '{trait}'...")
    eval_text = f"[S1]{trait_data['eval_prompt']}"
    eval_audio_path = os.path.join(audio_dir, "eval_prompt.wav")
    tts.synthesize(eval_text, eval_audio_path)
    output_data["eval_prompt"] = eval_audio_path
    
    # Save the output JSON with audio file paths
    output_json_path = os.path.join(output_dir, f"{trait}.json")
    with open(output_json_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    print(f"\nCompleted! Audio files and JSON saved to: {output_dir}")
    return output_dir

if __name__ == "__main__":
    # Examples:
    #   python tts.py --trait evil
    #   python tts.py --mode-class /path/to/mode_class --answer-time 10
    #   python tts.py --synthesize "[S1]Hello world" --output out.wav
    parser = argparse.ArgumentParser(description="Dia2 TTS - Text to Speech Synthesis")

    # Mutually exclusive modes
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--synthesize", type=str,
                       help="Text prompt to synthesize (use [S1]/[S2] tags for dialogue)")
    group.add_argument("--trait", type=str,
                       help="Trait name for batch processing (e.g., 'evil', 'optimistic')")
    group.add_argument("--mode-class", type=str, dest="mode_class",
                       help="Path to mode-class dataset dir (e.g., Full-Duplex-Bench/data/mode_class)")

    # Shared arguments
    parser.add_argument("--model", type=str, default="nari-labs/Dia2-2B",
                        help="Model ID (default: nari-labs/Dia2-2B)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="Model dtype (default: bfloat16)")

    # --synthesize specific
    parser.add_argument("--output", type=str,
                        help="Output WAV file path (required with --synthesize)")

    # --trait specific
    parser.add_argument("--type", type=str, default="extract",
                        help="Type of trait data (default: 'extract', used with --trait)")

    # --mode-class specific
    parser.add_argument("--answer-time", type=float, default=10.0,
                        help="Seconds of silence appended after complete_sentence (default: 10, used with --mode-class)")

    args = parser.parse_args()

    # Validate
    if args.synthesize and not args.output:
        parser.error("--synthesize requires --output")

    # Execute
    if args.synthesize:
        tts = TTS(model_id=args.model, dtype=args.dtype)
        tts.synthesize(args.synthesize, args.output)
    elif args.trait:
        trait_tts(args.trait, args.type)
    elif args.mode_class:
        mode_class_dataset_tts(args.mode_class, answer_time=args.answer_time)