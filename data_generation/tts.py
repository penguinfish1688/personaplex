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
    def __init__(
        self,
        model_id: str = "nari-labs/Dia2-2B",
        dtype: str = "bfloat16",
        prefix_speaker_1: str | None = None,
        prefix_speaker_2: str | None = None,
        include_prefix: bool = False,
    ):
        """
        Initialize the Dia2 TTS model.

        To fix the voice template, export the voice template audio paths as environment variables:

        export DIA2_PREFIX_SPEAKER_1="<some path>/dia2/example_prefix1.wav"
        export DIA2_PREFIX_SPEAKER_2="<some path>/dia2/example_prefix2.wav"

        Args:
            model_id: HuggingFace model ID (nari-labs/Dia2-1B or nari-labs/Dia2-2B)
            dtype: Model dtype (bfloat16, float16, or float32)
            prefix_speaker_1: Optional reference audio path for [S1] voice conditioning.
            prefix_speaker_2: Optional reference audio path for [S2] voice conditioning.
            include_prefix: Whether to keep prefix audio in final waveform.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Dia2 model: {model_id} on {self.device}...")

        # Voice conditioning defaults come from environment variables.
        env_prefix_1 = os.environ.get("DIA2_PREFIX_SPEAKER_1")
        env_prefix_2 = os.environ.get("DIA2_PREFIX_SPEAKER_2")
        self.prefix_speaker_1 = prefix_speaker_1 if prefix_speaker_1 is not None else env_prefix_1
        self.prefix_speaker_2 = prefix_speaker_2 if prefix_speaker_2 is not None else env_prefix_2

        def _normalize_path(path: str | None) -> str | None:
            if not path:
                return None
            return os.path.abspath(os.path.expanduser(path))

        self.prefix_speaker_1 = _normalize_path(self.prefix_speaker_1)
        self.prefix_speaker_2 = _normalize_path(self.prefix_speaker_2)
        self.include_prefix = include_prefix

        for label, path in (("prefix_speaker_1", self.prefix_speaker_1), ("prefix_speaker_2", self.prefix_speaker_2)):
            if path and not os.path.exists(path):
                raise FileNotFoundError(f"{label} file not found: {path}")

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

        # Optional prefix conditioning stabilizes voice identity across runs.
        result = self.model.generate(
            prompt,
            config=self.config,
            output_wav=output_wav_path,
            prefix_speaker_1=self.prefix_speaker_1,
            prefix_speaker_2=self.prefix_speaker_2,
            include_prefix=self.include_prefix,
            verbose=True,
        )

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

def user_interrupt_dataset_tts(dataset_path: str, initial_silence: float = 0.0, no_interrupt: bool = False) -> None:
    """
    take dataset path as arg for reference the dataset would be somethign like /home/penguinfish/personaplex/Full-Duplex-Bench/data/user_interrupt/interrupt_dataset
    in each json you have
    {
    "initial_silence": 3.0,
    "question_1": "Can you explain how the internet works from home routers to global data centers?",
    "response_duration_1": 6.261823826770641,
    "question_2": "goat is a building for people to live in, is this information correct?",
    "question_2_starting_word": "goat",
    "question_2_describing_word": "house",
    "question_2_consistent": false,
    "question_2_answer": false,
    "response_duration_2": 15.0
    }
    you synthesize input.wav that contains these in time order with the following structure:
    1) initial_silence seconds of silence (if >0, otherwise skip)
    2) question_1
    3) response_duration_1 seconds of silence
    4) question_2
    5) response_duration_2 seconds of silence
    and save the results input.wav to the same dir as <root>/*/user_interrupt_text.json

    
    """
    """
    Update this function so that when no_interrupt is true only syntheize:
   1) initial_silence seconds of silence (if >0, otherwise skip)
    4) question_2
    5) response_duration_2 seconds of silence
    and save the results input.wav to the same dir as <root>/*/user_interrupt_text.json

    save conversation_time.json under <dataset_path>/*/ with the following structure:
    {
        "Q1_start": initial_silence,
        "Q1_end": initial_silence + duration of question_1 audio,
        "Q2_start": Q1_end + response_duration_1,
        "Q2_end": Q2_start + duration of question_2 audio
    }
    """
    import scipy.io.wavfile as wavfile

    pattern_a = os.path.join(dataset_path, "*", "user_interrupts_text.json")
    pattern_b = os.path.join(dataset_path, "*", "user_interrupt_text.json")
    input_files = sorted(set(glob.glob(pattern_a) + glob.glob(pattern_b)), key=lambda p: os.path.basename(os.path.dirname(p)))

    if not input_files:
        print(f"No user interrupt JSON files found under {dataset_path}/*/")
        return

    tts = TTS()
    print(f"Found {len(input_files)} entries in {dataset_path}")

    for input_json_path in input_files:
        entry_dir = os.path.dirname(input_json_path)
        entry_id = os.path.basename(entry_dir)

        with open(input_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        q2 = data["question_2"]
        entry_initial_silence = float(data.get("initial_silence", initial_silence))
        response_duration_2 = float(data["response_duration_2"])
        response_duration_1 = float(data.get("response_duration_1", 0.0))

        if not no_interrupt:
            q1 = data["question_1"]

        q1_wav = os.path.join(entry_dir, "_q1_tmp.wav")
        q2_wav = os.path.join(entry_dir, "_q2_tmp.wav")
        output_wav = os.path.join(entry_dir, "input.wav")

        print(f"\n--- Entry {entry_id} ---")
        tts.synthesize(f"[S1]{q2}", q2_wav)

        tmp_wavs = [q2_wav]
        if not no_interrupt:
            tts.synthesize(f"[S1]{q1}", q1_wav)
            tmp_wavs.append(q1_wav)

        sr2, q2_audio = wavfile.read(q2_wav)
        q2_audio = np.asarray(q2_audio)
        q2_duration_sec = float(q2_audio.shape[0]) / float(sr2)

        if no_interrupt:
            # In no_interrupt mode, question 1 is omitted from the waveform timeline.
            q1_duration_sec = 0.0
            q1_start = float(max(0.0, entry_initial_silence))
            q1_end = q1_start
            q2_start = q1_end
            q2_end = q2_start + q2_duration_sec

            if q2_audio.ndim == 1:
                shape_initial = (int(max(0.0, entry_initial_silence) * sr2),)
                shape_end = (int(max(0.0, response_duration_2) * sr2),)
            else:
                channels = int(q2_audio.shape[-1])
                shape_initial = (int(max(0.0, entry_initial_silence) * sr2), channels)
                shape_end = (int(max(0.0, response_duration_2) * sr2), channels)

            initial_pad = np.zeros(shape_initial, dtype=q2_audio.dtype)
            end_pad = np.zeros(shape_end, dtype=q2_audio.dtype)
            combined = np.concatenate([initial_pad, q2_audio, end_pad], axis=0)
            wavfile.write(output_wav, sr2, combined)
            print(f"Saved {output_wav} (no_interrupt=True)")
        else:
            sr1, q1_audio = wavfile.read(q1_wav)
            q1_audio = np.asarray(q1_audio)
            q1_duration_sec = float(q1_audio.shape[0]) / float(sr1)
            if sr1 != sr2:
                raise ValueError(f"Sample rate mismatch for entry {entry_id}: q1={sr1}, q2={sr2}")

            if q1_audio.ndim != q2_audio.ndim:
                raise ValueError(f"Channel mismatch for entry {entry_id}: q1 ndim={q1_audio.ndim}, q2 ndim={q2_audio.ndim}")
            if q1_audio.ndim == 2 and q1_audio.shape[-1] != q2_audio.shape[-1]:
                raise ValueError(
                    f"Channel count mismatch for entry {entry_id}: q1={q1_audio.shape[-1]}, q2={q2_audio.shape[-1]}"
                )

            if q1_audio.ndim == 1:
                shape_initial = (int(max(0.0, entry_initial_silence) * sr1),)
                shape_mid = (int(max(0.0, response_duration_1) * sr1),)
                shape_end = (int(max(0.0, response_duration_2) * sr1),)
            else:
                channels = int(q1_audio.shape[-1])
                shape_initial = (int(max(0.0, entry_initial_silence) * sr1), channels)
                shape_mid = (int(max(0.0, response_duration_1) * sr1), channels)
                shape_end = (int(max(0.0, response_duration_2) * sr1), channels)

            initial_pad = np.zeros(shape_initial, dtype=q1_audio.dtype)
            mid_pad = np.zeros(shape_mid, dtype=q1_audio.dtype)
            end_pad = np.zeros(shape_end, dtype=q1_audio.dtype)

            combined = np.concatenate([initial_pad, q1_audio, mid_pad, q2_audio, end_pad], axis=0)
            wavfile.write(output_wav, sr1, combined)
            print(f"Saved {output_wav}")

            q1_start = float(max(0.0, entry_initial_silence))
            q1_end = q1_start + q1_duration_sec
            q2_start = q1_end + float(max(0.0, response_duration_1))
            q2_end = q2_start + q2_duration_sec

        conversation_time = {
            "Q1_start": q1_start,
            "Q1_end": q1_end,
            "Q2_start": q2_start,
            "Q2_end": q2_end,
        }
        conversation_time_path = os.path.join(entry_dir, "conversation_time.json")
        with open(conversation_time_path, "w", encoding="utf-8") as f:
            json.dump(conversation_time, f, indent=2, ensure_ascii=False)
        print(f"Saved {conversation_time_path}")

        for tmp_wav in tmp_wavs:
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)

    print(f"\nDone. Generated input.wav for {len(input_files)} entries.")


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