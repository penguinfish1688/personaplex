import os
import torch

try:
    from dia2 import Dia2, GenerationConfig, SamplingConfig
except ImportError:
    print("ERROR: dia2 library not found!")
    print("Install it with: pip install git+https://github.com/nari-labs/dia2.git")
    raise

class TTS:
    def __init__(self, model_id: str = "nari-labs/Dia2-1B", dtype: str = "bfloat16"):
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

def trait_tts(trait: str, type: str="extract"):
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
    
    # Determine paths based on type
    input_json_path = f"data_generation/trait_data_{type}/{trait}.json"
    output_dir = f"data_generation/trait_data_{type}_audio/{trait}"
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
    # cd ~/personaplex/personaplex ; python3 -m data_generation.tts "Hello, how are you today" "tmp/tts/example.wav"
    import sys

    if len(sys.argv) < 3:
        print("Usage: python tts.py <text_prompt> <output_wav_path>")
        sys.exit(1)
    
    text_prompt = sys.argv[1]
    output_wav = sys.argv[2]
    
    tts = TTS()
    tts.synthesize(text_prompt, output_wav)