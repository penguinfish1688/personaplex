import os
import torch
import soundfile as sf
from transformers import AutoModel, AutoTokenizer

class TTS:
    def __init__(self, model_id: str = "nari-labs/Dia-1.6B"):
        """
        Initialize the Dia TTS model using Hugging Face Transformers.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Dia model: {model_id} on {self.device}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        except (ValueError, OSError, ImportError) as e:
            print(f"\nError loading model {model_id}: {e}")
            if "sentencepiece" in str(e) or "tiktoken" in str(e):
                print("\nPOSSIBLE FIX: proper tokenizer backend is missing.")
                print("Try installing sentencepiece and upgrading transformers:")
                print("    pip install sentencepiece protobuf")
                print("    pip install --upgrade transformers")
            raise 

    def synthesize(self, prompt: str, output_wav_path: str):
        """
        Synthesize speech from a text prompt and save to a WAV file.
        
        Args:
            prompt (str): The text prompt to synthesize. Should include [S1], [S2] tags for dialogue.
            output_wav_path (str): The path to save the output audio file.
        """
        print(f"Synthesizing prompt: {prompt[:50]}...")
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(**inputs)
            
        # Extract audio
        if hasattr(output, "audio_values"):
            audio = output.audio_values.cpu().float().numpy().squeeze()
        elif hasattr(output, "waveform"):
            audio = output.waveform.cpu().float().numpy().squeeze()
        elif isinstance(output, torch.Tensor):
            audio = output.cpu().float().numpy().squeeze()
        else:
            # Fallback if output contains sequences
            # Assume it returns audio tensor if generic generate is overridden or correct model class used
            audio = output.cpu().float().numpy().squeeze()

        # Sampling rate from config or default
        sample_rate = getattr(self.model.config, "sampling_rate", 44100)
        
        os.makedirs(os.path.dirname(os.path.abspath(output_wav_path)), exist_ok=True)
        sf.write(output_wav_path, audio, sample_rate)
        print(f"Saved audio to {output_wav_path}")

if __name__ == "__main__":
    # cd ~/personaplex/moshi ; python3 -m moshi.persona_vector.data_generation.tts "Hello, how are you today" "tmp/tts/example.wav"
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python tts.py <text_prompt> <output_wav_path>")
        sys.exit(1)
    
    text_prompt = sys.argv[1]
    output_wav = sys.argv[2]
    
    tts = TTS()
    tts.synthesize(text_prompt, output_wav)