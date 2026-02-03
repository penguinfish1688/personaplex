
import torch
import torchaudio
import logging
from pathlib import Path

# Relative import to access the models package from moshi/moshi/persona_vector/data_generation
from ..models import loaders

logger = logging.getLogger(__name__)

class AudioCodec:
    """
    Handles conversion between audio files (WAV) and Moshi voice prompt tokens (.pt).
    Uses the Mimi neural audio codec for encoding and decoding.
    """
    def __init__(self, mimi_weight: str, device: str = "cuda"):
        """
        Initialize the codec with a pretrained Mimi model.

        Args:
            mimi_weight: Path to the mimi model weight file.
            device: Device to run the model on ('cuda' or 'cpu').
        """
        self.device = device
        self.sample_rate = loaders.SAMPLE_RATE  # 24000 Hz
        
        logger.info(f"Loading Mimi model from {mimi_weight} on {device}...")
        self.mimi = loaders.get_mimi(mimi_weight, device=device)
        self.mimi.eval()
        # Moshi uses 8 codebooks for its voice representation
        self.mimi.set_num_codebooks(8)

    def encode(self, audio_path: str, output_pt_path: str):
        """
        Convert an audio file to a .pt file containing Mimi codes.
        
        Args:
            audio_path: Path to input audio file (e.g. .wav).
            output_pt_path: Path to save the output .pt token file.
        """
        # Load audio using torchaudio
        wav, sr = torchaudio.load(audio_path)
        
        # Resample to Model SR (24kHz) if needed
        if sr != self.sample_rate:
            transform = torchaudio.transforms.Resample(sr, self.sample_rate)
            wav = transform(wav)

        # Mix down to mono if necessary (Mimi input is mono)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
            
        # Add batch dimension: [Channels, Time] -> [1, Channels, Time]
        wav = wav.unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Encode audio to discrete codes
            # Output shape is typically [Batch, Codebooks, Time]
            codes = self.mimi.encode(wav)
        
        # Move to CPU for saving
        codes = codes.cpu()
        
        # Ensure directory exists
        Path(output_pt_path).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(codes, output_pt_path)
        logger.info(f"Encoded {audio_path} -> {output_pt_path} (Shape: {codes.shape})")

    def decode(self, pt_path: str, output_audio_path: str):
        """
        Convert a .pt file containing Mimi codes back to an audio file.

        Args:
            pt_path: Path to input .pt token file.
            output_audio_path: Path to save the reconstructed .wav file.
        """
        # Load tokens
        codes = torch.load(pt_path, map_location=self.device)
        
        if not isinstance(codes, torch.Tensor):
            raise ValueError(f"File {pt_path} did not contain a torch Tensor.")

        # Ensure correct device and type
        codes = codes.to(self.device).long()

        # Input shape check. Mimi decode expects [Batch, Codebooks, Time]
        # If the saved tensor is [Codebooks, Time], add batch dimension
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)
            
        with torch.no_grad():
            # Decode codes back to PCM audio
            pcm = self.mimi.decode(codes)
            
        # Remove batch dimension: [1, 1, Time] -> [1, Time]
        pcm = pcm.squeeze(0).cpu()
        
        # Save to WAV
        Path(output_audio_path).parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(output_audio_path, pcm, self.sample_rate)
        logger.info(f"Decoded {pt_path} -> {output_audio_path}")


if __name__ == "__main__":
    import sys
    from huggingface_hub import hf_hub_download
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python audio_codec.py <input.pt> [output.wav]")
        print("  Converts a .pt voice token file to .wav audio")
        sys.exit(1)
    
    input_pt = sys.argv[1]
    output_wav = sys.argv[2] if len(sys.argv) > 2 else Path(input_pt).with_suffix('.wav')
    
    # Download Mimi model
    print("Downloading Mimi model...")
    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Convert
    codec = AudioCodec(mimi_weight, device=device)
    codec.decode(input_pt, str(output_wav))
    print(f"✓ Converted: {output_wav}")
