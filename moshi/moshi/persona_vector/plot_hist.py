# This script loads a mean_persona_vectors.pt file as output by gen_vector.py
# and plots histograms of cosine similarities between individual prompt diffs and the mean vector.

import torch
import sys
import os
import matplotlib.pyplot as plt
import argparse
import numpy as np

def plot_cosine_histograms(pt_file, output_dir):
    print(f"Loading {pt_file}...")
    data = torch.load(pt_file)
    
    if "raw_cosines" not in data:
        print("Error: 'raw_cosines' not found in the .pt file. Please regenerate using the updated gen_vector.py.")
        return

    raw_cosines = data["raw_cosines"]
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Text Layers ---
    text_cosines = raw_cosines["text"]
    num_text_layers = len(text_cosines)
    print(f"Plotting histograms for {num_text_layers} text layers...")

    # We can't plot ALL layers individually in one readable figure, so let's select representative ones
    # Early, Mid, Late, and the final layer
    selected_indices = [0, num_text_layers // 3, 2 * num_text_layers // 3, num_text_layers - 1]
    # Ensure indices are valid and unique
    selected_indices = sorted(list(set([i for i in selected_indices if i < num_text_layers])))
    
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(selected_indices):
        cos_vals = text_cosines[idx].numpy()
        plt.subplot(2, 2, i + 1)
        plt.hist(cos_vals, bins=30, range=(-1.0, 1.0), edgecolor='black', alpha=0.7)
        plt.title(f"Text Layer {idx}\nMean: {np.mean(cos_vals):.3f}, Std: {np.std(cos_vals):.3f}")
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Count")
        plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "text_layers_histograms.png"))
    plt.close()
    
    # Plot heatmap of cosine similarities across all text layers
    # Rows: Prompts, Cols: Layers
    # We need to stack them: [Num_Layers, Num_Prompts]
    # Note: different layers might have same number of prompts (they should)
    
    try:
        all_text_cosines = torch.stack(text_cosines).numpy() # [L, N]
        plt.figure(figsize=(12, 8))
        plt.imshow(all_text_cosines, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(label="Cosine Similarity")
        plt.title("Cosine Similarity Heatmap (Text Layers)")
        plt.xlabel("Prompt Index")
        plt.ylabel("Layer Index")
        plt.savefig(os.path.join(output_dir, "text_layers_heatmap.png"))
        plt.close()
    except Exception as e:
        print(f"Could not create heatmap: {e}")

    print(f"Plots saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Plot cosine similarity histograms from persona vector data.")
    parser.add_argument("--pt_file", type=str, required=True, help="Path to mean_persona_vectors.pt file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save plots")
    
    args = parser.parse_args()
    plot_cosine_histograms(args.pt_file, args.output_dir)

if __name__ == "__main__":
    main()
