# This script loads a mean_persona_vectors.pt file as output by gen_vector.py
# and plots histograms of cosine similarities between individual prompt diffs and the mean vector.

import torch
import sys
import os
import matplotlib.pyplot as plt
import argparse
import numpy as np
from typing import List
from ..models.lm import HiddenLayerOutputs # Necessary for unpickling

def plot_cosine_histograms(pt_file, output_dir):
    print(f"Loading {pt_file}...")
    torch.serialization.add_safe_globals([HiddenLayerOutputs])
    data = torch.load(pt_file, map_location=torch.device('cpu'), weights_only=False)
    
    # Extract file name
    filename = os.path.basename(pt_file)

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
    plt.savefig(os.path.join(output_dir, f"{filename}_text_layers_histograms.png"))
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
        plt.savefig(os.path.join(output_dir, f"{filename}_text_layers_heatmap.png"))
        plt.close()
    except Exception as e:
        print(f"Could not create heatmap: {e}")

    print(f"Plots saved to {output_dir}")

def calculate_similarity_between_traits(pt_files: List[str], output_dir: str):
    """
    Loads multiple persona vector .pt files.
    Computes cosine similarity per layer, plots per-layer heatmaps,
    and an average similarity heatmap.
    """
    if not pt_files:
        print("No files provided for similarity calculation.")
        return

    print(f"Comparing traits from {len(pt_files)} files...")
    torch.serialization.add_safe_globals([HiddenLayerOutputs])
    
    traits = []
    traits_layers_vectors = [] 
    
    for fpath in pt_files:
        if not os.path.exists(fpath):
            print(f"Warning: File not found: {fpath}")
            continue

        # Extract a readable name from filename (e.g., 'evil' from 'evil_mean_persona_vectors.pt')
        basename = os.path.basename(fpath)
        name = basename.replace("_mean_persona_vectors.pt", "").replace("mean_persona_vectors.pt", "").replace(".pt", "")
        if not name: 
            name = basename
        
        try:
            data = torch.load(fpath, map_location='cpu', weights_only=False)
            
            text_layers = None

            # Check for the dictionary structure found in inspection
            if isinstance(data, dict) and 'persona_vector' in data:
                pv = data['persona_vector']
                if isinstance(pv, dict) and 'text' in pv:
                    text_layers = pv['text']
                # Fallback if persona_vector is the HiddenLayerOutputs object or similar
                elif hasattr(pv, 'text_hidden_states'):
                     text_layers = pv.text_hidden_states
            
            # Handle other/older formats
            if text_layers is None:
                if isinstance(data, dict) and "mean_diff" in data:
                    mean_obj = data["mean_diff"]
                    if hasattr(mean_obj, 'text_hidden_states'):
                        text_layers = mean_obj.text_hidden_states
                elif isinstance(data, HiddenLayerOutputs):
                    text_layers = data.text_hidden_states
                else:
                    # Try to search for hidden layer outputs in values
                    if isinstance(data, dict):
                        for k, v in data.items():
                            if isinstance(v, HiddenLayerOutputs):
                                text_layers = v.text_hidden_states
                                break

            if text_layers is None:
                print(f"Skipping {fpath}: Could not locate text layers.")
                continue

            # Store the flattened layers for this trait
            flat_layers = [t.float().flatten() for t in text_layers]
            traits_layers_vectors.append(flat_layers)
            traits.append(name)
            
        except Exception as e:
            print(f"Error loading {fpath}: {e}")
                
    if not traits_layers_vectors:
        print("No valid vectors loaded.")
        return

    # Check consistency
    num_layers = len(traits_layers_vectors[0])
    for i, layers in enumerate(traits_layers_vectors):
        if len(layers) != num_layers:
            print(f"Error: Trait {traits[i]} has {len(layers)} layers, expected {num_layers}. Skipping inconsistent data is not implemented.")
            return

    os.makedirs(output_dir, exist_ok=True)
    all_sim_matrices = []

    def plot_matrix(matrix, title, filename):
        plt.figure(figsize=(12, 10))
        plt.imshow(matrix, interpolation='nearest', cmap='RdBu_r', vmin=-1, vmax=1)
        cbar = plt.colorbar()
        cbar.set_label("Cosine Similarity")

        indices = np.arange(len(traits))
        plt.xticks(indices, traits, rotation=45, ha='right')
        plt.yticks(indices, traits)
        
        for i in range(len(traits)):
            for j in range(len(traits)):
                val = matrix[i, j]
                color = "white" if abs(val) > 0.5 else "black"
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)
                
        plt.title(title)
        plt.tight_layout()
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Saved {save_path}")

    # Per-layer calculation
    print(f"Calculating similarity for {num_layers} layers...")
    for layer_idx in range(num_layers):
        layer_vecs = [tl[layer_idx] for tl in traits_layers_vectors]
        matrix = torch.stack(layer_vecs)
        
        norm = matrix.norm(p=2, dim=1, keepdim=True)
        matrix_norm = matrix / (norm + 1e-8)
        sim_matrix = torch.mm(matrix_norm, matrix_norm.t()).numpy()
        
        all_sim_matrices.append(sim_matrix)
        
        plot_matrix(sim_matrix, f"Similarity Matrix - Text Layer {layer_idx}", f"trait_sim_layer_{layer_idx:02d}.png")

    # Average Matrix
    avg_sim_matrix = np.mean(np.array(all_sim_matrices), axis=0)
    plot_matrix(avg_sim_matrix, "Average Cosine Similarity (Across All Layers)", "trait_sim_average.png")

def main():
    """Example usage:
    1. Single file analysis:
       python3 -m moshi.persona_vector.plot_hist --pt_file ... --output_dir ...

       ```bash
       cd /home/penguinfish/personaplex/moshi && python3 -m moshi.persona_vector.plot_hist --pt_file /home/penguinfish/personaplex/tmp/histogram/apathetic_mean_persona_vectors.pt --output_dir /home/penguinfish/personaplex/tmp/histogram/
       ```

    2. Multi trait comparison:
       python3 -m moshi.persona_vector.plot_hist --compare_files path/to/trait1.pt path/to/trait2.pt --output_dir ...

       ```bash
        cd /home/penguinfish/personaplex/moshi && python3 -m moshi.persona_vector.plot_hist --compare_files \
        /home/penguinfish/personaplex/tmp/histogram/evil_mean_persona_vectors.pt \
        /home/penguinfish/personaplex/tmp/histogram/hallucinating_mean_persona_vectors.pt \
        /home/penguinfish/personaplex/tmp/histogram/impolite_mean_persona_vectors.pt \
        /home/penguinfish/personaplex/tmp/histogram/apathetic_mean_persona_vectors.pt \
        /home/penguinfish/personaplex/tmp/histogram/sycophantic_mean_persona_vectors.pt \
        /home/penguinfish/personaplex/tmp/histogram/humorous_mean_persona_vectors.pt \
        /home/penguinfish/personaplex/tmp/histogram/optimistic_mean_persona_vectors.pt \
        /home/penguinfish/personaplex/tmp/histogram/random_mean_persona_vectors.pt \
        --output_dir /home/penguinfish/personaplex/tmp/histogram/trait_comparison
        ```



       

    """
    parser = argparse.ArgumentParser(description="Analyze persona vector .pt files.")
    
    parser.add_argument("--pt_file", type=str, help="Path to a single mean_persona_vectors.pt file for histogram analysis")
    parser.add_argument("--compare_files", type=str, nargs='+', help="List of .pt files to compare against each other")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save plots")
    
    args = parser.parse_args()
    
    if args.pt_file:
        plot_cosine_histograms(args.pt_file, args.output_dir)
        
    if args.compare_files:
        calculate_similarity_between_traits(args.compare_files, args.output_dir)

    if not args.pt_file and not args.compare_files:
        print("Please provide either --pt_file or --compare_files")

if __name__ == "__main__":
    main()