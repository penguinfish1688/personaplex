#!/bin/bash
#SBATCH --job-name=personaplex
#SBATCH --output=personaplex_%j.out
#SBATCH --error=personaplex_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#
# SLURM job script for running PersonaPlex on MIT ORCD
# Adjust the parameters above based on your needs

# Set your Hugging Face token here or load from environment
export HF_TOKEN=${HF_TOKEN:-"your_huggingface_token_here"}

# Print job information
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Starting at: $(date)"
echo "======================================"

# Print GPU information
echo "GPU Information:"
nvidia-smi

# Navigate to project directory
cd ~/personaplex || cd $SLURM_SUBMIT_DIR

# Create cache directories
mkdir -p "${HOME}/.cache/personaplex"
mkdir -p "${HOME}/.ssl/personaplex"

# Run the PersonaPlex container
echo "Starting PersonaPlex server..."
apptainer run \
    --nv \
    --bind "${HOME}/.cache/personaplex:/root/.cache" \
    --bind "${HOME}/.ssl/personaplex:/app/ssl" \
    --env "HF_TOKEN=$HF_TOKEN" \
    personaplex.sif

echo "======================================"
echo "Job finished at: $(date)"
echo "======================================"
