#!/bin/bash
# Run script for PersonaPlex Apptainer container on MIT ORCD
# Usage: 
#   ./run_apptainer.sh              # Start server
#   ./run_apptainer.sh --shell      # Interactive shell
#   ./run_apptainer.sh --help       # Show help

set -e

# Configuration
IMAGE_FILE="personaplex.sif"
CACHE_DIR="${HOME}/.cache/personaplex"
SSL_DIR="${HOME}/.ssl/personaplex"
WORKSPACE_DIR="${HOME}/workspace"
SSH_DIR="${HOME}/.ssh"
INTERACTIVE_MODE=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --shell|--interactive|-i)
            INTERACTIVE_MODE=true
            shift
            ;;
        --help|-h)
            echo "PersonaPlex Apptainer Runner"
            echo ""
            echo "Usage:"
            echo "  ./run_apptainer.sh              Run PersonaPlex server"
            echo "  ./run_apptainer.sh --shell      Enter interactive shell"
            echo "  ./run_apptainer.sh --help       Show this help"
            echo ""
            echo "Server arguments (pass after script options):"
            echo "  --cpu-offload                   Enable CPU offload"
            echo "  --port PORT                     Custom port (default: 8998)"
            echo ""
            echo "Examples:"
            echo "  ./run_apptainer.sh                    # Start server"
            echo "  ./run_apptainer.sh --cpu-offload      # Server with CPU offload"
            echo "  ./run_apptainer.sh --shell            # Interactive shell"
            exit 0
            ;;
    esac
done

# Create necessary directories
mkdir -p "$CACHE_DIR"
mkdir -p "$SSL_DIR"
mkdir -p "$WORKSPACE_DIR"

# Check if container image exists
if [ ! -f "$IMAGE_FILE" ]; then
    echo "Error: Container image '$IMAGE_FILE' not found!"
    echo "Please build it first using: ./build_apptainer.sh"
    exit 1
fi

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN environment variable is not set!"
    echo "Please set it with: export HF_TOKEN=your_huggingface_token"
    echo "You can get your token from: https://huggingface.co/settings/tokens"
    echo ""
    read -p "Do you want to continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if running on a system with apptainer/singularity
if ! command -v apptainer &> /dev/null && ! command -v singularity &> /dev/null; then
    echo "Error: Neither apptainer nor singularity found in PATH"
    exit 1
fi

# Use apptainer if available, otherwise singularity
CONTAINER_CMD=$(command -v apptainer || command -v singularity)

if [ "$INTERACTIVE_MODE" = true ]; then
    echo "Starting interactive shell in PersonaPlex container..."
    echo "Using: $CONTAINER_CMD"
    echo "Cache directory: $CACHE_DIR"
    echo "SSL directory: $SSL_DIR"
    echo "Workspace directory: $WORKSPACE_DIR"
    echo "SSH keys: $SSH_DIR"
    echo ""
    echo "You can now:"
    echo "  - Run Python commands"
    echo "  - Test the model"
    echo "  - Install packages (temporary)"
    echo "  - Clone repos with SSH: git clone git@github.com:user/repo.git"
    echo "  - Type 'exit' to leave"
    echo ""
    
    # Start interactive shell
    $CONTAINER_CMD shell \
        --nv \
        --bind "$CACHE_DIR:/root/.cache" \
        --bind "$SSL_DIR:/app/ssl" \
        --bind "$WORKSPACE_DIR:/workspace" \
        --bind "$SSH_DIR:/root/.ssh:ro" \
        --env "HF_TOKEN=$HF_TOKEN" \
        "$IMAGE_FILE"
else
    echo "Starting PersonaPlex container..."
    echo "Using: $CONTAINER_CMD"
    echo "Cache directory: $CACHE_DIR"
    echo "SSL directory: $SSL_DIR"
    echo "Workspace directory: $WORKSPACE_DIR"
    echo ""

    # Run the container with GPU support
    # --nv: Enable NVIDIA GPU support
    # --bind: Mount directories
    # --env: Pass environment variables (HF_TOKEN)
    $CONTAINER_CMD run \
        --nv \
        --bind "$CACHE_DIR:/root/.cache" \
        --bind "$SSL_DIR:/app/ssl" \
        --bind "$WORKSPACE_DIR:/workspace" \
        --bind "$SSH_DIR:/root/.ssh:ro" \
        --env "HF_TOKEN=$HF_TOKEN" \
        "$IMAGE_FILE" "$@"
fi
