#!/bin/bash
# Verification script to check if PersonaPlex container has all requirements

set -e

IMAGE_FILE="personaplex.sif"

if [ ! -f "$IMAGE_FILE" ]; then
    echo "Error: Container image '$IMAGE_FILE' not found!"
    echo "Please build it first using: ./build_apptainer.sh"
    exit 1
fi

echo "======================================"
echo "PersonaPlex Container Verification"
echo "======================================"
echo ""

# Use apptainer if available, otherwise singularity
CONTAINER_CMD=$(command -v apptainer || command -v singularity)

echo "✓ Container command: $CONTAINER_CMD"
echo ""

echo "Checking Python version..."
$CONTAINER_CMD exec "$IMAGE_FILE" /app/moshi/.venv/bin/python --version

echo ""
echo "Checking critical dependencies..."
echo ""

# Check PyTorch
echo -n "  PyTorch: "
$CONTAINER_CMD exec "$IMAGE_FILE" /app/moshi/.venv/bin/python -c "import torch; print(f'v{torch.__version__} (CUDA: {torch.cuda.is_available()})')"

# Check numpy
echo -n "  NumPy: "
$CONTAINER_CMD exec "$IMAGE_FILE" /app/moshi/.venv/bin/python -c "import numpy; print(f'v{numpy.__version__}')"

# Check HuggingFace Hub
echo -n "  HuggingFace Hub: "
$CONTAINER_CMD exec "$IMAGE_FILE" /app/moshi/.venv/bin/python -c "import huggingface_hub; print(f'v{huggingface_hub.__version__}')"

# Check sounddevice
echo -n "  sounddevice: "
$CONTAINER_CMD exec "$IMAGE_FILE" /app/moshi/.venv/bin/python -c "import sounddevice; print(f'v{sounddevice.__version__}')"

# Check sentencepiece
echo -n "  sentencepiece: "
$CONTAINER_CMD exec "$IMAGE_FILE" /app/moshi/.venv/bin/python -c "import sentencepiece; print(f'v{sentencepiece.__version__}')"

# Check einops
echo -n "  einops: "
$CONTAINER_CMD exec "$IMAGE_FILE" /app/moshi/.venv/bin/python -c "import einops; print(f'v{einops.__version__}')"

# Check aiohttp
echo -n "  aiohttp: "
$CONTAINER_CMD exec "$IMAGE_FILE" /app/moshi/.venv/bin/python -c "import aiohttp; print(f'v{aiohttp.__version__}')"

# Check accelerate (optional)
echo -n "  accelerate (optional): "
$CONTAINER_CMD exec "$IMAGE_FILE" /app/moshi/.venv/bin/python -c "import accelerate; print(f'v{accelerate.__version__}')" 2>/dev/null || echo "not installed"

# Check safetensors
echo -n "  safetensors: "
$CONTAINER_CMD exec "$IMAGE_FILE" /app/moshi/.venv/bin/python -c "import safetensors; print(f'v{safetensors.__version__}')"

# Check sphn
echo -n "  sphn: "
$CONTAINER_CMD exec "$IMAGE_FILE" /app/moshi/.venv/bin/python -c "import sphn; print(f'v{sphn.__version__}')"

echo ""
echo "Checking moshi package..."
$CONTAINER_CMD exec "$IMAGE_FILE" /app/moshi/.venv/bin/python -c "import moshi; print(f'  Moshi package: OK')"

echo ""
echo "Checking system libraries..."
$CONTAINER_CMD exec "$IMAGE_FILE" ldconfig -p | grep opus && echo "  ✓ libopus found" || echo "  ✗ libopus NOT found"

echo ""
echo "======================================"
echo "Verification Complete!"
echo "======================================"
echo ""
echo "To test GPU access, run:"
echo "  apptainer exec --nv $IMAGE_FILE nvidia-smi"
echo ""
echo "To start the server:"
echo "  export HF_TOKEN=your_token"
echo "  ./run_apptainer.sh"
