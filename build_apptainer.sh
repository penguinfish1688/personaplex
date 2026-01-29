#!/bin/bash
# Build script for PersonaPlex Apptainer container

set -e

# Check if running on a system with apptainer/singularity
if ! command -v apptainer &> /dev/null && ! command -v singularity &> /dev/null; then
    echo "Error: Neither apptainer nor singularity found in PATH"
    echo "Please install Apptainer/Singularity first"
    exit 1
fi

# Use apptainer if available, otherwise singularity
CONTAINER_CMD=$(command -v apptainer || command -v singularity)

echo "Building PersonaPlex Apptainer container..."
echo "Using: $CONTAINER_CMD"

# Build the container
# Note: This requires sudo/fakeroot or --remote build on systems that support it
$CONTAINER_CMD build personaplex.sif personaplex.def

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Build successful!"
    echo "Container image: personaplex.sif"
    echo ""
    echo "To run the container:"
    echo "  ./run_apptainer.sh"
    echo ""
    echo "Or manually:"
    echo "  export APPTAINERENV_HF_TOKEN=your_huggingface_token"
    echo "  apptainer run --nv personaplex.sif"
else
    echo ""
    echo "✗ Build failed!"
    exit 1
fi
