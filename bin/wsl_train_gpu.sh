#!/bin/bash
# ATLAS GPU Training via WSL2 + RTX 4090
# Usage: wsl -d Ubuntu-22.04 -- bash "/mnt/f/Old Files/.../bin/wsl_train_gpu.sh"

source ~/atlas-venv/bin/activate

# Set CUDA paths
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64

# Add NVIDIA pip package paths
SITE_PKGS=$(python3 -c "import site; print(site.getsitepackages()[0])")
for pkg in cudnn cuda_runtime cublas cuda_nvrtc cufft cusolver cusparse; do
    dir="$SITE_PKGS/nvidia/${pkg}/lib"
    [ -d "$dir" ] && export LD_LIBRARY_PATH="$dir:$LD_LIBRARY_PATH"
done

cd "/mnt/f/Old Files/Archived/Coding_Projects/Github_Projects/atlas-ml-trading"

PRESET=${1:-gpu}
MODE=${2:-tech}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ATLAS GPU Training | preset=$PRESET | mode=$MODE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ "$MODE" = "all" ]; then
    python train_gpu.py --preset "$PRESET" --all
elif [ "$MODE" = "validate" ]; then
    SYMBOL=${3:-AAPL}
    python train_gpu.py --validate "$SYMBOL" --preset "$PRESET"
else
    python train_gpu.py --preset "$PRESET"
fi
