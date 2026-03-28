#!/bin/bash
# ATLAS WSL2 GPU Training Setup
# Run this inside WSL2 Ubuntu to set up GPU-accelerated training
# Usage: bash /mnt/f/Old\ Files/Archived/Coding_Projects/Github_Projects/atlas-ml-trading/bin/setup_wsl_gpu.sh

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ATLAS - WSL2 GPU Training Setup (RTX 4090)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

PROJECT_DIR="/mnt/f/Old Files/Archived/Coding_Projects/Github_Projects/atlas-ml-trading"

# Step 1: Check GPU access from WSL2
echo "[1/6] Checking GPU access..."
if nvidia-smi > /dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    echo "  GPU detected: $GPU_NAME"
else
    echo "  ERROR: nvidia-smi not found. NVIDIA driver must be installed on Windows."
    echo "  WSL2 uses the Windows GPU driver automatically."
    echo "  Make sure your Windows NVIDIA driver is up to date."
    exit 1
fi

# Step 2: Install Python 3.11
echo ""
echo "[2/6] Setting up Python 3.11..."
if python3.11 --version > /dev/null 2>&1; then
    echo "  Python 3.11 already installed: $(python3.11 --version)"
else
    echo "  Installing Python 3.11..."
    sudo apt-get update -qq
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -qq
    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
    echo "  Installed: $(python3.11 --version)"
fi

# Step 3: Create venv
echo ""
echo "[3/6] Creating virtual environment..."
VENV_DIR="$PROJECT_DIR/venv_wsl"
if [ -d "$VENV_DIR" ]; then
    echo "  venv_wsl already exists, skipping"
else
    python3.11 -m venv "$VENV_DIR"
    echo "  Created venv at $VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Step 4: Install dependencies
echo ""
echo "[4/6] Installing dependencies..."
pip install --upgrade pip -q
pip install -r "$PROJECT_DIR/requirements.txt" -q
echo "  Dependencies installed"

# Step 5: Verify TensorFlow GPU
echo ""
echo "[5/6] Verifying TensorFlow GPU..."
python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f'  TensorFlow: {tf.__version__}')
print(f'  GPUs found: {len(gpus)}')
for gpu in gpus:
    try:
        details = tf.config.experimental.get_device_details(gpu)
        name = details.get('device_name', 'Unknown')
        print(f'  Device: {name}')
    except:
        print(f'  Device: {gpu.name}')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print('  GPU memory growth: enabled')
    print('  Status: READY FOR TRAINING')
else:
    print('  WARNING: No GPU detected in WSL2')
    print('  Check: https://docs.nvidia.com/cuda/wsl-user-guide/')
"

# Step 6: Summary
echo ""
echo "[6/6] Setup complete!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Ready! Activate with:"
echo "    source \"$VENV_DIR/bin/activate\""
echo ""
echo "  Then train with GPU:"
echo "    cd \"$PROJECT_DIR\""
echo "    python -c \""
echo "from ml.prediction_service import PredictionService"
echo "ps = PredictionService(provider='local')"
echo "# GPU training - 7 tech stocks, BiLSTM+Attention (~1.2M params)"
echo "symbols = ['AAPL','MSFT','GOOGL','AMZN','NVDA','META','TSLA']"
echo "for sym in symbols:"
echo "    result = ps.train_model_gpu(sym, preset='gpu', days=730, epochs=300)"
echo "    print(f'{sym}: {result[\"status\"]}')"
echo "\""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
