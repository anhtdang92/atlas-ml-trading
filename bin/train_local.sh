#!/bin/bash

# Local ML Training Launcher
# Train LSTM models on your local GPU (RTX 4090, 3090, etc.) or CPU.
# No Google Cloud required.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ATLAS - Local GPU Training Launcher"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

# Check TensorFlow and GPU
echo "Checking TensorFlow and GPU..."
GPU_INFO=$(python3 -c "
import sys
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            try:
                details = tf.config.experimental.get_device_details(gpu)
                name = details.get('device_name', 'Unknown GPU')
                print(f'  Found: {name}')
            except:
                print(f'  Found: {gpu.name}')
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f'  Total GPUs: {len(gpus)}')
        print('GPU')
    else:
        print('  No GPU detected - will train on CPU (slower)')
        print('CPU')
except ImportError:
    print('  TensorFlow not installed!')
    print('  Run: pip install tensorflow==2.15.0')
    print('MISSING')
    sys.exit(1)
" 2>/dev/null)

DEVICE=$(echo "$GPU_INFO" | tail -1)
echo ""

if [ "$DEVICE" = "MISSING" ]; then
    echo "Error: TensorFlow not installed. Run: pip install tensorflow==2.15.0"
    exit 1
fi

if [ "$DEVICE" = "GPU" ]; then
    echo "GPU acceleration available"
else
    echo "CPU mode (install CUDA toolkit for GPU support)"
fi
echo ""

# Show menu
echo "Choose training option:"
echo ""
echo "  === CPU / Basic (original model) ==="
echo "1. Quick Training - 7 tech stocks (base model, ~6K params)"
echo "2. Full Training - all 33 stocks (base model)"
echo ""
echo "  === GPU Optimized (RTX 4090 / 3090 / A100) ==="
echo "3. GPU Training - 7 tech stocks"
echo "   3-layer BiLSTM(256) + Attention, ~1.2M params"
echo "   Batch 256, mixed precision, cosine annealing"
echo ""
echo "4. GPU MAX Training - 7 tech stocks"
echo "   4-layer BiLSTM(512) + Attention + Conv1D, ~5M params"
echo "   Batch 512, mixed precision, 60-day lookback"
echo ""
echo "5. GPU Full Training - ALL 33 stocks"
echo "   Choose preset at prompt (gpu or gpu_max)"
echo ""
echo "6. GPU Ensemble - 5 models per stock (most accurate)"
echo "   Choose preset and stocks at prompt"
echo ""
echo "  === Validation ==="
echo "7. Walk-Forward Validation (single stock, GPU model)"
echo ""
echo "8. Single Stock GPU Training"
echo ""
echo "0. Exit"
echo ""
read -p "Enter choice [0-8]: " choice

case $choice in
    1)
        echo ""
        echo "Starting base model training (7 tech stocks, ~6K params each)..."
        echo ""
        cd "$PROJECT_DIR"
        python3 -c "
from ml.prediction_service import PredictionService
ps = PredictionService(provider='local')
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
for i, sym in enumerate(symbols):
    print(f'\n[{i+1}/{len(symbols)}] Training {sym}...')
    result = ps.train_model(sym, days=730, epochs=150)
    if result['status'] == 'success':
        print(f'  Trained: {result[\"training_samples\"]} samples, '
              f'{result[\"epochs_trained\"]} epochs, '
              f'val_loss={result[\"final_val_loss\"]:.6f}')
    else:
        print(f'  Failed: {result.get(\"message\", \"unknown error\")}')
print('\nTraining complete!')
"
        ;;
    2)
        echo ""
        echo "Starting base model training (all 33 stocks)..."
        echo ""
        cd "$PROJECT_DIR"
        python3 -c "
from ml.prediction_service import PredictionService
ps = PredictionService(provider='local')
results = ps.train_all_models(days=730, epochs=150)
success = sum(1 for r in results.values() if r['status'] == 'success')
print(f'\nComplete: {success}/{len(results)} models trained successfully')
"
        ;;
    3)
        echo ""
        echo "Starting GPU training (7 tech stocks, BiLSTM-256 + Attention)..."
        echo "Mixed precision enabled, batch size 256"
        echo ""
        cd "$PROJECT_DIR"
        python3 -c "
from ml.prediction_service import PredictionService
ps = PredictionService(provider='local')
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
for i, sym in enumerate(symbols):
    print(f'\n[{i+1}/{len(symbols)}] GPU Training {sym}...')
    result = ps.train_model_gpu(sym, preset='gpu', days=730, lookback=60, epochs=300)
    if result['status'] == 'success':
        m = result['metrics']
        print(f'  Params: {result.get(\"total_params\", 0):,}')
        print(f'  DA: {m[\"directional_accuracy\"]:.1%} | MAE: {m[\"mae\"]:.4f} | IC: {m[\"information_coefficient\"]:.4f}')
        print(f'  Epochs: {result[\"epochs_trained\"]} | Val loss: {result[\"final_val_loss\"]:.6f}')
    else:
        print(f'  Failed: {result.get(\"message\", \"unknown error\")}')
print('\nGPU Training complete!')
"
        ;;
    4)
        echo ""
        echo "Starting GPU MAX training (7 tech stocks, BiLSTM-512 + Attention + Conv1D)..."
        echo "Mixed precision, batch size 512, 60-day lookback"
        echo "This will push your GPU hard."
        echo ""
        cd "$PROJECT_DIR"
        python3 -c "
from ml.prediction_service import PredictionService
ps = PredictionService(provider='local')
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
for i, sym in enumerate(symbols):
    print(f'\n[{i+1}/{len(symbols)}] GPU MAX Training {sym}...')
    result = ps.train_model_gpu(sym, preset='gpu_max', days=730, lookback=60, epochs=300)
    if result['status'] == 'success':
        m = result['metrics']
        print(f'  Params: {result.get(\"total_params\", 0):,}')
        print(f'  DA: {m[\"directional_accuracy\"]:.1%} | MAE: {m[\"mae\"]:.4f} | IC: {m[\"information_coefficient\"]:.4f}')
        print(f'  Epochs: {result[\"epochs_trained\"]} | Val loss: {result[\"final_val_loss\"]:.6f}')
    else:
        print(f'  Failed: {result.get(\"message\", \"unknown error\")}')
print('\nGPU MAX Training complete!')
"
        ;;
    5)
        echo ""
        read -p "Choose preset (gpu / gpu_max) [gpu]: " PRESET
        PRESET=${PRESET:-gpu}
        echo ""
        echo "Starting GPU training for ALL 33 stocks (preset=$PRESET)..."
        echo ""
        cd "$PROJECT_DIR"
        python3 -c "
from ml.prediction_service import PredictionService
ps = PredictionService(provider='local')
results = ps.train_all_models_gpu(preset='$PRESET', days=730, lookback=60, epochs=300)
success = sum(1 for r in results.values() if r['status'] == 'success')
print(f'\nComplete: {success}/{len(results)} GPU models trained')
for sym, r in results.items():
    if r['status'] == 'success':
        m = r['metrics']
        print(f'  {sym}: DA={m[\"directional_accuracy\"]:.1%}, MAE={m[\"mae\"]:.4f}, params={r.get(\"total_params\",0):,}')
"
        ;;
    6)
        echo ""
        read -p "Choose preset (gpu / gpu_max) [gpu]: " PRESET
        PRESET=${PRESET:-gpu}
        read -p "Number of ensemble models [5]: " N_ENSEMBLE
        N_ENSEMBLE=${N_ENSEMBLE:-5}
        read -p "Stocks (comma-separated, or 'all') [AAPL,MSFT,NVDA]: " STOCKS
        STOCKS=${STOCKS:-AAPL,MSFT,NVDA}
        echo ""
        echo "Starting GPU Ensemble: ${N_ENSEMBLE}x ${PRESET} models per stock..."
        echo "This is the most accurate but slowest option."
        echo ""
        cd "$PROJECT_DIR"
        python3 -c "
from ml.prediction_service import PredictionService
from data.stock_api import get_all_symbols
ps = PredictionService(provider='local')
stocks_input = '$STOCKS'
if stocks_input.lower() == 'all':
    symbols = get_all_symbols()
else:
    symbols = [s.strip().upper() for s in stocks_input.split(',')]
for i, sym in enumerate(symbols):
    print(f'\n[{i+1}/{len(symbols)}] GPU Ensemble Training {sym} (${N_ENSEMBLE} models)...')
    result = ps.train_model_gpu(
        sym, preset='$PRESET', days=730, lookback=60, epochs=300,
        use_ensemble=True, n_ensemble=$N_ENSEMBLE,
    )
    if result['status'] == 'success':
        m = result['metrics']
        print(f'  DA: {m[\"directional_accuracy\"]:.1%} | MAE: {m[\"mae\"]:.4f}')
    else:
        print(f'  Failed: {result.get(\"message\", \"unknown error\")}')
print('\nGPU Ensemble Training complete!')
"
        ;;
    7)
        echo ""
        read -p "Enter stock symbol (e.g. AAPL): " SYMBOL
        SYMBOL=$(echo "$SYMBOL" | tr '[:lower:]' '[:upper:]')
        read -p "Choose preset (gpu / gpu_max) [gpu]: " PRESET
        PRESET=${PRESET:-gpu}
        echo ""
        echo "Running walk-forward validation for $SYMBOL (5 folds, $PRESET model)..."
        echo ""
        cd "$PROJECT_DIR"
        python3 -c "
from ml.prediction_service import PredictionService
ps = PredictionService(provider='local')
result = ps.walk_forward_validate('$SYMBOL', n_splits=5, days=730, epochs=100)
if result['status'] == 'success':
    agg = result['aggregate_metrics']
    print(f'\nWalk-Forward Results for $SYMBOL ({result[\"n_folds\"]} folds):')
    print(f'  Avg Directional Accuracy: {agg[\"avg_directional_accuracy\"]:.1%} (need >53%)')
    print(f'  Std Directional Accuracy: {agg[\"std_directional_accuracy\"]:.1%}')
    print(f'  Avg MAE: {agg[\"avg_mae\"]:.4f}')
    print(f'  Avg RMSE: {agg[\"avg_rmse\"]:.4f}')
    print(f'  Avg Info Coefficient: {agg[\"avg_ic\"]:.4f}')
    if agg['avg_directional_accuracy'] > 0.53:
        print('\n  PASS - Model shows edge over random.')
    else:
        print('\n  FAIL - Model does not beat random. Do not trade on this.')
else:
    print(f'Failed: {result.get(\"message\", \"unknown error\")}')
"
        ;;
    8)
        echo ""
        read -p "Enter stock symbol (e.g. AAPL): " SYMBOL
        SYMBOL=$(echo "$SYMBOL" | tr '[:lower:]' '[:upper:]')
        read -p "Choose preset (base / gpu / gpu_max) [gpu]: " PRESET
        PRESET=${PRESET:-gpu}
        echo ""
        echo "GPU Training $SYMBOL with preset=$PRESET..."
        echo ""
        cd "$PROJECT_DIR"
        python3 -c "
from ml.prediction_service import PredictionService
ps = PredictionService(provider='local')
result = ps.train_model_gpu('$SYMBOL', preset='$PRESET', days=730, lookback=60, epochs=300)
if result['status'] == 'success':
    m = result['metrics']
    print(f'\nTraining complete for $SYMBOL!')
    print(f'  Preset: {result[\"preset\"]}')
    print(f'  Params: {result.get(\"total_params\", 0):,}')
    print(f'  Samples: {result[\"training_samples\"]} train, {result[\"validation_samples\"]} val')
    print(f'  Epochs: {result[\"epochs_trained\"]}')
    print(f'  Train loss: {result[\"final_train_loss\"]:.6f}')
    print(f'  Val loss: {result[\"final_val_loss\"]:.6f}')
    print(f'  Directional accuracy: {m[\"directional_accuracy\"]:.1%}')
    print(f'  MAE: {m[\"mae\"]:.4f}')
    print(f'  Info Coefficient: {m[\"information_coefficient\"]:.4f}')
    print(f'  GPU: {result.get(\"gpu_info\", {}).get(\"gpu_name\", \"N/A\")}')
else:
    print(f'Failed: {result.get(\"message\", \"unknown error\")}')
"
        ;;
    0)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Done!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Next steps:"
echo "  1. Run the dashboard: streamlit run app.py"
echo "  2. Check ML Predictions tab for real predictions"
echo "  3. Run walk-forward validation (option 7) to prove edge"
echo "  4. Paper trade for 6 months before using real money"
echo ""
