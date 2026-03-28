"""
GPU Training Script for ATLAS
Run inside WSL2 with RTX 4090 for full GPU acceleration.

Usage:
    python train_gpu.py                    # Train 7 tech stocks with 'gpu' preset
    python train_gpu.py --preset gpu_max   # Train with GPU MAX preset (~5M params)
    python train_gpu.py --preset transformer  # Train with Transformer preset
    python train_gpu.py --all              # Train all 33 stocks
    python train_gpu.py --symbol NVDA      # Train single stock
    python train_gpu.py --validate AAPL    # Walk-forward validation
"""
import argparse
import time
import sys

from ml.prediction_service import PredictionService
from data.stock_api import get_all_symbols

TECH_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]

def train_stocks(symbols, preset, days=730, lookback=60, epochs=300):
    ps = PredictionService(provider="local")
    results = {}
    total = len(symbols)

    for i, sym in enumerate(symbols):
        print(f"\n{'='*60}")
        print(f"  [{i+1}/{total}] Training {sym} | preset={preset}")
        print(f"{'='*60}")
        start = time.time()

        try:
            if preset == "base":
                result = ps.train_model(sym, days=days, epochs=min(epochs, 150))
            else:
                result = ps.train_model_gpu(
                    sym, preset=preset, days=days,
                    lookback=lookback, epochs=epochs
                )

            elapsed = time.time() - start

            if result["status"] == "success":
                m = result.get("metrics", {})
                print(f"  Status: SUCCESS ({elapsed:.1f}s)")
                print(f"  Params: {result.get('total_params', 'N/A'):,}")
                print(f"  Epochs: {result['epochs_trained']}")
                print(f"  Train loss: {result['final_train_loss']:.6f}")
                print(f"  Val loss:   {result['final_val_loss']:.6f}")
                print(f"  Directional Accuracy: {m.get('directional_accuracy', 0):.1%}")
                print(f"  MAE: {m.get('mae', 0):.4f}")
                print(f"  Info Coefficient: {m.get('information_coefficient', 0):.4f}")
                results[sym] = "OK"
            else:
                print(f"  Status: FAILED ({elapsed:.1f}s)")
                print(f"  Reason: {result.get('message', 'unknown')}")
                results[sym] = "FAIL"
        except Exception as e:
            print(f"  Status: ERROR - {e}")
            results[sym] = "ERROR"

    print(f"\n{'='*60}")
    print(f"  TRAINING SUMMARY")
    print(f"{'='*60}")
    ok = sum(1 for v in results.values() if v == "OK")
    print(f"  {ok}/{len(results)} models trained successfully")
    for sym, status in results.items():
        icon = "+" if status == "OK" else "X"
        print(f"  [{icon}] {sym}: {status}")
    return results


def validate_stock(symbol, preset="gpu"):
    ps = PredictionService(provider="local")
    print(f"\nWalk-Forward Validation for {symbol} (5 folds, preset={preset})")
    print("This tests if the model generalizes across different time periods.\n")

    result = ps.walk_forward_validate(symbol, n_splits=5, days=730, epochs=100)

    if result["status"] == "success":
        agg = result["aggregate_metrics"]
        da = agg["avg_directional_accuracy"]
        print(f"Results ({result['n_folds']} folds):")
        print(f"  Avg Directional Accuracy: {da:.1%}")
        print(f"  Std Directional Accuracy: {agg['std_directional_accuracy']:.1%}")
        print(f"  Avg MAE:  {agg['avg_mae']:.4f}")
        print(f"  Avg RMSE: {agg['avg_rmse']:.4f}")
        print(f"  Avg IC:   {agg['avg_ic']:.4f}")
        print()
        if da > 0.53:
            print("  PASS: Model shows statistical edge (DA > 53%)")
            print("  Consider paper trading this stock.")
        else:
            print("  FAIL: No edge detected (DA <= 53%)")
            print("  Do NOT trade on this model's predictions.")
    else:
        print(f"  Failed: {result.get('message', 'unknown')}")


def main():
    parser = argparse.ArgumentParser(description="ATLAS GPU Training")
    parser.add_argument("--preset", default="gpu",
                        choices=["base", "gpu", "gpu_max", "transformer"])
    parser.add_argument("--all", action="store_true", help="Train all 33 stocks")
    parser.add_argument("--symbol", type=str, help="Train single stock")
    parser.add_argument("--validate", type=str, help="Walk-forward validate a stock")
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lookback", type=int, default=60)
    args = parser.parse_args()

    if args.validate:
        validate_stock(args.validate.upper(), args.preset)
    elif args.symbol:
        train_stocks([args.symbol.upper()], args.preset,
                     args.days, args.lookback, args.epochs)
    elif args.all:
        train_stocks(get_all_symbols(), args.preset,
                     args.days, args.lookback, args.epochs)
    else:
        train_stocks(TECH_SYMBOLS, args.preset,
                     args.days, args.lookback, args.epochs)


if __name__ == "__main__":
    main()
