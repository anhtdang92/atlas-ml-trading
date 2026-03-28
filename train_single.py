"""Quick single-stock training script."""
import sys
from ml.prediction_service import PredictionService

symbol = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
ps = PredictionService(provider="local")

print(f"Training {symbol} (base LSTM, ~6K params, 730 days, 150 epochs)...")
print("Running on CPU — expect ~3-5 minutes\n")

result = ps.train_model(symbol, days=730, epochs=150)

if result["status"] == "success":
    print(f"\nTraining COMPLETE for {symbol}!")
    print(f"  Training samples: {result['training_samples']}")
    print(f"  Validation samples: {result['validation_samples']}")
    print(f"  Epochs trained: {result['epochs_trained']}")
    print(f"  Final val loss: {result['final_val_loss']:.6f}")
    print(f"  Final train loss: {result['final_train_loss']:.6f}")
    if "metrics" in result:
        m = result["metrics"]
        print(f"  Directional accuracy: {m.get('directional_accuracy', 0):.1%}")
        print(f"  MAE: {m.get('mae', 0):.4f}")
        print(f"  Info Coefficient: {m.get('information_coefficient', 0):.4f}")
else:
    print(f"Failed: {result.get('message', 'unknown error')}")
