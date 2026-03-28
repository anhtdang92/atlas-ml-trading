"""Quick prediction + multi-stock training script."""
import sys
from ml.prediction_service import PredictionService

ps = PredictionService(provider="local")

# First, get prediction for any already-trained model
print("=" * 60)
print("  Getting NVDA prediction from trained model...")
print("=" * 60)
pred = ps.get_prediction("NVDA", allow_mock=False)
print(f"  Status: {pred['status']}")
print(f"  Predicted return: {pred.get('predicted_return', 'N/A')}")
print(f"  Confidence: {pred.get('confidence', 'N/A')}")
print(f"  Direction: {pred.get('direction', 'N/A')}")
print(f"  Source: {pred.get('prediction_source', 'N/A')}")
print()

# Now train more stocks
symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"]
for i, sym in enumerate(symbols):
    print(f"\n[{i+1}/{len(symbols)}] Training {sym}...")
    result = ps.train_model(sym, days=730, epochs=150)
    if result["status"] == "success":
        m = result.get("metrics", {})
        print(f"  OK: {result['epochs_trained']} epochs, "
              f"DA={m.get('directional_accuracy', 0):.1%}, "
              f"samples={result['training_samples']}")
    else:
        print(f"  FAIL: {result.get('message', 'unknown')}")

print("\n" + "=" * 60)
print("  Training complete! Run: streamlit run app.py")
print("=" * 60)
