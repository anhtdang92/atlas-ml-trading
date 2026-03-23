#!/bin/bash

# 📊 Check ML Training Progress

REGION="${GCP_REGION:-us-central1}"

# Get latest job ID dynamically
JOB_ID=$(gcloud ai custom-jobs list --region=$REGION --sort-by=~createTime --limit=1 --format="value(name)" 2>/dev/null)
if [ -z "$JOB_ID" ]; then
    echo "❌ No training jobs found. Run ./bin/train_now.sh first."
    exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  📊 ML Training Progress Checker"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Get job status
echo "🔍 Checking training status..."
echo ""

STATUS=$(gcloud ai custom-jobs describe $JOB_ID --format="value(state)")
CREATE_TIME=$(gcloud ai custom-jobs describe $JOB_ID --format="value(createTime)")

echo "Status: $STATUS"
echo "Started: $CREATE_TIME"
echo ""

case $STATUS in
    "JOB_STATE_PENDING")
        echo "⏳ Job is starting up (takes 2-5 minutes)"
        echo "   Container is being pulled and initialized..."
        echo ""
        echo "✅ What to do: Wait a few minutes, then run this script again"
        ;;
    "JOB_STATE_RUNNING")
        echo "🔄 Training is in progress!"
        echo ""
        echo "📋 Streaming recent logs..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        gcloud ai custom-jobs stream-logs $JOB_ID 2>&1 | head -20 || echo "Logs not available yet..."
        ;;
    "JOB_STATE_SUCCEEDED")
        echo "🎉 Training completed successfully!"
        echo ""
        echo "📦 Models have been trained and saved to Cloud Storage"
        echo ""
        echo "✅ Next step: Deploy the models to an endpoint"
        echo "   See docs/TRAIN_ML_MODELS.md for endpoint deployment steps"
        ;;
    "JOB_STATE_FAILED")
        echo "❌ Training failed"
        echo ""
        echo "📋 Error logs:"
        gcloud ai custom-jobs stream-logs $JOB_ID 2>&1 | tail -30
        ;;
    *)
        echo "❓ Unknown status: $STATUS"
        ;;
esac

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 Track in dashboard: http://localhost:8501 → ☁️ Cloud Progress"
echo "🌐 Web console: https://console.cloud.google.com/vertex-ai/training/custom-jobs"
echo ""
echo "🔄 Run this script again anytime: ./check_training.sh"
echo ""

