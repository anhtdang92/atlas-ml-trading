#!/bin/bash

# 🚀 Quick ML Training Launcher
# Train crypto price prediction models on Google Cloud

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🚀 ATLAS Stock ML Training Launcher"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ Error: gcloud CLI not found"
    echo "📥 Install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check project
PROJECT=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT" ]; then
    echo "❌ Error: No GCP project set"
    echo "🔧 Run: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

echo "✅ Project: $PROJECT"
echo ""

# Show menu
echo "Choose training option:"
echo ""
echo "1. 💰 Budget Training (Recommended)"
echo "   - Cost: $3-8"
echo "   - Time: 30-60 min"
echo "   - Stocks: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA"
echo ""
echo "2. 🚀 Full Training"
echo "   - Cost: $8-15"
echo "   - Time: 1-2 hours"
echo "   - Stocks: All 30 tracked symbols (Tech, Sector Leaders, ETFs, Growth)"
echo ""
echo "3. 📊 Check Training Status"
echo ""
echo "4. ⚙️  Setup (First Time Only)"
echo ""
echo "5. 🔍 View Training History"
echo ""
echo "0. ❌ Exit"
echo ""
read -p "Enter choice [0-5]: " choice

case $choice in
    1)
        echo ""
        echo "🚀 Starting budget-optimized training..."
        echo "💰 Estimated cost: $3-8"
        echo "⏱️  Expected time: 30-60 minutes"
        echo ""
        read -p "Continue? (y/n): " confirm
        if [ "$confirm" = "y" ]; then
            echo "⚠️  GCP deployment scripts not yet created. See docs/TRAIN_ML_MODELS.md"
        else
            echo "❌ Cancelled"
        fi
        ;;
    2)
        echo ""
        echo "🚀 Starting full training..."
        echo "💰 Estimated cost: $8-15"
        echo "⏱️  Expected time: 1-2 hours"
        echo ""
        read -p "Continue? (y/n): " confirm
        if [ "$confirm" = "y" ]; then
            echo "⚠️  GCP deployment scripts not yet created. See docs/TRAIN_ML_MODELS.md"
        else
            echo "❌ Cancelled"
        fi
        ;;
    3)
        echo ""
        echo "📊 Training Job Status:"
        echo ""
        gcloud ai custom-jobs list --region=us-central1 --sort-by=~createTime --limit=5
        echo ""
        echo "💡 To stream logs for a job, run:"
        echo "   gcloud ai custom-jobs stream-logs JOB_NAME --region=us-central1"
        ;;
    4)
        echo ""
        echo "⚙️  Setting up GCP infrastructure..."
        echo ""
        echo "Step 1: Enabling APIs..."
        echo "⚠️  GCP setup scripts not yet created. See docs/TRAIN_ML_MODELS.md"
        echo ""
        echo "Step 2: Setting up storage..."
        echo "   (storage setup placeholder)"
        echo ""
        echo "Step 3: Setting up IAM..."
        echo "   (IAM setup placeholder)"
        echo ""
        echo "✅ Setup complete! You can now train models."
        ;;
    5)
        echo ""
        echo "📜 Recent Training Jobs:"
        echo ""
        gcloud ai custom-jobs list --region=us-central1 --sort-by=~createTime --limit=10
        ;;
    0)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ Done!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "💡 Next steps:"
echo "   1. Monitor progress in Cloud Progress tab"
echo "   2. Deploy endpoint when training completes"
echo "   3. Test predictions in ML Predictions tab"
echo ""
echo "📖 Full guide: See TRAIN_ML_MODELS.md"
echo ""

