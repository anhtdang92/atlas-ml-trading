#!/bin/bash
# Quick Start Script for ATLAS Stock ML Trading Dashboard

echo "🚀 Starting ATLAS Stock ML Trading Dashboard..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Start the application
echo "🌟 Launching Streamlit dashboard..."
streamlit run app.py

echo "✅ Dashboard started! Open http://localhost:8501 in your browser."
