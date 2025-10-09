#!/bin/bash
# Development Setup Script

echo "🛠️ Setting up development environment..."

# Install development dependencies
pip install -r requirements.txt
pip install pytest flake8 black

# Run tests
echo "🧪 Running tests..."
pytest tests/

# Run linting
echo "🔍 Running linting..."
flake8 .

# Format code
echo "✨ Formatting code..."
black .

echo "✅ Development environment ready!"
