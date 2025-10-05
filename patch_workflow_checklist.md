# Autonomous Patch Workflow Checklist

## Pre-Patch Setup
```bash
# Navigate to project root
cd "/Users/anhdang/Documents/GitHub/Kraken Crypto ML/Kraken_Cloud_ML_Strat"

# Activate virtual environment
source venv/bin/activate

# Verify Python environment
python --version  # Should show Python 3.9+
pip list | grep -E "(streamlit|tensorflow|pandas|pytest)"
```

## Apply Patches
```bash
# Apply unified diff patch
git apply --check patch.diff  # Dry run first
git apply patch.diff          # Apply patch

# Or apply specific file changes
patch -p1 < patch.diff
```

## Run Tests
```bash
# Run all tests
pytest tests/ -v --tb=short

# Run specific test file
pytest tests/test_lstm.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run linting
flake8 . --max-line-length=100 --ignore=E203,W503

# Run Streamlit dashboard (manual test)
streamlit run app.py
```

## Validation Steps
```bash
# Check for syntax errors
python -m py_compile app.py
python -m py_compile ml/lstm_model.py
python -m py_compile data/kraken_api.py

# Verify imports work
python -c "import streamlit, tensorflow, pandas, numpy; print('All imports successful')"

# Test Kraken API connectivity (if keys configured)
python test_auth.py

# Run backtest to verify no regressions
python run_backtest.py
```

## Post-Patch Cleanup
```bash
# Check git status
git status

# Review changes
git diff --cached

# Commit if tests pass
git add .
git commit -m "[PATCH] Brief description of changes"

# Update documentation if needed
echo "Updated Sat Oct  4 20:42:52 CDT 2025" >> workflow_state.md
```

## Emergency Rollback
```bash
# Revert last commit
git reset --hard HEAD~1

# Revert specific file
git checkout HEAD -- app.py

# Restore from backup
git stash
git checkout main
```

