# Autonomous Patch Workflow State

## Goal
Implement an autonomous patch workflow system for the Crypto ML Trading Dashboard that applies targeted code changes using unified diffs, maintains test coverage, and follows project conventions without whole-file rewrites.

## Context
**Project**: Crypto ML Trading Dashboard - A machine learning-powered cryptocurrency trading system using LSTM neural networks for portfolio rebalancing via Kraken API.

**Current Status**: 
- Phase 1 Complete: Streamlit dashboard, Kraken API integration, real portfolio tracking
- Phase 2 In Progress: ML model development (LSTM architecture, feature engineering)
- Architecture: Streamlit UI → Business Logic → Data/API Layer (Kraken, BigQuery, GCP)
- Tech Stack: Python 3.9+, TensorFlow, Streamlit, Google Cloud Platform
- Testing: pytest framework, minimal existing test coverage
- Deployment: Local development + GCP Cloud Run ready

**Key Files**:
- `app.py`: Main Streamlit dashboard (1,100+ lines)
- `data/kraken_api.py`: Kraken API client with authentication
- `ml/`: LSTM model architecture and training
- `trading/`: Portfolio management and rebalancing logic
- `config/`: Configuration and secrets management
- `run_backtest.py`: Custom backtesting engine (29% baseline return)

## Rules
1. **Patch-Only Operations**: Use unified diff format for all code changes
2. **Minimal Edits**: Make smallest possible changes to achieve desired functionality
3. **Test-Driven**: Add/update unit tests for all modifications
4. **No Whole-File Rewrites**: Preserve existing code structure and formatting
5. **Style Compliance**: Follow PEP 8, project conventions, and existing patterns
6. **Documentation**: Update project-context.md for architectural changes
7. **Security**: Never commit secrets, use environment variables/Secret Manager
8. **Validation**: Run tests and linting before considering patches complete

## Plan
1. **Initialize Workflow**: Create documentation files and establish testing framework
2. **Enhance Testing**: Add comprehensive unit tests for core modules
3. **ML Development**: Implement LSTM model architecture and training pipeline
4. **Feature Engineering**: Add technical indicators and data preprocessing
5. **Integration**: Connect ML predictions to portfolio rebalancing logic
6. **Validation**: Test ML strategy against 29% baseline performance
7. **Deployment**: Prepare cloud deployment with automated testing

## Failing Tests
- [ ] No existing test suite - tests/ directory is empty
- [ ] Missing unit tests for critical functions:
  - Portfolio rebalancing calculations
  - ML model prediction functions
  - Kraken API integration methods
  - Data validation and preprocessing
- [ ] No integration tests for end-to-end workflows
- [ ] Missing test coverage for error handling and edge cases

## Next Action
1. Create comprehensive test suite for existing functionality
2. Implement LSTM model architecture in `ml/lstm_model.py`
3. Add feature engineering pipeline for technical indicators
4. Develop training script with proper model versioning
5. Integrate ML predictions with portfolio rebalancing logic

## Log
**2025-01-XX**: Workflow initialized
- Created `docs/brief.md` with requirements and acceptance criteria
- Created `workflow_state.md` with current project context
- Established patch-only operation rules
- Identified missing test coverage as primary risk
- Ready to begin ML model development phase
