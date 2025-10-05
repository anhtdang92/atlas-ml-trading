# Crypto ML Trading Dashboard - Requirements Brief

## Goal
Implement an autonomous patch workflow system that can apply targeted code changes to a cryptocurrency trading application using machine learning for portfolio rebalancing. The system should operate as a patch-only agent that makes minimal, precise edits using unified diffs without whole-file rewrites.

## Inputs/Constraints
- **Codebase**: Python-based crypto trading application with Streamlit dashboard, LSTM models, and Kraken API integration
- **Target**: Phase 1 complete (portfolio integration), Phase 2 in progress (ML model development)
- **Scope**: Patch-only modifications (unified diffs), test-driven development
- **Environment**: Local development with virtual environment, GCP cloud deployment ready
- **Budget**: $50 GCP credit for 3-4 months operation
- **Security**: API keys in Secret Manager, no secrets in code

## Target Behavior
- Apply patches using unified diff format with minimal line changes
- Add/update unit tests for all modifications
- Maintain existing functionality while implementing new features
- Follow PEP 8 style guide and project conventions
- Validate changes through automated testing before deployment
- Document all changes in project context and changelog

## Key APIs/Libraries
- **Trading**: python-kraken-sdk v2.3.0+ (https://docs.kraken.com/rest/)
- **ML**: TensorFlow/Keras 2.13.0+ (https://www.tensorflow.org/)
- **Cloud**: Google Cloud Platform (BigQuery, Cloud Run, Vertex AI)
- **UI**: Streamlit 1.28.0+ (https://docs.streamlit.io/)
- **Testing**: pytest 7.4.0+ (https://docs.pytest.org/)
- **Data**: pandas 2.0.0+, numpy 1.24.0+

## Acceptance Tests
- [ ] All existing tests pass after patch application
- [ ] New functionality has corresponding unit tests with >80% coverage
- [ ] Code passes flake8 linting without warnings
- [ ] Streamlit dashboard loads without errors
- [ ] Kraken API integration maintains authentication
- [ ] ML model training completes successfully
- [ ] Portfolio rebalancing calculations are accurate
- [ ] Paper trading mode executes without errors
- [ ] Cloud deployment remains functional
- [ ] No hardcoded secrets or API keys in code

## Risks
- **API Rate Limits**: Kraken API (15 req/min) may cause timeouts during testing
- **Cloud Costs**: GCP usage could exceed $50 budget if not monitored
- **Data Quality**: Missing or corrupted historical data could break ML training
- **Model Performance**: LSTM predictions may not beat 29% baseline return
- **Security**: API key exposure risk during development and deployment
- **Trading Risk**: Live trading could result in financial losses if not properly validated
- **Dependencies**: Version conflicts between ML libraries and cloud SDKs
- **Testing**: Limited test coverage in current codebase may hide regressions
