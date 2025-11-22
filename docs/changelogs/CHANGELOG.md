# Changelog

All notable changes to the Crypto ML Trading Dashboard project will be documented in this file.

## [Unreleased]

### Phase 3 - Vertex AI Deployment (Optional)
- Deploy LSTM models to Vertex AI endpoints
- Production-scale prediction serving
- Automated model retraining

## [0.3.0] - 2025-01-15

### Added - GCP ML Infrastructure
- **Vertex AI Integration**: Production-ready ML training and prediction serving
- **BigQuery Data Warehouse**: 6 partitioned tables for comprehensive data management
- **Cloud Storage**: 3 buckets with lifecycle policies for cost optimization
- **IAM Security**: Three service accounts with minimal permissions
- **Cost Management**: Optimized for $50 budget over 3-4 months
- **PredictionService**: Updated to support both local and Vertex AI providers
- **Docker Support**: Containerized training jobs for Vertex AI
- **Environment Configuration**: `.env` file with GCP settings
- **Setup Scripts**: Automated infrastructure deployment

## [0.2.0] - 2025-10-04

### Added - Portfolio Integration & Staking
- **Kraken API Authentication**: Full private endpoint support with HMAC-SHA512 signatures
- **Real Portfolio Integration**: Connect to actual Kraken account and view holdings
- **Staking Dashboard**: Separate section for staked/bonded assets
  - Tracks BTC bonded (.B), ETH bonded (.B), SOL futures (.F), DOT futures (.F)
  - Shows current market value of staked positions
  - Educational "What is Staking?" section
- **Enhanced KPI Cards**: Large, colorful cards showing:
  - Total liquid value
  - Total P&L
  - Staked value (NEW)
  - Total asset count
- **Manual Refresh**: Button to refresh data from Kraken on demand
- **Asset Mapping**: Smart detection of Kraken asset suffixes (.B, .F, .S, .M, .L)
- **Authentication Test Script**: `test_auth.py` to verify API keys
- **Authenticated API Client**: `data/kraken_auth.py` with secure key handling

### Changed
- **Portfolio View**: Now shows real data from Kraken instead of mock data
- **Improved UI**: Better color coding, larger text, more professional styling
- **Separated Holdings**: Liquid vs staked assets shown in different tables
- **Better Error Messages**: More helpful feedback when API calls fail

### Fixed
- API static method call issues
- Portfolio data not displaying correctly
- KPI readability issues
- Asset name mapping for Kraken's special formats

### Security
- API keys stored in `config/secrets.yaml` (gitignored)
- Never expose keys in logs or error messages
- Read-only API permissions documented
- HMAC signature authentication implemented

## [0.1.0] - 2025-10-04

### Added - Initial Release
- **Streamlit Dashboard**: Multi-page web application
  - Portfolio view
  - Live prices view
  - Predictions view (placeholder)
  - Rebalancing view (placeholder)
- **Kraken API Integration**: Public endpoints for price data
- **Interactive Charts**: Candlestick charts with Plotly
  - Multiple time intervals (1min to 1day)
  - Volume visualization
- **Live Price Tracking**: Real-time data for 6+ cryptocurrencies
- **HTML Dashboard**: Standalone alternative with vanilla JavaScript
- **API Testing Suite**: `kraken_test.py` for connectivity testing
- **Project Documentation**:
  - Comprehensive README.md
  - project-context.md for architecture
  - .cursorrules for AI development guidelines
  - QUICKSTART.md for quick reference
- **Configuration**: YAML-based config system
- **Python Dependencies**: Complete requirements.txt

### Infrastructure
- Virtual environment setup
- Git repository initialization
- .gitignore for security
- Project folder structure

---

## Version History Summary

- **v0.2.0** (Oct 4, 2025): Portfolio Integration & Staking Support
- **v0.1.0** (Oct 4, 2025): Initial Dashboard & Public API Integration

---

## Upcoming Features

### Phase 2 - ML Model (Next)
- Historical data collection from Kraken
- LSTM model architecture (2 layers, 50 units)
- Feature engineering (MA, RSI, volume)
- Model training and predictions
- Integration into dashboard

### Phase 3 - Advanced Features
- Staking rewards tracking
- APY calculations
- Historical performance charts
- Trade history analysis

### Phase 4 - Trading Features
- Paper trading mode
- Rebalancing recommendations
- Risk analysis
- Trade execution

### Phase 5 - Cloud Deployment
- Google Cloud Run deployment
- BigQuery data storage
- Vertex AI model training
- Cloud Scheduler automation

