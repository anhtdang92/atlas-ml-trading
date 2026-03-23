# Changelog

All notable changes to the Stock ML Trading Dashboard project will be documented in this file.

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

### Added - Portfolio Integration & Real-Time Data
- **Yahoo Finance Integration**: Full stock data support via yfinance (free, no API key needed)
- **Real Portfolio Integration**: Connect to actual brokerage account and view holdings
- **Stock Categories Dashboard**: Separate sections for Tech, Sector Leaders, ETFs, Growth stocks
  - Tracks AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA and more
  - Shows current market value of all positions
  - Sector allocation visualization
- **Enhanced KPI Cards**: Large, colorful cards showing:
  - Total portfolio value
  - Total P&L
  - Day change (NEW)
  - Total asset count
- **Manual Refresh**: Button to refresh data on demand
- **Stock Universe**: ~30 stocks across Tech, Sector Leaders, ETFs, and Growth categories
- **Authentication Test Script**: `test_auth.py` to verify API keys
- **Stock Data Client**: `data/stock_api.py` with yfinance integration

### Changed
- **Portfolio View**: Now shows real data instead of mock data
- **Improved UI**: Better color coding, larger text, more professional styling
- **Separated Holdings**: Different categories shown in organized tables
- **Better Error Messages**: More helpful feedback when API calls fail

### Fixed
- API static method call issues
- Portfolio data not displaying correctly
- KPI readability issues
- Asset name mapping for stock ticker formats

### Security
- API keys stored in `config/secrets.yaml` (gitignored)
- Never expose keys in logs or error messages
- Read-only API permissions documented

## [0.1.0] - 2025-10-04

### Added - Initial Release
- **Streamlit Dashboard**: Multi-page web application
  - Portfolio view
  - Live prices view
  - Predictions view (placeholder)
  - Rebalancing view (placeholder)
- **Yahoo Finance Integration**: Real-time stock data via yfinance
- **Interactive Charts**: Candlestick charts with Plotly
  - Multiple time intervals (1min to 1day)
  - Volume visualization
- **Live Price Tracking**: Real-time data for ~30 stocks
- **HTML Dashboard**: Standalone alternative with vanilla JavaScript
- **API Testing Suite**: `test_stock_api.py` for connectivity testing
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

- **v0.2.0** (Oct 4, 2025): Portfolio Integration & Stock Data Support
- **v0.1.0** (Oct 4, 2025): Initial Dashboard & Yahoo Finance Integration

---

## Upcoming Features

### Phase 2 - ML Model (Next)
- Historical data collection from Yahoo Finance
- LSTM model architecture (2 layers, 64 units)
- Feature engineering (25 technical indicators: MAs, RSI, MACD, Bollinger Bands, volume, momentum, volatility, ATR)
- Model training and 21-day predictions
- Integration into dashboard

### Phase 3 - Advanced Features
- Portfolio performance tracking
- Sector allocation analysis
- Historical performance charts
- Trade history analysis

### Phase 4 - Trading Features
- Paper trading mode
- Rebalancing recommendations
- Risk analysis
- Trade execution (zero-commission)

### Phase 5 - Cloud Deployment
- Google Cloud Run deployment
- BigQuery data storage
- Vertex AI model training
- Cloud Scheduler automation
