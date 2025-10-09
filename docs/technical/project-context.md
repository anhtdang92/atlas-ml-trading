# Project Context: Crypto ML Trading Dashboard

**Last Updated**: October 7, 2025  
**Project Status**: Production-Ready with Real ML Predictions  
**Current Phase**: Live Google Cloud ML Trading System

---

## 📋 Project Summary

A production-ready cryptocurrency trading system that leverages Google Cloud's Vertex AI for real ML predictions and automatically rebalances portfolios via Kraken API. The system features a Streamlit dashboard with live cloud integration, requiring no local ML dependencies - everything runs in Google Cloud with real-time predictions and cost optimization.

**Primary Goal**: Create a profitable, automated crypto trading system while learning ML, cloud deployment, and quantitative finance.

---

## 🏗️ Architecture Overview

### System Components

```
┌─────────────────┐
│   Streamlit     │ ← User Interface Layer
│   Dashboard     │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  Application    │ ← Business Logic Layer
│  Logic Layer    │
├─────────────────┤
│ - Portfolio Mgr │
│ - Rebalancer    │
│ - ML Predictor  │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  Data & API     │ ← External Integration Layer
│     Layer       │
├─────────────────┤
│ - Kraken API    │
│ - BigQuery      │
│ - Cloud Storage │
└─────────────────┘
```

### Cloud-First Architecture (GCP)

- **Streamlit Dashboard**: Local app with cloud integration (no ML dependencies)
- **Vertex AI**: All ML training and predictions (pre-built containers)
- **BigQuery**: Data warehouse with partitioned tables for cost optimization
- **Cloud Storage**: Model artifacts with lifecycle policies
- **Cloud Functions**: Automated trading execution
- **Secret Manager**: Secure API credential storage
- **IAM**: Minimal-permission service accounts
- **Cost Optimization**: Preemptible instances, auto-scaling, lifecycle policies

---

## 🎯 Core Features & Design Decisions

### 1. Data Collection Strategy

**Decision**: Use Kraken API as primary data source, CoinAPI as backup

**Rationale**:
- Kraken API is free and provides OHLCV data
- Direct integration with trading platform ensures data consistency
- CoinAPI can fill historical gaps if needed

**Implementation**:
- Fetch daily OHLCV data for 4-12 cryptocurrencies
- Store in BigQuery partitioned by date for cost efficiency
- Run daily Cloud Scheduler job at 1 AM CDT to fetch previous day's data

**Schema** (`crypto_data.historical_prices`):
```
- timestamp: TIMESTAMP (partition key)
- symbol: STRING
- open: FLOAT64
- high: FLOAT64
- low: FLOAT64
- close: FLOAT64
- volume: FLOAT64
- data_source: STRING
- created_at: TIMESTAMP
```

### 2. Cloud-First ML Strategy

**Decision**: All ML operations run in Google Cloud using pre-built containers

**Rationale**:
- No local TensorFlow installation required
- Google's pre-built containers handle all dependencies
- Automatic scaling and cost optimization
- Professional-grade ML infrastructure

**Training Strategy**:
- Vertex AI Custom Jobs with pre-built TensorFlow containers
- Preemptible instances for 60-80% cost savings
- Automatic hyperparameter tuning and model selection
- Rolling retraining with BigQuery data integration

**Model Deployment**:
- Vertex AI Endpoints with auto-scaling (scale to zero)
- Model versioning in Vertex AI Model Registry
- A/B testing and gradual rollout capabilities
- Integrated with BigQuery for prediction logging

### 3. Portfolio Rebalancing Logic

**Decision**: ML-enhanced equal-weight with risk constraints

**Base Strategy**:
- Start with equal allocation (25% each for 4 coins)
- Adjust weights based on ML predicted returns
- Higher predicted return → increase allocation (up to 40%)
- Lower/negative predicted return → decrease allocation (down to 10%)

**Risk Controls**:
- Max position size: 40% per cryptocurrency
- Min position size: 10% (maintain diversification)
- Min trade size: $50 (avoid excessive fee drag)
- Max single trade: $2,000 (reduce slippage risk)

**Rebalancing Frequency**:
- **When**: Every Sunday 10 PM CDT
- **Why**: Low trading volume = less slippage, pre-week positioning

**Fee Consideration**:
- Kraken fees: ~0.16% maker, ~0.26% taker
- Only rebalance if expected gain > 2% (covers fees + buffer)

### 4. Kraken API Integration

**Library**: `python-kraken-sdk` v2.3.0+

**Authentication**:
- API keys stored in Secret Manager
- Keys have permissions: Query, Trade (not Withdraw)
- Rotate keys every 90 days

**Rate Limits**:
- 15 requests per minute (API tier 2 assumed)
- Implement exponential backoff on 429 errors
- Queue orders if multiple symbols need rebalancing

**Order Types**:
- Use market orders for simplicity and guaranteed execution
- Future: Implement limit orders for lower fees

### 5. Streamlit Dashboard Design

**Page Structure**:
```python
# app.py main structure
def main():
    st.set_page_config(page_title="Crypto ML Dashboard", layout="wide")
    
    # Sidebar for navigation
    page = st.sidebar.selectbox("View", ["Portfolio", "Predictions", "Rebalancing"])
    
    if page == "Portfolio":
        show_portfolio_view()
    elif page == "Predictions":
        show_predictions_view()
    elif page == "Rebalancing":
        show_rebalancing_view()
```

**Caching Strategy**:
- Cache BigQuery queries for 5 minutes
- Cache model predictions for 1 day (update after weekly run)
- Cache portfolio data for 1 minute

**UI/UX Principles**:
- Mobile-responsive (single column on small screens)
- Dark mode support
- Real-time price updates with WebSocket (future)
- Export data as CSV for external analysis

### 6. Local Backtesting Engine

**Purpose**: Validate strategy on real Kraken historical data

**Backtest Results (90-day baseline)**:
- Period: July 2025 - October 2025
- Initial capital: $5,000
- Final value: $6,450.76
- **Total Return: +29.02%**
- **Sharpe Ratio: 3.35** ✅
- **Max Drawdown: 5.82%** ✅
- Total Trades: 23
- Rebalances: 13 (weekly)
- Fees: $19.25 (0.38%)

**Key Metrics**:
- Equal-weight baseline established
- All risk metrics exceeded targets
- Low volatility despite crypto market
- Realistic Kraken fees included (0.16% maker, 0.26% taker)

**Implementation** (`run_backtest.py`):
1. Fetch historical data from Kraken API
2. Simulate weekly rebalancing
3. Track portfolio value and trades
4. Calculate performance metrics
5. Establish baseline for ML to beat

### 7. Cloud Automation Workflow

**Weekly Rebalancing Cloud Function**:
```
Sunday 10 PM CDT → Cloud Scheduler triggers Cloud Function

Cloud Function steps:
1. Fetch latest data from BigQuery
2. Load trained models from Cloud Storage
3. Generate predictions for each symbol
4. Calculate target portfolio weights
5. Determine buy/sell orders
6. Execute trades via Kraken API (if confidence > threshold)
7. Log trades to BigQuery
8. Send notification email/Slack
```

**Notification Content**:
- Executed trades summary
- New portfolio allocation
- Model confidence scores
- Week-over-week performance

---

## ☁️ Cloud-First ML Infrastructure (PRODUCTION READY)

### Key Innovation: No Local ML Dependencies

**Revolutionary Approach**: This project proves that you don't need TensorFlow, PyTorch, or any heavy ML libraries installed locally. Everything runs in Google Cloud using pre-built containers.

**Benefits**:
- ✅ **Zero local setup** - No dependency hell
- ✅ **Cost efficient** - Pay only for what you use
- ✅ **Always up-to-date** - Google maintains the containers
- ✅ **Scalable** - Automatic scaling based on demand
- ✅ **Professional grade** - Enterprise ML infrastructure

### Vertex AI Integration

**Decision**: Use Vertex AI for scalable ML training and prediction serving

**Implementation**:
- **Training Jobs**: Custom Docker containers with LSTM training
- **Model Registry**: Versioned model storage and management
- **Prediction Endpoints**: Auto-scaling endpoints for real-time predictions
- **Cost Optimization**: Preemptible instances, scale-to-zero, lifecycle policies

**Benefits**:
- Production-grade ML infrastructure
- Automatic scaling and load balancing
- Integrated with BigQuery and Cloud Storage
- Cost-effective with proper optimization

### BigQuery Data Warehouse

**Schema**: 6 partitioned tables for comprehensive data management
- `historical_prices`: OHLCV data from Kraken API
- `predictions`: ML model predictions with confidence scores
- `trades`: Complete trading history and execution logs
- `model_metrics`: Training performance and validation metrics
- `portfolio_snapshots`: Portfolio state over time
- `rebalancing_events`: Rebalancing decisions and outcomes

### Cloud Storage Architecture

**Buckets**:
- `crypto-ml-trading-487-models`: Model artifacts (1-year retention)
- `crypto-ml-trading-487-training-data`: Training data (30-day retention)
- `crypto-ml-trading-487-backups`: System backups (90-day retention)

**Lifecycle Policies**: Automatic cost optimization through data archiving and deletion

### Security & IAM

**Service Accounts**:
- `ml-training-sa`: Vertex AI training jobs (minimal permissions)
- `ml-prediction-sa`: Vertex AI predictions (read-only access)
- `crypto-app-sa`: Streamlit app access (BigQuery read, Storage read, Secret Manager)

**Security Features**:
- Least-privilege IAM roles
- Encrypted storage and data transfer
- Service account key rotation (90-day cycle)
- Secret Manager for API keys

### Cost Management

**Target**: $18-37/month (3-4 months on $50 budget)

**Optimizations**:
- Preemptible instances for training (60-80% savings)
- Auto-scaling endpoints (scale to zero)
- Partitioned BigQuery tables
- Lifecycle policies for storage
- Billing alerts at $10, $25, $50

---

## 🔧 Technical Decisions Log

### Python Version: 3.9+
- **Why**: TensorFlow 2.x requires 3.9+, GCP Cloud Run supports it

### Streamlit vs. Flask/React
- **Chosen**: Streamlit
- **Why**: Rapid prototyping, built-in data viz, Python-native (no JS needed)
- **Trade-off**: Less customizable UI, but sufficient for MVP

### BigQuery vs. PostgreSQL
- **Chosen**: BigQuery
- **Why**: Serverless, scales automatically, integrates with GCP ecosystem
- **Trade-off**: More expensive for small datasets, but free tier covers MVP usage

### Local ML vs. Cloud-First Approach
- **Chosen**: Cloud-First (Google Cloud Vertex AI)
- **Why**: No local dependencies, professional infrastructure, cost-effective
- **Trade-off**: Requires internet connection, but eliminates dependency management

### Custom Backtesting vs. QuantConnect
- **Chosen**: Custom Python engine
- **Why**: Full control, integrates with our stack, uses real Kraken data, free
- **Trade-off**: Less features than pro platform, but we built exactly what we need
- **Benefit**: Production-ready code, no external dependencies

### Market Orders vs. Limit Orders
- **Chosen**: Market orders (for now)
- **Why**: Guaranteed execution, simpler implementation
- **Trade-off**: Higher fees, but acceptable for weekly rebalancing
- **Future**: Add limit orders to reduce costs

---

## 📊 Data Flow Diagrams

### Training Pipeline
```
Kraken API → BigQuery → Vertex AI Training Job → Cloud Storage (model)
                          ↓
                    BigQuery (metrics log)
```

### Prediction Pipeline
```
Cloud Storage (model) → Prediction Script → BigQuery (predictions table)
                        ↑
                    BigQuery (recent data)
```

### Trading Pipeline
```
BigQuery (predictions) → Rebalancer → Kraken API (orders)
            ↑                             ↓
    BigQuery (portfolio)          BigQuery (trade log)
```

---

## 🧪 Testing Strategy

### Unit Tests
- `tests/test_data_fetcher.py`: Mock Kraken API responses
- `tests/test_lstm.py`: Test model training and prediction functions
- `tests/test_rebalancer.py`: Test portfolio calculations (critical!)
- `tests/test_kraken_client.py`: Mock API calls

### Integration Tests
- End-to-end paper trading simulation
- BigQuery read/write operations
- Cloud Storage model save/load

### Manual Testing Checklist
- [ ] Dashboard loads without errors
- [ ] Prices update correctly
- [ ] Model predictions display
- [ ] Rebalancing calculations are accurate (verify manually)
- [ ] Paper trades log correctly
- [ ] Notifications send successfully

---

## 🚀 Deployment Strategy

### Development Environment
- Local Streamlit for rapid iteration
- Local Python virtual environment
- Mock Kraken API responses

### Staging Environment (Paper Trading)
- Deploy to Cloud Run with `--paper-trading` flag
- Use separate BigQuery dataset (`crypto_staging`)
- Monitor for 2 weeks before live deployment

### Production Environment
- Deploy to Cloud Run with auto-scaling
- Enable live trading after successful staging validation
- Set up monitoring dashboards in GCP
- Configure billing alerts ($10, $25, $50 thresholds)

---

## 💰 Cost Estimates

**Target**: Run for 3-4 months on $50 GCP credit

### Monthly Cost Breakdown (Estimated)
- Cloud Run: $0 (within free tier: 2M requests/month)
- BigQuery: $5 (storage + queries)
- Cloud Storage: $1 (model files)
- Vertex AI: $10 (weekly training runs)
- Cloud Scheduler: $0 (within free tier)
- Cloud Functions: $0 (within free tier)
- **Total**: ~$16/month → $50 covers ~3 months ✓

### Cost Optimization Tips
- Use preemptible instances for Vertex AI training
- Partition BigQuery tables by date
- Delete old model versions after 1 month
- Limit query result caching

---

## 🔐 Security Considerations

### API Keys
- **Storage**: Google Secret Manager
- **Access**: Cloud Function service account only
- **Rotation**: Every 90 days
- **Permissions**: Query + Trade (no Withdraw)

### Database
- Use parameterized queries exclusively
- No user input directly in SQL
- Encrypt sensitive data (API keys) at rest

### Application
- No sensitive data in logs
- Mask API keys in error messages
- HTTPS only for Streamlit dashboard

---

## 📚 Learning Resources Used

### Machine Learning
- [TensorFlow Time Series Tutorial](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [LSTM for Stock Prediction (Medium)](https://medium.com/tag/lstm-stock-prediction)

### Trading & Data
- [Kraken API Docs](https://docs.kraken.com/rest/)

### Cloud
- [GCP Cloud Run Quickstart](https://cloud.google.com/run/docs/quickstarts)
- [Vertex AI Training Guide](https://cloud.google.com/vertex-ai/docs/training/overview)

---

## 🐛 Known Issues & TODOs

### Current Issues
- [ ] None critical - Phase 1 working smoothly!

### Completed Recently
- [x] Kraken API authentication
- [x] Real portfolio integration
- [x] Staking/bonded assets tracking
- [x] Improved KPI visualization
- [x] Manual refresh functionality
- [x] Separated liquid vs staked holdings
- [x] Custom backtesting engine
- [x] Baseline strategy validated (29% return)

### Future Enhancements
- [ ] Historical staking rewards tracking
- [ ] Staking APY calculations
- [ ] Add more technical indicators (MACD, Bollinger Bands)
- [ ] Implement sentiment analysis from Twitter/Reddit
- [ ] Multi-model ensemble (LSTM + XGBoost)
- [ ] WebSocket real-time price updates
- [ ] Mobile-responsive dashboard improvements
- [ ] Stop-loss and take-profit automation
- [ ] Tax reporting CSV export
- [ ] Alert system for price movements

---

## 📝 Developer Notes

**Developer Background**:
- 4 years SQL experience (T-SQL, Oracle)
- Strong Power BI/data visualization skills
- Python proficiency (data analysis, ML coursework from Georgia Tech)
- New to GCP but familiar with cloud concepts
- Located in Spring, TX (Kraken supports US trading)

**Project Goals**:
1. **Learning**: Gain hands-on experience with ML deployment, cloud infrastructure, and algorithmic trading
2. **Financial**: Generate supplemental income through automated trading (aspirational)
3. **Portfolio**: Build impressive project for resume/GitHub

**Time Commitment**:
- Development: 10-15 hours/week
- Monitoring (once live): 2-3 hours/week

---

## 📅 Development Timeline

### Phase 1: Data Pipeline (Weeks 1-2)
- Set up GCP project and billing
- Implement Kraken API data fetcher
- Create BigQuery tables and schemas
- Test daily data collection

### Phase 2: ML Foundation (Weeks 3-4)
- Build LSTM model architecture
- Implement feature engineering
- Train initial models locally
- Evaluate model performance

### Phase 3: Dashboard Core (Weeks 5-6)
- Create basic Streamlit app structure
- Implement portfolio view with mock data
- Add price charts and visualizations

### Phase 4: Predictions Integration (Week 7)
- Connect ML predictions to dashboard
- Display confidence intervals
- Show model performance metrics

### Phase 5: Rebalancing Logic (Weeks 8-9)
- Implement portfolio calculation engine
- Build trade recommendation system
- Test extensively with paper trading

### Phase 6: Extended Backtesting (Week 10)
- ✅ Built custom Python backtesting engine
- ✅ Established baseline: 29% return, 3.35 Sharpe
- [ ] Extend to 1+ year historical data
- [ ] Test ML predictions vs baseline

### Phase 7: Cloud Deployment (Weeks 11-12)
- Containerize application
- Deploy to Cloud Run
- Set up Cloud Scheduler and Functions

### Phase 8: Live Trading (Weeks 13-14)
- Enable live trading after validation
- Monitor performance closely
- Iterate based on results

**Total Estimated Time**: 14 weeks (~3.5 months)

---

## 🎓 Lessons Learned

### Phase 1 & Portfolio Integration:

**Technical Learnings:**
1. **Kraken API Authentication**: HMAC-SHA512 signature generation is tricky but well documented
2. **Staking Assets**: Kraken uses suffixes (.B, .F, .S) to indicate bonded/futures positions
3. **Streamlit Caching**: `@st.cache_data` is crucial for performance with API calls
4. **Asset Mapping**: Need to handle Kraken's naming (XXBT vs XBT vs BT.B all = BTC)

**UI/UX Learnings:**
1. **Large KPIs Work Better**: Users prefer big, colorful cards over small metrics
2. **Separation is Clarity**: Showing liquid vs staked separately reduces confusion
3. **Manual Refresh**: Users want control over when data updates (vs auto-refresh)
4. **Progressive Disclosure**: Expandable sections for details (like "What is Staking?") work well

**Security Learnings:**
1. **Never Commit Secrets**: .gitignore is critical - always check before commits
2. **Read-Only First**: Start with query-only API permissions, add trading later
3. **API Key Rotation**: Plan to rotate keys every 90 days for security

**Development Learnings:**
1. **Test with Real Data Early**: Mock data hides real-world edge cases
2. **Debug Views Help**: Adding expandable debug sections helped troubleshooting
3. **Incremental Features**: Building in small steps made debugging easier

---

## 📞 Questions to Resolve

- [ ] Should we use daily or hourly data for training? (Daily chosen for now)
- [ ] What confidence threshold for auto-executing trades? (TBD after backtesting)
- [ ] Should we implement stop-losses? (Phase 2 consideration)

---

**This document should be updated after each major milestone or architectural decision.**

