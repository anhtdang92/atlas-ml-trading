# Project Context: Crypto ML Trading Dashboard

**Last Updated**: October 4, 2025  
**Project Status**: Phase 1 Complete + Portfolio Integration Complete  
**Current Phase**: Ready for Phase 2 - ML Model Development

---

## 📋 Project Summary

A machine learning-powered cryptocurrency trading system that uses LSTM neural networks to predict weekly price movements and automatically rebalance a portfolio via Kraken API. The system features a Streamlit dashboard for visualization, GCP-based cloud infrastructure, and QuantConnect backtesting integration.

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

### Cloud Architecture (GCP)

- **Cloud Run**: Hosts Streamlit application (containerized)
- **Vertex AI**: Trains LSTM models on schedule
- **BigQuery**: Stores historical data, trading logs, model metrics
- **Cloud Storage**: Stores trained models and backups
- **Cloud Scheduler**: Triggers weekly rebalancing function
- **Cloud Functions**: Executes automated trading logic
- **Secret Manager**: Securely stores Kraken API credentials

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

### 2. LSTM Model Architecture

**Decision**: 2-layer LSTM with 50 units each, dropout 0.2

**Rationale**:
- Sufficient complexity for time series patterns without overfitting
- 7-day input window balances recent trends vs. noise
- Dropout prevents overfitting on crypto's volatile data

**Training Strategy**:
- Rolling window: Use days 1-7 to predict day 8's return
- Features: Raw OHLCV + MA(7,14,30) + RSI + volume momentum
- Train/val split: 80/20 with temporal ordering preserved
- Retrain weekly to adapt to market regime changes

**Model Versioning**:
- Save to Cloud Storage: `gs://bucket/models/{symbol}/model_{timestamp}.h5`
- Track metadata in BigQuery: training date, data range, validation RMSE
- Keep last 4 versions per symbol for rollback capability

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

### 6. QuantConnect Backtesting

**Purpose**: Validate strategy on historical data before live deployment

**Backtest Configuration**:
- Time period: January 2022 - Present (~3 years)
- Initial capital: $5,000
- Brokerage: `KrakenBrokerageModel`
- Benchmark: Equal-weight buy-and-hold

**Key Metrics to Track**:
- Total Return
- Sharpe Ratio (target: >1.5)
- Max Drawdown (target: <30%)
- Win Rate
- Profit Factor
- Trade frequency

**Algorithm Logic** (`backtest_strategy.py`):
1. Load historical ML predictions (pre-generated)
2. Rebalance weekly based on predictions
3. Log all trades to local CSV
4. Compare against buy-and-hold baseline

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

### TensorFlow vs. PyTorch
- **Chosen**: TensorFlow/Keras
- **Why**: Better integration with Vertex AI, simpler API for LSTMs
- **Trade-off**: PyTorch more flexible for research, but not needed here

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

### Trading
- [Kraken API Docs](https://docs.kraken.com/rest/)
- [QuantConnect Docs](https://www.quantconnect.com/docs/v2)

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

### Phase 6: Backtesting (Week 10)
- Implement QuantConnect strategy
- Run backtests on historical data
- Analyze and optimize strategy

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

