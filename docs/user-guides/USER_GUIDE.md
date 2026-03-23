# User Guide
## ATLAS - Stock ML Intelligence System

**Welcome to your advanced stock trading dashboard!**

This guide will help you navigate and use all the features of your ML-powered trading system.

---

## **Getting Started**

### **Accessing the Dashboard**
1. **Start the application:**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser:**
   - Navigate to `http://localhost:8501`
   - The dashboard will load automatically

3. **First-time setup:**
   - All systems are pre-configured
   - Paper trading is enabled by default (safe mode)
   - No additional setup required
   - Yahoo Finance (yfinance) provides free data with no API key needed

---

## **Dashboard Overview**

### **Navigation Menu (Sidebar)**
- **Portfolio Overview** - Current holdings and performance
- **Live Prices** - Real-time stock prices by category
- **ML Predictions** - AI-powered price predictions
- **Portfolio Rebalancing** - Automated portfolio optimization
- **Cloud Progress** - Training and deployment status
- **ATLAS Console** - Advanced system controls

### **Main Content Area**
- **Dynamic pages** that change based on your selection
- **Real-time data updates** every 30 seconds
- **Interactive charts** and visualizations
- **Responsive design** that works on all devices

---

## **ML Predictions Page**

### **Prediction Modes**

#### **Hybrid Mode (Recommended)**
**Best of both worlds - combines real ML with reliable fallbacks**

**Features:**
- **Real ML Models** when available (Vertex AI or local)
- **Enhanced Mock Predictions** with technical analysis as fallback
- **Automatic failover** ensures predictions always work
- **Source transparency** - see which method was used

**When to use:** Always - this is the most reliable option

#### **Enhanced Mock Mode**
**Reliable predictions using technical analysis**

**Features:**
- **Real-time Yahoo Finance data**
- **Technical analysis** (RSI, MACD, Bollinger Bands, moving averages, momentum, volatility, ATR)
- **Dynamic confidence scoring**
- **Always available** - no training required

**When to use:** When you want consistent, reliable predictions

#### **Vertex AI Mode**
**Real trained machine learning models**

**Features:**
- **Actual ML models** trained on historical data
- **Advanced LSTM neural networks**
- **Cloud-powered predictions**
- **Fallback to mock** if models aren't deployed

**When to use:** When real ML models are deployed and working

### **Using Predictions**

1. **Select a stock:**
   - Choose "All" for comprehensive overview
   - Or select specific symbols (AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, etc.)

2. **View 21-day forecasts:**
   - Predictions are optimized for position trading with a 21-day horizon

3. **View results:**
   - **Prediction cards** show expected returns
   - **Color coding:** Green (bullish), Red (bearish)
   - **Confidence scores** indicate prediction reliability
   - **Source badges** show which method was used

### **Understanding Prediction Results**

#### **Return Predictions:**
- **Positive values** (green): Expected price increase
- **Negative values** (red): Expected price decrease
- **Percentage format:** Easy to understand (e.g., +3.5% = 3.5% gain expected)

#### **Confidence Scores:**
- **0.8-1.0:** Very high confidence (reliable predictions)
- **0.6-0.8:** High confidence (good predictions)
- **0.4-0.6:** Medium confidence (moderate reliability)
- **Below 0.4:** Low confidence (use with caution)

#### **Source Indicators:**
- **Hybrid:** Combined approach used
- **Vertex AI:** Real ML model prediction
- **Enhanced Mock:** Technical analysis prediction
- **Local ML:** Local trained model used

---

## **Portfolio Rebalancing Page**

### **Rebalancing Modes**

#### **Hybrid Rebalancing (Recommended)**
**Uses hybrid predictions for optimal allocation**

**Benefits:**
- **Best prediction accuracy** from combined sources
- **Robust fallbacks** ensure rebalancing always works
- **ML-enhanced allocations** for better returns
- **Risk-controlled adjustments**

#### **Enhanced Mock Rebalancing**
**Uses technical analysis for allocation decisions**

**Benefits:**
- **Consistent performance** with reliable predictions
- **Always available** regardless of ML model status
- **Technical indicator-based** adjustments
- **Stable rebalancing** logic

#### **Vertex AI Rebalancing**
**Uses real ML models for allocation optimization**

**Benefits:**
- **Advanced ML insights** for allocation decisions
- **Data-driven optimization** based on trained models
- **Cloud-powered processing** for complex calculations
- **Fallback available** if models aren't ready

### **Rebalancing Process**

1. **Current Portfolio Analysis:**
   - View your current holdings
   - See current allocation percentages
   - Identify drift from target allocation

2. **Target Allocation Calculation:**
   - **Base allocation:** Weighted across ~30 stocks
   - **ML adjustments:** Enhanced based on predictions
   - **Risk controls:** Position limits applied (max 15%, min 2%)
   - **Final allocation:** Optimized target weights

3. **Order Generation:**
   - **Buy orders:** For underweight positions
   - **Sell orders:** For overweight positions
   - **Order sizing:** Calculated precisely ($100 minimum trade)
   - **Zero-commission:** No trading fees with modern brokers

4. **Execution Options:**
   - **Paper Trading:** Simulate trades (default, safe)
   - **Live Trading:** Execute real trades (requires confirmation)

### **Understanding Rebalancing Results**

#### **Allocation Comparison:**
- **Current vs Target:** See how your portfolio needs to change
- **Drift Analysis:** Identify which positions are off-target
- **Rebalancing Threshold:** Only rebalance when drift > 5%

#### **Order Details:**
- **Symbol:** Which stock to trade
- **Action:** Buy or Sell
- **Quantity:** How much to trade
- **Estimated Cost:** Zero-commission trading
- **Reason:** Why this trade is recommended

#### **Risk Metrics:**
- **Total Portfolio Value:** Current worth
- **Expected Return:** Based on ML predictions
- **Risk Score:** Portfolio volatility assessment
- **Sharpe Ratio:** Risk-adjusted return measure

---

## **Live Prices Page**

### **Real-Time Data**
- **Current prices** from Yahoo Finance
- **Daily change** in price and percentage
- **Volume data** for trading activity
- **Price charts** with interactive features

### **Stock Categories**
- **Tech (FAANG+):** AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA
- **Sector Leaders:** JPM, UNH, XOM, CAT, PG, HD, NEE, AMT, LIN
- **ETFs:** SPY, QQQ, DIA, IWM, XLK, XLF, XLE, XLV, ARKK
- **Growth:** PLTR, CRWD, SNOW, SQ, COIN

### **Interactive Charts**
- **Zoom and pan** on price history
- **Time range selection** (1D, 7D, 30D)
- **Volume overlay** for market analysis
- **Mobile-friendly** responsive design

---

## **Portfolio Overview Page**

### **Portfolio Summary**
- **Total value** of your holdings
- **Day change** in portfolio value
- **Allocation breakdown** by stock
- **Performance metrics** and charts

### **Allocation Visualization**
- **Pie charts** showing current allocation
- **Target vs actual** comparison
- **Rebalancing recommendations** if needed
- **Historical performance** tracking

---

## **Cloud Progress Page**

### **Training Status**
- **Current training jobs** and their progress
- **Model deployment** status
- **Training metrics** and results
- **Cost tracking** for cloud resources

### **Endpoint Status**
- **Vertex AI endpoints** availability
- **Model deployment** progress
- **Prediction service** health
- **Performance metrics**

---

## **ATLAS Console (Advanced)**

### **System Controls**
- **Prediction service** configuration
- **Model training** controls
- **Data refresh** operations
- **System diagnostics**

### **Developer Tools**
- **Log viewing** and analysis
- **Configuration** management
- **API testing** tools
- **Performance monitoring**

---

## **Configuration Options**

### **Trading Settings**
- **Paper Trading Mode:** Safe simulation (default: ON)
- **Live Trading Mode:** Real trades (requires confirmation)
- **Position Limits:** Maximum allocation per symbol (15% max)
- **Rebalancing Threshold:** When to trigger rebalancing

### **Prediction Settings**
- **Default Mode:** Hybrid (recommended)
- **Prediction Horizon:** 21 days (position trading)
- **Confidence Threshold:** 0.6 minimum for ML adjustments
- **Fallback Behavior:** Automatic or manual

### **Risk Management**
- **Maximum Position Weight:** 15% per symbol
- **Minimum Position Weight:** 2% per symbol
- **ML Weight Factor:** 30% ML influence on allocation
- **Trading Fees:** Zero-commission (most modern brokers)

---

## **Safety Features**

### **Paper Trading (Default)**
- **All trades are simulated** - no real money at risk
- **Full functionality** without financial risk
- **Learning environment** to understand the system
- **Performance tracking** without consequences

### **Live Trading Safeguards**
- **Confirmation dialogs** for all real trades
- **Position limits** prevent over-concentration
- **Risk controls** built into all decisions
- **Audit trail** for all trading activity

### **Error Handling**
- **Graceful fallbacks** when services fail
- **User-friendly error messages**
- **Automatic recovery** from temporary issues
- **Comprehensive logging** for troubleshooting

---

## **Tips for Best Results**

### **Getting Started:**
1. **Start with Paper Trading** to learn the system
2. **Use Hybrid Mode** for the most reliable predictions
3. **Monitor predictions** for a few days before live trading
4. **Understand the risk controls** and position limits

### **Optimizing Performance:**
1. **Regular rebalancing** based on ML recommendations
2. **Monitor prediction accuracy** over time
3. **Adjust confidence thresholds** based on your risk tolerance
4. **Review portfolio performance** weekly

### **Risk Management:**
1. **Never invest more than you can afford to lose**
2. **Diversify across sectors** using the full stock universe
3. **Monitor market conditions** and adjust accordingly
4. **Use position limits** to prevent over-concentration

---

## **Troubleshooting**

### **Common Issues:**

#### **"No predictions available"**
- **Cause:** API connectivity issues
- **Solution:** Check internet connection, refresh page
- **Fallback:** System will retry automatically

#### **"Prediction service failed"**
- **Cause:** Temporary service unavailability
- **Solution:** System automatically falls back to enhanced mock
- **Note:** Predictions will still work with fallback

#### **"Model not trained"**
- **Cause:** ML models haven't been trained yet
- **Solution:** Use enhanced mock mode, or wait for training completion
- **Note:** Enhanced mock predictions are still highly effective

#### **"Rebalancing not recommended"**
- **Cause:** Portfolio is within rebalancing threshold
- **Solution:** Wait for more significant drift, or manually adjust
- **Note:** This prevents unnecessary trading activity

### **Getting Help:**
- **Check the logs** in ATLAS Console for detailed error information
- **Review system status** in Cloud Progress page
- **Monitor prediction accuracy** to understand system performance
- **Use paper trading** to test changes safely

---

## **Additional Resources**

### **Understanding the Technology:**
- **LSTM Neural Networks:** How the ML models work
- **Technical Analysis:** RSI, MACD, Bollinger Bands, moving averages, momentum, volatility, ATR (25 indicators)
- **Portfolio Theory:** Modern portfolio optimization principles
- **Risk Management:** Position sizing and diversification

### **Best Practices:**
- **Start small** and scale up as you gain confidence
- **Monitor performance** regularly and adjust strategies
- **Stay informed** about market conditions and news
- **Diversify** your overall investment portfolio

---

**You now have access to an advanced stock trading dashboard powered by ML intelligence. Start with paper trading, learn the system, and gradually scale up your usage as you become comfortable with the technology.**

---

*User Guide - ATLAS Stock ML Intelligence System*
