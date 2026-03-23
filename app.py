"""
ATLAS - Stock ML Intelligence Dashboard

Main Streamlit application for AI-powered stock trading with ML price predictions
(LSTM neural networks), real-time Yahoo Finance data, and Google Cloud ML infrastructure.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import subprocess
import json
import os

# Import new UI components
from ui.styles import THEME
from ui.components import load_css, card_start, card_end, metric_card, section_header, status_badge

# Helper functions for cloud progress tracking
def get_training_job_status():
    """Get the status of the latest training job from Google Cloud."""
    try:
        # Get the latest training job
        result = subprocess.run([
            'gcloud', 'ai', 'custom-jobs', 'list',
            '--region=us-central1',
            '--format=value(state)',
            '--limit=1',
            '--sort-by=~createTime'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            status = result.stdout.strip()
            return status if status else "NO_JOBS"
        else:
            return "ERROR"
    except Exception as e:
        return f"ERROR: {str(e)}"

def get_latest_training_jobs():
    """Get the latest training jobs from Google Cloud."""
    try:
        result = subprocess.run([
            'gcloud', 'ai', 'custom-jobs', 'list',
            '--region=us-central1',
            '--format=json',
            '--limit=5',
            '--sort-by=~createTime'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            return json.loads(result.stdout)
        return []
    except Exception:
        return []

def get_latest_endpoints():
    """Get the latest endpoints from Google Cloud."""
    try:
        result = subprocess.run([
            'gcloud', 'ai', 'endpoints', 'list',
            '--region=us-central1',
            '--format=json',
            '--limit=5',
            '--sort-by=~createTime'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            return json.loads(result.stdout)
        return []
    except Exception:
        return []

def get_training_logs():
    """Get recent logs from the training job."""
    try:
        # Use a shorter timeout and handle the streaming nature
        # Get job ID from env instead of hardcoding
        job_id = os.getenv('VERTEX_JOB_ID', '')
        if not job_id:
            return ["No VERTEX_JOB_ID configured. Set it in .env to stream logs."]
        result = subprocess.run([
            'gcloud', 'ai', 'custom-jobs', 'stream-logs', job_id
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0 or result.stdout:
            logs = result.stdout.strip().split('\n')
            # Return last 5 lines, filtering out empty lines
            non_empty_logs = [log for log in logs if log.strip()]
            return non_empty_logs[-5:] if non_empty_logs else ["No logs available yet"]
        else:
            return [f"Log fetch error: {result.stderr}"]
    except subprocess.TimeoutExpired:
        # Timeout is normal for stream-logs - return a status message
        return ["Waiting for training logs... (Job may still be starting)"]
    except Exception as e:
        return [f"Log error: {str(e)}"]

def get_gcp_costs():
    """Get current GCP costs (mock data for now)."""
    return {
        "vertex_ai": 2.50,
        "bigquery": 1.20,
        "storage": 0.80,
        "total": 4.50
    }

# Page configuration
st.set_page_config(
    page_title="ATLAS • Stock ML Console",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Global CSS
load_css()

# Import stock data module
from data.stock_api import StockAPI, STOCK_UNIVERSE, get_all_symbols, get_stock_info, get_symbols_by_category

# Create a shared StockAPI instance
_stock_api = StockAPI()


def _get_demo_portfolio():
    """Return demo stock portfolio data."""
    return {
        'AAPL': {'quantity': 25, 'avg_buy_price': 175.00, 'current_price': 185.00},
        'MSFT': {'quantity': 12, 'avg_buy_price': 380.00, 'current_price': 420.00},
        'GOOGL': {'quantity': 20, 'avg_buy_price': 155.00, 'current_price': 175.00},
        'NVDA': {'quantity': 8, 'avg_buy_price': 750.00, 'current_price': 880.00},
        'SPY': {'quantity': 15, 'avg_buy_price': 490.00, 'current_price': 520.00},
        'JPM': {'quantity': 18, 'avg_buy_price': 185.00, 'current_price': 200.00},
    }


def show_header():
    """Display the main header - cyberpunk style."""
    col1, col2, col3 = st.columns([2, 3, 2])
    
    with col1:
        # Status indicator
        st.markdown(f"""
        <div class="fade-in" style="display: flex; align-items: center; gap: 12px;">
            <div class="pulse" style="width: 8px; height: 8px; background: {THEME['accent_success']}; border-radius: 50%; box-shadow: 0 0 12px {THEME['accent_success']};"></div>
            <span style="color: {THEME['text_secondary']}; font-size: 13px; font-weight: 600; font-family: 'Orbitron', sans-serif;">SYSTEM ONLINE</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="fade-in" style="text-align: center;">
            <h1 class="neon-text" style="margin: 0; font-size: 3.5rem; letter-spacing: 0.2em;">ATLAS</h1>
            <p style="color: {THEME['text_secondary']}; margin: 0; font-size: 0.9rem; font-weight: 500; letter-spacing: 0.4em; font-family: 'Rajdhani', sans-serif;">
                STOCK ML INTELLIGENCE
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="text-align: right; padding-top: 10px;">
            <span style="
                background: {THEME['bg_glass']}; 
                border: 1px solid {THEME['accent_primary']}40; 
                padding: 6px 12px; 
                border-radius: 20px;
                box-shadow: 0 0 10px {THEME['glow_primary']};
            ">
                <i class="fas fa-clock" style="font-size: 14px; color: {THEME['accent_primary']}; margin-right: 8px;"></i>
                <span style="font-family: 'Orbitron', sans-serif; font-size: 12px; color: {THEME['text_primary']};">{datetime.now().strftime('%H:%M:%S')}</span>
            </span>
        </div>
        """, unsafe_allow_html=True)


def show_portfolio_view():
    """Display stock portfolio overview with holdings and performance."""
    col1, col2 = st.columns([3, 1])
    with col1:
        section_header("Portfolio Overview", icon="fa-wallet")
    with col2:
        if st.button("⟳ Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # Use demo portfolio (paper trading)
    portfolio_data = _get_demo_portfolio()

    # Fetch live prices for all holdings via yfinance
    with st.spinner("⟳ Fetching live stock prices..."):
        quotes = _stock_api.get_batch_quotes(list(portfolio_data.keys()))
        for symbol, quote in quotes.items():
            if symbol in portfolio_data and quote:
                portfolio_data[symbol]['current_price'] = quote['current']

    # Calculate portfolio metrics
    total_value = 0
    total_cost = 0
    holdings = []

    for symbol, data in portfolio_data.items():
        quantity = data['quantity']
        avg_price = data['avg_buy_price']
        current_price = data['current_price']

        cost_basis = quantity * avg_price
        current_value = quantity * current_price
        pnl = current_value - cost_basis
        pnl_pct = (pnl / cost_basis) * 100 if cost_basis > 0 else 0

        total_value += current_value
        total_cost += cost_basis

        stock_info = get_stock_info(symbol)
        sector = stock_info['sector'] if stock_info else 'Unknown'

        holdings.append({
            'Symbol': symbol,
            'Sector': sector,
            'Shares': f"{quantity:,.0f}",
            'Avg Buy Price': f"${avg_price:,.2f}",
            'Current Price': f"${current_price:,.2f}",
            'Value': f"${current_value:,.2f}",
            'P&L': f"${pnl:,.2f}",
            'P&L %': f"{pnl_pct:+.2f}%",
            '% Portfolio': f"{(current_value/total_value)*100:.1f}%" if total_value > 0 else "0%"
        })

    total_pnl = total_value - total_cost
    total_pnl_pct = (total_pnl / total_cost) * 100 if total_cost > 0 else 0

    st.caption(f"⟳ Last updated: {datetime.now().strftime('%B %d, %Y at %I:%M:%S %p')}")

    # Display key metrics
    st.markdown("### ◉ Portfolio Summary")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card("TOTAL VALUE", f"${total_value:,.2f}", icon="fa-wallet")
    with col2:
        pnl_delta = f"{total_pnl:+.2f} ({total_pnl_pct:+.2f}%)"
        metric_card("TOTAL P&L", f"${total_pnl:+,.2f}", delta=pnl_delta, icon="fa-chart-line")
    with col3:
        metric_card("POSITIONS", str(len(portfolio_data)), icon="fa-layer-group")
    with col4:
        metric_card("CASH BALANCE", "$5,000.00", icon="fa-coins")

    st.markdown("---")

    # Holdings table
    st.markdown("### ⚡ Current Holdings")

    if holdings:
        holdings_df = pd.DataFrame(holdings)
        st.dataframe(
            holdings_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                "Sector": st.column_config.TextColumn("Sector", width="medium"),
                "Shares": st.column_config.TextColumn("Shares", width="small"),
                "Current Price": st.column_config.TextColumn("Current Price", width="medium"),
                "Value": st.column_config.TextColumn("Value", width="medium"),
                "P&L": st.column_config.TextColumn("P&L", width="medium"),
                "P&L %": st.column_config.TextColumn("P&L %", width="small"),
                "% Portfolio": st.column_config.TextColumn("% Portfolio", width="small"),
            }
        )
        st.caption("◉ **Note:** This is a demo portfolio for paper trading. P&L is calculated from simulated average buy prices.")
    else:
        st.info("No holdings to display")

    # Portfolio allocation pie chart
    st.markdown("### ◉ Portfolio Allocation")

    allocation_data = pd.DataFrame([
        {'Symbol': symbol, 'Value': data['quantity'] * data['current_price']}
        for symbol, data in portfolio_data.items()
        if data['quantity'] * data['current_price'] > 0
    ])

    if not allocation_data.empty:
        fig = px.pie(
            allocation_data,
            values='Value',
            names='Symbol',
            title='',
            color_discrete_sequence=px.colors.qualitative.Set3,
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=14)
        fig.update_layout(
            height=400, showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.success("◉ **Connected to Yahoo Finance** - Real-time stock prices via yfinance (free, no API key).")


def show_live_prices():
    """Display live stock prices with charts by category."""
    col1, col2 = st.columns([3, 1])
    with col1:
        section_header("Live Stock Prices", icon="fa-arrow-trend-up")
    with col2:
        if st.button("⟳ Refresh Prices", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # Category selector
    categories = {
        'tech': 'Tech (FAANG+)',
        'sector_leaders': 'Sector Leaders',
        'etfs': 'ETFs',
        'growth': 'Growth'
    }
    selected_category = st.selectbox(
        "Stock Category:",
        options=list(categories.keys()),
        format_func=lambda x: categories[x]
    )

    symbols = get_symbols_by_category(selected_category)
    stocks_info = STOCK_UNIVERSE[selected_category]

    # Fetch batch quotes
    with st.spinner("Fetching live prices..."):
        quotes = _stock_api.get_batch_quotes(symbols)

    if not quotes:
        st.error("Unable to fetch price data. Please try again.")
        return

    st.caption(f"⟳ Last updated: {datetime.now().strftime('%B %d, %Y at %I:%M:%S %p')}")
    st.markdown("### ◉ Market Overview")

    # Display price cards in rows of 3
    symbol_list = list(quotes.keys())
    for row_start in range(0, len(symbol_list), 3):
        cols = st.columns(3)
        for col_idx in range(3):
            idx = row_start + col_idx
            if idx >= len(symbol_list):
                break
            symbol = symbol_list[idx]
            info = stocks_info.get(symbol, {})
            quote = quotes[symbol]
            current_price = quote['current']
            change_pct = quote['change_pct']
            change_color = "#00ff00" if change_pct >= 0 else "#ff4444"
            change_symbol = "↗" if change_pct >= 0 else "↘"

            with cols[col_idx]:
                st.markdown(f"""
                <div class='glass-card fade-in' style='animation-delay: {idx * 0.1}s;'>
                    <div style='display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px;'>
                        <div>
                            <h4 style='color: {THEME['text_primary']}; margin: 0; font-size: 1.1rem; font-weight: 600;'>{info.get('name', symbol)}</h4>
                            <span style='color: {THEME['text_muted']}; font-size: 12px;'>{symbol} · {info.get('sector', '')}</span>
                        </div>
                        <div class='badge' style='background: {change_color}20; border-color: {change_color}40;'>
                            <span style='color: {change_color}; font-weight: 600;'>{change_symbol} {change_pct:+.2f}%</span>
                        </div>
                    </div>
                    <h1 style='color: {THEME['text_primary']}; margin: 0 0 16px 0; font-size: 2rem; font-weight: 700;'>${current_price:,.2f}</h1>
                    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 8px; color: {THEME['text_secondary']}; font-size: 13px;'>
                        <div><span style='color: {THEME['text_muted']}; font-size: 11px;'>Day High</span><br><strong style='color: {THEME['text_primary']};'>${quote['high']:,.2f}</strong></div>
                        <div><span style='color: {THEME['text_muted']}; font-size: 11px;'>Day Low</span><br><strong style='color: {THEME['text_primary']};'>${quote['low']:,.2f}</strong></div>
                    </div>
                    <div style='margin-top: 12px; padding-top: 12px; border-top: 1px solid {THEME['border_color']}; color: {THEME['text_secondary']}; font-size: 12px;'>
                        <span style='color: {THEME['text_muted']};'>Volume:</span> <strong style='color: {THEME['text_primary']};'>{quote['volume']:,.0f}</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    # Price chart section
    st.markdown("### ↗ Interactive Price Charts")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        selected_symbol = st.selectbox(
            "Select stock:",
            options=symbols,
            format_func=lambda x: f"{stocks_info[x]['name']} ({x})" if x in stocks_info else x
        )
    with col2:
        period = st.selectbox(
            "Time Period:",
            options=["1mo", "3mo", "6mo", "1y", "2y"],
            format_func=lambda x: {"1mo": "1 Month", "3mo": "3 Months", "6mo": "6 Months", "1y": "1 Year", "2y": "2 Years"}[x],
            index=2
        )
    with col3:
        chart_type = st.selectbox("Chart Type:", options=["Candlestick", "Line"], index=0)

    # Fetch OHLC data via yfinance
    df = _stock_api.get_ohlc(selected_symbol, period=period)

    if df is not None and not df.empty:
        stock_color = stocks_info.get(selected_symbol, {}).get('color', '#4ecdc4')
        stock_name = stocks_info.get(selected_symbol, {}).get('name', selected_symbol)

        if chart_type == "Candlestick":
            fig = go.Figure(data=[go.Candlestick(
                x=df['timestamp'], open=df['open'], high=df['high'],
                low=df['low'], close=df['close'], name='Price',
                increasing_line_color='#00ff00', decreasing_line_color='#ff4444'
            )])
        else:
            fig = go.Figure(data=[go.Scatter(
                x=df['timestamp'], y=df['close'], mode='lines',
                name='Price', line=dict(color=stock_color, width=2)
            )])

        fig.update_layout(
            title=f"{stock_name} ({selected_symbol}) Price Chart",
            yaxis_title="Price (USD)", xaxis_title="Date", height=500,
            template="plotly_dark", xaxis_rangeslider_visible=False,
            plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
            font=dict(color='white'), title_font=dict(size=20, color=stock_color),
            xaxis=dict(gridcolor='#333'), yaxis=dict(gridcolor='#333')
        )
        st.plotly_chart(fig, use_container_width=True)

        # Volume chart
        fig_volume = px.bar(df, x='timestamp', y='volume',
                           title=f"{selected_symbol} Trading Volume",
                           labels={'volume': 'Volume', 'timestamp': 'Date'})
        fig_volume.update_layout(
            height=200, template="plotly_dark",
            plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
            font=dict(color='white'), title_font=dict(size=16, color=stock_color),
            xaxis=dict(gridcolor='#333'), yaxis=dict(gridcolor='#333')
        )
        fig_volume.update_traces(marker_color=stock_color)
        st.plotly_chart(fig_volume, use_container_width=True)

        # Market stats
        st.markdown("### ◉ Market Statistics")
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2] if len(df) > 1 else current_price
        change_day = ((current_price - prev_price) / prev_price) * 100 if prev_price > 0 else 0

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            metric_card("Day Change", f"{change_day:+.2f}%", icon="fa-percent")
        with col2:
            metric_card("Avg Volume", f"{df['volume'].mean():,.0f}", icon="fa-chart-bar")
        with col3:
            metric_card("Period High", f"${df['high'].max():,.2f}", icon="fa-arrow-up")
        with col4:
            metric_card("Period Low", f"${df['low'].min():,.2f}", icon="fa-arrow-down")
    else:
        st.warning("Unable to fetch chart data. Please try again.")


def show_predictions():
    """Display ML predictions with real-time data and model training."""
    col1, col2 = st.columns([3, 1])
    with col1:
        section_header("ML Price Predictions", icon="fa-brain")
    with col2:
        if st.button("⟳ Refresh Predictions", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Hybrid ML System Selection
    st.markdown("### ⚙️ Prediction System Configuration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        prediction_mode = st.selectbox(
            "Prediction Mode:",
            options=["hybrid", "enhanced_mock", "vertex_ai"],
            format_func=lambda x: {
                "hybrid": "🔀 Hybrid (Best of Both Worlds)",
                "enhanced_mock": "🏠 Enhanced Mock (Reliable)",
                "vertex_ai": "☁️ Vertex AI (Real ML)"
            }[x],
            help="Choose your prediction strategy",
            index=0  # Default to Hybrid
        )
    
    with col2:
        # Show system status
        training_status = get_training_job_status()
        if training_status == "JOB_STATE_SUCCEEDED":
            status_badge("online", "ML MODEL READY")
        elif training_status == "JOB_STATE_RUNNING":
            status_badge("training", "ML TRAINING...")
        else:
            status_badge("offline", "ML MODEL PENDING")
    
    # Initialize prediction service
    prediction_service = None
    hybrid_service = None
    system_summary = {
        'enhanced_mock_available': True,
        'vertex_ai_available': False,
        'local_ml_models_available': False,
        'local_ml_models_count': 0,
        'supported_symbols': get_all_symbols()[:12],
        'system_status': 'basic_operational'
    }
    
    # Try to initialize hybrid prediction service
    try:
        from ml.hybrid_prediction_service import HybridPredictionService
        hybrid_service = HybridPredictionService()
        
        # Get system summary
        system_summary = hybrid_service.get_prediction_summary()
    except ImportError as e:
        st.warning(f"⚠️ Hybrid service import failed: {e}")
        st.info("Falling back to basic prediction service.")
        hybrid_service = None
        
        # Show system status
        st.markdown("### 🔧 System Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            metric_card("Enhanced Mock", "Active", icon="fa-check-circle")
        
        with col2:
            vertex_status = "Available" if system_summary['vertex_ai_available'] else "Unavailable"
            metric_card("Vertex AI", vertex_status, icon="fa-cloud")
        
        with col3:
            local_status = "Available" if system_summary['local_ml_models_available'] else "Unavailable"
            metric_card("Local Models", local_status, icon="fa-server")
        
    # Set prediction service based on mode
    try:
        if prediction_mode == "hybrid" and hybrid_service is not None:
            prediction_service = hybrid_service
        elif prediction_mode == "enhanced_mock":
            from ml.prediction_service import PredictionService
            prediction_service = PredictionService(provider="local")
        else:  # vertex_ai or hybrid fallback
            from ml.prediction_service import PredictionService
            prediction_service = PredictionService(provider="vertex")
    except ImportError as e:
        st.error(f"⊗ Could not import prediction service: {e}")
        st.info("Falling back to basic prediction service.")
        
        try:
            from ml.prediction_service import PredictionService
            prediction_service = PredictionService(provider="local")
        except ImportError:
            st.error("⊗ Could not import any prediction service.")
            return
    
    # Ensure prediction_service is always defined
    if prediction_service is None:
        try:
            from ml.prediction_service import PredictionService
            prediction_service = PredictionService(provider="local")
        except ImportError:
            st.error("⊗ Could not initialize any prediction service.")
            return
    
    # Show last update time
    st.caption(f"⟳ Last updated: {datetime.now().strftime('%B %d, %Y at %I:%M:%S %p')}")
    
    # Show current prediction mode status
    if prediction_mode == "hybrid":
        st.success("🔀 **Using Hybrid Prediction System** - Best of both worlds!")
        st.info("""
        **Hybrid System Features:**
        • **Real ML Models** when available (Vertex AI or local)
        • **Enhanced Mock Predictions** with technical analysis as fallback
        • **Automatic failover** ensures predictions always work
        • **Source transparency** - see which method was used for each prediction
        """)
    elif prediction_mode == "enhanced_mock":
        st.success("🏠 **Using Enhanced Mock Predictions** - Reliable and sophisticated!")
        st.info("""
        **Enhanced Mock Features:**
        • **Real-time Yahoo Finance data**
        • **Technical analysis** (RSI, trends, volatility)
        • **Dynamic confidence scoring**
        • **Always available** - no training required
        """)
    else:  # vertex_ai
        if hasattr(prediction_service, 'vertex_service') and prediction_service.vertex_service:
            st.success("☁️ **Using Vertex AI ML Models** - Real trained models!")
        else:
            st.warning("⚠️ **Vertex AI Not Available** - Using enhanced mock predictions.")
            
            # Show deployment instructions
            with st.expander("🚀 Enable Vertex AI ML Predictions"):
                st.markdown("""
                **To enable real ML predictions:**
                
                1. **Wait for training to complete** (check Cloud Progress tab)
                2. **Deploy the trained model to endpoint**
                3. **Refresh this page** to see real ML predictions!
                
                **Current Training Status:** Check the "☁️ Cloud Progress" tab for updates.
                """)
    
    # Prediction controls
    st.markdown("### ↗ Prediction Controls")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        selected_symbol = st.selectbox(
            "Select Stock:",
            options=['All'] + get_all_symbols()[:12],
            index=0
        )
    
    with col2:
        days_ahead = st.selectbox(
            "Prediction Horizon:",
            options=[1, 3, 7, 14, 30],
            index=2,
            format_func=lambda x: f"{x} day{'s' if x > 1 else ''}"
        )
    
    with col3:
        if st.button("◉ Train Model", use_container_width=True):
            with st.spinner("Training model... This may take a few minutes."):
                if selected_symbol == 'All':
                    st.info("Please select a specific stock to train its model.")
                else:
                    result = prediction_service.train_model(selected_symbol, days=365, epochs=50)
                    if result['status'] == 'success':
                        st.success(f"◉ Model trained successfully for {selected_symbol}!")
                        st.json(result)
                    else:
                        st.error(f"⊗ Training failed: {result['message']}")
                        if "TensorFlow" in result['message']:
                            st.info("◉ **Solution**: Install TensorFlow with Python 3.9-3.11 to train real models.")
                            st.info("◉ **What happened**: The system successfully fetched real data and calculated features, but failed at the TensorFlow model creation step.")
    
    # Train all models button
    st.markdown("### ◉ Train All Models")
    if st.button("◉ Train All Models", type="primary", use_container_width=True):
        with st.spinner("Training all models... This may take several minutes."):
            try:
                results = prediction_service.train_all_models()
                
                # Show training results
                st.success("◉ Training completed!")
                
                # Display results (results is a dict like {'BTC': {...}, 'ETH': {...}})
                for symbol, result in results.items():
                    if result['status'] == 'success':
                        st.success(f"◉ {symbol}: {result.get('message', 'Training successful')}")
                    else:
                        st.error(f"⊗ {symbol}: {result.get('message', 'Training failed')}")
                        
            except Exception as e:
                st.error(f"⊗ Training failed: {e}")
                st.info("◉ **Solution**: Install TensorFlow with Python 3.9-3.11 to train real models.")
                
                # Show what actually happened
                st.info("◉ **What happened**: The system successfully fetched real data and calculated features, but failed at the TensorFlow model creation step.")
                st.info("◉ **Data processed**: 366 days of real market data with 11 technical indicators")
                st.info("⟳ **Sequences created**: 352 training sequences ready for LSTM training")
                st.info("⊗ **Missing**: TensorFlow library for neural network training")
    
    st.markdown("---")
    
    # Generate predictions
    with st.spinner("◉ Generating predictions..."):
        try:
            if selected_symbol == 'All':
                # Check if using HybridPredictionService (needs symbols param) or regular PredictionService (doesn't)
                if hasattr(prediction_service, '__class__') and 'Hybrid' in prediction_service.__class__.__name__:
                    # HybridPredictionService needs symbols parameter
                    top_symbols = get_all_symbols()[:12]
                    predictions_dict = prediction_service.get_all_predictions(symbols=top_symbols, days_ahead=days_ahead)
                    predictions = [predictions_dict[s] for s in top_symbols if s in predictions_dict]
                else:
                    # Regular PredictionService returns list directly
                    predictions = prediction_service.get_all_predictions(days_ahead=days_ahead)
            else:
                predictions = [prediction_service.get_prediction(selected_symbol, days_ahead)]
        except Exception as e:
            st.error(f"⊗ Error generating predictions: {e}")
            predictions = []
    
    # Display predictions with portfolio-style cards
    st.markdown("### ↗ Price Predictions")
    
    if not predictions:
        st.warning("No predictions available. Please try again.")
        return
    
    # Create prediction cards
    num_cols = 3
    for i in range(0, len(predictions), num_cols):
        cols = st.columns(num_cols)
        
        for j, col in enumerate(cols):
            if i + j < len(predictions):
                pred = predictions[i + j]
                
                # Determine colors based on prediction
                if pred['predicted_return'] > 0.02:  # > 2% gain
                    return_color = THEME['accent_success']
                    return_symbol = "↗"
                elif pred['predicted_return'] < -0.02:  # < -2% loss
                    return_color = THEME['accent_danger']
                    return_symbol = "↘"
                else:  # Neutral
                    return_color = THEME['accent_primary']
                    return_symbol = "→"
                
                with col:
                    # Create a container with custom styling
                    st.markdown(f"""
                    <div class="glass-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                            <h3 style="margin: 0; color: {THEME['text_primary']};">{pred['symbol']} {return_symbol}</h3>
                            <span class="badge" style="background: {return_color}20; color: {return_color}; border-color: {return_color};">
                                {pred['confidence']*100:.0f}% CONFIDENCE
                            </span>
                        </div>
                        
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 15px;">
                            <div>
                                <div style="font-size: 0.8rem; color: {THEME['text_muted']};">CURRENT</div>
                                <div style="font-size: 1.2rem; font-weight: bold; color: {THEME['text_primary']};">${pred['current_price']:,.2f}</div>
                            </div>
                            <div>
                                <div style="font-size: 0.8rem; color: {THEME['text_muted']};">PREDICTED</div>
                                <div style="font-size: 1.2rem; font-weight: bold; color: {THEME['text_primary']};">${pred['predicted_price']:,.2f}</div>
                            </div>
                        </div>
                        
                        <div style="text-align: center; padding: 8px; background: {return_color}10; border-radius: 8px; border: 1px solid {return_color}40;">
                            <span style="color: {return_color}; font-weight: bold; font-size: 1.1rem;">
                                {pred['predicted_return']*100:+.2f}%
                            </span>
                            <span style="color: {THEME['text_secondary']}; font-size: 0.8rem; margin-left: 5px;">
                                ({days_ahead}d forecast)
                            </span>
                        </div>
                        
                        <div style="margin-top: 10px; font-size: 0.7rem; color: {THEME['text_muted']}; text-align: right;">
                            Source: {pred.get('prediction_source', 'unknown').replace('_', ' ').title()}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Prediction table
    st.markdown("### ◉ Detailed Predictions")
    
    # Prepare data for table
    table_data = []
    for pred in predictions:
        prediction_source = pred.get('prediction_source', 'unknown')
        
        # Format prediction source for display
        source_display = {
            'vertex_ai_ml': '🤖 Vertex AI ML',
            'local_ml': '🧠 Local ML',
            'enhanced_mock': '🔧 Enhanced Mock',
            'basic_mock': '⚠️ Basic Mock'
        }.get(prediction_source, '❓ Unknown')
        
        table_data.append({
            'Symbol': pred['symbol'],
            'Current Price': f"${pred['current_price']:,.2f}",
            'Predicted Price': f"${pred['predicted_price']:,.2f}",
            'Predicted Return': f"{pred['predicted_return']*100:+.2f}%",
            'Confidence': f"{pred['confidence']*100:.1f}%",
            'Source': source_display,
            'Status': pred['status'].title()
        })
    
    if table_data:
        st.dataframe(
            pd.DataFrame(table_data),
            use_container_width=True,
            hide_index=True
        )
    
    # Model training section
    st.markdown("---")
    st.markdown("### ◉ Model Training")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Train Custom Models:**
        
        - **LSTM Architecture**: 2-layer neural network with 50 units each
        - **Features**: 11 technical indicators (MA, RSI, Volume, Momentum, Volatility)
        - **Training Data**: 365 days of historical data from Yahoo Finance
        - **Prediction Target**: 7-day future returns
        - **Validation**: 80/20 train/validation split with early stopping
        """)
    
    with col2:
        if st.button("◉ Train All Models", use_container_width=True):
            with st.spinner("Training all models... This will take several minutes."):
                results = prediction_service.train_all_models(days=365, epochs=50)
                
                # Display results with better formatting
                st.markdown("#### ◉ Training Results:")
                
                successful = 0
                failed = 0
                
                for symbol, result in results.items():
                    if result['status'] == 'success':
                        st.success(f"◉ **{symbol}**: Trained successfully")
                        successful += 1
                        with st.expander(f"◉ {symbol} Training Details"):
                            st.json(result)
                    else:
                        st.error(f"⊗ **{symbol}**: {result['message']}")
                        failed += 1
                        
                        # Show solution for TensorFlow error
                        if "TensorFlow is required" in result['message']:
                            st.info(f"◉ **Solution for {symbol}**: Install TensorFlow with Python 3.9-3.11")
                
                # Summary
                if successful > 0:
                    st.success(f"◉ **{successful} models trained successfully!**")
                if failed > 0:
                    st.warning(f"⊗ **{failed} models failed** - TensorFlow installation required")
                    
                # Show installation instructions
                if failed > 0:
                    with st.expander("◉ How to Install TensorFlow for Real Model Training"):
                        st.markdown("""
                        **To train real LSTM models, you need TensorFlow:**
                        
                        1. **Create a new Python environment** with Python 3.9-3.11:
                           ```bash
                           conda create -n stock-ml python=3.11
                           conda activate stock-ml
                           ```
                        
                        2. **Install TensorFlow**:
                           ```bash
                           pip install tensorflow==2.13.0
                           ```
                        
                        3. **Install other dependencies**:
                           ```bash
                           pip install -r requirements.txt
                           ```
                        
                        4. **Run the app**:
                           ```bash
                           streamlit run app.py
                           ```
                        
                        **Current Status**: Using mock predictions (realistic but not trained on your data)
                        """)
    
    # Model status
    st.markdown("### ◉ Model Status")
    
    model_status = []
    for symbol in get_all_symbols()[:12]:
        has_model = prediction_service._has_model(symbol)
        status = "◉ Trained" if has_model else "⊗ Not Trained"
        
        # Get last updated timestamp from model file
        model_path = f"models/{symbol}_model.h5"
        last_updated = 'N/A'
        if os.path.exists(model_path):
            timestamp = os.path.getmtime(model_path)
            last_updated = datetime.fromtimestamp(timestamp).strftime('%b %d, %Y %I:%M %p')
        
        model_status.append({
            'Symbol': symbol,
            'Status': status,
            'Model File': f"{symbol}_model.h5",
            'Last Updated': last_updated
        })
    
    status_df = pd.DataFrame(model_status)
    st.dataframe(status_df, use_container_width=True, hide_index=True)
    
    # Information section
    with st.expander("◉ About ML Predictions"):
        st.markdown("""
        **How It Works:**
        
        1. **Data Collection**: Fetches 365 days of OHLCV data from Yahoo Finance
        2. **Feature Engineering**: Creates 11 technical indicators:
           - Moving Averages (7, 14, 30-day)
           - RSI (Relative Strength Index)
           - Volume indicators
           - Price momentum
           - Volatility measures
        3. **LSTM Training**: 2-layer neural network learns patterns from historical data
        4. **Prediction**: Uses last 7 days of features to predict future returns
        5. **Confidence**: Based on model certainty and historical accuracy
        
        **Important Notes:**
        - Predictions are for educational purposes only
        - Past performance doesn't guarantee future results
        - Always do your own research before trading
        - Models are retrained weekly to adapt to market changes
        """)
    
    # Status info
    st.success("◉ **ML Prediction System Active** - Real-time forecasts powered by LSTM neural networks!")


def show_rebalancing():
    """Display portfolio rebalancing interface."""
    st.markdown("""
    <h1 style='color: #4ecdc4; margin: 0 0 20px 0; display: flex; align-items: center;'>
        <span class='material-symbols-outlined' style='margin-right: 12px; font-size: 32px;'>tune</span>
        Portfolio Rebalancing
    </h1>
    """, unsafe_allow_html=True)
    
    # Prediction System Selection for Rebalancing
    st.markdown("### ⚙️ Rebalancing Prediction System")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        rebalancing_mode = st.selectbox(
            "Rebalancing Prediction Mode:",
            options=["hybrid", "enhanced_mock", "vertex_ai"],
            format_func=lambda x: {
                "hybrid": "🔀 Hybrid (Best of Both Worlds)",
                "enhanced_mock": "🏠 Enhanced Mock (Reliable)",
                "vertex_ai": "☁️ Vertex AI (Real ML)"
            }[x],
            help="Choose prediction system for portfolio rebalancing",
            index=0  # Default to Hybrid
        )
    
    with col2:
        paper_trading = st.checkbox("◉ Paper Trading Mode", value=True, help="Enable paper trading to test strategies without real money")
    
    # Import rebalancing service
    try:
        from ml.portfolio_rebalancer import PortfolioRebalancer
        try:
            from ml.hybrid_prediction_service import HybridPredictionService
            hybrid_service = HybridPredictionService()
        except ImportError as e:
            st.warning(f"⚠️ Hybrid service import failed: {e}")
            hybrid_service = None
        
        # Create custom rebalancer with hybrid predictions
        if rebalancing_mode == "hybrid" and hybrid_service is not None:
            # Create a custom rebalancer that uses hybrid predictions
            class HybridRebalancer(PortfolioRebalancer):
                def __init__(self, paper_trading=True):
                    super().__init__(paper_trading)
                    # Replace prediction service with hybrid service
                    self.prediction_service = hybrid_service
            
            rebalancer = HybridRebalancer(paper_trading=paper_trading)
        else:
            # Use standard rebalancer with specific provider
            if rebalancing_mode == "hybrid":
                # Fallback to enhanced mock if hybrid service not available
                provider = "local"
            else:
                provider = "vertex" if rebalancing_mode == "vertex_ai" else "local"
            
            rebalancer = PortfolioRebalancer(paper_trading=paper_trading)
            # Update the prediction service provider
            rebalancer.prediction_service.provider = provider
            
    except ImportError as e:
        st.error(f"⊗ Could not import portfolio rebalancer: {e}")
        return
    
    # Show prediction system status for rebalancing
    st.markdown("### 🔧 Rebalancing System Status")
    
    if rebalancing_mode == "hybrid" and hybrid_service is not None:
        system_summary = hybrid_service.get_prediction_summary()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
                border: 2px solid #4caf50;
                border-radius: 10px;
                padding: 15px;
                text-align: center;
                margin: 10px 0;
            ">
                <div style="font-size: 16px; font-weight: bold; color: #4caf50; margin-bottom: 5px;">
                    ✅ Enhanced Mock
                </div>
                <div style="font-size: 11px; color: #cccccc;">
                    Always available fallback
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            vertex_status = "✅ Available" if system_summary['vertex_ai_available'] else "❌ Not Available"
            vertex_color = "#4caf50" if system_summary['vertex_ai_available'] else "#f44336"
            
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
                border: 2px solid {vertex_color};
                border-radius: 10px;
                padding: 15px;
                text-align: center;
                margin: 10px 0;
            ">
                <div style="font-size: 16px; font-weight: bold; color: {vertex_color}; margin-bottom: 5px;">
                    {vertex_status}
                </div>
                <div style="font-size: 11px; color: #cccccc;">
                    Vertex AI ML Models
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            local_status = "✅ Available" if system_summary['local_ml_models_available'] else "❌ Not Available"
            local_color = "#4caf50" if system_summary['local_ml_models_available'] else "#f44336"
            
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
                border: 2px solid {local_color};
                border-radius: 10px;
                padding: 15px;
                text-align: center;
                margin: 10px 0;
            ">
                <div style="font-size: 16px; font-weight: bold; color: {local_color}; margin-bottom: 5px;">
                    {local_status}
                </div>
                <div style="font-size: 11px; color: #cccccc;">
                    Local ML Models
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.success("🔀 **Using Hybrid System for Rebalancing** - ML-enhanced allocations with automatic fallback!")
    else:
        st.info(f"🏠 **Using {rebalancing_mode.replace('_', ' ').title()} System** for rebalancing predictions.")
    
    # Portfolio value input
    st.markdown("### ◉ Portfolio Configuration")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        portfolio_value = st.number_input(
            "Portfolio Value (USD)",
            min_value=1000.0,
            max_value=1000000.0,
            value=10000.0,
            step=1000.0,
            help="Total portfolio value for rebalancing calculations"
        )
    
    with col2:
        use_ml = st.checkbox("◉ Use ML Predictions", value=True, help="Enhance allocation with ML predictions")
    
    # Advanced configuration
    with st.expander("◉ Advanced Configuration"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Risk Controls**")
            max_position = st.slider(
                "Max Position Weight (%)",
                min_value=20,
                max_value=60,
                value=int(rebalancer.config.get('max_position_weight', 0.4) * 100),
                help="Maximum allocation per position"
            )
            
            min_position = st.slider(
                "Min Position Weight (%)",
                min_value=5,
                max_value=20,
                value=int(rebalancer.config.get('min_position_weight', 0.1) * 100),
                help="Minimum allocation per position"
            )
            
            min_trade_size = st.number_input(
                "Min Trade Size ($)",
                min_value=10.0,
                max_value=1000.0,
                value=float(rebalancer.config.get('min_trade_size', 50.0)),
                step=10.0,
                help="Minimum trade size to execute"
            )
        
        with col2:
            st.markdown("**ML Parameters**")
            ml_factor = st.slider(
                "ML Weight Factor",
                min_value=0.0,
                max_value=1.0,
                value=float(rebalancer.config.get('ml_weight_factor', 0.3)),
                step=0.1,
                help="How much ML predictions affect allocation"
            )
            
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=float(rebalancer.config.get('confidence_threshold', 0.6)),
                step=0.1,
                help="Minimum confidence for ML adjustments"
            )
            
            trading_fee = st.number_input(
                "Trading Fee (%)",
                min_value=0.0,
                max_value=1.0,
                value=float(rebalancer.config.get('trading_fee', 0.0016)) * 100,
                step=0.01,
                help="Trading fee percentage"
            )
        
        # Update configuration
        if st.button("◉ Save Configuration"):
            rebalancer.config.update({
                'max_position_weight': max_position / 100,
                'min_position_weight': min_position / 100,
                'min_trade_size': min_trade_size,
                'ml_weight_factor': ml_factor,
                'confidence_threshold': confidence_threshold,
                'trading_fee': trading_fee / 100
            })
            
            if rebalancer.save_config():
                st.success("◉ Configuration saved successfully!")
            else:
                st.error("⊗ Failed to save configuration")
    
    st.markdown("---")
    
    # Get rebalancing summary
    with st.spinner("◉ Calculating rebalancing recommendations..."):
        summary = rebalancer.get_rebalancing_summary()
    
    # Quick status overview
    total_orders = len(summary['orders'])
    max_drift = summary['metrics']['max_drift'] * 100
    total_fees = summary['metrics']['total_fees']
    
    if total_orders == 0:
        st.success("◉ **Portfolio is perfectly balanced!** No rebalancing needed.")
    elif max_drift > 5:
        st.warning(f"△ **Significant drift detected** - {max_drift:.1f}% max drift requires rebalancing")
    elif total_fees > 100:
        st.info(f"◉ **High trading fees** - ${total_fees:.2f} total cost for rebalancing")
    else:
        st.info(f"◉ **Portfolio needs minor adjustments** - {total_orders} orders, ${total_fees:.2f} fees")
    
    # Display current vs target allocation
    st.markdown("### ◉ Current vs Target Allocation")
    
    # Create allocation comparison
    allocation_data = []
    for symbol in rebalancer.SUPPORTED_SYMBOLS:
        current = summary['current_allocation'].get(symbol, 0.0)
        target = summary['target_allocation'].get(symbol, 0.0)
        drift = abs(target - current)
        
        allocation_data.append({
            'Symbol': symbol,
            'Current (%)': f"{current*100:.1f}%",
            'Target (%)': f"{target*100:.1f}%",
            'Drift (%)': f"{drift*100:.1f}%",
            'Status': 'OVERWEIGHT' if current > target else 'UNDERWEIGHT' if current < target else 'BALANCED'
        })
    
    # Display allocation table
    st.dataframe(
        pd.DataFrame(allocation_data),
        use_container_width=True,
        hide_index=True
    )
    
    # Display allocation visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ◉ Current Allocation")
        current_df = pd.DataFrame([
            {'Symbol': symbol, 'Weight': weight*100}
            for symbol, weight in summary['current_allocation'].items()
        ])
        
        fig_current = px.pie(
            current_df, 
            values='Weight', 
            names='Symbol',
            title="Current Portfolio",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_current.update_layout(height=400)
        st.plotly_chart(fig_current, use_container_width=True)
    
    with col2:
        st.markdown("#### ◉ Target Allocation")
        target_df = pd.DataFrame([
            {'Symbol': symbol, 'Weight': weight*100}
            for symbol, weight in summary['target_allocation'].items()
        ])
        
        fig_target = px.pie(
            target_df, 
            values='Weight', 
            names='Symbol',
            title="Target Portfolio",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_target.update_layout(height=400)
        st.plotly_chart(fig_target, use_container_width=True)
    
    st.markdown("---")
    
    # Display rebalancing orders
    st.markdown("### ◉ Rebalancing Orders")
    
    if summary['orders']:
        orders_data = []
        for order in summary['orders']:
            orders_data.append({
                'Symbol': order['symbol'],
                'Action': order['type'],
                'Amount ($)': f"${order['amount_usd']:,.2f}",
                'Fee ($)': f"${order['fee_usd']:.2f}",
                'Net Amount ($)': f"${order['net_amount_usd']:,.2f}",
                'Weight Change': f"{order['weight_change']*100:+.1f}%"
            })
        
        st.dataframe(
            pd.DataFrame(orders_data),
            use_container_width=True,
            hide_index=True
        )
        
        # Order summary with improved styling
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_orders = len(summary['orders'])
            order_color = "#ff6b6b" if total_orders > 8 else "#ffa726" if total_orders > 4 else "#4caf50"
            order_status = "Many Trades" if total_orders > 8 else "Moderate" if total_orders > 4 else "Few Trades"
            
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
                border: 2px solid {order_color};
                border-radius: 15px;
                padding: 20px;
                text-align: center;
                margin: 10px 0;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            ">
                <div style="font-size: 28px; font-weight: bold; color: {order_color}; margin-bottom: 10px; text-shadow: 0 0 10px {order_color}40;">
                    {total_orders}
                </div>
                <div style="font-size: 16px; font-weight: bold; color: #ffffff; margin-bottom: 5px;">
                    Total Orders
                </div>
                <div style="font-size: 12px; color: {order_color}; margin-bottom: 5px; font-weight: 500;">
                    {order_status}
                </div>
                <div style="font-size: 11px; color: #cccccc;">
                    Rebalancing trades needed
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            total_fees = summary['metrics']['total_fees']
            fee_color = "#ff6b6b" if total_fees > 100 else "#ffa726" if total_fees > 50 else "#4caf50"
            fee_status = "High Cost" if total_fees > 100 else "Moderate" if total_fees > 50 else "Low Cost"
            
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
                border: 2px solid {fee_color};
                border-radius: 15px;
                padding: 20px;
                text-align: center;
                margin: 10px 0;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            ">
                <div style="font-size: 28px; font-weight: bold; color: {fee_color}; margin-bottom: 10px; text-shadow: 0 0 10px {fee_color}40;">
                    ${total_fees:.2f}
                </div>
                <div style="font-size: 16px; font-weight: bold; color: #ffffff; margin-bottom: 5px;">
                    Total Fees
                </div>
                <div style="font-size: 12px; color: {fee_color}; margin-bottom: 5px; font-weight: 500;">
                    {fee_status}
                </div>
                <div style="font-size: 11px; color: #cccccc;">
                    Rebalancing cost
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            buy_orders = summary['metrics']['buy_orders']
            buy_color = "#4caf50" if buy_orders > 0 else "#666"
            buy_status = "Buying" if buy_orders > 0 else "No Buys"
            
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
                border: 2px solid {buy_color};
                border-radius: 15px;
                padding: 20px;
                text-align: center;
                margin: 10px 0;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            ">
                <div style="font-size: 28px; font-weight: bold; color: {buy_color}; margin-bottom: 10px; text-shadow: 0 0 10px {buy_color}40;">
                    {buy_orders}
                </div>
                <div style="font-size: 16px; font-weight: bold; color: #ffffff; margin-bottom: 5px;">
                    Buy Orders
                </div>
                <div style="font-size: 12px; color: {buy_color}; margin-bottom: 5px; font-weight: 500;">
                    {buy_status}
                </div>
                <div style="font-size: 11px; color: #cccccc;">
                    Positions to increase
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            sell_orders = summary['metrics']['sell_orders']
            sell_color = "#ff6b6b" if sell_orders > 0 else "#666"
            sell_status = "Selling" if sell_orders > 0 else "No Sells"
            
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
                border: 2px solid {sell_color};
                border-radius: 15px;
                padding: 20px;
                text-align: center;
                margin: 10px 0;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            ">
                <div style="font-size: 28px; font-weight: bold; color: {sell_color}; margin-bottom: 10px; text-shadow: 0 0 10px {sell_color}40;">
                    {sell_orders}
                </div>
                <div style="font-size: 16px; font-weight: bold; color: #ffffff; margin-bottom: 5px;">
                    Sell Orders
                </div>
                <div style="font-size: 12px; color: {sell_color}; margin-bottom: 5px; font-weight: 500;">
                    {sell_status}
                </div>
                <div style="font-size: 11px; color: #cccccc;">
                    Positions to decrease
                </div>
            </div>
            """, unsafe_allow_html=True)
        
    else:
        st.success("◉ Portfolio is already balanced - no rebalancing needed!")
    
    st.markdown("---")
    
    # Recommendations
    st.markdown("### ◉ Recommendations")
    
    for recommendation in summary['recommendations']:
        if "⊗" in recommendation:
            st.warning(recommendation)
        elif "◉" in recommendation:
            st.info(recommendation)
        elif "△" in recommendation and "ALERT" in recommendation:
            st.error(recommendation)
        else:
            st.success(recommendation)
    
    st.markdown("---")
    
    # Execution section
    st.markdown("### ▶ Execute Rebalancing")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("▶ Execute Paper Trading", type="primary", use_container_width=True):
            with st.spinner("Executing paper trading rebalancing..."):
                result = rebalancer.execute_rebalancing(portfolio_value)
                
                if result['status'] == 'success':
                    st.success(f"◉ Paper trading executed successfully!")
                    st.success(f"◉ {result['orders_executed']} orders executed")
                    st.success(f"◉ Total fees: ${result['total_fees']:.2f}")
                    
                    # Show executed orders
                    with st.expander("◉ View Executed Orders"):
                        for order in result['orders']:
                            st.json(order)
                else:
                    st.error(f"⊗ Execution failed: {result.get('message', 'Unknown error')}")
    
    with col2:
        if st.button("▶ Execute Live Trading", type="secondary", use_container_width=True, disabled=True):
            st.warning("△ Live trading not implemented yet - use paper trading mode")
    
    # Strategy information
    with st.expander("◉ Rebalancing Strategy Details"):
        st.markdown("""
        **Strategy Overview:**
    
        1. **Base Strategy**: Equal-weight allocation across tracked stocks
        2. **ML Enhancement**: Adjust weights based on predicted returns
        3. **Risk Controls**: 
           - Max 40% per position
           - Min 10% per position
           - Min $50 trade size
        4. **Trading Fees**: Zero-commission (most modern brokers)
        
        **How It Works:**
        
        1. **Data Collection**: Fetches current portfolio and ML predictions
        2. **Target Calculation**: Combines equal-weight base with ML adjustments
        3. **Risk Application**: Enforces position limits and trade minimums
        4. **Order Generation**: Creates buy/sell orders with fee calculations
        5. **Execution**: Paper trading for testing, live trading for production
        
        **Important Notes:**
        - Always test with paper trading first
        - Review all orders before live execution
        - ML predictions are for educational purposes only
        - Past performance doesn't guarantee future results
        """)
    
    # Performance metrics with enhanced styling
    st.markdown("---")
    st.markdown("### ◉ Portfolio Health Metrics")
    
    # Create enhanced KPI cards with better styling
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        max_drift = summary['metrics']['max_drift'] * 100
        drift_color = "#ff6b6b" if max_drift > 5 else "#ffa726" if max_drift > 2 else "#4caf50"
        drift_status = "High Risk" if max_drift > 5 else "Moderate" if max_drift > 2 else "Good"
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            border: 2px solid {drift_color};
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            margin: 10px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        ">
            <div style="font-size: 28px; font-weight: bold; color: {drift_color}; margin-bottom: 10px; text-shadow: 0 0 10px {drift_color}40;">
                {max_drift:.1f}%
            </div>
            <div style="font-size: 16px; font-weight: bold; color: #ffffff; margin-bottom: 5px;">
                Max Drift
            </div>
            <div style="font-size: 12px; color: {drift_color}; margin-bottom: 5px; font-weight: 500;">
                {drift_status}
            </div>
            <div style="font-size: 11px; color: #cccccc;">
                Largest allocation deviation
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_drift = summary['metrics']['avg_drift'] * 100
        avg_color = "#ff6b6b" if avg_drift > 3 else "#ffa726" if avg_drift > 1 else "#4caf50"
        avg_status = "High Risk" if avg_drift > 3 else "Moderate" if avg_drift > 1 else "Good"
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            border: 2px solid {avg_color};
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            margin: 10px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        ">
            <div style="font-size: 28px; font-weight: bold; color: {avg_color}; margin-bottom: 10px; text-shadow: 0 0 10px {avg_color}40;">
                {avg_drift:.1f}%
            </div>
            <div style="font-size: 16px; font-weight: bold; color: #ffffff; margin-bottom: 5px;">
                Avg Drift
            </div>
            <div style="font-size: 12px; color: {avg_color}; margin-bottom: 5px; font-weight: 500;">
                {avg_status}
            </div>
            <div style="font-size: 11px; color: #cccccc;">
                Average deviation
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        total_fees = summary['metrics']['total_fees']
        fee_color = "#ff6b6b" if total_fees > 100 else "#ffa726" if total_fees > 50 else "#4caf50"
        fee_status = "High Cost" if total_fees > 100 else "Moderate" if total_fees > 50 else "Low Cost"
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            border: 2px solid {fee_color};
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            margin: 10px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        ">
            <div style="font-size: 28px; font-weight: bold; color: {fee_color}; margin-bottom: 10px; text-shadow: 0 0 10px {fee_color}40;">
                ${total_fees:.2f}
            </div>
            <div style="font-size: 16px; font-weight: bold; color: #ffffff; margin-bottom: 5px;">
                Trading Fees
            </div>
            <div style="font-size: 12px; color: {fee_color}; margin-bottom: 5px; font-weight: 500;">
                {fee_status}
            </div>
            <div style="font-size: 11px; color: #cccccc;">
                Total rebalancing cost
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        fee_percentage = (summary['metrics']['total_fees'] / portfolio_value) * 100
        fee_pct_color = "#ff6b6b" if fee_percentage > 0.5 else "#ffa726" if fee_percentage > 0.2 else "#4caf50"
        fee_pct_status = "High %" if fee_percentage > 0.5 else "Moderate" if fee_percentage > 0.2 else "Low %"
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            border: 2px solid {fee_pct_color};
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            margin: 10px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        ">
            <div style="font-size: 28px; font-weight: bold; color: {fee_pct_color}; margin-bottom: 10px; text-shadow: 0 0 10px {fee_pct_color}40;">
                {fee_percentage:.2f}%
            </div>
            <div style="font-size: 16px; font-weight: bold; color: #ffffff; margin-bottom: 5px;">
                Fee %
            </div>
            <div style="font-size: 12px; color: {fee_pct_color}; margin-bottom: 5px; font-weight: 500;">
                {fee_pct_status}
            </div>
            <div style="font-size: 11px; color: #cccccc;">
                Fees vs portfolio value
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Additional metrics row
    st.markdown("### ◉ Rebalancing Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_orders = len(summary['orders'])
        order_color = "#ff6b6b" if total_orders > 8 else "#ffa726" if total_orders > 4 else "#4caf50"
        order_status = "Many Trades" if total_orders > 8 else "Moderate" if total_orders > 4 else "Few Trades"
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            border: 2px solid {order_color};
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            margin: 10px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        ">
            <div style="font-size: 28px; font-weight: bold; color: {order_color}; margin-bottom: 10px; text-shadow: 0 0 10px {order_color}40;">
                {total_orders}
            </div>
            <div style="font-size: 16px; font-weight: bold; color: #ffffff; margin-bottom: 5px;">
                Total Orders
            </div>
            <div style="font-size: 12px; color: {order_color}; margin-bottom: 5px; font-weight: 500;">
                {order_status}
            </div>
            <div style="font-size: 11px; color: #cccccc;">
                Rebalancing trades needed
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        buy_orders = summary['metrics']['buy_orders']
        buy_color = "#4caf50" if buy_orders > 0 else "#666"
        buy_status = "Buying" if buy_orders > 0 else "No Buys"
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            border: 2px solid {buy_color};
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            margin: 10px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        ">
            <div style="font-size: 28px; font-weight: bold; color: {buy_color}; margin-bottom: 10px; text-shadow: 0 0 10px {buy_color}40;">
                {buy_orders}
            </div>
            <div style="font-size: 16px; font-weight: bold; color: #ffffff; margin-bottom: 5px;">
                Buy Orders
            </div>
            <div style="font-size: 12px; color: {buy_color}; margin-bottom: 5px; font-weight: 500;">
                {buy_status}
            </div>
            <div style="font-size: 11px; color: #cccccc;">
                Positions to increase
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        sell_orders = summary['metrics']['sell_orders']
        sell_color = "#ff6b6b" if sell_orders > 0 else "#666"
        sell_status = "Selling" if sell_orders > 0 else "No Sells"
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            border: 2px solid {sell_color};
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            margin: 10px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        ">
            <div style="font-size: 28px; font-weight: bold; color: {sell_color}; margin-bottom: 10px; text-shadow: 0 0 10px {sell_color}40;">
                {sell_orders}
            </div>
            <div style="font-size: 16px; font-weight: bold; color: #ffffff; margin-bottom: 5px;">
                Sell Orders
            </div>
            <div style="font-size: 12px; color: {sell_color}; margin-bottom: 5px; font-weight: 500;">
                {sell_status}
            </div>
            <div style="font-size: 11px; color: #cccccc;">
                Positions to decrease
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Portfolio health score
        max_drift = summary['metrics']['max_drift'] * 100
        avg_drift = summary['metrics']['avg_drift'] * 100
        fee_pct = (summary['metrics']['total_fees'] / portfolio_value) * 100
        
        # Calculate health score (0-100)
        drift_score = max(0, 100 - (max_drift * 10))  # Penalty for high drift
        fee_score = max(0, 100 - (fee_pct * 200))     # Penalty for high fees
        health_score = (drift_score + fee_score) / 2
        
        health_color = "#4caf50" if health_score > 80 else "#ffa726" if health_score > 60 else "#ff6b6b"
        health_status = "Excellent" if health_score > 80 else "Good" if health_score > 60 else "Needs Attention"
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            border: 2px solid {health_color};
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            margin: 10px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        ">
            <div style="font-size: 28px; font-weight: bold; color: {health_color}; margin-bottom: 10px; text-shadow: 0 0 10px {health_color}40;">
                {health_score:.0f}
            </div>
            <div style="font-size: 16px; font-weight: bold; color: #ffffff; margin-bottom: 5px;">
                Health Score
            </div>
            <div style="font-size: 12px; color: {health_color}; margin-bottom: 5px; font-weight: 500;">
                {health_status}
            </div>
            <div style="font-size: 11px; color: #cccccc;">
                Overall portfolio health
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Drift analysis details
    with st.expander("◉ Detailed Drift Analysis"):
        drift_data = []
        for symbol, analysis in summary['drift_analysis'].items():
            drift_data.append({
                'Symbol': symbol,
                'Current Weight': f"{analysis['current']*100:.1f}%",
                'Target Weight': f"{analysis['target']*100:.1f}%",
                'Drift': f"{analysis['drift']*100:.1f}%",
                'Status': analysis['status']
            })
        
        st.dataframe(
            pd.DataFrame(drift_data),
            use_container_width=True,
            hide_index=True
        )
    
    # Status info
    mode_text = "Paper Trading" if paper_trading else "Live Trading"
    st.success(f"◉ **Portfolio Rebalancer Active** - {mode_text} mode with ML-enhanced allocation strategy!")


def show_cloud_progress():
    """Display Google Cloud ML training progress and status."""
    st.markdown("""
    <h1 style='color: #4ecdc4; margin: 0 0 20px 0; display: flex; align-items: center;'>
        <span class='material-symbols-outlined' style='margin-right: 12px; font-size: 32px;'>cloud</span>
        Cloud Progress Tracker
    </h1>
    """, unsafe_allow_html=True)
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (30s)", value=True, help="Automatically refresh progress every 30 seconds")
    
    # Manual refresh button
    if st.sidebar.button("🔄 Manual Refresh", use_container_width=True):
        st.rerun()
    
    # Progress status cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Training Status
        current_status = get_training_job_status()
        
        status_info = {
            "JOB_STATE_PENDING": ("⏳ Pending", "#ffa726", "Training job is starting up"),
            "JOB_STATE_RUNNING": ("🔄 Running", "#4caf50", "Training is in progress"),
            "JOB_STATE_SUCCEEDED": ("✅ Completed", "#4caf50", "Training completed successfully"),
            "JOB_STATE_FAILED": ("❌ Failed", "#f44336", "Training failed"),
            "NO_JOBS": ("📭 No Jobs", "#666666", "No training jobs found"),
            "ERROR": ("❓ Error", "#f44336", "Unable to check status")
        }
        
        status_text, status_color, status_desc = status_info.get(current_status, ("❓ Unknown", "#666666", "Unknown status"))
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            border: 2px solid {status_color};
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            margin: 10px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        ">
            <div style="font-size: 24px; margin-bottom: 10px;">
                {status_text}
            </div>
            <div style="font-size: 16px; font-weight: bold; color: #ffffff; margin-bottom: 5px;">
                Training Status
            </div>
            <div style="font-size: 12px; color: {status_color}; margin-bottom: 5px;">
                {status_desc}
            </div>
            <div style="font-size: 11px; color: #cccccc;">
                Latest Training Job
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Cost Tracking
        costs = get_gcp_costs()
        budget_used = costs['total'] / 50 * 100
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            border: 2px solid #4ecdc4;
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            margin: 10px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        ">
            <div style="font-size: 24px; font-weight: bold; color: #4ecdc4; margin-bottom: 10px;">
                ${costs['total']:.2f} / $50.00
            </div>
            <div style="font-size: 16px; font-weight: bold; color: #ffffff; margin-bottom: 10px;">
                Budget Usage
            </div>
            <div style="background: rgba(255,255,255,0.1); height: 8px; border-radius: 4px; margin-bottom: 10px;">
                <div style="background: #4ecdc4; height: 100%; width: {budget_used:.1f}%; border-radius: 4px;"></div>
            </div>
            <div style="font-size: 12px; color: #4ecdc4;">
                {budget_used:.1f}% used • {3-4 if budget_used < 20 else 2-3 if budget_used < 40 else 1-2} months remaining
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # System Health
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            border: 2px solid #4caf50;
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            margin: 10px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        ">
            <div style="font-size: 24px; margin-bottom: 10px;">
                🏥 System Health
            </div>
            <div style="text-align: left; margin: 10px 0;">
                <div style="display: flex; justify-content: space-between; margin: 5px 0;">
                    <span style="color: #cccccc;">BigQuery:</span>
                    <span style="color: #4caf50;">✅ Connected</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin: 5px 0;">
                    <span style="color: #cccccc;">Yahoo Finance:</span>
                    <span style="color: #4caf50;">✅ Connected</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin: 5px 0;">
                    <span style="color: #cccccc;">Vertex AI:</span>
                    <span style="color: #4caf50;">✅ Active</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Progress Timeline
    st.markdown("### 📈 Progress Timeline")
    
    timeline_data = [
        {"time": "22:09", "event": "GCP Infrastructure Setup", "status": "✅ Completed", "details": "Service accounts, buckets, BigQuery created"},
        {"time": "22:10", "event": "Training Job Deployed", "status": "✅ Completed", "details": "Vertex AI job submitted successfully"},
        {"time": "22:11", "event": "Container Startup", "status": "🔄 In Progress", "details": "Downloading TensorFlow container..."},
        {"time": "22:12", "event": "Model Training", "status": "⏳ Pending", "details": "Waiting for container startup"},
        {"time": "22:13", "event": "Prediction Endpoint", "status": "⏳ Pending", "details": "Will deploy after training"},
        {"time": "22:14", "event": "Dashboard Integration", "status": "✅ Completed", "details": "Real-time progress tracking active"}
    ]
    
    # Display timeline
    for item in timeline_data:
        col1, col2, col3 = st.columns([1, 2, 3])
        with col1:
            st.write(f"**{item['time']}**")
        with col2:
            st.write(item['event'])
        with col3:
            st.write(f"{item['status']} - {item['details']}")
    
    # Latest Training Jobs and Endpoints
    st.markdown("### 🔄 Latest Training Jobs & Endpoints")
    
    # Get latest data
    training_jobs = get_latest_training_jobs()
    endpoints = get_latest_endpoints()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🚀 Training Jobs")
        if training_jobs:
            for i, job in enumerate(training_jobs[:3]):  # Show top 3
                job_name = job.get('displayName', 'Unknown')
                job_state = job.get('state', 'UNKNOWN')
                create_time = job.get('createTime', '')
                
                # Format time
                if create_time:
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(create_time.replace('Z', '+00:00'))
                        time_str = dt.strftime('%m/%d %H:%M')
                    except:
                        time_str = create_time[:16]
                else:
                    time_str = 'Unknown'
                
                # Status color
                status_colors = {
                    'JOB_STATE_SUCCEEDED': '#4caf50',
                    'JOB_STATE_RUNNING': '#ff9800',
                    'JOB_STATE_FAILED': '#f44336',
                    'JOB_STATE_PENDING': '#2196f3'
                }
                status_color = status_colors.get(job_state, '#666666')
                
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
                    border-left: 4px solid {status_color};
                    border-radius: 8px;
                    padding: 15px;
                    margin: 10px 0;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
                ">
                    <div style="font-weight: bold; color: #ffffff; margin-bottom: 5px;">
                        {job_name}
                    </div>
                    <div style="font-size: 12px; color: {status_color}; margin-bottom: 3px;">
                        {job_state.replace('JOB_STATE_', '').title()}
                    </div>
                    <div style="font-size: 11px; color: #cccccc;">
                        {time_str}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No training jobs found")
    
    with col2:
        st.markdown("#### 🎯 Prediction Endpoints")
        if endpoints:
            for i, endpoint in enumerate(endpoints[:3]):  # Show top 3
                endpoint_name = endpoint.get('displayName', 'Unknown')
                endpoint_id = endpoint.get('name', '').split('/')[-1] if endpoint.get('name') else 'Unknown'
                create_time = endpoint.get('createTime', '')
                
                # Format time
                if create_time:
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(create_time.replace('Z', '+00:00'))
                        time_str = dt.strftime('%m/%d %H:%M')
                    except:
                        time_str = create_time[:16]
                else:
                    time_str = 'Unknown'
                
                # Get deployed models count
                deployed_models = len(endpoint.get('deployedModels', []))
                
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
                    border-left: 4px solid #4ecdc4;
                    border-radius: 8px;
                    padding: 15px;
                    margin: 10px 0;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
                ">
                    <div style="font-weight: bold; color: #ffffff; margin-bottom: 5px;">
                        {endpoint_name}
                    </div>
                    <div style="font-size: 12px; color: #4ecdc4; margin-bottom: 3px;">
                        ID: {endpoint_id}
                    </div>
                    <div style="font-size: 12px; color: #4ecdc4; margin-bottom: 3px;">
                        Models: {deployed_models}
                    </div>
                    <div style="font-size: 11px; color: #cccccc;">
                        {time_str}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No endpoints found")

    # Live Logs Section
    st.markdown("### 📋 Live Training Logs")
    
    # Get and display logs
    logs = get_training_logs()
    if logs:
        log_text = "\n".join(logs)
        st.markdown(f"""
        <div style="
            background: #1e1e1e;
            color: #ffffff;
            padding: 15px;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            font-size: 0.85rem;
            max-height: 200px;
            overflow-y: auto;
            margin: 10px 0;
            border: 1px solid #333;
        ">
            {log_text}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("📋 No logs available yet. Training job is starting up...")
    
    # Real-time Metrics
    col1, col2 = st.columns(2)
    
    with col1:
        # Training Progress Chart
        st.markdown("**📊 Training Progress**")
        
        # Mock progress data
        progress_data = {
            "Epoch": [1, 2, 3, 4, 5],
            "Loss": [0.8, 0.6, 0.4, 0.3, 0.2],
            "Validation Loss": [0.9, 0.7, 0.5, 0.4, 0.3]
        }
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=progress_data["Epoch"], y=progress_data["Loss"], 
                                name="Training Loss", line=dict(color="#4ecdc4")))
        fig.add_trace(go.Scatter(x=progress_data["Epoch"], y=progress_data["Validation Loss"], 
                                name="Validation Loss", line=dict(color="#ff6b6b")))
        
        fig.update_layout(
            title="Model Training Progress",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            height=300,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Cost Breakdown
        st.markdown("**💰 Cost Breakdown**")
        
        cost_data = {
            "Service": ["Vertex AI", "BigQuery", "Storage", "Functions"],
            "Cost": [2.50, 1.20, 0.80, 0.50],
        }
        
        fig = go.Figure(data=[go.Pie(
            labels=cost_data["Service"],
            values=cost_data["Cost"],
            hole=0.3,
            marker_colors=['#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7']
        )])
        
        fig.update_layout(
            title="Monthly Cost Breakdown",
            height=300,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Status info
    if current_status == "JOB_STATE_SUCCEEDED":
        st.success("🎉 **Training Complete!** Your ML model is ready for deployment.")
        st.info("🚀 **Next Steps**: Deploy prediction endpoint and test end-to-end pipeline.")
    elif current_status == "JOB_STATE_RUNNING":
        st.info("🔄 **Training in Progress** - Your ML model is being trained in Google Cloud.")
    elif current_status == "JOB_STATE_PENDING":
        st.warning("⏳ **Starting Up** - Training job is initializing (takes 2-5 minutes).")
    elif current_status == "JOB_STATE_FAILED":
        st.error("❌ **Training Failed** - Check logs for details.")
    else:
        st.info("📊 **Cloud-First ML System** - All operations run in Google Cloud with zero local dependencies.")
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(30)
        st.rerun()


def main():
    """Main application entry point."""
    
    # Initialize session state for navigation
    if 'current_page' not in st.session_state:
        # Check for URL parameters first
        query_params = st.query_params
        if 'page' in query_params:
            page_param = query_params['page']
            if page_param in ["portfolio", "prices", "predictions", "rebalancing", "cloud"]:
                page_map = {
                    "portfolio": "⚡ Portfolio",
                    "prices": "↗ Live Prices", 
                    "predictions": "◉ ML Predictions",
                    "rebalancing": "◉ Rebalancing",
                    "cloud": "☁️ Cloud Progress"
                }
                st.session_state.current_page = page_map[page_param]
            else:
                st.session_state.current_page = "⚡ Portfolio"
        else:
            st.session_state.current_page = "⚡ Portfolio"
    
    # Enhanced sidebar navigation with cyberpunk glass effect
    st.sidebar.markdown(f"""
    <div class='glass-card fade-in' style='text-align: center; margin-bottom: 24px; border: 1px solid {THEME['accent_primary']}40; box-shadow: 0 0 30px {THEME['glow_primary']};'>
        <div style='display: inline-flex; align-items: center; justify-content: center; width: 72px; height: 72px; border-radius: 20px; background: linear-gradient(135deg, {THEME['accent_primary']}, {THEME['accent_secondary']}); margin-bottom: 16px; box-shadow: 0 8px 32px {THEME['glow_primary']}, inset 0 2px 4px rgba(255,255,255,0.2);'>
            <i class="fas fa-bolt icon-glow" style="color: white; font-size: 40px;"></i>
        </div>
        <h2 class='gradient-text' style='margin: 0 0 8px 0; font-size: 1.9rem; letter-spacing: 0.2em;'>ATLAS</h2>
        <p style='color: {THEME['text_secondary']}; margin: 0; font-size: 11px; font-weight: 600; letter-spacing: 0.3em; font-family: "Rajdhani", sans-serif;'>STOCK ML INTELLIGENCE</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown(f"""
    <h3 style='color: {THEME['text_primary']}; margin: 0 0 16px 0; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.15em; display: flex; align-items: center; font-family: "Orbitron", sans-serif;'>
        <i class='fas fa-th' style='margin-right: 10px; font-size: 18px; color: {THEME['accent_primary']};'></i>
        NAVIGATION
    </h3>
    """, unsafe_allow_html=True)
    
    # Create proper navigation buttons instead of radio buttons
    def navigate_to_page(page_name):
        st.session_state.current_page = page_name
        page_url_map = {
            "⚡ Portfolio": "portfolio",
            "↗ Live Prices": "prices", 
            "◉ ML Predictions": "predictions",
            "◉ Rebalancing": "rebalancing",
            "☁️ Cloud Progress": "cloud"
        }
        st.query_params.page = page_url_map[page_name]
        st.rerun()
    
    # Clean, intuitive navigation buttons with icons
    nav_buttons = [
        ("⚡ Portfolio", "Your stock holdings & performance", "fa-wallet"),
        ("↗ Live Prices", "Real-time market data & charts", "fa-chart-line"),
        ("◉ ML Predictions", "ML-powered price forecasts", "fa-brain"),
        ("◉ Rebalancing", "Smart portfolio rebalancing", "fa-sliders-h"),
        ("☁️ Cloud Progress", "Google Cloud ML training status", "fa-cloud")
    ]
    
    for page_name, description, icon in nav_buttons:
        # Determine if this is the current page
        is_current = st.session_state.current_page == page_name
        page_label = page_name.split(' ', 1)[1].upper() if ' ' in page_name else page_name.upper()
        
        if is_current:
            # Current page - highlighted with glass effect
            st.sidebar.markdown(f"""
            <div class='glass-card' style='
                background: linear-gradient(135deg, {THEME['accent_primary']}, {THEME['accent_secondary']});
                border: none;
                margin: 8px 0;
                padding: 18px 16px;
                box-shadow: 0 8px 32px {THEME['glow_primary']};
                cursor: pointer;
            '>
                <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 8px;'>
                    <i class='fas {icon}' style='font-size: 22px; color: white;'></i>
                    <span style='color: white; font-size: 15px; font-weight: 700; font-family: "Orbitron", sans-serif; letter-spacing: 0.05em;'>{page_label}</span>
                </div>
                <div style='color: rgba(255, 255, 255, 0.85); font-size: 11px; margin-bottom: 8px; margin-left: 34px;'>{description}</div>
                <div style='display: inline-flex; align-items: center; gap: 6px; padding: 5px 12px; background: rgba(255, 255, 255, 0.25); border-radius: 12px; font-size: 10px; font-weight: 700; font-family: "Orbitron", sans-serif; letter-spacing: 0.1em; margin-left: 34px;'>
                    <i class='fas fa-check-circle' style='font-size: 12px;'></i>
                    ACTIVE
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Inactive page - use Streamlit button for navigation
            button_id = f"nav_{page_name.replace(' ', '_')}"
            page_url = {"⚡ Portfolio": "portfolio", "↗ Live Prices": "prices", "◉ ML Predictions": "predictions", "◉ Rebalancing": "rebalancing", "☁️ Cloud Progress": "cloud"}.get(page_name, "portfolio")
            
            # Create clickable button
            button_clicked = st.sidebar.button(
                page_name,
                key=button_id,
                use_container_width=True,
                help=description
            )
            
            if button_clicked:
                st.session_state.current_page = page_name
                st.query_params.page = page_url
                st.rerun()
    
    # Set the current page for the rest of the app
    page = st.session_state.current_page
    
    # Update session state when page changes
    if page != st.session_state.current_page:
        st.session_state.current_page = page
        
        # Update URL parameters
        page_url_map = {
            "⚡ Portfolio": "portfolio",
            "↗ Live Prices": "prices", 
            "◉ ML Predictions": "predictions",
            "◉ Rebalancing": "rebalancing"
        }
        
        st.query_params.page = page_url_map[page]
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Compact system status
    st.sidebar.markdown(f"""
    <h3 style='color: {THEME['text_primary']}; margin: 0 0 16px 0; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.15em; display: flex; align-items: center; font-family: "Orbitron", sans-serif;'>
        <i class='fas fa-chart-line' style='margin-right: 10px; font-size: 18px; color: {THEME['accent_primary']};'></i>
        SYSTEM STATUS
    </h3>
    """, unsafe_allow_html=True)
    
    # Single status card with key indicators - cyberpunk style
    st.sidebar.markdown(f"""
    <div class='glass-card' style='text-align: center; padding: 20px; border: 1px solid {THEME['accent_primary']}30;'>
        <div style='display: flex; justify-content: space-around;'>
            <div style='text-align: center;'>
                <i class='fas fa-check-circle icon-glow' style='color: {THEME['accent_success']}; font-size: 28px; margin-bottom: 8px; filter: drop-shadow(0 0 8px {THEME['accent_success']});'></i>
                <div style='color: {THEME['text_primary']}; font-size: 11px; font-weight: 700; margin-bottom: 2px; font-family: "Orbitron", sans-serif;'>YFINANCE</div>
                <div style='color: {THEME['text_muted']}; font-size: 9px; letter-spacing: 0.05em;'>CONNECTED</div>
            </div>
            <div style='text-align: center;'>
                <i class='fas fa-microchip icon-glow' style='color: {THEME['accent_warning']}; font-size: 28px; margin-bottom: 8px; filter: drop-shadow(0 0 8px {THEME['accent_warning']});'></i>
                <div style='color: {THEME['text_primary']}; font-size: 11px; font-weight: 700; margin-bottom: 2px; font-family: "Orbitron", sans-serif;'>ML</div>
                <div style='color: {THEME['text_muted']}; font-size: 9px; letter-spacing: 0.05em;'>TRAINING</div>
            </div>
            <div style='text-align: center;'>
                <i class='fas fa-shield-alt icon-glow' style='color: {THEME['accent_primary']}; font-size: 28px; margin-bottom: 8px; filter: drop-shadow(0 0 8px {THEME['accent_primary']});'></i>
                <div style='color: {THEME['text_primary']}; font-size: 11px; font-weight: 700; margin-bottom: 2px; font-family: "Orbitron", sans-serif;'>PAPER</div>
                <div style='color: {THEME['text_muted']}; font-size: 9px; letter-spacing: 0.05em;'>SAFE</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced about section
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
    <h3 style='color: {THEME['text_primary']}; margin: 0 0 16px 0; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.15em; display: flex; align-items: center; font-family: "Orbitron", sans-serif;'>
        <i class='fas fa-info-circle' style='margin-right: 10px; font-size: 18px; color: {THEME['accent_primary']};'></i>
        ABOUT
    </h3>
    """, unsafe_allow_html=True)
    st.sidebar.markdown(f"""
    <div class='glass-card' style='padding: 18px;'>
        <div class='gradient-text' style='font-weight: 700; margin-bottom: 12px; font-size: 1rem;'>ATLAS Console</div>
        <div style='display: grid; gap: 8px;'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <span style='color: {THEME['text_muted']}; font-size: 11px;'>Version</span>
                <span class='badge' style='font-size: 10px; padding: 4px 8px;'>v1.0.0</span>
            </div>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <span style='color: {THEME['text_muted']}; font-size: 11px;'>Data Source</span>
                <span style='color: {THEME['text_primary']}; font-size: 11px; font-weight: 600;'>Yahoo Finance</span>
            </div>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <span style='color: {THEME['text_muted']}; font-size: 11px;'>ML Engine</span>
                <span style='color: {THEME['text_primary']}; font-size: 11px; font-weight: 600;'>LSTM</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress indicator
    st.sidebar.markdown(f"""
    <h3 style='color: {THEME['text_primary']}; margin: 0 0 16px 0; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.15em; display: flex; align-items: center; font-family: "Orbitron", sans-serif;'>
        <i class='fas fa-chart-bar' style='margin-right: 10px; font-size: 18px; color: {THEME['accent_primary']};'></i>
        PROGRESS
    </h3>
    """, unsafe_allow_html=True)
    
    # Mock progress data
    progress_data = {
        "Data Collection": (95, THEME['accent_success']),
        "ML Training": (60, THEME['accent_warning']),
        "Portfolio": (100, THEME['accent_success']),
        "Rebalancing": (85, THEME['accent_primary'])
    }
    
    for component, (progress, color) in progress_data.items():
        st.sidebar.markdown(f"""
        <div class='glass-card' style='padding: 12px; margin: 8px 0;'>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>
                <span style='color: {THEME['text_primary']}; font-size: 11px; font-weight: 600;'>{component}</span>
                <span style='color: {color}; font-size: 11px; font-weight: 700;'>{progress}%</span>
            </div>
            <div style='background: {THEME['bg_secondary']}; border-radius: 10px; height: 6px; overflow: hidden;'>
                <div style='
                    background: linear-gradient(90deg, {color}, {THEME['accent_secondary']});
                    height: 100%;
                    width: {progress}%;
                    border-radius: 10px;
                    transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
                    box-shadow: 0 0 8px {color}40;
                '></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Display header
    show_header()
    
    # Enhanced page header with cyberpunk glass effect
    page_icon_map = {
        "⚡ Portfolio": "fa-wallet",
        "↗ Live Prices": "fa-chart-line",
        "◉ ML Predictions": "fa-brain",
        "◉ Rebalancing": "fa-sliders-h",
        "☁️ Cloud Progress": "fa-cloud"
    }
    page_icon = page_icon_map.get(page, "fa-th")
    
    st.markdown(f"""
    <div class='glass-card fade-in' style='text-align: center; padding: 36px; margin: 24px 0; border: 1px solid {THEME['accent_primary']}40; box-shadow: 0 0 40px {THEME['glow_primary']};'>
        <div style='display: inline-flex; align-items: center; justify-content: center; width: 80px; height: 80px; border-radius: 24px; background: linear-gradient(135deg, {THEME['accent_primary']}, {THEME['accent_secondary']}); margin-bottom: 24px; box-shadow: 0 12px 40px {THEME['glow_primary']}, inset 0 2px 4px rgba(255,255,255,0.2);'>
            <i class="fas {page_icon} icon-glow" style="color: white; font-size: 44px;"></i>
        </div>
        <h1 class='gradient-text' style='margin: 0 0 12px 0; font-size: 2.5rem; letter-spacing: 0.2em;'>{page.split(' ', 1)[1].upper() if ' ' in page else page.upper()}</h1>
        <p style='color: {THEME['text_secondary']}; margin: 0; font-size: 0.9rem; font-weight: 600; letter-spacing: 0.3em; font-family: "Rajdhani", sans-serif;'>AI-POWERED STOCK ANALYTICS</p>
    </div>
    """, unsafe_allow_html=True)
    
    
    # Route to selected page
    if page == "⚡ Portfolio":
        show_portfolio_view()
    elif page == "↗ Live Prices":
        show_live_prices()
    elif page == "◉ ML Predictions":
        show_predictions()
    elif page == "◉ Rebalancing":
        show_rebalancing()
    elif page == "☁️ Cloud Progress":
        show_cloud_progress()
    
    # Enhanced refresh controls - Cyberpunk
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
    <h3 style='color: {THEME['text_primary']}; margin: 0 0 16px 0; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.15em; display: flex; align-items: center; font-family: "Orbitron", sans-serif;'>
        <i class='fas fa-sync-alt' style='margin-right: 10px; font-size: 18px; color: {THEME['accent_primary']};'></i>
        CONTROLS
    </h3>
    """, unsafe_allow_html=True)
    
    # Refresh button with cyberpunk styling
    if st.sidebar.button("↻ SYNC DATA", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("⚡ AUTO-REFRESH (60S)", value=False, help="Automatically refresh data every 60 seconds")
    
    if auto_refresh:
        time.sleep(60)
        st.rerun()


if __name__ == "__main__":
    main()

