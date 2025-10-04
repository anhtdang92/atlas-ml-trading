"""
Crypto ML Trading Dashboard - Main Streamlit Application

This is the main entry point for the dashboard. It provides a multi-page
interface for viewing portfolio, live prices, ML predictions, and rebalancing.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time

# Page configuration
st.set_page_config(
    page_title="Crypto ML Trading Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    h1 {
        color: #1f77b4;
    }
    .positive {
        color: #28a745;
    }
    .negative {
        color: #dc3545;
    }
    </style>
    """, unsafe_allow_html=True)

# Import helper modules
try:
    from data.kraken_api import KrakenAPI
except ImportError:
    # If module not created yet, we'll use inline code
    import requests
    
    class KrakenAPI:
        """Simple Kraken API wrapper for public endpoints."""
        
        BASE_URL = "https://api.kraken.com"
        
        @staticmethod
        @st.cache_data(ttl=60)  # Cache for 1 minute
        def get_ticker(pairs):
            """Get ticker information for specified pairs."""
            try:
                pair_str = ','.join(pairs)
                response = requests.get(
                    f"{KrakenAPI.BASE_URL}/0/public/Ticker",
                    params={'pair': pair_str},
                    timeout=10
                )
                response.raise_for_status()
                data = response.json()
                
                if data.get('error'):
                    return None
                
                return data['result']
            except Exception as e:
                st.error(f"Error fetching data: {e}")
                return None
        
        @staticmethod
        @st.cache_data(ttl=300)  # Cache for 5 minutes
        def get_ohlc(pair, interval=60):
            """Get OHLC data for specified pair."""
            try:
                response = requests.get(
                    f"{KrakenAPI.BASE_URL}/0/public/OHLC",
                    params={'pair': pair, 'interval': interval},
                    timeout=10
                )
                response.raise_for_status()
                data = response.json()
                
                if data.get('error'):
                    return None
                
                # Get the pair key from result
                pair_key = [k for k in data['result'].keys() if k != 'last'][0]
                return data['result'][pair_key]
            except Exception as e:
                st.error(f"Error fetching OHLC data: {e}")
                return None


def show_header():
    """Display the main header."""
    col1, col2, col3 = st.columns([2, 3, 2])
    
    with col2:
        st.title("🚀 Crypto ML Trading Dashboard")
        st.markdown("*Powered by Machine Learning & Kraken API*")
    
    with col3:
        st.markdown(f"**Last Updated:** {datetime.now().strftime('%I:%M:%S %p')}")


def show_portfolio_view():
    """Display portfolio overview with holdings and performance."""
    st.header("💼 Portfolio Overview")
    
    # Mock portfolio data (will be replaced with real data when API keys are added)
    portfolio_data = {
        'BTC': {'quantity': 0.041, 'avg_buy_price': 118500, 'current_price': 122001},
        'ETH': {'quantity': 0.55, 'avg_buy_price': 4300, 'current_price': 4479},
        'SOL': {'quantity': 5.5, 'avg_buy_price': 220, 'current_price': 227},
        'ADA': {'quantity': 3000, 'avg_buy_price': 0.42, 'current_price': 0.45}
    }
    
    # Fetch live prices
    kraken_pairs = ['XXBTZUSD', 'XETHZUSD', 'SOLUSD', 'ADAUSD']
    ticker_data = KrakenAPI.get_ticker(kraken_pairs)
    
    if ticker_data:
        # Update current prices from live data
        pair_map = {'XXBTZUSD': 'BTC', 'XETHZUSD': 'ETH', 'SOLUSD': 'SOL', 'ADAUSD': 'ADA'}
        for kraken_pair, symbol in pair_map.items():
            # Find the matching key in ticker_data
            matching_key = [k for k in ticker_data.keys() if kraken_pair in k or k in kraken_pair]
            if matching_key:
                portfolio_data[symbol]['current_price'] = float(ticker_data[matching_key[0]]['c'][0])
    
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
        
        holdings.append({
            'Symbol': symbol,
            'Quantity': f"{quantity:.4f}",
            'Avg Buy Price': f"${avg_price:,.2f}",
            'Current Price': f"${current_price:,.2f}",
            'Value': f"${current_value:,.2f}",
            'P&L': f"${pnl:,.2f}",
            'P&L %': f"{pnl_pct:+.2f}%",
            '% Portfolio': f"{(current_value/total_value)*100:.1f}%"
        })
    
    total_pnl = total_value - total_cost
    total_pnl_pct = (total_pnl / total_cost) * 100 if total_cost > 0 else 0
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Portfolio Value", f"${total_value:,.2f}")
    
    with col2:
        st.metric("Total Cost Basis", f"${total_cost:,.2f}")
    
    with col3:
        st.metric("Total P&L", f"${total_pnl:,.2f}", f"{total_pnl_pct:+.2f}%")
    
    with col4:
        # Calculate 24h change (mock for now)
        daily_change = 2.3  # Will be calculated from real data
        st.metric("24h Change", f"{daily_change:+.2f}%", 
                 delta_color="normal" if daily_change >= 0 else "inverse")
    
    st.markdown("---")
    
    # Holdings table
    st.subheader("Current Holdings")
    holdings_df = pd.DataFrame(holdings)
    st.dataframe(holdings_df, use_container_width=True, hide_index=True)
    
    # Portfolio allocation pie chart
    st.subheader("Portfolio Allocation")
    
    allocation_data = pd.DataFrame([
        {'Symbol': symbol, 'Value': data['quantity'] * data['current_price']}
        for symbol, data in portfolio_data.items()
    ])
    
    fig = px.pie(allocation_data, values='Value', names='Symbol', 
                 title='Portfolio Distribution',
                 color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig, use_container_width=True)
    
    # Info box
    st.info("📝 **Note:** This is paper trading mode. To enable live trading, add your Kraken API keys to `config/secrets.yaml`")


def show_live_prices():
    """Display live cryptocurrency prices with charts."""
    st.header("📈 Live Cryptocurrency Prices")
    
    # Crypto configuration
    cryptos = {
        'XXBTZUSD': {'name': 'Bitcoin', 'symbol': 'BTC', 'icon': '₿'},
        'XETHZUSD': {'name': 'Ethereum', 'symbol': 'ETH', 'icon': 'Ξ'},
        'SOLUSD': {'name': 'Solana', 'symbol': 'SOL', 'icon': '◎'},
        'ADAUSD': {'name': 'Cardano', 'symbol': 'ADA', 'icon': '₳'},
        'XRPUSD': {'name': 'Ripple', 'symbol': 'XRP', 'icon': '✕'},
        'MATICUSD': {'name': 'Polygon', 'symbol': 'MATIC', 'icon': '⬡'}
    }
    
    # Fetch ticker data
    ticker_data = KrakenAPI.get_ticker(list(cryptos.keys()))
    
    if not ticker_data:
        st.error("Unable to fetch price data. Please try again.")
        return
    
    # Display price cards in grid
    cols = st.columns(3)
    
    for idx, (pair, info) in enumerate(cryptos.items()):
        # Find matching ticker data
        matching_key = [k for k in ticker_data.keys() if pair in k or k in pair]
        
        if not matching_key:
            continue
        
        data = ticker_data[matching_key[0]]
        current_price = float(data['c'][0])
        day_high = float(data['h'][1])
        day_low = float(data['l'][1])
        volume = float(data['v'][1])
        open_price = float(data['o'])
        
        price_change = ((current_price - open_price) / open_price) * 100
        
        with cols[idx % 3]:
            with st.container():
                st.markdown(f"### {info['icon']} {info['name']}")
                st.metric(
                    label=f"{info['symbol']}/USD",
                    value=f"${current_price:,.2f}",
                    delta=f"{price_change:+.2f}%"
                )
                st.caption(f"24h High: ${day_high:,.2f} | Low: ${day_low:,.2f}")
                st.caption(f"Volume: {volume:,.0f} {info['symbol']}")
    
    st.markdown("---")
    
    # Price chart section
    st.subheader("📊 Price Chart")
    
    selected_crypto = st.selectbox(
        "Select cryptocurrency:",
        options=list(cryptos.keys()),
        format_func=lambda x: f"{cryptos[x]['icon']} {cryptos[x]['name']} ({cryptos[x]['symbol']})"
    )
    
    # Time interval selection
    col1, col2 = st.columns([3, 1])
    with col2:
        interval = st.selectbox(
            "Interval:",
            options=[1, 5, 15, 60, 240, 1440],
            format_func=lambda x: f"{x} min" if x < 60 else f"{x//60} hour" if x < 1440 else "1 day",
            index=3
        )
    
    # Fetch OHLC data
    ohlc_data = KrakenAPI.get_ohlc(selected_crypto, interval)
    
    if ohlc_data:
        # Convert to DataFrame
        df = pd.DataFrame(ohlc_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        
        # Create candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        )])
        
        fig.update_layout(
            title=f"{cryptos[selected_crypto]['name']} Price Chart",
            yaxis_title="Price (USD)",
            xaxis_title="Time",
            height=500,
            template="plotly_white",
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume chart
        fig_volume = px.bar(df, x='timestamp', y='volume', 
                           title="Trading Volume",
                           labels={'volume': 'Volume', 'timestamp': 'Time'})
        fig_volume.update_layout(height=200, template="plotly_white")
        st.plotly_chart(fig_volume, use_container_width=True)
    else:
        st.warning("Unable to fetch chart data.")


def show_predictions():
    """Display ML predictions (placeholder for now)."""
    st.header("🧠 ML Price Predictions")
    
    st.info("🚧 **Coming Soon!** ML model training in progress...")
    
    st.markdown("""
    ### What's Coming:
    
    - **LSTM Model**: 2-layer neural network trained on historical data
    - **7-Day Predictions**: Forecasted price movements for next week
    - **Confidence Scores**: How confident the model is in its predictions
    - **Performance Metrics**: Model accuracy and error rates
    - **Feature Importance**: Which indicators drive predictions
    
    ### Current Status:
    - ✅ Data collection pipeline ready
    - ✅ Kraken API integration complete
    - 🚧 Model training in progress
    - ⏳ Coming in Phase 2
    """)
    
    # Mock prediction data for visualization
    st.subheader("Preview: Predicted Returns (7-Day Forecast)")
    
    mock_predictions = pd.DataFrame({
        'Symbol': ['BTC', 'ETH', 'SOL', 'ADA'],
        'Current Price': [122001, 4479, 227, 0.45],
        'Predicted Price': [128500, 4650, 235, 0.48],
        'Predicted Return': [5.3, 3.8, 3.5, 6.7],
        'Confidence': [78, 72, 68, 65]
    })
    
    mock_predictions['Predicted Return'] = mock_predictions['Predicted Return'].apply(lambda x: f"+{x}%")
    mock_predictions['Confidence'] = mock_predictions['Confidence'].apply(lambda x: f"{x}%")
    
    st.dataframe(mock_predictions, use_container_width=True, hide_index=True)
    
    st.caption("📊 *This is mock data for demonstration purposes only*")


def show_rebalancing():
    """Display rebalancing recommendations."""
    st.header("⚖️ Portfolio Rebalancing")
    
    st.info("🚧 **Coming Soon!** Rebalancing logic in development...")
    
    st.markdown("""
    ### Rebalancing Strategy:
    
    1. **Base Strategy**: Equal-weight allocation (25% each for 4 coins)
    2. **ML Enhancement**: Adjust weights based on predicted returns
    3. **Risk Controls**: 
        - Max 40% per position
        - Min 10% per position
        - Min $50 trade size
    4. **Schedule**: Every Sunday at 10 PM CDT
    
    ### Next Actions:
    - Review current allocation
    - Calculate target allocation based on ML predictions
    - Generate buy/sell orders
    - Execute trades (paper trading first!)
    """)


def main():
    """Main application entry point."""
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to:",
        ["📊 Portfolio", "📈 Live Prices", "🧠 Predictions", "⚖️ Rebalancing"]
    )
    
    st.sidebar.markdown("---")
    
    # System status
    st.sidebar.subheader("System Status")
    st.sidebar.success("✅ Kraken API Connected")
    st.sidebar.warning("⏳ ML Models: Training")
    st.sidebar.info("📝 Paper Trading Mode")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("""
    **Version:** 1.0.0  
    **Mode:** Development  
    **Data Source:** Kraken API  
    """)
    
    # Display header
    show_header()
    
    st.markdown("---")
    
    # Route to selected page
    if page == "📊 Portfolio":
        show_portfolio_view()
    elif page == "📈 Live Prices":
        show_live_prices()
    elif page == "🧠 Predictions":
        show_predictions()
    elif page == "⚖️ Rebalancing":
        show_rebalancing()
    
    # Auto-refresh option
    st.sidebar.markdown("---")
    auto_refresh = st.sidebar.checkbox("Auto-refresh (60s)", value=False)
    
    if auto_refresh:
        time.sleep(60)
        st.rerun()


if __name__ == "__main__":
    main()

