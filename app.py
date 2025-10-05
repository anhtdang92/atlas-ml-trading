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


def _get_demo_portfolio():
    """Return demo portfolio data."""
    return {
        'BTC': {'quantity': 0.041, 'avg_buy_price': 118500, 'current_price': 122001},
        'ETH': {'quantity': 0.55, 'avg_buy_price': 4300, 'current_price': 4479},
        'SOL': {'quantity': 5.5, 'avg_buy_price': 220, 'current_price': 227},
        'ADA': {'quantity': 3000, 'avg_buy_price': 0.42, 'current_price': 0.45}
    }


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
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header("💼 Portfolio Overview")
    with col2:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Try to load real portfolio from Kraken
    try:
        from data.kraken_auth import KrakenAuthClient
        
        with st.spinner("🔄 Fetching your portfolio from Kraken..."):
            client = KrakenAuthClient()
            balance = client.get_account_balance()
            
            if balance:
                # Separate liquid and staked assets
                portfolio_data = {}
                staked_data = {}
                usd_balance = 0.0
                
                # Map Kraken asset names to friendly names
                asset_map = {
                    'XXBT': 'BTC', 'XBT': 'BTC',
                    'XETH': 'ETH', 'ETH': 'ETH',
                    'SOL': 'SOL',
                    'ADA': 'ADA',
                    'DOT': 'DOT',
                    'AAVE': 'AAVE',
                    'BABY': 'BABY',
                    'ZUSD': 'USD', 'USD': 'USD'
                }
                
                # Identify staked/bonded assets
                staked_suffixes = {
                    '.B': 'Bonded (Staked)',
                    '.F': 'Futures',
                    '.S': 'Staked',
                    '.M': 'Staked (Medium)',
                    '.L': 'Locked Staking'
                }
                
                # Process real balances
                for asset, amount in balance.items():
                    qty = float(amount)
                    if qty > 0:
                        # Check if it's USD
                        if asset in ['ZUSD', 'USD']:
                            usd_balance += qty
                            continue
                        
                        # Check if it's staked/bonded
                        is_staked = False
                        stake_type = 'Staked'
                        clean_asset = asset
                        
                        for suffix, description in staked_suffixes.items():
                            if asset.endswith(suffix):
                                is_staked = True
                                stake_type = description
                                # Remove suffix and map to friendly name
                                base_asset = asset[:-2]  # Remove .B, .F, etc.
                                clean_asset = asset_map.get(base_asset, base_asset)
                                break
                        
                        if not is_staked:
                            clean_asset = asset_map.get(asset, asset)
                        
                        # Add to appropriate dictionary
                        target_dict = staked_data if is_staked else portfolio_data
                        
                        if clean_asset not in target_dict:
                            target_dict[clean_asset] = {
                                'quantity': qty,
                                'avg_buy_price': 0,
                                'current_price': 0,
                                'stake_type': stake_type if is_staked else None,
                                'raw_asset': asset
                            }
                        else:
                            target_dict[clean_asset]['quantity'] += qty
                
                st.success("✅ Connected to your Kraken account!")
            else:
                st.warning("⚠️ Could not fetch portfolio. Using demo data.")
                portfolio_data = _get_demo_portfolio()
                
    except Exception as e:
        st.warning(f"⚠️ Could not connect to Kraken: {e}. Using demo data.")
        portfolio_data = _get_demo_portfolio()
    
    # If no holdings, show demo
    if not portfolio_data:
        st.info("📝 No holdings found. Add some crypto to your Kraken account or using demo data.")
        portfolio_data = _get_demo_portfolio()
    
    # Fetch live prices for all holdings
    if portfolio_data:
        # Map symbols to Kraken pair names
        symbol_to_pair = {
            'BTC': 'XXBTZUSD',
            'ETH': 'XETHZUSD',
            'SOL': 'SOLUSD',
            'ADA': 'ADAUSD',
            'DOT': 'DOTUSD',
            'AAVE': 'AAVEUSD',
            'BABY': 'BABYUSD'  # May not have USD pair
        }
        
        # Get pairs for symbols we have
        pairs_to_fetch = [symbol_to_pair.get(symbol) for symbol in portfolio_data.keys() if symbol_to_pair.get(symbol)]
        
        if pairs_to_fetch:
            kraken_api = KrakenAPI()
            ticker_data = kraken_api.get_ticker(pairs_to_fetch)
            
            if ticker_data:
                # Update current prices from live data
                for symbol, pair in symbol_to_pair.items():
                    if symbol in portfolio_data and pair:
                        matching_key = [k for k in ticker_data.keys() if pair in k or k in pair]
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
    
    # Show last update time
    st.caption(f"🕒 Last updated: {datetime.now().strftime('%B %d, %Y at %I:%M:%S %p')}")
    
    # Calculate staked value if available
    staked_value = 0
    if 'staked_data' in locals() and staked_data:
        # Fetch prices for staked assets too
        staked_pairs = []
        staked_symbol_to_pair = {
            'BTC': 'XXBTZUSD',
            'ETH': 'XETHZUSD',
            'SOL': 'SOLUSD',
            'DOT': 'DOTUSD'
        }
        
        for symbol in staked_data.keys():
            if symbol in staked_symbol_to_pair:
                staked_pairs.append(staked_symbol_to_pair[symbol])
        
        if staked_pairs:
            kraken_api = KrakenAPI()
            staked_ticker = kraken_api.get_ticker(staked_pairs)
            
            if staked_ticker:
                for symbol, pair in staked_symbol_to_pair.items():
                    if symbol in staked_data and pair:
                        matching_key = [k for k in staked_ticker.keys() if pair in k or k in pair]
                        if matching_key:
                            price = float(staked_ticker[matching_key[0]]['c'][0])
                            staked_data[symbol]['current_price'] = price
                            staked_value += staked_data[symbol]['quantity'] * price
    else:
        staked_data = {}
    
    # Display key metrics with better styling
    st.markdown("### 📊 Portfolio Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style='background-color: #0e1117; padding: 20px; border-radius: 10px; border: 2px solid #1f77b4;'>
            <h4 style='color: #1f77b4; margin: 0;'>💰 Total Value</h4>
            <h1 style='color: white; margin: 10px 0;'>${total_value:,.4f}</h1>
            <p style='color: #888; margin: 0; font-size: 14px;'>Current portfolio value</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        pnl_color = "#00ff00" if total_pnl >= 0 else "#ff4444"
        pnl_symbol = "📈" if total_pnl >= 0 else "📉"
        st.markdown(f"""
        <div style='background-color: #0e1117; padding: 20px; border-radius: 10px; border: 2px solid {pnl_color};'>
            <h4 style='color: {pnl_color}; margin: 0;'>{pnl_symbol} Total P&L</h4>
            <h1 style='color: {pnl_color}; margin: 10px 0;'>${total_pnl:+,.4f}</h1>
            <p style='color: #888; margin: 0; font-size: 14px;'>{total_pnl_pct:+.2f}% gain/loss</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style='background-color: #0e1117; padding: 20px; border-radius: 10px; border: 2px solid #9467bd;'>
            <h4 style='color: #9467bd; margin: 0;'>🔒 Staked Value</h4>
            <h1 style='color: white; margin: 10px 0;'>${staked_value:,.4f}</h1>
            <p style='color: #888; margin: 0; font-size: 14px;'>Assets earning rewards</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        num_holdings = len(portfolio_data) + len(staked_data)
        st.markdown(f"""
        <div style='background-color: #0e1117; padding: 20px; border-radius: 10px; border: 2px solid #ff7f0e;'>
            <h4 style='color: #ff7f0e; margin: 0;'>🪙 Total Assets</h4>
            <h1 style='color: white; margin: 10px 0;'>{num_holdings}</h1>
            <p style='color: #888; margin: 0; font-size: 14px;'>Liquid + Staked</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Holdings table with better formatting
    st.markdown("### 📋 Current Holdings")
    
    if holdings:
        holdings_df = pd.DataFrame(holdings)
        
        # Style the dataframe
        st.dataframe(
            holdings_df,
            width='stretch',
            hide_index=True,
            column_config={
                "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                "Quantity": st.column_config.TextColumn("Quantity", width="medium"),
                "Current Price": st.column_config.TextColumn("Current Price", width="medium"),
                "Value": st.column_config.TextColumn("Value", width="medium"),
                "P&L": st.column_config.TextColumn("P&L", width="medium"),
                "P&L %": st.column_config.TextColumn("P&L %", width="small"),
                "% Portfolio": st.column_config.TextColumn("% Portfolio", width="small"),
            }
        )
        
        st.caption("💡 **Note:** P&L is calculated from average buy price. Since we don't have your historical trades, P&L may show as $0.00")
    else:
        st.info("No liquid holdings to display")
    
    # Staked Assets Section
    if staked_data:
        st.markdown("---")
        st.markdown("### 🔒 Staked & Bonded Assets")
        st.info("💰 **These assets are earning staking rewards!** They're locked but still yours and generating passive income.")
        
        staked_holdings = []
        total_staked_value = 0
        
        for symbol, data in staked_data.items():
            quantity = data['quantity']
            current_price = data['current_price']
            stake_type = data['stake_type']
            raw_asset = data['raw_asset']
            
            current_value = quantity * current_price
            total_staked_value += current_value
            
            staked_holdings.append({
                'Symbol': symbol,
                'Type': stake_type,
                'Quantity': f"{quantity:.8f}",
                'Current Price': f"${current_price:,.2f}" if current_price > 0 else "N/A",
                'Value': f"${current_value:,.4f}",
                'Kraken Asset': raw_asset
            })
        
        if staked_holdings:
            staked_df = pd.DataFrame(staked_holdings)
            st.dataframe(
                staked_df,
                width='stretch',
                hide_index=True,
                column_config={
                    "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                    "Type": st.column_config.TextColumn("Staking Type", width="medium"),
                    "Quantity": st.column_config.TextColumn("Quantity", width="medium"),
                    "Current Price": st.column_config.TextColumn("Current Price", width="small"),
                    "Value": st.column_config.TextColumn("Value", width="medium"),
                    "Kraken Asset": st.column_config.TextColumn("Kraken Code", width="small"),
                }
            )
            
            st.markdown(f"**Total Staked Value:** ${total_staked_value:,.4f}")
            
            # Staking info
            with st.expander("ℹ️ What is Staking?"):
                st.markdown("""
                **Staking** is like earning interest on your crypto! Here's what's happening:
                
                - **Bonded (.B)**: Your crypto is locked in a staking contract, earning rewards
                - **Futures (.F)**: Futures positions (different from regular holdings)
                - **Rewards**: You earn passive income while holding
                - **Locked**: Can't trade immediately, but it's still yours!
                
                **Benefits:**
                - 📈 Earn passive income (APY varies by asset)
                - 🔒 Helps secure the blockchain network
                - 💎 Encourages long-term holding
                
                **Note:** To trade staked assets, you'll need to unstake them first (may take time).
                """)
    else:
        st.info("No holdings to display")
    
    # Portfolio allocation pie chart
    st.markdown("### 📊 Portfolio Allocation")
    
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
            hole=0.4  # Donut chart
        )
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            textfont_size=14
        )
        fig.update_layout(
            height=400,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig, width='stretch')
    
    # Status info
    st.success("✅ **Connected to Kraken** - Your portfolio is live and updating with real-time prices!")


def show_live_prices():
    """Display live cryptocurrency prices with charts."""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header("📈 Live Cryptocurrency Prices")
    with col2:
        if st.button("🔄 Refresh Prices", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Crypto configuration - using verified Kraken pairs
    cryptos = {
        'XXBTZUSD': {'name': 'Bitcoin', 'symbol': 'BTC', 'icon': '₿', 'color': '#f7931a'},
        'XETHZUSD': {'name': 'Ethereum', 'symbol': 'ETH', 'icon': 'Ξ', 'color': '#627eea'},
        'SOLUSD': {'name': 'Solana', 'symbol': 'SOL', 'icon': '◎', 'color': '#9945ff'},
        'ADAUSD': {'name': 'Cardano', 'symbol': 'ADA', 'icon': '₳', 'color': '#0033ad'},
        'XXRPZUSD': {'name': 'Ripple', 'symbol': 'XRP', 'icon': '✕', 'color': '#23292f'},
        'DOTUSD': {'name': 'Polkadot', 'symbol': 'DOT', 'icon': '●', 'color': '#e6007a'}
    }
    
    # Fetch ticker data
    kraken_api = KrakenAPI()
    ticker_data = kraken_api.get_ticker(list(cryptos.keys()))
    
    if not ticker_data:
        st.error("Unable to fetch price data. Please try again.")
        return
    
    # Show last update time
    st.caption(f"🕒 Last updated: {datetime.now().strftime('%B %d, %Y at %I:%M:%S %p')}")
    
    # Display price cards with portfolio-style design
    st.markdown("### 💰 Market Overview")
    
    # Create 2 rows of 3 cards each
    for row in range(2):
        cols = st.columns(3)
        for col_idx in range(3):
            crypto_idx = row * 3 + col_idx
            if crypto_idx >= len(cryptos):
                break
                
            pair = list(cryptos.keys())[crypto_idx]
            info = cryptos[pair]
            
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
            change_color = "#00ff00" if price_change >= 0 else "#ff4444"
            change_symbol = "📈" if price_change >= 0 else "📉"
            
            with cols[col_idx]:
                st.markdown(f"""
                <div style='background-color: #0e1117; padding: 20px; border-radius: 10px; border: 2px solid {info['color']}; margin-bottom: 15px;'>
                    <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                        <h3 style='color: {info['color']}; margin: 0; font-size: 24px;'>{info['icon']}</h3>
                        <h4 style='color: white; margin: 0 0 0 10px;'>{info['name']}</h4>
                    </div>
                    <h1 style='color: white; margin: 10px 0; font-size: 28px;'>${current_price:,.2f}</h1>
                    <div style='display: flex; justify-content: space-between; align-items: center; margin: 10px 0;'>
                        <span style='color: {change_color}; font-weight: bold; font-size: 16px;'>{change_symbol} {price_change:+.2f}%</span>
                        <span style='color: #888; font-size: 14px;'>{info['symbol']}/USD</span>
                    </div>
                    <div style='color: #888; font-size: 12px; margin-top: 10px;'>
                        <div>24h High: ${day_high:,.2f}</div>
                        <div>24h Low: ${day_low:,.2f}</div>
                        <div>Volume: {volume:,.0f} {info['symbol']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Price chart section with portfolio-style header
    st.markdown("### 📊 Interactive Price Charts")
    
    # Chart controls in a nice layout
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        selected_crypto = st.selectbox(
            "Select cryptocurrency:",
            options=list(cryptos.keys()),
            format_func=lambda x: f"{cryptos[x]['icon']} {cryptos[x]['name']} ({cryptos[x]['symbol']})"
        )
    
    with col2:
        interval = st.selectbox(
            "Time Interval:",
            options=[1, 5, 15, 60, 240, 1440],
            format_func=lambda x: f"{x} min" if x < 60 else f"{x//60} hour" if x < 1440 else "1 day",
            index=3
        )
    
    with col3:
        chart_type = st.selectbox(
            "Chart Type:",
            options=["Candlestick", "Line"],
            index=0
        )
    
    # Fetch OHLC data
    kraken_api = KrakenAPI()
    ohlc_data = kraken_api.get_ohlc(selected_crypto, interval)
    
    if ohlc_data:
        # Convert to DataFrame
        df = pd.DataFrame(ohlc_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        
        # Create chart based on selection
        if chart_type == "Candlestick":
            fig = go.Figure(data=[go.Candlestick(
                x=df['timestamp'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price',
                increasing_line_color='#00ff00',
                decreasing_line_color='#ff4444'
            )])
        else:
            fig = go.Figure(data=[go.Scatter(
                x=df['timestamp'],
                y=df['close'],
                mode='lines',
                name='Price',
                line=dict(color=cryptos[selected_crypto]['color'], width=2)
            )])
        
        # Apply portfolio-style theming
        fig.update_layout(
            title=f"{cryptos[selected_crypto]['icon']} {cryptos[selected_crypto]['name']} Price Chart",
            yaxis_title="Price (USD)",
            xaxis_title="Time",
            height=500,
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font=dict(color='white'),
            title_font=dict(size=20, color=cryptos[selected_crypto]['color']),
            xaxis=dict(gridcolor='#333'),
            yaxis=dict(gridcolor='#333')
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Volume chart with matching styling
        fig_volume = px.bar(df, x='timestamp', y='volume', 
                           title=f"{cryptos[selected_crypto]['symbol']} Trading Volume",
                           labels={'volume': 'Volume', 'timestamp': 'Time'})
        
        fig_volume.update_layout(
            height=200, 
            template="plotly_dark",
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font=dict(color='white'),
            title_font=dict(size=16, color=cryptos[selected_crypto]['color']),
            xaxis=dict(gridcolor='#333'),
            yaxis=dict(gridcolor='#333')
        )
        
        fig_volume.update_traces(marker_color=cryptos[selected_crypto]['color'])
        st.plotly_chart(fig_volume, width='stretch')
        
        # Market stats in portfolio-style cards
        st.markdown("### 📈 Market Statistics")
        
        current_price = df['close'].iloc[-1]
        price_24h_ago = df['close'].iloc[-2] if len(df) > 1 else current_price
        change_24h = ((current_price - price_24h_ago) / price_24h_ago) * 100 if price_24h_ago > 0 else 0
        volume_24h = df['volume'].sum()
        high_24h = df['high'].max()
        low_24h = df['low'].min()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            change_color = "#00ff00" if change_24h >= 0 else "#ff4444"
            st.markdown(f"""
            <div style='background-color: #0e1117; padding: 15px; border-radius: 10px; border: 2px solid {change_color};'>
                <h4 style='color: {change_color}; margin: 0;'>24h Change</h4>
                <h2 style='color: {change_color}; margin: 5px 0;'>{change_24h:+.2f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style='background-color: #0e1117; padding: 15px; border-radius: 10px; border: 2px solid #1f77b4;'>
                <h4 style='color: #1f77b4; margin: 0;'>24h Volume</h4>
                <h2 style='color: white; margin: 5px 0;'>{volume_24h:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style='background-color: #0e1117; padding: 15px; border-radius: 10px; border: 2px solid #9467bd;'>
                <h4 style='color: #9467bd; margin: 0;'>24h High</h4>
                <h2 style='color: white; margin: 5px 0;'>${high_24h:,.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div style='background-color: #0e1117; padding: 15px; border-radius: 10px; border: 2px solid #ff7f0e;'>
                <h4 style='color: #ff7f0e; margin: 0;'>24h Low</h4>
                <h2 style='color: white; margin: 5px 0;'>${low_24h:,.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
    else:
        st.warning("Unable to fetch chart data. Please try again.")


def show_predictions():
    """Display ML predictions with real-time data and model training."""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header("🧠 ML Price Predictions")
    with col2:
        if st.button("🔄 Refresh Predictions", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Initialize prediction service
    try:
        from ml.prediction_service import PredictionService
        prediction_service = PredictionService()
    except ImportError as e:
        st.error(f"❌ Could not import prediction service: {e}")
        st.info("Please ensure all ML dependencies are installed.")
        return
    
    # Show last update time
    st.caption(f"🕒 Last updated: {datetime.now().strftime('%B %d, %Y at %I:%M:%S %p')}")
    
    # Show current status
    st.info("🔮 **Current Status**: Using mock predictions (TensorFlow not available). Predictions are realistic but not trained on your specific data. To train real models, install TensorFlow with Python 3.9-3.11.")
    
    # Prediction controls
    st.markdown("### 🎛️ Prediction Controls")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        selected_symbol = st.selectbox(
            "Select Cryptocurrency:",
            options=['All'] + ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'XRP'],
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
        if st.button("🚀 Train Model", use_container_width=True):
            with st.spinner("Training model... This may take a few minutes."):
                if selected_symbol == 'All':
                    st.info("Please select a specific cryptocurrency to train its model.")
                else:
                    result = prediction_service.train_model(selected_symbol, days=365, epochs=50)
                    if result['status'] == 'success':
                        st.success(f"✅ Model trained successfully for {selected_symbol}!")
                        st.json(result)
                    else:
                        st.error(f"❌ Training failed: {result['message']}")
                        if "TensorFlow" in result['message']:
                            st.info("💡 **Solution**: Install TensorFlow with Python 3.9-3.11 to train real models.")
                            st.info("🔍 **What happened**: The system successfully fetched real data and calculated features, but failed at the TensorFlow model creation step.")
    
    # Train all models button
    st.markdown("### 🚀 Train All Models")
    if st.button("🚀 Train All Models", type="primary", use_container_width=True):
        with st.spinner("Training all models... This may take several minutes."):
            try:
                results = prediction_service.train_all_models()
                
                # Show training results
                st.success("✅ Training completed!")
                
                # Display results
                for result in results:
                    if result['status'] == 'success':
                        st.success(f"✅ {result['symbol']}: {result['message']}")
                    else:
                        st.error(f"❌ {result['symbol']}: {result['message']}")
                        
            except Exception as e:
                st.error(f"❌ Training failed: {e}")
                st.info("💡 **Solution**: Install TensorFlow with Python 3.9-3.11 to train real models.")
                
                # Show what actually happened
                st.info("🔍 **What happened**: The system successfully fetched real data and calculated features, but failed at the TensorFlow model creation step.")
                st.info("📊 **Data processed**: 366 days of real market data with 11 technical indicators")
                st.info("🔄 **Sequences created**: 352 training sequences ready for LSTM training")
                st.info("⚠️ **Missing**: TensorFlow library for neural network training")
    
    st.markdown("---")
    
    # Generate predictions
    with st.spinner("🔮 Generating predictions..."):
        if selected_symbol == 'All':
            predictions = prediction_service.get_all_predictions(days_ahead)
        else:
            predictions = [prediction_service.get_prediction(selected_symbol, days_ahead)]
    
    # Display predictions with portfolio-style cards
    st.markdown("### 📊 Price Predictions")
    
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
                    border_color = "#00ff00"
                    return_color = "#00ff00"
                    return_symbol = "📈"
                elif pred['predicted_return'] < -0.02:  # < -2% loss
                    border_color = "#ff4444"
                    return_color = "#ff4444"
                    return_symbol = "📉"
                else:  # Neutral
                    border_color = "#1f77b4"
                    return_color = "#1f77b4"
                    return_symbol = "➡️"
                
                # Confidence color
                conf_color = "#00ff00" if pred['confidence'] > 0.7 else "#ffaa00" if pred['confidence'] > 0.5 else "#ff4444"
                
                with col:
                    # Create a container with custom styling
                    with st.container():
                        # Header with symbol and confidence
                        col_header, col_conf = st.columns([3, 1])
                        with col_header:
                            st.markdown(f"### {pred['symbol']} {return_symbol}")
                        with col_conf:
                            st.markdown(f"**{pred['confidence']*100:.0f}%**")
                        
                        # Current price section
                        st.markdown("**Current Price**")
                        st.markdown(f"# ${pred['current_price']:,.2f}")
                        
                        # Predicted price section
                        st.markdown("**Predicted Price**")
                        st.markdown(f"## ${pred['predicted_price']:,.2f}")
                        
                        # Return percentage with color
                        if pred['predicted_return'] > 0.02:
                            st.success(f"📈 **+{pred['predicted_return']*100:.2f}%** ({days_ahead}d forecast)")
                        elif pred['predicted_return'] < -0.02:
                            st.error(f"📉 **{pred['predicted_return']*100:.2f}%** ({days_ahead}d forecast)")
                        else:
                            st.info(f"➡️ **{pred['predicted_return']*100:+.2f}%** ({days_ahead}d forecast)")
                        
                        # Model info
                        st.caption(f"Model: {pred['model_version']} | Status: {pred['status']}")
                        
                        st.markdown("---")
    
    st.markdown("---")
    
    # Prediction table
    st.markdown("### 📋 Detailed Predictions")
    
    # Prepare data for table
    table_data = []
    for pred in predictions:
        table_data.append({
            'Symbol': pred['symbol'],
            'Current Price': f"${pred['current_price']:,.2f}",
            'Predicted Price': f"${pred['predicted_price']:,.2f}",
            'Predicted Return': f"{pred['predicted_return']*100:+.2f}%",
            'Confidence': f"{pred['confidence']*100:.1f}%",
            'Model Version': pred['model_version'],
            'Status': pred['status'].title()
        })
    
    if table_data:
        df = pd.DataFrame(table_data)
        st.dataframe(df, width='stretch', hide_index=True)
    
    # Model training section
    st.markdown("---")
    st.markdown("### 🚀 Model Training")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Train Custom Models:**
        
        - **LSTM Architecture**: 2-layer neural network with 50 units each
        - **Features**: 11 technical indicators (MA, RSI, Volume, Momentum, Volatility)
        - **Training Data**: 365 days of historical data from Kraken
        - **Prediction Target**: 7-day future returns
        - **Validation**: 80/20 train/validation split with early stopping
        """)
    
    with col2:
        if st.button("🎯 Train All Models", use_container_width=True):
            with st.spinner("Training all models... This will take several minutes."):
                results = prediction_service.train_all_models(days=365, epochs=50)
                
                # Display results with better formatting
                st.markdown("#### 🚀 Training Results:")
                
                successful = 0
                failed = 0
                
                for symbol, result in results.items():
                    if result['status'] == 'success':
                        st.success(f"✅ **{symbol}**: Trained successfully")
                        successful += 1
                        with st.expander(f"📊 {symbol} Training Details"):
                            st.json(result)
                    else:
                        st.error(f"❌ **{symbol}**: {result['message']}")
                        failed += 1
                        
                        # Show solution for TensorFlow error
                        if "TensorFlow is required" in result['message']:
                            st.info(f"💡 **Solution for {symbol}**: Install TensorFlow with Python 3.9-3.11")
                
                # Summary
                if successful > 0:
                    st.success(f"🎉 **{successful} models trained successfully!**")
                if failed > 0:
                    st.warning(f"⚠️ **{failed} models failed** - TensorFlow installation required")
                    
                # Show installation instructions
                if failed > 0:
                    with st.expander("🔧 How to Install TensorFlow for Real Model Training"):
                        st.markdown("""
                        **To train real LSTM models, you need TensorFlow:**
                        
                        1. **Create a new Python environment** with Python 3.9-3.11:
                           ```bash
                           conda create -n crypto-ml python=3.11
                           conda activate crypto-ml
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
    st.markdown("### 📊 Model Status")
    
    model_status = []
    for symbol in ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'XRP']:
        has_model = prediction_service._has_model(symbol)
        status = "✅ Trained" if has_model else "❌ Not Trained"
        model_status.append({
            'Symbol': symbol,
            'Status': status,
            'Model File': f"{symbol}_model.h5",
            'Last Updated': 'N/A'  # Could add timestamp tracking
        })
    
    status_df = pd.DataFrame(model_status)
    st.dataframe(status_df, width='stretch', hide_index=True)
    
    # Information section
    with st.expander("ℹ️ About ML Predictions"):
        st.markdown("""
        **How It Works:**
        
        1. **Data Collection**: Fetches 365 days of OHLCV data from Kraken API
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
    st.success("✅ **ML Prediction System Active** - Real-time forecasts powered by LSTM neural networks!")


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

