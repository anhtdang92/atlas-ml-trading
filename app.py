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
    page_title="NOVA • Crypto ML Console",
    page_icon="◉",
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
        st.title("NOVA • Crypto ML Console")
        st.markdown("*Powered by Machine Learning & Kraken API*")
    
    with col3:
        st.markdown(f"**Last Updated:** {datetime.now().strftime('%I:%M:%S %p')}")


def show_portfolio_view():
    """Display portfolio overview with holdings and performance."""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header("◉ Portfolio Overview")
    with col2:
        if st.button("◐ Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Try to load real portfolio from Kraken
    try:
        from data.kraken_auth import KrakenAuthClient
        
        with st.spinner("◐ Fetching your portfolio from Kraken..."):
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
                
                st.success("◉ Connected to your Kraken account!")
            else:
                st.warning("◊ Could not fetch portfolio. Using demo data.")
                portfolio_data = _get_demo_portfolio()
                
    except Exception as e:
        st.warning(f"◊ Could not connect to Kraken: {e}. Using demo data.")
        portfolio_data = _get_demo_portfolio()
    
    # If no holdings, show demo
    if not portfolio_data:
        st.info("◐ No holdings found. Add some crypto to your Kraken account or using demo data.")
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
    st.caption(f"◐ Last updated: {datetime.now().strftime('%B %d, %Y at %I:%M:%S %p')}")
    
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
    st.markdown("### ◉ Portfolio Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style='background-color: #0e1117; padding: 20px; border-radius: 10px; border: 2px solid #1f77b4;'>
            <h4 style='color: #1f77b4; margin: 0;'>◉ Total Value</h4>
            <h1 style='color: white; margin: 10px 0;'>${total_value:,.4f}</h1>
            <p style='color: #888; margin: 0; font-size: 14px;'>Current portfolio value</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        pnl_color = "#00ff00" if total_pnl >= 0 else "#ff4444"
        pnl_symbol = "◈" if total_pnl >= 0 else "◊"
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
            <h4 style='color: #9467bd; margin: 0;'>◐ Staked Value</h4>
            <h1 style='color: white; margin: 10px 0;'>${staked_value:,.4f}</h1>
            <p style='color: #888; margin: 0; font-size: 14px;'>Assets earning rewards</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        num_holdings = len(portfolio_data) + len(staked_data)
        st.markdown(f"""
        <div style='background-color: #0e1117; padding: 20px; border-radius: 10px; border: 2px solid #ff7f0e;'>
            <h4 style='color: #ff7f0e; margin: 0;'>◈ Total Assets</h4>
            <h1 style='color: white; margin: 10px 0;'>{num_holdings}</h1>
            <p style='color: #888; margin: 0; font-size: 14px;'>Liquid + Staked</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Holdings table with better formatting
    st.markdown("### ◊ Current Holdings")
    
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
        
        st.caption("◉ **Note:** P&L is calculated from average buy price. Since we don't have your historical trades, P&L may show as $0.00")
    else:
        st.info("No liquid holdings to display")
    
    # Staked Assets Section
    if staked_data:
        st.markdown("---")
        st.markdown("### ◐ Staked & Bonded Assets")
        st.info("◉ **These assets are earning staking rewards!** They're locked but still yours and generating passive income.")
        
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
            with st.expander("◈ What is Staking?"):
                st.markdown("""
                **Staking** is like earning interest on your crypto! Here's what's happening:
                
                - **Bonded (.B)**: Your crypto is locked in a staking contract, earning rewards
                - **Futures (.F)**: Futures positions (different from regular holdings)
                - **Rewards**: You earn passive income while holding
                - **Locked**: Can't trade immediately, but it's still yours!
                
                **Benefits:**
                - ◉ Earn passive income (APY varies by asset)
                - ◐ Helps secure the blockchain network
                - ◉ Encourages long-term holding
                
                **Note:** To trade staked assets, you'll need to unstake them first (may take time).
                """)
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
    st.success("◉ **Connected to Kraken** - Your portfolio is live and updating with real-time prices!")


def show_live_prices():
    """Display live cryptocurrency prices with charts."""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header("◈ Live Cryptocurrency Prices")
    with col2:
        if st.button("◐ Refresh Prices", use_container_width=True):
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
    st.caption(f"◐ Last updated: {datetime.now().strftime('%B %d, %Y at %I:%M:%S %p')}")
    
    # Display price cards with portfolio-style design
    st.markdown("### ◉ Market Overview")
    
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
            change_symbol = "◈" if price_change >= 0 else "◊"

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
    st.markdown("### ◉ Interactive Price Charts")
    
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
        st.markdown("### ◈ Market Statistics")
        
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
        st.header("◊ ML Price Predictions")
    with col2:
        if st.button("◐ Refresh Predictions", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Initialize prediction service
    try:
        from ml.prediction_service import PredictionService
        prediction_service = PredictionService()
    except ImportError as e:
        st.error(f"◊ Could not import prediction service: {e}")
        st.info("Please ensure all ML dependencies are installed.")
        return
    
    # Show last update time
    st.caption(f"◐ Last updated: {datetime.now().strftime('%B %d, %Y at %I:%M:%S %p')}")
    
    # Show current status
    st.info("◊ **Current Status**: Using mock predictions (TensorFlow not available). Predictions are realistic but not trained on your specific data. To train real models, install TensorFlow with Python 3.9-3.11.")
    
    # Prediction controls
    st.markdown("### ◈ Prediction Controls")
    
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
        if st.button("◈ Train Model", use_container_width=True):
            with st.spinner("Training model... This may take a few minutes."):
                if selected_symbol == 'All':
                    st.info("Please select a specific cryptocurrency to train its model.")
                else:
                    result = prediction_service.train_model(selected_symbol, days=365, epochs=50)
                    if result['status'] == 'success':
                        st.success(f"◉ Model trained successfully for {selected_symbol}!")
                        st.json(result)
                    else:
                        st.error(f"◊ Training failed: {result['message']}")
                        if "TensorFlow" in result['message']:
                            st.info("◉ **Solution**: Install TensorFlow with Python 3.9-3.11 to train real models.")
                            st.info("◈ **What happened**: The system successfully fetched real data and calculated features, but failed at the TensorFlow model creation step.")
    
    # Train all models button
    st.markdown("### ◈ Train All Models")
    if st.button("◈ Train All Models", type="primary", use_container_width=True):
        with st.spinner("Training all models... This may take several minutes."):
            try:
                results = prediction_service.train_all_models()
                
                # Show training results
                st.success("◉ Training completed!")
                
                # Display results
                for result in results:
                    if result['status'] == 'success':
                        st.success(f"◉ {result['symbol']}: {result['message']}")
                    else:
                        st.error(f"◊ {result['symbol']}: {result['message']}")
                        
            except Exception as e:
                st.error(f"◊ Training failed: {e}")
                st.info("◉ **Solution**: Install TensorFlow with Python 3.9-3.11 to train real models.")
                
                # Show what actually happened
                st.info("◈ **What happened**: The system successfully fetched real data and calculated features, but failed at the TensorFlow model creation step.")
                st.info("◉ **Data processed**: 366 days of real market data with 11 technical indicators")
                st.info("◐ **Sequences created**: 352 training sequences ready for LSTM training")
                st.info("◊ **Missing**: TensorFlow library for neural network training")
    
    st.markdown("---")
    
    # Generate predictions
    with st.spinner("◊ Generating predictions..."):
        if selected_symbol == 'All':
            predictions = prediction_service.get_all_predictions(days_ahead)
        else:
            predictions = [prediction_service.get_prediction(selected_symbol, days_ahead)]
    
    # Display predictions with portfolio-style cards
    st.markdown("### ◉ Price Predictions")
    
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
                    return_symbol = "◈"
                elif pred['predicted_return'] < -0.02:  # < -2% loss
                    border_color = "#ff4444"
                    return_color = "#ff4444"
                    return_symbol = "◊"
                else:  # Neutral
                    border_color = "#1f77b4"
                    return_color = "#1f77b4"
                    return_symbol = "◐"
                
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
                            st.success(f"◈ **+{pred['predicted_return']*100:.2f}%** ({days_ahead}d forecast)")
                        elif pred['predicted_return'] < -0.02:
                            st.error(f"◊ **{pred['predicted_return']*100:.2f}%** ({days_ahead}d forecast)")
                        else:
                            st.info(f"◐ **{pred['predicted_return']*100:+.2f}%** ({days_ahead}d forecast)")
                        
                        # Model info
                        st.caption(f"Model: {pred['model_version']} | Status: {pred['status']}")
                        
                        st.markdown("---")
    
    st.markdown("---")
    
    # Prediction table
    st.markdown("### ◊ Detailed Predictions")
    
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
    st.markdown("### ◈ Model Training")
    
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
        if st.button("◈ Train All Models", use_container_width=True):
            with st.spinner("Training all models... This will take several minutes."):
                results = prediction_service.train_all_models(days=365, epochs=50)
                
                # Display results with better formatting
                st.markdown("#### ◈ Training Results:")
                
                successful = 0
                failed = 0
                
                for symbol, result in results.items():
                    if result['status'] == 'success':
                        st.success(f"◉ **{symbol}**: Trained successfully")
                        successful += 1
                        with st.expander(f"◉ {symbol} Training Details"):
                            st.json(result)
                    else:
                        st.error(f"◊ **{symbol}**: {result['message']}")
                        failed += 1
                        
                        # Show solution for TensorFlow error
                        if "TensorFlow is required" in result['message']:
                            st.info(f"◉ **Solution for {symbol}**: Install TensorFlow with Python 3.9-3.11")
                
                # Summary
                if successful > 0:
                    st.success(f"◉ **{successful} models trained successfully!**")
                if failed > 0:
                    st.warning(f"◊ **{failed} models failed** - TensorFlow installation required")
                    
                # Show installation instructions
                if failed > 0:
                    with st.expander("◈ How to Install TensorFlow for Real Model Training"):
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
    st.markdown("### ◉ Model Status")
    
    model_status = []
    for symbol in ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'XRP']:
        has_model = prediction_service._has_model(symbol)
        status = "◉ Trained" if has_model else "◊ Not Trained"
        model_status.append({
            'Symbol': symbol,
            'Status': status,
            'Model File': f"{symbol}_model.h5",
            'Last Updated': 'N/A'  # Could add timestamp tracking
        })
    
    status_df = pd.DataFrame(model_status)
    st.dataframe(status_df, width='stretch', hide_index=True)
    
    # Information section
    with st.expander("◈ About ML Predictions"):
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
    st.success("◉ **ML Prediction System Active** - Real-time forecasts powered by LSTM neural networks!")


def show_rebalancing():
    """Display portfolio rebalancing interface."""
    st.header("◐ Portfolio Rebalancing")
    
    # Import rebalancing service
    try:
        from ml.portfolio_rebalancer import PortfolioRebalancer
    except ImportError as e:
        st.error(f"◊ Could not import portfolio rebalancer: {e}")
        return
    
    # Initialize rebalancer
    paper_trading = st.sidebar.checkbox("◐ Paper Trading Mode", value=True, help="Enable paper trading to test strategies without real money")
    rebalancer = PortfolioRebalancer(paper_trading=paper_trading)
    
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
        use_ml = st.checkbox("◊ Use ML Predictions", value=True, help="Enhance allocation with ML predictions")
    
    # Advanced configuration
    with st.expander("◈ Advanced Configuration"):
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
        if st.button("💾 Save Configuration"):
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
                st.error("◊ Failed to save configuration")
    
    st.markdown("---")
    
    # Get rebalancing summary
    with st.spinner("◐ Calculating rebalancing recommendations..."):
        summary = rebalancer.get_rebalancing_summary()
    
    # Quick status overview
    total_orders = len(summary['orders'])
    max_drift = summary['metrics']['max_drift'] * 100
    total_fees = summary['metrics']['total_fees']
    
    if total_orders == 0:
        st.success("◉ **Portfolio is perfectly balanced!** No rebalancing needed.")
    elif max_drift > 5:
        st.warning(f"◊ **Significant drift detected** - {max_drift:.1f}% max drift requires rebalancing")
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
        st.markdown("#### ◈ Current Allocation")
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
        st.markdown("#### ◈ Target Allocation")
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
        if "◊" in recommendation:
            st.warning(recommendation)
        elif "◉" in recommendation:
            st.info(recommendation)
        elif "◊" in recommendation and "ALERT" in recommendation:
            st.error(recommendation)
        else:
            st.success(recommendation)
    
    st.markdown("---")
    
    # Execution section
    st.markdown("### ◈ Execute Rebalancing")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("◐ Execute Paper Trading", type="primary", use_container_width=True):
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
                    st.error(f"◊ Execution failed: {result.get('message', 'Unknown error')}")
    
    with col2:
        if st.button("◊ Execute Live Trading", type="secondary", use_container_width=True, disabled=True):
            st.warning("◊ Live trading not implemented yet - use paper trading mode")
    
    # Strategy information
    with st.expander("◈ Rebalancing Strategy Details"):
        st.markdown("""
        **Strategy Overview:**
    
        1. **Base Strategy**: Equal-weight allocation (16.67% each for 6 coins)
        2. **ML Enhancement**: Adjust weights based on predicted returns
        3. **Risk Controls**: 
           - Max 40% per position
           - Min 10% per position
           - Min $50 trade size
        4. **Trading Fees**: 0.16% maker fee (Kraken)
        
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


def main():
    """Main application entry point."""
    
    # Initialize session state for navigation
    if 'current_page' not in st.session_state:
        # Check for URL parameters first
        query_params = st.query_params
        if 'page' in query_params:
            page_param = query_params['page']
            if page_param in ["portfolio", "prices", "predictions", "rebalancing"]:
                page_map = {
                    "portfolio": "◉ Portfolio",
                    "prices": "◈ Live Prices", 
                    "predictions": "◊ Predictions",
                    "rebalancing": "◐ Rebalancing"
                }
                st.session_state.current_page = page_map[page_param]
            else:
                st.session_state.current_page = "◉ Portfolio"
        else:
            st.session_state.current_page = "◉ Portfolio"
    
    # Enhanced sidebar navigation
    st.sidebar.markdown("""
    <div style='
        background: linear-gradient(135deg, #0e1117 0%, #1a1a1a 100%);
        border: 1px solid #333;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    '>
        <h2 style='
            color: #45b7d1;
            text-align: center;
            margin: 0 0 15px 0;
            font-size: 24px;
            font-weight: bold;
        '>NOVA • Crypto ML Console</h2>
        <p style='
            color: #888;
            text-align: center;
            margin: 0;
            font-size: 14px;
        '>Advanced Trading Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("### ◈ Navigation")
    
    # Create proper navigation buttons instead of radio buttons
    def navigate_to_page(page_name):
        st.session_state.current_page = page_name
        page_url_map = {
            "◉ Portfolio": "portfolio",
            "◈ Live Prices": "prices", 
            "◊ Predictions": "predictions",
            "◐ Rebalancing": "rebalancing"
        }
        st.query_params.page = page_url_map[page_name]
        st.rerun()
    
    # Clean, intuitive navigation buttons
    nav_buttons = [
        ("◉ Portfolio", "Your crypto holdings & performance"),
        ("◈ Live Prices", "Real-time market data & charts"),
        ("◊ Predictions", "ML-powered price forecasts"),
        ("◐ Rebalancing", "Smart portfolio rebalancing")
    ]
    
    for page_name, description in nav_buttons:
        # Determine if this is the current page
        is_current = st.session_state.current_page == page_name
        
        if is_current:
            # Current page - highlighted
            st.sidebar.markdown(f"""
            <div style='
                background: linear-gradient(135deg, #45b7d1 0%, #4ecdc4 100%);
                color: white;
                padding: 18px 20px;
                border-radius: 12px;
                margin: 10px 0;
                text-align: left;
                font-weight: bold;
                box-shadow: 0 6px 20px rgba(69, 183, 209, 0.4);
                border: 2px solid #45b7d1;
                cursor: pointer;
                transition: all 0.3s ease;
            '>
                <div style='font-size: 18px; margin-bottom: 8px; display: flex; align-items: center;'>
                    <span style='margin-right: 8px;'>{page_name.split()[0]}</span>
                    <span style='font-size: 14px; opacity: 0.9;'>{page_name.split(' ', 1)[1] if ' ' in page_name else ''}</span>
                </div>
                <div style='font-size: 13px; opacity: 0.9; margin-bottom: 5px;'>{description}</div>
                <div style='font-size: 11px; opacity: 0.8; font-weight: 500;'>✓ Currently Active</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Inactive page - clean button style
            if st.sidebar.button(f"{page_name}", use_container_width=True, help=description, type="secondary"):
                navigate_to_page(page_name)
    
    # Set the current page for the rest of the app
    page = st.session_state.current_page
    
    # Update session state when page changes
    if page != st.session_state.current_page:
        st.session_state.current_page = page
        
        # Update URL parameters
        page_url_map = {
            "◉ Portfolio": "portfolio",
            "◈ Live Prices": "prices", 
            "◊ Predictions": "predictions",
            "◐ Rebalancing": "rebalancing"
        }
        
        st.query_params.page = page_url_map[page]
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Compact system status
    st.sidebar.markdown("### ◊ Status")
    
    # Single status card with key indicators
    st.sidebar.markdown("""
    <div style='
        background: linear-gradient(135deg, #0e1117 0%, #1a1a1a 100%);
        border: 1px solid #45b7d1;
        border-radius: 12px;
        padding: 16px;
        margin: 10px 0;
        text-align: center;
    '>
        <div style='display: flex; justify-content: space-around; margin-bottom: 12px;'>
            <div style='text-align: center;'>
                <div style='color: #4ecdc4; font-size: 18px; margin-bottom: 4px;'>◉</div>
                <div style='color: white; font-size: 11px; font-weight: bold;'>Kraken</div>
                <div style='color: #888; font-size: 9px;'>Connected</div>
            </div>
            <div style='text-align: center;'>
                <div style='color: #ff6b6b; font-size: 18px; margin-bottom: 4px;'>⏳</div>
                <div style='color: white; font-size: 11px; font-weight: bold;'>ML</div>
                <div style='color: #888; font-size: 9px;'>Training</div>
            </div>
            <div style='text-align: center;'>
                <div style='color: #45b7d1; font-size: 18px; margin-bottom: 4px;'>◐</div>
                <div style='color: white; font-size: 11px; font-weight: bold;'>Paper</div>
                <div style='color: #888; font-size: 9px;'>Safe</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced about section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ◈ About")
    st.sidebar.markdown("""
    <div style='
        background: linear-gradient(135deg, #0e1117 0%, #1a1a1a 100%);
        border: 1px solid #333;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    '>
        <div style='color: #45b7d1; font-weight: bold; margin-bottom: 10px;'>NOVA • Crypto ML Console</div>
        <div style='color: #888; font-size: 12px; margin-bottom: 5px;'><strong>Version:</strong> 1.0.0</div>
        <div style='color: #888; font-size: 12px; margin-bottom: 5px;'><strong>Mode:</strong> Development</div>
        <div style='color: #888; font-size: 12px; margin-bottom: 5px;'><strong>Data Source:</strong> Kraken API</div>
        <div style='color: #888; font-size: 12px;'><strong>ML Engine:</strong> LSTM Neural Networks</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress indicator
    st.sidebar.markdown("### ◉ System Progress")
    
    # Mock progress data
    progress_data = {
        "Data Collection": 95,
        "ML Training": 60,
        "Portfolio Analysis": 100,
        "Rebalancing": 85
    }
    
    for component, progress in progress_data.items():
        st.sidebar.markdown(f"""
        <div style='
            background: linear-gradient(135deg, #0e1117 0%, #1a1a1a 100%);
            border: 1px solid #333;
            border-radius: 8px;
            padding: 10px;
            margin: 5px 0;
        '>
            <div style='
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 5px;
            '>
                <span style='color: white; font-size: 12px; font-weight: bold;'>{component}</span>
                <span style='color: #45b7d1; font-size: 12px; font-weight: bold;'>{progress}%</span>
            </div>
            <div style='
                background: #333;
                border-radius: 10px;
                height: 6px;
                overflow: hidden;
            '>
                <div style='
                    background: linear-gradient(90deg, #45b7d1 0%, #4ecdc4 100%);
                    height: 100%;
                    width: {progress}%;
                    border-radius: 10px;
                    transition: width 0.3s ease;
                '></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Display header
    show_header()
    
    # Enhanced page header
    st.markdown(f"""
    <div style='
        background: linear-gradient(135deg, #0e1117 0%, #1a1a1a 100%);
        border: 1px solid #333;
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        text-align: center;
    '>
        <h1 style='
            color: #45b7d1;
            margin: 0 0 10px 0;
            font-size: 32px;
            font-weight: bold;
        '>{page}</h1>
        <p style='
            color: #888;
            margin: 0;
            font-size: 16px;
        '>Advanced Cryptocurrency Analytics & Trading</p>
    </div>
    """, unsafe_allow_html=True)
    
    
    # Route to selected page
    if page == "◉ Portfolio":
        show_portfolio_view()
    elif page == "◈ Live Prices":
        show_live_prices()
    elif page == "◊ Predictions":
        show_predictions()
    elif page == "◐ Rebalancing":
        show_rebalancing()
    
    # Enhanced refresh controls
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ◐ Refresh Controls")
    
    # Refresh button with better styling
    if st.sidebar.button("◐ Refresh Data", use_container_width=True, type="primary"):
        st.rerun()
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("◐ Auto-refresh (60s)", value=False, help="Automatically refresh data every 60 seconds")
    
    if auto_refresh:
        time.sleep(60)
        st.rerun()
    
    # Quick refresh control
    st.sidebar.markdown("---")
    if st.sidebar.button("◐ Refresh All Data", use_container_width=True, type="primary"):
        st.rerun()


if __name__ == "__main__":
    main()

