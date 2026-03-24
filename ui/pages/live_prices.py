"""Live Prices page - Real-time stock prices by category with interactive charts."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from ui.styles import THEME
from ui.components import section_header, metric_card, apply_chart_theme
from data.stock_api import STOCK_UNIVERSE, get_symbols_by_category


def show_live_prices(_stock_api):
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

        apply_chart_theme(fig, title_color=stock_color)
        fig.update_layout(
            title=f"{stock_name} ({selected_symbol}) Price Chart",
            yaxis_title="Price (USD)", xaxis_title="Date", height=500,
            xaxis_rangeslider_visible=False,
            title_font=dict(size=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Volume chart
        fig_volume = px.bar(df, x='timestamp', y='volume',
                           title=f"{selected_symbol} Trading Volume",
                           labels={'volume': 'Volume', 'timestamp': 'Date'})
        apply_chart_theme(fig_volume, title_color=stock_color)
        fig_volume.update_layout(height=200, title_font=dict(size=16))
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
