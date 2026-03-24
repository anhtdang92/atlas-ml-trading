"""Live Prices page - Real-time stock prices with technical indicator overlays and multi-stock comparison."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime

from ui.styles import THEME, CHART_COLORS
from ui.components import section_header, metric_card, apply_chart_theme
from data.stock_api import STOCK_UNIVERSE, get_symbols_by_category, get_all_symbols


def show_live_prices(_stock_api):
    """Display live stock prices with charts by category."""
    col1, col2 = st.columns([3, 1])
    with col1:
        section_header("Live Stock Prices", icon="fa-arrow-trend-up")
    with col2:
        if st.button("Refresh Prices", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # Category selector
    categories = {
        'tech': 'Tech (FAANG+)',
        'sector_leaders': 'Sector Leaders',
        'defensive': 'Defensive',
        'etfs': 'ETFs',
        'growth': 'Growth',
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

    st.caption(f"Last updated: {datetime.now().strftime('%B %d, %Y at %I:%M:%S %p')}")
    st.markdown("### Market Overview")

    # Price cards in rows of 3
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
            change_color = THEME['accent_success'] if change_pct >= 0 else THEME['accent_danger']
            change_arrow = "▲" if change_pct >= 0 else "▼"

            with cols[col_idx]:
                st.markdown(f"""
                <div class='glass-card fade-in' style='animation-delay: {idx * 0.08}s;'>
                    <div style='display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;'>
                        <div>
                            <div style='color: {THEME['text_primary']}; font-size: 1rem; font-weight: 600;'>{info.get('name', symbol)}</div>
                            <span style='color: {THEME['text_muted']}; font-size: 11px;'>{symbol} · {info.get('sector', '')}</span>
                        </div>
                        <span style='color: {change_color}; font-weight: 600; font-size: 0.9rem;'>{change_arrow} {change_pct:+.2f}%</span>
                    </div>
                    <div style='color: {THEME['text_primary']}; font-size: 1.8rem; font-weight: 700; margin-bottom: 12px;'>${current_price:,.2f}</div>
                    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 6px; font-size: 12px;'>
                        <div><span style='color: {THEME['text_muted']};'>H</span> <strong style='color: {THEME['text_primary']};'>${quote['high']:,.2f}</strong></div>
                        <div><span style='color: {THEME['text_muted']};'>L</span> <strong style='color: {THEME['text_primary']};'>${quote['low']:,.2f}</strong></div>
                    </div>
                    <div style='margin-top: 10px; padding-top: 10px; border-top: 1px solid {THEME['border_color']}; font-size: 11px; color: {THEME['text_muted']};'>
                        Vol: <strong style='color: {THEME['text_secondary']};'>{quote['volume']:,.0f}</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    # ----- PRICE CHART with Volume Overlay and Indicators -----
    st.markdown("### Interactive Price Charts")

    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

    with col1:
        selected_symbol = st.selectbox(
            "Select stock:",
            options=symbols,
            format_func=lambda x: f"{stocks_info[x]['name']} ({x})" if x in stocks_info else x
        )
    with col2:
        period = st.selectbox(
            "Period:",
            options=["1mo", "3mo", "6mo", "1y", "2y"],
            format_func=lambda x: {"1mo": "1M", "3mo": "3M", "6mo": "6M", "1y": "1Y", "2y": "2Y"}[x],
            index=2
        )
    with col3:
        chart_type = st.selectbox("Type:", options=["Candlestick", "Line"], index=0)
    with col4:
        show_indicators = st.selectbox("Indicators:", options=["None", "Bollinger Bands", "RSI", "Both"], index=0)

    df = _stock_api.get_ohlc(selected_symbol, period=period)

    if df is not None and not df.empty:
        stock_color = stocks_info.get(selected_symbol, {}).get('color', '#4ecdc4')
        stock_name = stocks_info.get(selected_symbol, {}).get('name', selected_symbol)

        # Determine subplot layout
        show_rsi = show_indicators in ("RSI", "Both")
        n_rows = 2 if not show_rsi else 3
        row_heights = [0.6, 0.2] if not show_rsi else [0.5, 0.15, 0.2]

        fig = make_subplots(
            rows=n_rows, cols=1, shared_xaxes=True,
            row_heights=row_heights, vertical_spacing=0.03,
            subplot_titles=[f"{stock_name} ({selected_symbol})", "Volume"] + (["RSI (14)"] if show_rsi else []),
        )

        # Price trace
        if chart_type == "Candlestick":
            fig.add_trace(go.Candlestick(
                x=df['timestamp'], open=df['open'], high=df['high'],
                low=df['low'], close=df['close'], name='Price',
                increasing_line_color=THEME['accent_success'], decreasing_line_color=THEME['accent_danger'],
            ), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(
                x=df['timestamp'], y=df['close'], mode='lines',
                name='Price', line=dict(color=stock_color, width=2),
            ), row=1, col=1)

        # Bollinger Bands overlay
        if show_indicators in ("Bollinger Bands", "Both"):
            ma20 = df['close'].rolling(20).mean()
            std20 = df['close'].rolling(20).std()
            upper = ma20 + 2 * std20
            lower = ma20 - 2 * std20

            fig.add_trace(go.Scatter(
                x=df['timestamp'], y=upper, mode='lines', name='BB Upper',
                line=dict(color=THEME['accent_secondary'], width=1, dash='dot'),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df['timestamp'], y=ma20, mode='lines', name='BB Mid',
                line=dict(color=THEME['accent_primary'], width=1, dash='dash'),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df['timestamp'], y=lower, mode='lines', name='BB Lower',
                line=dict(color=THEME['accent_secondary'], width=1, dash='dot'),
                fill='tonexty', fillcolor='rgba(188, 19, 254, 0.05)',
            ), row=1, col=1)

        # Volume overlay (same chart, subplot row 2)
        vol_colors = [THEME['accent_success'] if c >= o else THEME['accent_danger']
                      for c, o in zip(df['close'], df['open'])]
        fig.add_trace(go.Bar(
            x=df['timestamp'], y=df['volume'], name='Volume',
            marker_color=vol_colors, opacity=0.6,
        ), row=2, col=1)

        # RSI subplot
        if show_rsi:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            fig.add_trace(go.Scatter(
                x=df['timestamp'], y=rsi, mode='lines', name='RSI',
                line=dict(color=THEME['accent_warning'], width=1.5),
            ), row=3, col=1)
            # Overbought/oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color=THEME['accent_danger'],
                          opacity=0.5, row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color=THEME['accent_success'],
                          opacity=0.5, row=3, col=1)
            fig.update_yaxes(range=[0, 100], row=3, col=1)

        apply_chart_theme(fig)
        fig.update_layout(
            height=550 if not show_rsi else 650,
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(orientation="h", y=1.02, x=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Market stats
        st.markdown("### Market Statistics")
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

    # ----- MULTI-STOCK COMPARISON -----
    st.markdown("---")
    st.markdown("### Multi-Stock Comparison")

    compare_symbols = st.multiselect(
        "Compare stocks (normalized to 100):",
        options=get_all_symbols(),
        default=symbols[:3] if len(symbols) >= 3 else symbols,
        max_selections=6,
    )

    if compare_symbols and len(compare_symbols) >= 2:
        compare_period = st.selectbox("Comparison period:", ["3mo", "6mo", "1y", "2y"], index=1, key="compare_period")

        fig_compare = go.Figure()
        for i, sym in enumerate(compare_symbols):
            cdf = _stock_api.get_ohlc(sym, period=compare_period)
            if cdf is not None and not cdf.empty:
                # Normalize to 100 at start
                normalized = (cdf['close'] / cdf['close'].iloc[0]) * 100
                fig_compare.add_trace(go.Scatter(
                    x=cdf['timestamp'], y=normalized, mode='lines',
                    name=sym, line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
                ))

        apply_chart_theme(fig_compare)
        fig_compare.update_layout(
            title="Normalized Performance (base = 100)",
            yaxis_title="Normalized Price",
            height=400,
            hovermode='x unified',
        )
        st.plotly_chart(fig_compare, use_container_width=True)
