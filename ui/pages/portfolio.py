"""Portfolio page - Stock portfolio with real-time prices, P&L tracking, sector breakdown, and analytics."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from ui.styles import THEME, CHART_COLORS
from ui.components import (
    section_header, metric_card, apply_chart_theme,
    skeleton_loader, pnl_text, pnl_pct_text, sector_exposure_bar,
)
from data.stock_api import get_stock_info, SECTOR_MAP


def _get_demo_portfolio():
    """Return demo stock portfolio data.

    v2.0: Diversified across tech, financials, defensive staples, bonds, and gold.
    Removed SPY to avoid mega-cap overlap with individual holdings.
    """
    return {
        'AAPL': {'quantity': 15, 'avg_buy_price': 175.00, 'current_price': 185.00},
        'MSFT': {'quantity': 8, 'avg_buy_price': 380.00, 'current_price': 420.00},
        'NVDA': {'quantity': 5, 'avg_buy_price': 750.00, 'current_price': 880.00},
        'JPM': {'quantity': 15, 'avg_buy_price': 185.00, 'current_price': 200.00},
        'JNJ': {'quantity': 20, 'avg_buy_price': 155.00, 'current_price': 162.00},
        'KO': {'quantity': 40, 'avg_buy_price': 58.00, 'current_price': 62.00},
        'PG': {'quantity': 15, 'avg_buy_price': 155.00, 'current_price': 165.00},
        'XOM': {'quantity': 20, 'avg_buy_price': 105.00, 'current_price': 112.00},
        'TLT': {'quantity': 25, 'avg_buy_price': 92.00, 'current_price': 95.00},
        'GLD': {'quantity': 12, 'avg_buy_price': 185.00, 'current_price': 198.00},
        'IWM': {'quantity': 10, 'avg_buy_price': 195.00, 'current_price': 205.00},
    }


def show_portfolio_view(_stock_api):
    """Display stock portfolio overview with holdings and performance."""
    col1, col2 = st.columns([3, 1])
    with col1:
        section_header("Portfolio Overview", icon="fa-wallet")
    with col2:
        if st.button("Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    portfolio_data = _get_demo_portfolio()

    # Fetch live prices
    with st.spinner("Fetching live stock prices..."):
        quotes = _stock_api.get_batch_quotes(list(portfolio_data.keys()))
        for symbol, quote in quotes.items():
            if symbol in portfolio_data and quote:
                portfolio_data[symbol]['current_price'] = quote['current']

    # Calculate portfolio metrics
    total_value = 0
    total_cost = 0
    holdings = []
    sector_values = {}

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

        # Accumulate sector values
        sector_values[sector] = sector_values.get(sector, 0) + current_value

        holdings.append({
            'Symbol': symbol,
            'Sector': sector,
            'Shares': quantity,
            'Avg Buy': avg_price,
            'Price': current_price,
            'Value': current_value,
            'P&L ($)': pnl,
            'P&L (%)': pnl_pct,
            'Weight': 0,  # filled below
        })

    total_pnl = total_value - total_cost
    total_pnl_pct = (total_pnl / total_cost) * 100 if total_cost > 0 else 0

    # Fill weight
    for h in holdings:
        h['Weight'] = (h['Value'] / total_value * 100) if total_value > 0 else 0

    st.caption(f"Last updated: {datetime.now().strftime('%B %d, %Y at %I:%M:%S %p')}")

    # ----- KEY METRICS ROW -----
    st.markdown("### Portfolio Summary")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card("TOTAL VALUE", f"${total_value:,.2f}", icon="fa-wallet")
    with col2:
        pnl_delta = f"{total_pnl:+.2f} ({total_pnl_pct:+.2f}%)"
        metric_card("TOTAL P&L", f"${total_pnl:+,.2f}", delta=pnl_delta, icon="fa-chart-line")
    with col3:
        metric_card("POSITIONS", str(len(portfolio_data)), icon="fa-layer-group")
    with col4:
        cash = total_value * 0.05  # 5% cash reserve
        metric_card("CASH RESERVE", f"${cash:,.2f}", icon="fa-coins")

    st.markdown("---")

    # ----- HOLDINGS TABLE with colored P&L -----
    st.markdown("### Current Holdings")

    if holdings:
        df = pd.DataFrame(holdings)
        df = df.sort_values('Value', ascending=False)

        # Format for display
        display_df = pd.DataFrame({
            'Symbol': df['Symbol'],
            'Sector': df['Sector'],
            'Shares': df['Shares'].apply(lambda x: f"{x:,.0f}"),
            'Avg Buy': df['Avg Buy'].apply(lambda x: f"${x:,.2f}"),
            'Price': df['Price'].apply(lambda x: f"${x:,.2f}"),
            'Value': df['Value'].apply(lambda x: f"${x:,.2f}"),
            'P&L ($)': df['P&L ($)'].apply(lambda x: f"${x:+,.2f}"),
            'P&L (%)': df['P&L (%)'].apply(lambda x: f"{x:+.2f}%"),
            'Weight': df['Weight'].apply(lambda x: f"{x:.1f}%"),
        })

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                "Sector": st.column_config.TextColumn("Sector", width="medium"),
                "Shares": st.column_config.TextColumn("Shares", width="small"),
                "Price": st.column_config.TextColumn("Price", width="small"),
                "Value": st.column_config.TextColumn("Value", width="medium"),
                "P&L ($)": st.column_config.TextColumn("P&L ($)", width="small"),
                "P&L (%)": st.column_config.TextColumn("P&L (%)", width="small"),
                "Weight": st.column_config.TextColumn("Wt%", width="small"),
            }
        )

        # ----- P&L SUMMARY BAR -----
        winners = [h for h in holdings if h['P&L ($)'] > 0]
        losers = [h for h in holdings if h['P&L ($)'] < 0]
        st.markdown(
            f"<div style='display:flex; gap:20px; font-size:0.9rem;'>"
            f"<span class='pnl-positive'>{len(winners)} winners</span>"
            f"<span class='pnl-negative'>{len(losers)} losers</span>"
            f"<span style='color:{THEME[\"text_muted\"]};'>Best: {max(holdings, key=lambda x: x['P&L (%)'])['Symbol']} "
            f"({max(holdings, key=lambda x: x['P&L (%)'])['P&L (%)']:+.1f}%)</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.caption("Note: Demo portfolio for paper trading. P&L calculated from simulated buy prices.")
    else:
        st.info("No holdings to display")

    st.markdown("---")

    # ----- CHARTS: Allocation + Sector -----
    col_alloc, col_sector = st.columns(2)

    with col_alloc:
        st.markdown("### Allocation by Stock")
        alloc_data = pd.DataFrame([
            {'Symbol': h['Symbol'], 'Value': h['Value']}
            for h in holdings if h['Value'] > 0
        ])
        if not alloc_data.empty:
            fig = px.pie(
                alloc_data, values='Value', names='Symbol',
                color_discrete_sequence=CHART_COLORS, hole=0.4,
            )
            fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=12)
            apply_chart_theme(fig)
            fig.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col_sector:
        st.markdown("### Allocation by Sector")
        sector_df = pd.DataFrame([
            {'Sector': s, 'Value': v} for s, v in sector_values.items()
        ]).sort_values('Value', ascending=False)
        if not sector_df.empty:
            fig = px.pie(
                sector_df, values='Value', names='Sector',
                color_discrete_sequence=CHART_COLORS[5:] + CHART_COLORS[:5], hole=0.4,
            )
            fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=12)
            apply_chart_theme(fig)
            fig.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # ----- SECTOR EXPOSURE BAR -----
    st.markdown("### Sector Exposure")
    sector_weights = {s: v / total_value for s, v in sector_values.items()} if total_value > 0 else {}
    limits = load_sector_limits()
    sector_exposure_bar(sector_weights, limits)

    # ----- PORTFOLIO ANALYTICS -----
    st.markdown("---")
    st.markdown("### Portfolio Analytics")
    col1, col2, col3, col4 = st.columns(4)

    # Simple analytics from the demo data
    weights = np.array([h['Weight'] / 100 for h in holdings])
    returns = np.array([h['P&L (%)'] / 100 for h in holdings])

    with col1:
        # Weighted return
        weighted_return = np.sum(weights * returns)
        metric_card("WEIGHTED RETURN", f"{weighted_return*100:+.2f}%", icon="fa-percent")
    with col2:
        # Concentration (HHI)
        hhi = np.sum(weights ** 2)
        effective_n = 1 / hhi if hhi > 0 else 0
        metric_card("EFFECTIVE DIVERSIFICATION", f"{effective_n:.1f} stocks", icon="fa-project-diagram")
    with col3:
        # Top holding weight
        max_weight = max(h['Weight'] for h in holdings)
        metric_card("LARGEST POSITION", f"{max_weight:.1f}%", icon="fa-arrow-up")
    with col4:
        # Number of sectors
        n_sectors = len(sector_values)
        metric_card("SECTORS", str(n_sectors), icon="fa-th")

    st.success("Connected to Yahoo Finance - Real-time stock prices via yfinance (free, no API key).")


def load_sector_limits():
    """Load sector limits from config, with fallback defaults."""
    try:
        import json
        from pathlib import Path
        config_path = Path("config/rebalancing_config.json")
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            return config.get("sector_limits", {})
    except Exception:
        pass
    return {}
