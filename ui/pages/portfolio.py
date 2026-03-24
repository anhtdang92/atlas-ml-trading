"""Portfolio page - Demo stock portfolio with real-time prices, P&L tracking, and allocation."""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

from ui.styles import THEME
from ui.components import section_header, metric_card, apply_chart_theme
from data.stock_api import get_stock_info


def _get_demo_portfolio():
    """Return demo stock portfolio data.

    v2.0: Diversified across tech, financials, defensive staples, bonds, and gold.
    Removed SPY to avoid mega-cap overlap with individual holdings.
    Added JNJ (healthcare defensive), KO (consumer staples), TLT (bonds), GLD (gold).
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
            color_discrete_sequence=['#00f3ff', '#bc13fe', '#00ff9d', '#ffb800', '#ff0055', '#45b7d1'],
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=14)
        apply_chart_theme(fig)
        fig.update_layout(
            height=400, showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.success("◉ **Connected to Yahoo Finance** - Real-time stock prices via yfinance (free, no API key).")
