"""Rebalancing page - ML-enhanced portfolio allocation with paper/live trading."""

import streamlit as st
import pandas as pd
import plotly.express as px
from ui.styles import CHART_COLORS

from ui.styles import THEME
from ui.components import section_header, status_card, kpi_box, apply_chart_theme


def show_rebalancing(_stock_api):
    """Display portfolio rebalancing interface."""
    section_header("Portfolio Rebalancing", icon="fa-sliders-h")

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
            status_card("Enhanced Mock",
                        f'<div style="font-size: 16px; font-weight: bold; color: #4caf50; margin-bottom: 5px;">Active</div>',
                        "#4caf50", subtitle="Always available fallback")

        with col2:
            vertex_available = system_summary['vertex_ai_available']
            vertex_color = THEME['accent_success'] if vertex_available else THEME['accent_danger']
            vertex_label = "Available" if vertex_available else "Not Available"
            status_card("Vertex AI ML",
                        f'<div style="font-size: 16px; font-weight: bold; color: {vertex_color}; margin-bottom: 5px;">{vertex_label}</div>',
                        vertex_color, subtitle="Cloud ML models")

        with col3:
            local_available = system_summary['local_ml_models_available']
            local_color = THEME['accent_success'] if local_available else THEME['accent_danger']
            local_label = "Available" if local_available else "Not Available"
            status_card("Local ML Models",
                        f'<div style="font-size: 16px; font-weight: bold; color: {local_color}; margin-bottom: 5px;">{local_label}</div>',
                        local_color, subtitle="Locally trained models")

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
            color_discrete_sequence=CHART_COLORS
        )
        apply_chart_theme(fig_current)
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
            color_discrete_sequence=CHART_COLORS
        )
        apply_chart_theme(fig_target)
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

        # Order summary
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_orders = len(summary['orders'])
            order_color = "#ff6b6b" if total_orders > 8 else "#ffa726" if total_orders > 4 else "#4caf50"
            order_status = "Many Trades" if total_orders > 8 else "Moderate" if total_orders > 4 else "Few Trades"
            kpi_box(str(total_orders), "Total Orders", order_color, order_status, "Rebalancing trades needed")

        with col2:
            total_fees = summary['metrics']['total_fees']
            fee_color = "#ff6b6b" if total_fees > 100 else "#ffa726" if total_fees > 50 else "#4caf50"
            fee_status = "High Cost" if total_fees > 100 else "Moderate" if total_fees > 50 else "Low Cost"
            kpi_box(f"${total_fees:.2f}", "Total Fees", fee_color, fee_status, "Rebalancing cost")

        with col3:
            buy_orders = summary['metrics']['buy_orders']
            buy_color = "#4caf50" if buy_orders > 0 else "#666"
            buy_status = "Buying" if buy_orders > 0 else "No Buys"
            kpi_box(str(buy_orders), "Buy Orders", buy_color, buy_status, "Positions to increase")

        with col4:
            sell_orders = summary['metrics']['sell_orders']
            sell_color = "#ff6b6b" if sell_orders > 0 else "#666"
            sell_status = "Selling" if sell_orders > 0 else "No Sells"
            kpi_box(str(sell_orders), "Sell Orders", sell_color, sell_status, "Positions to decrease")

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

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        max_drift = summary['metrics']['max_drift'] * 100
        drift_color = "#ff6b6b" if max_drift > 5 else "#ffa726" if max_drift > 2 else "#4caf50"
        drift_status = "High Risk" if max_drift > 5 else "Moderate" if max_drift > 2 else "Good"
        kpi_box(f"{max_drift:.1f}%", "Max Drift", drift_color, drift_status, "Largest allocation deviation")

    with col2:
        avg_drift = summary['metrics']['avg_drift'] * 100
        avg_color = "#ff6b6b" if avg_drift > 3 else "#ffa726" if avg_drift > 1 else "#4caf50"
        avg_status = "High Risk" if avg_drift > 3 else "Moderate" if avg_drift > 1 else "Good"
        kpi_box(f"{avg_drift:.1f}%", "Avg Drift", avg_color, avg_status, "Average deviation")

    with col3:
        total_fees = summary['metrics']['total_fees']
        fee_color = "#ff6b6b" if total_fees > 100 else "#ffa726" if total_fees > 50 else "#4caf50"
        fee_status = "High Cost" if total_fees > 100 else "Moderate" if total_fees > 50 else "Low Cost"
        kpi_box(f"${total_fees:.2f}", "Trading Fees", fee_color, fee_status, "Total rebalancing cost")

    with col4:
        fee_percentage = (summary['metrics']['total_fees'] / portfolio_value) * 100
        fee_pct_color = "#ff6b6b" if fee_percentage > 0.5 else "#ffa726" if fee_percentage > 0.2 else "#4caf50"
        fee_pct_status = "High %" if fee_percentage > 0.5 else "Moderate" if fee_percentage > 0.2 else "Low %"
        kpi_box(f"{fee_percentage:.2f}%", "Fee %", fee_pct_color, fee_pct_status, "Fees vs portfolio value")

    # Rebalancing Summary
    st.markdown("### ◉ Rebalancing Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_orders = len(summary['orders'])
        order_color = "#ff6b6b" if total_orders > 8 else "#ffa726" if total_orders > 4 else "#4caf50"
        order_status = "Many Trades" if total_orders > 8 else "Moderate" if total_orders > 4 else "Few Trades"
        kpi_box(str(total_orders), "Total Orders", order_color, order_status, "Rebalancing trades needed")

    with col2:
        buy_orders = summary['metrics']['buy_orders']
        buy_color = "#4caf50" if buy_orders > 0 else "#666"
        buy_status = "Buying" if buy_orders > 0 else "No Buys"
        kpi_box(str(buy_orders), "Buy Orders", buy_color, buy_status, "Positions to increase")

    with col3:
        sell_orders = summary['metrics']['sell_orders']
        sell_color = "#ff6b6b" if sell_orders > 0 else "#666"
        sell_status = "Selling" if sell_orders > 0 else "No Sells"
        kpi_box(str(sell_orders), "Sell Orders", sell_color, sell_status, "Positions to decrease")

    with col4:
        # Portfolio health score
        max_drift = summary['metrics']['max_drift'] * 100
        avg_drift = summary['metrics']['avg_drift'] * 100
        fee_pct = (summary['metrics']['total_fees'] / portfolio_value) * 100

        drift_score = max(0, 100 - (max_drift * 10))
        fee_score = max(0, 100 - (fee_pct * 200))
        health_score = (drift_score + fee_score) / 2

        health_color = "#4caf50" if health_score > 80 else "#ffa726" if health_score > 60 else "#ff6b6b"
        health_status = "Excellent" if health_score > 80 else "Good" if health_score > 60 else "Needs Attention"
        kpi_box(f"{health_score:.0f}", "Health Score", health_color, health_status, "Overall portfolio health")

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
