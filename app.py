"""
ATLAS - Stock ML Intelligence Dashboard

Main Streamlit application for AI-powered stock trading with ML price predictions
(LSTM neural networks), real-time Yahoo Finance data, and Google Cloud ML infrastructure.

This module serves as the thin router/entry point. Page logic lives in ui/pages/.
"""

import streamlit as st
import time
from datetime import datetime

# Import UI components
from ui.styles import THEME
from ui.components import load_css

# Import page modules
from ui.pages.portfolio import show_portfolio_view
from ui.pages.live_prices import show_live_prices
from ui.pages.predictions import show_predictions
from ui.pages.rebalancing import show_rebalancing
from ui.pages.cloud_progress import show_cloud_progress

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
from data.stock_api import StockAPI

# Create a shared StockAPI instance
_stock_api = StockAPI()


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
        show_portfolio_view(_stock_api)
    elif page == "↗ Live Prices":
        show_live_prices(_stock_api)
    elif page == "◉ ML Predictions":
        show_predictions(_stock_api)
    elif page == "◉ Rebalancing":
        show_rebalancing(_stock_api)
    elif page == "☁️ Cloud Progress":
        show_cloud_progress(_stock_api)

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
