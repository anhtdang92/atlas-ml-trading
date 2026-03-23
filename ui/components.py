"""
ATLAS UI Components - Reusable widgets and layout helpers
"""
import streamlit as st
from ui.styles import THEME


def load_css():
    """Inject global CSS."""
    from ui.styles import GLOBAL_CSS
    st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">', unsafe_allow_html=True)
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def card_start():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)


def card_end():
    st.markdown('</div>', unsafe_allow_html=True)


def metric_card(label, value, delta=None, delta_color="normal", icon=None):
    delta_html = ""
    if delta:
        color = THEME['text_secondary']
        icon_arrow = ""
        if delta_color == "normal":
            if delta.startswith("+"):
                color = THEME['accent_success']
                icon_arrow = "▲"
            elif delta.startswith("-"):
                color = THEME['accent_danger']
                icon_arrow = "▼"
        delta_html = f'<div style="color: {color}; font-size: 0.9rem; margin-top: 4px;">{icon_arrow} {delta}</div>'

    icon_html = ""
    if icon:
        icon_html = f'<i class="fa-solid {icon}" style="font-size: 1.5rem; color: {THEME["accent_primary"]}; margin-bottom: 10px;"></i>'

    st.markdown(f"""
    <div class="glass-card" style="text-align: center; padding: 15px;">
        {icon_html}
        <div style="color: {THEME['text_secondary']}; font-family: 'Orbitron'; font-size: 0.8rem; letter-spacing: 0.1em; text-transform: uppercase;">{label}</div>
        <div style="color: {THEME['text_primary']}; font-family: 'Rajdhani'; font-size: 1.8rem; font-weight: 700; margin: 5px 0; text-shadow: 0 0 10px {THEME['glow_primary']};">
            {value}
        </div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def section_header(title, icon=None):
    icon_html = f'<i class="fa-solid {icon}" style="margin-right: 10px; color: {THEME["accent_secondary"]};"></i>' if icon else ""
    st.markdown(f"""
    <h2 style="border-bottom: 1px solid {THEME['border_color']}; padding-bottom: 10px; margin-top: 30px; margin-bottom: 20px;">
        {icon_html}{title}
    </h2>
    """, unsafe_allow_html=True)


def status_badge(status, text=None):
    colors = {
        "online": THEME['accent_success'],
        "offline": THEME['text_muted'],
        "error": THEME['accent_danger'],
        "training": THEME['accent_warning'],
        "market_open": THEME['accent_success'],
        "market_closed": THEME['accent_warning'],
    }
    color = colors.get(status, THEME['accent_primary'])
    display_text = text if text else status.upper()

    return f"""
    <span style="
        background: rgba(0,0,0,0.3);
        border: 1px solid {color};
        color: {color};
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-family: 'Orbitron';
        box-shadow: 0 0 5px {color};
    ">
        <i class="fa-solid fa-circle" style="font-size: 6px; vertical-align: middle; margin-right: 4px;"></i> {display_text}
    </span>
    """
