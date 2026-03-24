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
        icon_html = f'<i class="fas {icon}" style="font-size: 1.5rem; color: {THEME["accent_primary"]}; margin-bottom: 10px;"></i>'

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
    icon_html = f'<i class="fas {icon}" style="margin-right: 10px; color: {THEME["accent_secondary"]};"></i>' if icon else ""
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
        <i class="fas fa-circle" style="font-size: 6px; vertical-align: middle; margin-right: 4px;"></i> {display_text}
    </span>
    """


def kpi_box(value, label, color, status_text=None, subtitle=None):
    """Reusable KPI box component. Replaces all duplicated inline HTML KPI cards.

    Args:
        value: The main display value (e.g. "42", "$100.00", "85%")
        label: The metric label (e.g. "Total Orders")
        color: Border/accent color hex (e.g. "#4caf50")
        status_text: Optional status line below value (e.g. "Good", "High Risk")
        subtitle: Optional small description text at bottom
    """
    status_html = ""
    if status_text:
        status_html = f'<div style="font-size: 12px; color: {color}; margin-bottom: 5px; font-weight: 500;">{status_text}</div>'

    subtitle_html = ""
    if subtitle:
        subtitle_html = f'<div style="font-size: 11px; color: {THEME["text_secondary"]};">{subtitle}</div>'

    st.markdown(f"""
    <div class="kpi-box" style="border: 2px solid {color};">
        <div style="font-size: 28px; font-weight: bold; color: {color}; margin-bottom: 10px; text-shadow: 0 0 10px {color}40;">
            {value}
        </div>
        <div style="font-size: 16px; font-weight: bold; color: {THEME['text_primary']}; margin-bottom: 5px;">
            {label}
        </div>
        {status_html}
        {subtitle_html}
    </div>
    """, unsafe_allow_html=True)


def status_card(title, value_html, color, subtitle=None):
    """Reusable status card for cloud/system status displays.

    Args:
        title: Card title (e.g. "Training Status")
        value_html: HTML content for the main value area (can include emoji/icons)
        color: Border color hex
        subtitle: Optional small text at bottom
    """
    subtitle_html = ""
    if subtitle:
        subtitle_html = f'<div style="font-size: 11px; color: {THEME["text_secondary"]};">{subtitle}</div>'

    st.markdown(f"""
    <div class="status-card" style="border: 2px solid {color};">
        {value_html}
        <div style="font-size: 16px; font-weight: bold; color: {THEME['text_primary']}; margin-bottom: 5px;">
            {title}
        </div>
        {subtitle_html}
    </div>
    """, unsafe_allow_html=True)


def job_card(name, status_text, time_str, color, extra_info=None):
    """Card for displaying training jobs or endpoints.

    Args:
        name: Job/endpoint display name
        status_text: Status label
        time_str: Formatted timestamp
        color: Left-border accent color
        extra_info: Optional additional info line
    """
    extra_html = ""
    if extra_info:
        extra_html = f'<div style="font-size: 12px; color: {color}; margin-bottom: 3px;">{extra_info}</div>'

    st.markdown(f"""
    <div class="job-card" style="border-left: 4px solid {color};">
        <div style="font-weight: bold; color: {THEME['text_primary']}; margin-bottom: 5px;">
            {name}
        </div>
        <div style="font-size: 12px; color: {color}; margin-bottom: 3px;">
            {status_text}
        </div>
        {extra_html}
        <div style="font-size: 11px; color: {THEME['text_secondary']};">
            {time_str}
        </div>
    </div>
    """, unsafe_allow_html=True)


def apply_chart_theme(fig, title_color=None):
    """Apply consistent ATLAS dark theme to any Plotly figure.

    Args:
        fig: Plotly figure object
        title_color: Optional override for title color (defaults to accent_primary)
    """
    from ui.styles import CHART_LAYOUT
    fig.update_layout(**CHART_LAYOUT)
    if title_color:
        fig.update_layout(title_font=dict(color=title_color))
    return fig
