"""
ATLAS UI Styles - Modern Financial Dashboard Theme
"""

THEME = {
    'bg_primary': '#050505',
    'bg_secondary': '#0a0a0a',
    'bg_card': 'rgba(15, 15, 15, 0.7)',
    'bg_glass': 'rgba(10, 10, 10, 0.5)',
    'text_primary': '#ffffff',
    'text_secondary': '#a0a0a0',
    'text_muted': '#505050',
    'accent_primary': '#00f3ff',    # Neon Cyan
    'accent_secondary': '#bc13fe',  # Neon Purple
    'accent_success': '#00ff9d',    # Neon Green
    'accent_warning': '#ffb800',    # Neon Orange
    'accent_danger': '#ff0055',     # Neon Red
    'border_color': 'rgba(0, 243, 255, 0.1)',
    'shadow': 'rgba(0, 0, 0, 0.5)',
    'glow_primary': 'rgba(0, 243, 255, 0.4)',
    'glow_secondary': 'rgba(188, 19, 254, 0.4)',
}

GLOBAL_CSS = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&display=swap');

    .stApp {{
        background-color: {THEME['bg_primary']};
        background-image:
            radial-gradient(circle at 10% 20%, rgba(0, 243, 255, 0.03) 0%, transparent 20%),
            radial-gradient(circle at 90% 80%, rgba(188, 19, 254, 0.03) 0%, transparent 20%),
            linear-gradient(0deg, rgba(0,0,0,0.2) 0%, transparent 1px),
            linear-gradient(90deg, rgba(0,0,0,0.2) 0%, transparent 1px);
        background-size: 100% 100%, 100% 100%, 40px 40px, 40px 40px;
    }}

    *:not(i) {{
        font-family: 'Rajdhani', sans-serif !important;
        letter-spacing: 0.03em;
    }}

    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Orbitron', sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: {THEME['text_primary']} !important;
    }}

    [data-testid="stSidebar"] {{
        background-color: {THEME['bg_secondary']} !important;
        border-right: 1px solid {THEME['border_color']};
        box-shadow: 5px 0 30px rgba(0,0,0,0.5);
    }}

    [data-testid="stSidebar"] .stMarkdown h1 {{
        font-size: 1.5rem;
        background: linear-gradient(90deg, {THEME['accent_primary']}, {THEME['accent_secondary']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}

    .glass-card {{
        background: {THEME['bg_card']};
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid {THEME['border_color']};
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }}

    .glass-card::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; width: 100%; height: 2px;
        background: linear-gradient(90deg, transparent, {THEME['accent_primary']}, transparent);
        opacity: 0.5;
    }}

    .glass-card:hover {{
        transform: translateY(-2px);
        border-color: {THEME['accent_primary']};
        box-shadow: 0 15px 40px -10px {THEME['glow_primary']};
    }}

    [data-testid="stMetric"] {{
        background: {THEME['bg_glass']};
        padding: 15px;
        border-radius: 10px;
        border: 1px solid {THEME['border_color']};
        transition: all 0.3s ease;
    }}

    [data-testid="stMetric"]:hover {{
        border-color: {THEME['accent_primary']};
        box-shadow: 0 0 15px {THEME['glow_primary']};
    }}

    [data-testid="stMetricLabel"] {{
        color: {THEME['text_secondary']} !important;
        font-size: 0.9rem !important;
        font-family: 'Orbitron', sans-serif !important;
    }}

    [data-testid="stMetricValue"] {{
        color: {THEME['text_primary']} !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        text-shadow: 0 0 10px {THEME['glow_primary']};
    }}

    .stButton > button {{
        background: linear-gradient(45deg, rgba(0, 243, 255, 0.1), rgba(188, 19, 254, 0.1));
        border: 1px solid {THEME['accent_primary']};
        color: {THEME['accent_primary']};
        font-family: 'Orbitron', sans-serif;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        transition: all 0.3s ease;
        border-radius: 4px;
        padding: 0.5rem 1rem;
    }}

    .stButton > button:hover {{
        background: linear-gradient(45deg, {THEME['accent_primary']}, {THEME['accent_secondary']});
        color: #000;
        border-color: transparent;
        box-shadow: 0 0 20px {THEME['glow_primary']};
        transform: scale(1.02);
    }}

    [data-testid="stDataFrame"] {{
        background: {THEME['bg_card']};
        border: 1px solid {THEME['border_color']};
        border-radius: 8px;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: transparent;
    }}

    .stTabs [data-baseweb="tab"] {{
        background-color: rgba(255,255,255,0.05);
        border-radius: 4px;
        border: 1px solid transparent;
        color: {THEME['text_secondary']};
        padding: 8px 16px;
    }}

    .stTabs [aria-selected="true"] {{
        background-color: rgba(0, 243, 255, 0.1) !important;
        border-color: {THEME['accent_primary']} !important;
        color: {THEME['accent_primary']} !important;
        box-shadow: 0 0 15px {THEME['glow_primary']};
    }}

    @keyframes pulse-glow {{
        0% {{ box-shadow: 0 0 5px {THEME['glow_primary']}; }}
        50% {{ box-shadow: 0 0 20px {THEME['glow_primary']}; }}
        100% {{ box-shadow: 0 0 5px {THEME['glow_primary']}; }}
    }}

    .pulse {{ animation: pulse-glow 2s infinite; }}

    .neon-text {{
        color: {THEME['accent_primary']};
        text-shadow: 0 0 10px {THEME['glow_primary']};
    }}

    .neon-text-purple {{
        color: {THEME['accent_secondary']};
        text-shadow: 0 0 10px {THEME['glow_secondary']};
    }}

    .text-success {{ color: {THEME['accent_success']}; }}
    .text-warning {{ color: {THEME['accent_warning']}; }}
    .text-danger {{ color: {THEME['accent_danger']}; }}

    </style>
"""
