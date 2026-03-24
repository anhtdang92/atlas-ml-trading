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
    'text_muted': '#888888',       # WCAG AA contrast on #0a0a0a (was #707070)
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

# Extended palette for charts with many categories (33 stocks)
CHART_COLORS = [
    '#00f3ff', '#bc13fe', '#00ff9d', '#ffb800', '#ff0055', '#45b7d1',
    '#ff6b6b', '#c084fc', '#34d399', '#fbbf24', '#f472b6', '#60a5fa',
    '#a78bfa', '#4ade80', '#fb923c', '#e879f9', '#38bdf8', '#facc15',
    '#f87171', '#2dd4bf', '#818cf8', '#a3e635', '#fb7185', '#22d3ee',
    '#d946ef', '#84cc16', '#f59e0b', '#ef4444', '#10b981', '#8b5cf6',
    '#06b6d4', '#eab308', '#ec4899',
]

# Standard chart layout for consistent Plotly styling
CHART_LAYOUT = dict(
    template="plotly_dark",
    plot_bgcolor='rgba(14, 17, 23, 0.8)',
    paper_bgcolor='rgba(0, 0, 0, 0)',
    font=dict(color='#e0e0e0', family='Rajdhani, sans-serif'),
    xaxis=dict(gridcolor='rgba(255,255,255,0.06)', zeroline=False),
    yaxis=dict(gridcolor='rgba(255,255,255,0.06)', zeroline=False),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(
        bgcolor='rgba(0,0,0,0)',
        bordercolor='rgba(255,255,255,0.1)',
        font=dict(size=11),
    ),
)

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

    /* Animations */
    @keyframes pulse-glow {{
        0% {{ box-shadow: 0 0 5px {THEME['glow_primary']}; }}
        50% {{ box-shadow: 0 0 20px {THEME['glow_primary']}; }}
        100% {{ box-shadow: 0 0 5px {THEME['glow_primary']}; }}
    }}
    .pulse {{ animation: pulse-glow 2s infinite; }}

    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(8px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    .fade-in {{ animation: fadeIn 0.4s ease-out forwards; }}

    @keyframes slideUp {{
        from {{ opacity: 0; transform: translateY(16px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    .slide-up {{ animation: slideUp 0.5s ease-out forwards; }}

    @keyframes shimmer {{
        0% {{ background-position: -200% 0; }}
        100% {{ background-position: 200% 0; }}
    }}

    /* Gradient text */
    .gradient-text {{
        background: linear-gradient(90deg, {THEME['accent_primary']}, {THEME['accent_secondary']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}

    /* Neon text classes */
    .neon-text {{
        color: {THEME['accent_primary']};
        text-shadow: 0 0 10px {THEME['glow_primary']};
    }}
    .neon-text-purple {{
        color: {THEME['accent_secondary']};
        text-shadow: 0 0 10px {THEME['glow_secondary']};
    }}

    /* Status text classes */
    .text-success {{ color: {THEME['accent_success']}; }}
    .text-warning {{ color: {THEME['accent_warning']}; }}
    .text-danger {{ color: {THEME['accent_danger']}; }}

    /* KPI box */
    .kpi-box {{
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .kpi-box:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.4);
    }}

    /* Status card for cloud/system */
    .status-card {{
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        margin: 10px 0;
        transition: transform 0.2s ease;
    }}
    .status-card:hover {{
        transform: translateY(-1px);
    }}

    /* Job/endpoint cards */
    .job-card {{
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        transition: transform 0.2s ease;
    }}
    .job-card:hover {{
        transform: translateX(4px);
    }}

    /* Scrollbar styling */
    ::-webkit-scrollbar {{
        width: 6px;
        height: 6px;
    }}
    ::-webkit-scrollbar-track {{
        background: {THEME['bg_secondary']};
    }}
    ::-webkit-scrollbar-thumb {{
        background: {THEME['accent_primary']}40;
        border-radius: 3px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: {THEME['accent_primary']}80;
    }}

    /* Accessible focus styles */
    :focus-visible {{
        outline: 2px solid {THEME['accent_primary']};
        outline-offset: 2px;
    }}

    /* Reduced motion for accessibility */
    @media (prefers-reduced-motion: reduce) {{
        .fade-in, .slide-up, .pulse {{ animation: none !important; }}
        .glass-card:hover {{ transform: none; }}
    }}

    /* Skeleton loader placeholder */
    .skeleton {{
        background: linear-gradient(90deg, #1a1a1a 25%, #2a2a2a 50%, #1a1a1a 75%);
        background-size: 200% 100%;
        animation: shimmer 1.5s infinite;
        border-radius: 8px;
    }}
    .skeleton-text {{ height: 16px; margin-bottom: 8px; }}
    .skeleton-card {{ height: 120px; margin-bottom: 16px; }}
    .skeleton-chart {{ height: 300px; }}

    /* P&L coloring for tables */
    .pnl-positive {{ color: {THEME['accent_success']}; font-weight: 600; }}
    .pnl-negative {{ color: {THEME['accent_danger']}; font-weight: 600; }}
    .pnl-neutral {{ color: {THEME['text_secondary']}; }}

    /* Confidence bar */
    .conf-bar {{
        height: 6px;
        border-radius: 3px;
        background: #1a1a1a;
        overflow: hidden;
    }}
    .conf-bar-fill {{
        height: 100%;
        border-radius: 3px;
        transition: width 0.3s ease;
    }}

    /* Responsive: stack columns on small screens */
    @media (max-width: 768px) {{
        [data-testid="stHorizontalBlock"] {{
            flex-direction: column !important;
        }}
        [data-testid="stHorizontalBlock"] > div {{
            width: 100% !important;
            flex: 1 1 100% !important;
        }}
    }}

    /* Low-confidence warning pulse */
    .low-confidence {{
        border: 1px solid {THEME['accent_danger']};
        box-shadow: 0 0 8px rgba(255, 0, 85, 0.3);
    }}

    </style>
"""
