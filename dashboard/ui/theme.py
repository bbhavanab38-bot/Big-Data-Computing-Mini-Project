"""Design System Theme Definitions, Tokens, and Global Styles.

Banking Retention Intelligence Platform
Color Palette:
  - Background: #0B1120 (Command Center Dark)
  - Sidebar: #101827 (Deep Charcoal)
  - Cards: #172235 (Slate Navy)
  - Secondary Background: #1D2B40
  - Primary Accent: #38BDF8 (Sky Blue)
  - Secondary Accent: #8B5CF6 (Violet)
  - Success: #22C55E (Emerald Green)
  - Warning: #F59E0B (Amber)
  - Danger: #EF4444 (Rose Red)
  - Text Primary: #F8FAFC (Snow)
  - Text Secondary: #94A3B8 (Muted Slate)
"""

BG_COLOR = "#0B1120"
SIDEBAR_COLOR = "#101827"
CARD_BG = "#172235"
SECONDARY_BG = "#1D2B40"
PRIMARY_ACCENT = "#38BDF8"
SECONDARY_ACCENT = "#8B5CF6"
SUCCESS = "#22C55E"
WARNING = "#F59E0B"
DANGER = "#EF4444"
TEXT_PRIMARY = "#F8FAFC"
TEXT_SECONDARY = "#94A3B8"
BORDER_COLOR = "rgba(148, 163, 184, 0.12)"
BORDER_HOVER = "rgba(56, 189, 248, 0.35)"

PALETTE_CHART = [
    PRIMARY_ACCENT,    # #38BDF8
    SECONDARY_ACCENT,  # #8B5CF6
    SUCCESS,           # #22C55E
    WARNING,           # #F59E0B
    DANGER,            # #EF4444
    "#60A5FA",         # Light Blue
    "#C084FC",         # Light Violet
]


def get_custom_css() -> str:
    """Generate scoped CSS string for the enterprise dark finance theme."""
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}

    /* Global container adjustments */
    .block-container {{
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        max-width: 1440px !important;
    }}

    /* Sidebar container and styling */
    [data-testid="stSidebar"] {{
        background-color: {SIDEBAR_COLOR} !important;
        border-right: 1px solid {BORDER_COLOR} !important;
    }}

    [data-testid="stSidebarNav"] {{
        padding-top: 0.5rem !important;
        padding-left: 0.65rem !important;
        padding-right: 0.65rem !important;
    }}

    /* Sidebar navigation section headers (uppercase tracked) */
    [data-testid="stSidebarNavSeparator"] {{
        margin-top: 0.85rem !important;
        margin-bottom: 0.25rem !important;
        border-top: 1px solid rgba(148, 163, 184, 0.08) !important;
    }}

    /* Section header text */
    [data-testid="stSidebarNavItems"] > div > span,
    [data-testid="stSidebarNav"] > ul > li > div > span {{
        font-size: 0.68rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        color: #64748B !important;
        padding: 0.45rem 0.5rem 0.2rem 0.5rem !important;
        display: block !important;
    }}

    /* Sidebar Nav Link item */
    [data-testid="stSidebarNavLink"] {{
        border-radius: 8px !important;
        padding: 0.42rem 0.65rem !important;
        margin: 0.12rem 0 !important;
        transition: background-color 0.15s ease, border-color 0.15s ease, color 0.15s ease !important;
        text-decoration: none !important;
    }}

    [data-testid="stSidebarNavLink"] span {{
        font-size: 0.84rem !important;
        font-weight: 500 !important;
        color: #CBD5E1 !important;
        text-transform: none !important;
        letter-spacing: normal !important;
    }}

    /* Nav link hover state */
    [data-testid="stSidebarNavLink"]:hover {{
        background-color: rgba(56, 189, 248, 0.08) !important;
    }}

    [data-testid="stSidebarNavLink"]:hover span {{
        color: #F8FAFC !important;
    }}

    /* Active page highlight */
    [data-testid="stSidebarNavLink"][aria-current="page"] {{
        background: linear-gradient(90deg, rgba(56, 189, 248, 0.18) 0%, rgba(56, 189, 248, 0.04) 100%) !important;
        border-left: 3px solid {PRIMARY_ACCENT} !important;
    }}

    [data-testid="stSidebarNavLink"][aria-current="page"] span {{
        color: {PRIMARY_ACCENT} !important;
        font-weight: 600 !important;
    }}

    /* Sidebar icons */
    [data-testid="stSidebarNavLink"] span[data-testid="stIconMaterial"] {{
        font-size: 1.15rem !important;
        margin-right: 0.45rem !important;
        color: #94A3B8 !important;
        vertical-align: middle !important;
    }}

    [data-testid="stSidebarNavLink"][aria-current="page"] span[data-testid="stIconMaterial"] {{
        color: {PRIMARY_ACCENT} !important;
    }}

    /* Card Box Component */
    .retention-card {{
        background-color: {CARD_BG};
        border: 1px solid {BORDER_COLOR};
        border-radius: 12px;
        padding: 1.15rem 1.25rem;
        margin-bottom: 1rem;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }}

    .retention-card:hover {{
        border-color: {BORDER_HOVER};
        box-shadow: 0 4px 16px -2px rgba(0, 0, 0, 0.4);
    }}

    /* KPI Card Elements */
    .kpi-title {{
        font-size: clamp(0.65rem, 0.95vw, 0.72rem) !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        color: {TEXT_SECONDARY} !important;
        margin-bottom: 0.35rem !important;
        display: flex;
        align-items: center;
        justify-content: space-between;
        white-space: nowrap !important;
        overflow: hidden;
        text-overflow: ellipsis;
    }}

    .kpi-value {{
        font-size: clamp(1.15rem, 1.8vw, 1.75rem) !important;
        font-weight: 600 !important;
        color: {TEXT_PRIMARY} !important;
        line-height: 1.2 !important;
        margin-bottom: 0.35rem !important;
        white-space: nowrap !important;
        overflow: hidden;
        text-overflow: ellipsis;
    }}

    .kpi-delta {{
        font-size: 0.76rem !important;
        font-weight: 500 !important;
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
    }}

    .delta-positive {{ color: {SUCCESS} !important; }}
    .delta-negative {{ color: {DANGER} !important; }}
    .delta-neutral {{ color: {TEXT_SECONDARY} !important; }}

    .delta-basis {{
        font-size: 0.70rem !important;
        color: {TEXT_SECONDARY} !important;
        margin-left: 0.25rem;
    }}

    /* Insight Banner */
    .insight-banner {{
        background: linear-gradient(90deg, rgba(56, 189, 248, 0.12) 0%, rgba(139, 92, 246, 0.08) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 12px;
        padding: 0.85rem 1.25rem;
        margin-bottom: 1.25rem;
        display: flex;
        align-items: center;
        gap: 0.85rem;
    }}

    .insight-banner-icon {{
        font-size: 1.25rem;
        color: {PRIMARY_ACCENT};
        line-height: 1;
    }}

    .insight-banner-badge {{
        background: rgba(56, 189, 248, 0.2);
        color: {PRIMARY_ACCENT};
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        margin-right: 0.5rem;
    }}

    .insight-banner-text {{
        font-size: 0.85rem;
        color: {TEXT_PRIMARY};
        line-height: 1.4;
    }}

    /* Section Header */
    .section-title {{
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        color: {TEXT_PRIMARY} !important;
        letter-spacing: -0.01em !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.25rem !important;
    }}

    .section-subtitle {{
        font-size: 0.80rem !important;
        color: {TEXT_SECONDARY} !important;
        margin-bottom: 0.85rem !important;
    }}

    /* Badges & Chips */
    .badge-synthetic {{
        background: rgba(148, 163, 184, 0.15);
        border: 1px solid rgba(148, 163, 184, 0.25);
        color: {TEXT_SECONDARY};
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 0.25rem 0.55rem;
        border-radius: 6px;
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
    }}

    .pill-status {{
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
    }}

    .pill-live {{
        background: rgba(34, 197, 94, 0.15);
        border: 1px solid rgba(34, 197, 94, 0.35);
        color: {SUCCESS};
    }}

    .pill-warn {{
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.35);
        color: {WARNING};
    }}

    .risk-chip {{
        font-size: 0.70rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        padding: 0.2rem 0.55rem;
        border-radius: 4px;
        display: inline-block;
        text-align: center;
    }}

    .risk-low {{
        background: rgba(34, 197, 94, 0.15);
        color: {SUCCESS};
        border: 1px solid rgba(34, 197, 94, 0.25);
    }}

    .risk-medium {{
        background: rgba(245, 158, 11, 0.15);
        color: {WARNING};
        border: 1px solid rgba(245, 158, 11, 0.25);
    }}

    .risk-high {{
        background: rgba(239, 68, 68, 0.15);
        color: {DANGER};
        border: 1px solid rgba(239, 68, 68, 0.25);
    }}

    /* In-cell progress bar container */
    .bar-container {{
        background: rgba(148, 163, 184, 0.12);
        border-radius: 4px;
        height: 6px;
        width: 100%;
        overflow: hidden;
        margin-top: 4px;
    }}

    .bar-fill {{
        height: 100%;
        border-radius: 4px;
    }}

    /* Streamlit widget tuning */
    .stSelectbox, .stMultiSelect, .stSlider, .stTextInput, .stNumberInput {{
        margin-bottom: 0.75rem !important;
    }}

    div[data-testid="stMetricValue"] {{
        font-size: clamp(1.05rem, 1.4vw, 1.45rem) !important;
        font-weight: 600 !important;
        color: {TEXT_PRIMARY} !important;
        white-space: nowrap !important;
    }}

    div[data-testid="stMetricLabel"] {{
        font-size: clamp(0.62rem, 0.85vw, 0.70rem) !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        color: {TEXT_SECONDARY} !important;
        white-space: nowrap !important;
    }}

    /* Streamlit button and download button responsiveness */
    .stButton button, [data-testid="stDownloadButton"] button {{
        font-size: clamp(0.72rem, 1vw, 0.82rem) !important;
        font-weight: 600 !important;
        padding: 0.35rem 0.65rem !important;
        border-radius: 8px !important;
        white-space: nowrap !important;
    }}

    @media (max-width: 1100px) {{
        .block-container {{
            padding-left: 1.25rem !important;
            padding-right: 1.25rem !important;
        }}
        .retention-card {{
            padding: 0.85rem 0.95rem !important;
        }}
    }}

    /* Table borders */
    [data-testid="stDataFrame"] {{
        border: 1px solid {BORDER_COLOR} !important;
        border-radius: 10px !important;
    }}

    /* Hide Streamlit Deploy button, header decorations and default toolbar */
    .stDeployButton,
    [data-testid="stAppDeployButton"],
    [data-testid="stDecoration"],
    [data-testid="stToolbar"] [data-testid="stAppDeployButton"],
    header [data-testid="stAppDeployButton"],
    #MainMenu {{
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
    }}
    header[data-testid="stHeader"] {{
        background: transparent !important;
    }}
    </style>
    """

