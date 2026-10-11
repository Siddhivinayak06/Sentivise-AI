"""Reusable UI components and premium high-contrast dark-theme CSS styling for Streamlit."""
from typing import Any, Optional, Dict, List, Tuple
import streamlit as st
from config import THEME_COLORS

# Using pure CSS string (no f-string brace escaping issues)
CUSTOM_CSS = """
<style>
/* ===== IMPORTS ===== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ===== GLOBAL RESETS & DARK THEME ===== */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0B0F19 !important;
    color: #F8FAFC !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

[data-testid="stHeader"] {
    background-color: transparent !important;
}

/* ===== MAIN CONTENT CONTAINER ===== */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3.5rem;
    max-width: 1280px;
}

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #0B0F19;
}
::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.15);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.25);
}

/* ===== KPI METRIC CARDS (High-Contrast Glassmorphism) ===== */
.kpi-card {
    background: linear-gradient(135deg, rgba(19, 27, 46, 0.9), rgba(15, 23, 42, 0.8));
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 1.25rem 1.4rem;
    margin-bottom: 0.85rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #6366F1, #8B5CF6, #38BDF8);
    border-radius: 14px 14px 0 0;
}
.kpi-card:hover {
    border-color: rgba(99, 102, 241, 0.45);
    transform: translateY(-3px);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4), 0 0 20px rgba(99, 102, 241, 0.2);
}
.kpi-title {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: #94A3B8;
    margin-bottom: 0.4rem;
}
.kpi-value {
    font-size: 1.5rem;
    font-weight: 800;
    color: #FFFFFF !important;
    line-height: 1.25;
    min-height: 2.2rem;
    display: flex;
    align-items: center;
    letter-spacing: -0.01em;
}
.kpi-subtitle {
    font-size: 0.78rem;
    color: #CBD5E1;
    margin-top: 0.35rem;
    line-height: 1.35;
}

/* ===== CONTENT CARDS (Glassmorphism) ===== */
.content-card {
    background: linear-gradient(135deg, rgba(19, 27, 46, 0.85), rgba(15, 23, 42, 0.75));
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
}
.content-card:hover {
    border-color: rgba(99, 102, 241, 0.45);
    box-shadow: 0 6px 25px rgba(0, 0, 0, 0.35);
}

/* ===== RECOMMENDATION CARD ===== */
.rec-card {
    background: linear-gradient(135deg, rgba(19, 27, 46, 0.85), rgba(15, 23, 42, 0.75));
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-left: 4px solid #6366F1;
    border-radius: 0 12px 12px 0;
    padding: 1.25rem;
    margin-bottom: 0.85rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}
.rec-card:hover {
    border-left-color: #06B6D4;
    transform: translateX(4px);
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.15);
}

/* ===== ISSUE CARD ===== */
.issue-card {
    background: linear-gradient(135deg, rgba(19, 27, 46, 0.9), rgba(15, 23, 42, 0.8));
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1.25rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}
.issue-card:hover {
    border-color: rgba(99, 102, 241, 0.45);
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4), 0 0 15px rgba(99, 102, 241, 0.15);
}

/* ===== PILL BADGES ===== */
.badge {
    display: inline-block;
    padding: 0.3rem 0.75rem;
    font-size: 0.72rem;
    font-weight: 700;
    border-radius: 9999px;
    margin-right: 0.5rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.badge-positive {
    background: rgba(16, 185, 129, 0.18);
    color: #10B981;
    border: 1px solid rgba(16, 185, 129, 0.35);
}
.badge-negative {
    background: rgba(244, 63, 94, 0.18);
    color: #F43F5E;
    border: 1px solid rgba(244, 63, 94, 0.35);
}
.badge-warning {
    background: rgba(245, 158, 11, 0.18);
    color: #F59E0B;
    border: 1px solid rgba(245, 158, 11, 0.35);
}
.badge-info {
    background: rgba(99, 102, 241, 0.18);
    color: #A5B4FC;
    border: 1px solid rgba(99, 102, 241, 0.35);
}
.badge-neutral {
    background: rgba(148, 163, 184, 0.15);
    color: #CBD5E1;
    border: 1px solid rgba(148, 163, 184, 0.25);
}

/* ===== SIDEBAR HIGH CONTRAST & MAXIMUM VISIBILITY ===== */
[data-testid="stSidebar"] {
    background-color: #0D111D !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
}
/* Hide default auto-generated multipage navigation */
[data-testid="stSidebarNav"] {
    display: none !important;
}
/* Native st.page_link styling - force bright crisp white on ALL child elements */
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"],
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] *,
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] p,
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] span,
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] div {
    color: #F8FAFC !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    opacity: 1 !important;
    text-decoration: none !important;
}
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] {
    border-radius: 8px !important;
    padding: 0.45rem 0.75rem !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:hover {
    background: rgba(99, 102, 241, 0.25) !important;
    transform: translateX(4px);
}
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:hover *,
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:hover p {
    color: #38BDF8 !important;
}
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][aria-current="page"] {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.4), rgba(139, 92, 246, 0.3)) !important;
    border: 1px solid rgba(99, 102, 241, 0.55) !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.25) !important;
}
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][aria-current="page"] *,
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][aria-current="page"] p {
    color: #FFFFFF !important;
    font-weight: 800 !important;
}

/* ===== TABS PREMIUM PILL STYLING ===== */
[data-testid="stTabs"] {
    margin-top: 0.5rem;
    margin-bottom: 1.5rem;
}
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 8px !important;
    background: rgba(19, 27, 46, 0.9) !important;
    padding: 6px !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius: 8px !important;
    padding: 8px 16px !important;
    color: #CBD5E1 !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    border: none !important;
    background: transparent !important;
    transition: all 0.2s ease !important;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {
    color: #FFFFFF !important;
    background: rgba(99, 102, 241, 0.2) !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.45), rgba(139, 92, 246, 0.35)) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(99, 102, 241, 0.55) !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
    font-weight: 700 !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"] {
    display: none !important;
}

/* ===== STREAMLIT BUTTONS ===== */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.35) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
    color: #FFFFFF !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.35) !important;
}
.stButton > button[kind="secondary"] {
    background: rgba(19, 27, 46, 0.85) !important;
    color: #F8FAFC !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
}

/* ===== EXPANDERS ===== */
[data-testid="stExpander"] {
    background: rgba(19, 27, 46, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    overflow: hidden;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary * {
    color: #F8FAFC !important;
    font-weight: 700 !important;
}

/* ===== WIDGET LABELS (HIGH VISIBILITY) ===== */
label,
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] *,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span,
[data-testid="stWidgetLabel"] label,
.stSelectbox label,
.stSelectbox label *,
.stTextArea label,
.stTextArea label *,
.stTextInput label,
.stTextInput label *,
.stSlider label,
.stSlider label *,
.stNumberInput label,
.stNumberInput label *,
.stFileUploader label,
.stFileUploader label * {
    color: #F8FAFC !important;
    font-size: 0.94rem !important;
    font-weight: 700 !important;
    opacity: 1 !important;
    margin-bottom: 0.35rem !important;
}

/* ===== CHECKBOX & RADIO (HIGH VISIBILITY) ===== */
[data-testid="stCheckbox"],
[data-testid="stCheckbox"] *,
[data-testid="stCheckbox"] p,
[data-testid="stCheckbox"] span,
[data-testid="stCheckbox"] label,
.stCheckbox label,
.stCheckbox label *,
[data-testid="stRadio"] label,
[data-testid="stRadio"] p,
[data-testid="stRadio"] span {
    color: #F8FAFC !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}

/* ===== FORM INPUTS & TEXTAREA (HIGH VISIBILITY) ===== */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-baseweb="textarea"] textarea,
[data-baseweb="input"] input {
    background-color: #131B2E !important;
    color: #FFFFFF !important;
    font-size: 0.95rem !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 10px !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.35) !important;
}
textarea::placeholder,
input::placeholder,
[data-baseweb="textarea"] textarea::placeholder,
[data-baseweb="input"] input::placeholder {
    color: #94A3B8 !important;
    opacity: 1 !important;
    font-style: italic !important;
}

/* ===== SELECTBOXES (HIGH VISIBILITY) ===== */
[data-baseweb="select"] {
    background-color: #131B2E !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 10px !important;
}
[data-baseweb="select"] *,
[data-baseweb="select"] span,
[data-baseweb="select"] div {
    color: #FFFFFF !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}
[data-baseweb="popover"],
[data-baseweb="menu"] {
    background-color: #131B2E !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 10px !important;
}
[data-baseweb="menu"] li,
[data-baseweb="menu"] li * {
    color: #F8FAFC !important;
    font-weight: 500 !important;
}
[data-baseweb="menu"] li:hover {
    background-color: rgba(99, 102, 241, 0.3) !important;
}

/* ===== SECTION HEADER ===== */
.section-header {
    font-size: 1.25rem;
    font-weight: 800;
    color: #F8FAFC;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ===== SCORE RING ===== */
.score-ring {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 60px;
    height: 60px;
    border-radius: 50%;
    border: 3px solid #6366F1;
    font-size: 1.35rem;
    font-weight: 800;
    color: #FFFFFF;
    background: rgba(99, 102, 241, 0.15);
    box-shadow: 0 0 15px rgba(99, 102, 241, 0.25);
}

/* ===== ANIMATED PULSE DOT ===== */
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.9); }
}
.pulse-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    animation: pulse 2s infinite;
    margin-right: 6px;
    vertical-align: middle;
}
.pulse-dot-green { background-color: #10B981; box-shadow: 0 0 8px #10B981; }
.pulse-dot-red { background-color: #F43F5E; box-shadow: 0 0 8px #F43F5E; }
.pulse-dot-blue { background-color: #3B82F6; box-shadow: 0 0 8px #3B82F6; }

/* ===== STATUS CHIP ===== */
.status-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.status-open {
    background: rgba(244, 63, 94, 0.18);
    color: #F43F5E;
    border: 1px solid rgba(244, 63, 94, 0.35);
}
.status-investigating {
    background: rgba(245, 158, 11, 0.18);
    color: #F59E0B;
    border: 1px solid rgba(245, 158, 11, 0.35);
}
.status-resolved {
    background: rgba(16, 185, 129, 0.18);
    color: #10B981;
    border: 1px solid rgba(16, 185, 129, 0.35);
}

/* ===== FEATURE GRID & ITEMS ===== */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
}
@media (max-width: 900px) {
    .feature-grid {
        grid-template-columns: 1fr;
    }
}
.feature-item {
    background: linear-gradient(135deg, rgba(19, 27, 46, 0.85), rgba(15, 23, 42, 0.75));
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 1.25rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}
.feature-item:hover {
    border-color: rgba(99, 102, 241, 0.45);
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.15);
}
.feature-icon {
    font-size: 1.7rem;
    margin-bottom: 0.5rem;
}
.feature-title {
    font-size: 0.98rem;
    font-weight: 700;
    color: #F8FAFC;
    margin-bottom: 0.3rem;
}
.feature-desc {
    font-size: 0.82rem;
    color: #CBD5E1;
    line-height: 1.45;
}
</style>
"""

def inject_custom_css():
    """Injects the premium high-contrast dark CSS stylesheet into the active page."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def render_header(title: str, subtitle: str, badge: str = "Enterprise Intelligence"):
    """Renders a premium page header with high-contrast gradient title and active badge."""
    html = (
        f'<div style="margin-bottom: 1.8rem;">'
        f'<div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.5rem;">'
        f'<span class="badge badge-info">{badge}</span>'
        f'<span class="pulse-dot pulse-dot-green"></span>'
        f'<span style="font-size: 0.72rem; color: #CBD5E1; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 700;">System Active</span>'
        f'</div>'
        f'<h1 style="font-size: 2.2rem; font-weight: 900; margin: 0; line-height: 1.25; '
        f'background: linear-gradient(135deg, #FFFFFF 25%, #C7D2FE 70%, #38BDF8 100%); '
        f'-webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">'
        f'{title}</h1>'
        f'<p style="font-size: 0.96rem; color: #CBD5E1; margin-top: 0.45rem; margin-bottom: 0; line-height: 1.5; font-weight: 400;">'
        f'{subtitle}</p>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

def render_kpi(col, title: str, value: Any, subtitle: str = "", delta: str = "", delta_color: str = "normal"):
    """Renders a modern glassmorphism KPI card with crisp legible value inside a Streamlit column."""
    with col:
        html = (
            f'<div class="kpi-card">'
            f'<div class="kpi-title">{title}</div>'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-subtitle">{subtitle}</div>'
            f'</div>'
        )
        st.markdown(html, unsafe_allow_html=True)

def render_empty_state(message: str, hint: str = "Please run an analysis or upload a dataset to view records."):
    """Renders a friendly and clear empty state container."""
    html = (
        f'<div style="text-align: center; padding: 3rem 1.5rem; border: 2px dashed rgba(255, 255, 255, 0.12); '
        f'border-radius: 16px; background: rgba(19, 27, 46, 0.4); margin: 1.5rem 0; transition: all 0.3s ease;">'
        f'<div style="font-size: 2.5rem; margin-bottom: 0.6rem; opacity: 0.6;">📂</div>'
        f'<h4 style="color: #F8FAFC; font-weight: 700; margin-bottom: 0.3rem;">{message}</h4>'
        f'<p style="color: #94A3B8; font-size: 0.88rem; margin-bottom: 0;">{hint}</p>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

def render_section_header(icon: str, title: str):
    """Renders a styled section header with icon."""
    st.markdown(
        f'<div class="section-header">{icon} {title}</div>',
        unsafe_allow_html=True
    )

def render_feature_card(icon: str, title: str, description: str) -> str:
    """Renders a feature card for the home page capabilities grid."""
    return (
        f'<div class="feature-item">'
        f'<div class="feature-icon">{icon}</div>'
        f'<div class="feature-title">{title}</div>'
        f'<div class="feature-desc">{description}</div>'
        f'</div>'
    )
