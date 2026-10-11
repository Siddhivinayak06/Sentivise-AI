"""Main Streamlit Application Entrypoint for AI-Based Product Review Sentiment Analysis and Intelligent Decision Support System."""
import streamlit as st
import pandas as pd
from config import APP_TITLE, APP_SUBTITLE, APP_VERSION, THEME_COLORS
from database.schema import initialize_database
from database.repository import Repository
from services.decision_service import DecisionService
from services.model_loader import get_model_readiness_status
from components.ui_elements import (
    inject_custom_css,
    render_header,
    render_kpi,
    render_feature_card,
    render_section_header,
)
from components.charts import plot_sentiment_donut
from components.sidebar import render_sidebar

# Configure Streamlit page
st.set_page_config(
    page_title=f"{APP_TITLE} (v{APP_VERSION})",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database schema and seed initial issues if empty
initialize_database()
DecisionService.seed_initial_issues_if_needed()

# Inject styling
inject_custom_css()
render_sidebar()

# Page Header
render_header(
    title=APP_TITLE,
    subtitle=APP_SUBTITLE,
    badge="Production Analytics & Decision Support"
)

# Fetch System KPIs from database
kpis = Repository.get_system_kpis()

# Render Top KPIs
c1, c2, c3, c4 = st.columns(4)
render_kpi(c1, "Total Reviews Analyzed", f"{kpis['total_analyzed']:,}", "From single & batch records")
render_kpi(c2, "Positive Ratio", f"{kpis['positive_percentage']:.1f}%", f"{kpis['total_positive']:,} Positive Reviews")
render_kpi(c3, "Negative Ratio", f"{kpis['negative_percentage']:.1f}%", f"{kpis['total_negative']:,} Negative Reviews")
render_kpi(c4, "Active Business Issues", f"{kpis['active_issues']}", f"{kpis['high_priority_issues']} High/Critical Priority")

st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

# Overview & Action Grid
left_col, right_col = st.columns([7, 5])

with left_col:
    render_section_header("🎯", "Platform Capabilities")
    st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)

    # Feature Cards Grid
    features = [
        ("🔍", "Review Analyzer", "Classify feedback with ML/DNN models and trigger automated business actions"),
        ("📦", "Batch Analytics", "Upload CSV datasets for validated, deduplicated, high-throughput inference"),
        ("💡", "Customer Insights", "Uncover complaint themes across Quality, Delivery, Pricing, Usability, Support"),
        ("⚖️", "Decision Center", "Prioritize issues with transparent scoring and track resolution workflows"),
        ("🧠", "Knowledge Engine", "Forward and backward chaining inference with auditable proof traces"),
        ("🔬", "Model Evaluation", "Compare 5 ML models + DNN using real evaluation metrics and confusion matrices"),
        ("🔎", "Explainable AI", "SHAP, LIME, and counterfactual explanations for model transparency"),
        ("🗺️", "Search Lab", "Explore resolution workflows using BFS, DFS, and A* informed search"),
    ]

    cards_html = (
        '<div class="feature-grid">'
        + "".join(render_feature_card(icon, title, desc) for icon, title, desc in features)
        + '</div>'
    )
    st.markdown(cards_html, unsafe_allow_html=True)

    # Quick Action Buttons
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    qa1, qa2, qa3 = st.columns(3)
    with qa1:
        if st.button("🔍 Analyze Single Review", use_container_width=True, type="primary"):
            st.switch_page("pages/2_Review_Analyzer.py")
    with qa2:
        if st.button("📦 Upload Batch Reviews", use_container_width=True):
            st.switch_page("pages/3_Batch_Analytics.py")
    with qa3:
        if st.button("⚖️ Open Decision Center", use_container_width=True):
            st.switch_page("pages/5_Decision_Center.py")

with right_col:
    render_section_header("📊", "Sentiment Telemetry")
    st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)
    fig = plot_sentiment_donut(
        pos_count=kpis["total_positive"],
        neg_count=kpis["total_negative"],
        title="Stored Feedback Sentiment Split"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Model Health Summary
    status_info = get_model_readiness_status()
    ready = sum(1 for v in status_info.values() if v["ready"])
    total = len(status_info)
    pct = int(ready / total * 100) if total > 0 else 0

    model_status_html = (
        f'<div class="content-card" style="margin-top: 0.5rem;">'
        f'<div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; '
        f'color: {THEME_COLORS["text_muted"]}; margin-bottom: 0.6rem;">'
        f'Model Pipeline Status'
        f'</div>'
        f'<div style="display: flex; align-items: baseline; gap: 0.5rem; margin-bottom: 0.5rem;">'
        f'<span style="font-size: 1.8rem; font-weight: 800; '
        f'background: linear-gradient(135deg, {THEME_COLORS["primary"]}, {THEME_COLORS["secondary"]}); '
        f'-webkit-background-clip: text; -webkit-text-fill-color: transparent;">'
        f'{ready}/{total}'
        f'</span>'
        f'<span style="font-size: 0.85rem; color: {THEME_COLORS["text_muted"]};">artifacts loaded</span>'
        f'</div>'
        f'<div style="background: rgba(51, 65, 85, 0.6); border-radius: 8px; height: 6px; overflow: hidden;">'
        f'<div style="background: linear-gradient(90deg, {THEME_COLORS["gradient_start"]}, {THEME_COLORS["gradient_end"]}); '
        f'width: {pct}%; height: 100%; border-radius: 8px; transition: width 0.5s ease;"></div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(model_status_html, unsafe_allow_html=True)

st.markdown("---")

# Recent Activity Audit Trail
st.markdown(
    f'<div class="section-header">📋 Recent Platform Activity</div>',
    unsafe_allow_html=True
)
recent_logs = Repository.get_audit_logs(limit=8)
if recent_logs:
    logs_df = pd.DataFrame(recent_logs)[["created_at", "action_type", "entity_type", "description"]]
    logs_df.columns = ["Timestamp", "Action", "Entity", "Description"]
    st.dataframe(logs_df, use_container_width=True, hide_index=True)
else:
    st.info("No audit activities recorded yet. Run a single review or upload a batch file to start logging events.")
