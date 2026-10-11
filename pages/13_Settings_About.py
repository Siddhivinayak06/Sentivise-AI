"""Page 13: Settings and About page detailing system architecture, data retention, and Responsible AI policies."""
import streamlit as st
import pandas as pd
from database.repository import Repository
from services.model_loader import get_model_readiness_status
from config import APP_TITLE, APP_SUBTITLE, APP_VERSION, DATABASE_PATH, THEME_COLORS
from components.ui_elements import (
    inject_custom_css,
    render_header,
    render_kpi,
    render_section_header,
)
from components.sidebar import render_sidebar

st.set_page_config(page_title="Settings & About", page_icon="⚙️", layout="wide")
inject_custom_css()
render_sidebar()

render_header(
    title="Platform Settings, Architecture & Responsible AI",
    subtitle="System configuration, database maintenance, artifact health, and ethical AI deployment principles",
    badge="System Administration"
)

tab_about, tab_artifacts, tab_db, tab_ethics = st.tabs([
    "ℹ️ About Platform & Architecture",
    "🩺 Artifact Health Status",
    "🗄️ Database Maintenance & Purge",
    "⚖️ Responsible AI & Ethical Boundaries"
])

# Tab 1: About & Architecture
with tab_about:
    render_section_header("🏷️", f"{APP_TITLE} (v{APP_VERSION})")
    
    st.markdown(
        f"""
        <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid {THEME_COLORS['border']};
            border-radius: 16px; padding: 1.5rem; margin-bottom: 1.5rem;">
            <h4 style="color: {THEME_COLORS['secondary']}; font-weight: 800; margin-top: 0; margin-bottom: 0.6rem;">
                🎯 Strategic Purpose & Architecture
            </h4>
            <p style="color: {THEME_COLORS['text']}; line-height: 1.6; margin-bottom: 1rem;">
                The platform bridges predictive machine learning, deep learning, explainability (XAI), and symbolic knowledge representation to deliver an enterprise-grade customer intelligence and decision support platform.
            </p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem;">
                <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid {THEME_COLORS['border']}; border-radius: 10px; padding: 1rem;">
                    <div style="font-weight: 700; color: {THEME_COLORS['primary']}; margin-bottom: 0.3rem;">🖥️ Frontend Interface</div>
                    <div style="font-size: 0.85rem; color: {THEME_COLORS['text_muted']};">Streamlit with custom responsive dark CSS, glassmorphism cards, and interactive Plotly telemetry.</div>
                </div>
                <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid {THEME_COLORS['border']}; border-radius: 10px; padding: 1rem;">
                    <div style="font-weight: 700; color: {THEME_COLORS['secondary']}; margin-bottom: 0.3rem;">🗄️ Persistence Layer</div>
                    <div style="font-size: 0.85rem; color: {THEME_COLORS['text_muted']};">SQLite engine with Write-Ahead Logging (WAL) and thread-safe connection pooling.</div>
                </div>
                <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid {THEME_COLORS['border']}; border-radius: 10px; padding: 1rem;">
                    <div style="font-weight: 700; color: {THEME_COLORS['info']}; margin-bottom: 0.3rem;">🧠 Dual-Model ML/DL</div>
                    <div style="font-size: 0.85rem; color: {THEME_COLORS['text_muted']};">Scikit-Learn Calibrated Classifiers + Keras Deep Neural Network with 100 features.</div>
                </div>
                <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid {THEME_COLORS['border']}; border-radius: 10px; padding: 1rem;">
                    <div style="font-weight: 700; color: {THEME_COLORS['warning']}; margin-bottom: 0.3rem;">🔬 Explainable AI (XAI)</div>
                    <div style="font-size: 0.85rem; color: {THEME_COLORS['text_muted']};">SHAP Shapley values, LIME local perturbations, and automated minimal counterfactuals.</div>
                </div>
                <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid {THEME_COLORS['border']}; border-radius: 10px; padding: 1rem;">
                    <div style="font-weight: 700; color: {THEME_COLORS['positive']}; margin-bottom: 0.3rem;">⚙️ Symbolic Expert System</div>
                    <div style="font-size: 0.85rem; color: {THEME_COLORS['text_muted']};">Dual-engine Knowledge Base with Forward Chaining deduction and Backward Chaining hypothesis proof.</div>
                </div>
                <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid {THEME_COLORS['border']}; border-radius: 10px; padding: 1rem;">
                    <div style="font-weight: 700; color: {THEME_COLORS['accent']}; margin-bottom: 0.3rem;">🗺️ State-Space Planning</div>
                    <div style="font-size: 0.85rem; color: {THEME_COLORS['text_muted']};">Graph escalation resolution planning comparing BFS, DFS, and A* Informed Search.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Tab 2: Artifact Health
with tab_artifacts:
    render_section_header("🩺", "Model & Pipeline Artifact Inventory")
    status_dict = get_model_readiness_status()
    status_rows = []
    for name, info in status_dict.items():
        status_rows.append({
            "Model Name": name,
            "Architecture Type": info["type"],
            "Artifact Path": info["path"],
            "File Size (KB)": info["size_kb"],
            "Readiness Status": "✅ Ready" if info["ready"] else "❌ Missing"
        })
    st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

# Tab 3: Database Maintenance
with tab_db:
    render_section_header("🗄️", "Database Management & Hygiene")
    st.caption(f"Active SQLite File: `{DATABASE_PATH}`")

    kpis = Repository.get_system_kpis()
    c1, c2, c3 = st.columns(3)
    render_kpi(c1, "Single Reviews", f"{kpis['total_analyzed'] - kpis['distinct_products']}", "analysis_history table")
    render_kpi(c2, "Tracked Issues", f"{kpis['active_issues']}", "issues table")
    render_kpi(c3, "Audit Entries", "Active", "audit_logs table")

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    
    st.markdown(
        f"""
        <div style="background: rgba(251, 113, 133, 0.08); border: 1px solid rgba(251, 113, 133, 0.3);
            border-radius: 12px; padding: 1.25rem; margin-bottom: 1rem;">
            <div style="font-weight: 800; color: {THEME_COLORS['negative']}; font-size: 1.05rem; margin-bottom: 0.3rem;">
                ⚠️ Administrative Danger Zone
            </div>
            <div style="font-size: 0.88rem; color: {THEME_COLORS['text_muted']}; margin-bottom: 0.8rem;">
                Purging data permanently wipes stored historical records and active issue tracking. This action cannot be reversed.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    del_c1, del_c2 = st.columns(2)
    with del_c1:
        if st.button("🗑️ Purge Single Analysis History", type="secondary", use_container_width=True):
            Repository.clear_table("analysis_history")
            st.success("Cleared table: `analysis_history`.")
            st.rerun()
    with del_c2:
        if st.button("🗑️ Purge Batch Reviews", type="secondary", use_container_width=True):
            Repository.clear_table("batch_reviews")
            st.success("Cleared table: `batch_reviews`.")
            st.rerun()

# Tab 4: Responsible AI
with tab_ethics:
    render_section_header("⚖️", "Responsible AI Guidelines & Operating Boundaries")
    st.markdown(
        f"""
        <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid {THEME_COLORS['border']};
            border-radius: 14px; padding: 1.5rem;">
            <ol style="margin: 0; padding-left: 1.2rem; color: {THEME_COLORS['text']}; line-height: 1.8;">
                <li><strong>Human-in-the-Loop Decision Authority:</strong> The platform functions strictly as an intelligent <em>decision support system</em>. It recommends remediation steps but never triggers financial refunds, vendor cancellations, or external customer messages without authorized human sign-off.</li>
                <li><strong>Transparency in Scoring:</strong> Business priority scores are transparent, weighted heuristics (frequency, negative polarity, severity, recency, product volume) and are never falsely presented as mathematical certainty.</li>
                <li><strong>Explainability & Auditing:</strong> Every classification is accompanied by token-level feature attributions (SHAP/LIME) and recorded into an immutable SQLite audit log.</li>
                <li><strong>Handling Ambiguity:</strong> Reviews near the uncertainty boundary (0.45 &le; P &le; 0.55) trigger an automatic <strong>Uncertainty Alert</strong> rather than giving false high-confidence predictions.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True
    )
