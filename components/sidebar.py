"""Standardized premium sidebar component with system health indicators and native navigation."""
import streamlit as st
from config import APP_TITLE, APP_VERSION, THEME_COLORS
from services.model_loader import get_model_readiness_status

def render_sidebar():
    """Renders the premium dark-themed common sidebar across all application pages."""
    with st.sidebar:
        # 1. Brand Logo Header
        st.markdown(
            f"""<div style="padding: 0.4rem 0 0.9rem 0; border-bottom: 1px solid rgba(255, 255, 255, 0.1); margin-bottom: 1rem;">
<div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.35rem;">
<span style="font-size: 1.6rem;">🧠</span>
<div style="font-size: 1.25rem; font-weight: 900; background: linear-gradient(135deg, #FFFFFF 20%, #C7D2FE 70%, #38BDF8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
Sentivise AI</div></div>
<div style="display: flex; align-items: center; justify-content: space-between; padding-left: 2.2rem;">
<span style="font-size: 0.72rem; color: #CBD5E1; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 700;">v{APP_VERSION}</span>
<span style="font-size: 0.68rem; background: rgba(16, 185, 129, 0.2); color: #10B981; padding: 0.15rem 0.55rem; border-radius: 9999px; font-weight: 800; border: 1px solid rgba(16, 185, 129, 0.4);">
ONLINE</span></div></div>""",
            unsafe_allow_html=True
        )

        # 2. System Readiness Widget
        status_info = get_model_readiness_status()
        ready_count = sum(1 for v in status_info.values() if v["ready"])
        total_models = len(status_info)
        health_pct = int((ready_count / total_models) * 100) if total_models > 0 else 0

        st.markdown(
            f"""<div style="background: linear-gradient(135deg, rgba(19, 27, 46, 0.95), rgba(15, 23, 42, 0.85)); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 12px; padding: 0.85rem 1rem; margin-bottom: 1.25rem;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
<span style="font-size: 0.72rem; font-weight: 800; color: #CBD5E1; text-transform: uppercase; letter-spacing: 0.08em;">Model Readiness</span>
<span style="font-size: 0.75rem; color: #38BDF8; font-weight: 800;">{ready_count}/{total_models} Active</span></div>
<div style="background: rgba(51, 65, 85, 0.6); border-radius: 6px; height: 5px; overflow: hidden; margin-bottom: 0.6rem;">
<div style="background: linear-gradient(90deg, #6366F1, #38BDF8); width: {health_pct}%; height: 100%; border-radius: 6px;"></div></div>
<div style="display: flex; justify-content: space-between; font-size: 0.74rem; color: #E2E8F0; font-weight: 500;">
<span>SQLite Database</span>
<span style="color: #10B981; font-weight: 700;">● Connected</span></div></div>""",
            unsafe_allow_html=True
        )

        # 3. Categorized Page Navigation via st.page_link
        st.markdown(
            """<div style="font-size: 0.72rem; font-weight: 800; color: #A5B4FC; text-transform: uppercase; letter-spacing: 0.12em; margin: 0.5rem 0 0.4rem 0.5rem;">
Analytics & Decisions</div>""",
            unsafe_allow_html=True
        )
        st.page_link("app.py", label="Home Portal", icon="🏠")
        st.page_link("pages/1_Executive_Dashboard.py", label="Executive Dashboard", icon="📈")
        st.page_link("pages/2_Review_Analyzer.py", label="Review Analyzer", icon="🔍")
        st.page_link("pages/3_Batch_Analytics.py", label="Batch Analytics", icon="📦")
        st.page_link("pages/4_Customer_Insights.py", label="Customer Insights", icon="💡")
        st.page_link("pages/5_Decision_Center.py", label="Decision Center", icon="⚖️")

        st.markdown(
            """<div style="font-size: 0.72rem; font-weight: 800; color: #A5B4FC; text-transform: uppercase; letter-spacing: 0.12em; margin: 1rem 0 0.4rem 0.5rem;">
AI & Reasoning Lab</div>""",
            unsafe_allow_html=True
        )
        st.page_link("pages/6_Knowledge_Rules.py", label="Knowledge & Rules", icon="🧠")
        st.page_link("pages/7_ML_Evaluation.py", label="ML Evaluation", icon="🔬")
        st.page_link("pages/8_DNN_Model.py", label="DNN Architecture", icon="🧬")
        st.page_link("pages/9_Explainable_AI.py", label="Explainable AI", icon="🔎")
        st.page_link("pages/10_Search_Lab.py", label="Search Lab", icon="🗺️")

        st.markdown(
            """<div style="font-size: 0.72rem; font-weight: 800; color: #A5B4FC; text-transform: uppercase; letter-spacing: 0.12em; margin: 1rem 0 0.4rem 0.5rem;">
System Governance</div>""",
            unsafe_allow_html=True
        )
        st.page_link("pages/11_Data_Management.py", label="Data Management", icon="🗄️")
        st.page_link("pages/12_Reports_Audit.py", label="Reports & Audit", icon="📑")
        st.page_link("pages/13_Settings_About.py", label="Settings & Ethics", icon="⚙️")

        # 4. Footer
        st.markdown(
            """<div style="text-align: center; padding: 1.25rem 0 0.5rem 0; border-top: 1px solid rgba(255, 255, 255, 0.08); margin-top: 1.25rem;">
<div style="font-size: 0.68rem; color: #94A3B8; line-height: 1.5;">
Enterprise Intelligence Platform<br/>Python 3.13 &bull; Streamlit &bull; SQLite</div></div>""",
            unsafe_allow_html=True
        )
