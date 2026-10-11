"""Page 12: Reports and Audit Trail providing downloadable executive briefings and system event history."""
import streamlit as st
import pandas as pd
from datetime import datetime

from services.report_service import ReportService
from database.repository import Repository
from components.ui_elements import (
    inject_custom_css,
    render_header,
    render_kpi,
    render_section_header,
)
from components.sidebar import render_sidebar
from config import THEME_COLORS

st.set_page_config(page_title="Reports & Audit Trail", page_icon="📑", layout="wide")
inject_custom_css()
render_sidebar()

render_header(
    title="Executive Reports & Compliance Audit Trail",
    subtitle="Generate comprehensive platform intelligence summaries and review operational audit logs",
    badge="Compliance & Governance"
)

tab_reports, tab_audit = st.tabs([
    "📊 Downloadable Executive Reports",
    "📜 System Event Audit Log"
])

# Tab 1: Downloadable Reports
with tab_reports:
    render_section_header("📥", "Generate Platform Intelligence Reports")
    st.caption("Export complete executive briefings encompassing KPIs, top business issues, and model metrics.")

    col_rep1, col_rep2 = st.columns(2)

    with col_rep1:
        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid {THEME_COLORS['border']};
                border-radius: 12px; padding: 1.25rem; margin-bottom: 1rem;">
                <div style="font-weight: 700; color: {THEME_COLORS['text']}; font-size: 1.05rem; margin-bottom: 0.3rem;">
                    1. JSON Intelligence Payload
                </div>
                <div style="font-size: 0.85rem; color: {THEME_COLORS['text_muted']}; margin-bottom: 0.8rem;">
                    Machine-readable export containing telemetry KPIs, active issues, and recent predictions.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        json_str = ReportService.generate_executive_report_json()
        st.download_button(
            label="⬇️ Download Executive Report (JSON)",
            data=json_str,
            file_name=f"executive_sentiment_report_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json",
            type="primary",
            use_container_width=True
        )

    with col_rep2:
        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid {THEME_COLORS['border']};
                border-radius: 12px; padding: 1.25rem; margin-bottom: 1rem;">
                <div style="font-weight: 700; color: {THEME_COLORS['text']}; font-size: 1.05rem; margin-bottom: 0.3rem;">
                    2. Markdown Executive Briefing
                </div>
                <div style="font-size: 0.85rem; color: {THEME_COLORS['text_muted']}; margin-bottom: 0.8rem;">
                    Formatted report ready for stakeholder distribution and executive presentations.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        md_str = ReportService.generate_executive_report_markdown()
        st.download_button(
            label="⬇️ Download Executive Briefing (Markdown)",
            data=md_str,
            file_name=f"executive_sentiment_briefing_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown",
            type="primary",
            use_container_width=True
        )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    render_section_header("👁️", "Executive Briefing Preview")
    st.markdown(
        f"""
        <div style="background: rgba(30, 41, 59, 0.4); border: 1px solid {THEME_COLORS['border']};
            border-radius: 12px; padding: 1.5rem; margin-top: 0.5rem;">
        """,
        unsafe_allow_html=True
    )
    st.markdown(md_str)
    st.markdown("</div>", unsafe_allow_html=True)

# Tab 2: Audit Logs
with tab_audit:
    render_section_header("📜", "System Event Audit Trail")
    st.caption("Immutable record of critical operational actions including model predictions, batch imports, and issue status transitions.")

    logs = Repository.get_audit_logs(limit=200)

    if logs:
        df_logs = pd.DataFrame(logs)[["id", "created_at", "action_type", "entity_type", "entity_id", "description"]]
        df_logs.columns = ["Event ID", "Timestamp", "Action Type", "Entity", "Entity ID", "Description"]
        st.dataframe(df_logs, use_container_width=True, hide_index=True)

        st.download_button(
            label="Download Complete Audit Trail as CSV",
            data=df_logs.to_csv(index=False).encode('utf-8'),
            file_name=f"audit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No audit logs recorded yet.")
