"""Page 11: Data Management for inspecting datasets, validating data quality, and viewing import logs."""
import streamlit as st
import pandas as pd
from database.repository import Repository
from config import CLEAN_REVIEWS_CSV, THEME_COLORS
from components.ui_elements import (
    inject_custom_css,
    render_header,
    render_kpi,
    render_section_header,
)
from components.charts import plot_sentiment_donut
from components.sidebar import render_sidebar

st.set_page_config(page_title="Data Management", page_icon="📁", layout="wide")
inject_custom_css()
render_sidebar()

render_header(
    title="Data Management & Quality Governance",
    subtitle="Audit review datasets, inspect missing values, validate label distributions, and track import history",
    badge="Data Operations"
)

tab_inspect, tab_imports, tab_corpus = st.tabs([
    "📊 Local Dataset Inspection",
    "📜 Import History & Batches",
    "💾 Export & Privacy Governance"
])

# Tab 1: Local Dataset Inspection
with tab_inspect:
    render_section_header("🔍", "Cleaned Dataset Inspection (clean_product_reviews.csv)")

    try:
        sample_df = pd.read_csv(CLEAN_REVIEWS_CSV, nrows=1000)
        total_cols = len(sample_df.columns)
        null_counts = sample_df.isnull().sum()
        dup_count = sample_df.duplicated(subset=['clean_review']).sum()

        c1, c2, c3, c4 = st.columns(4)
        render_kpi(c1, "Audited Sample", f"{len(sample_df):,} Rows", "Sampled from repository file")
        render_kpi(c2, "Schema Width", f"{total_cols} Columns", "Structured feature columns")
        render_kpi(c3, "Missing Values", f"{null_counts.sum()}", "In inspected sample")
        render_kpi(c4, "Duplicate Reviews", f"{dup_count}", "Repeated text occurrences")

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        st.markdown("#### Sample Schema & First 5 Records")
        st.dataframe(sample_df.head(5), use_container_width=True)

        st.markdown("#### Data Quality & Column Details")
        col_summary = []
        for col in sample_df.columns:
            col_summary.append({
                "Column Name": col,
                "Data Type": str(sample_df[col].dtype),
                "Missing Values": int(sample_df[col].isnull().sum()),
                "Unique Values": int(sample_df[col].nunique())
            })
        st.dataframe(pd.DataFrame(col_summary), use_container_width=True, hide_index=True)

        if "sentiment_numeric" in sample_df.columns:
            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
            render_section_header("⚖️", "Label Distribution (Ground Truth)")
            pos_cnt = int((sample_df["sentiment_numeric"] == 1).sum())
            neg_cnt = int((sample_df["sentiment_numeric"] == 0).sum())
            fig_lbl = plot_sentiment_donut(pos_cnt, neg_cnt, title="Ground Truth Label Balance (0=Neg, 1=Pos)")
            st.plotly_chart(fig_lbl, use_container_width=True)

    except Exception as e:
        st.error(f"Error reading local dataset: {e}")

# Tab 2: Import History
with tab_imports:
    render_section_header("📜", "Batch Import History Logs")
    imports = Repository.get_batch_imports()
    if imports:
        imp_df = pd.DataFrame(imports)
        imp_df.columns = ["Import ID", "File Name", "Total Rows", "Valid Rows", "Invalid / Skipped", "Timestamp"]
        st.dataframe(imp_df, use_container_width=True, hide_index=True)
    else:
        st.info("No bulk batch files have been imported yet. Use the Batch Review Analytics page to upload a CSV.")

# Tab 3: Privacy & Export
with tab_corpus:
    render_section_header("🔐", "Data Governance & Privacy Safeguards")
    st.markdown(
        f"""
        <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid {THEME_COLORS['border']};
            border-radius: 12px; padding: 1.25rem; margin-bottom: 1.5rem;">
            <ul style="margin: 0; padding-left: 1.2rem; color: {THEME_COLORS['text']}; line-height: 1.7;">
                <li><strong>PII Sanitization:</strong> Customer review texts are stripped of emails, phone numbers, and addresses during automated preprocessing.</li>
                <li><strong>Retention Policy:</strong> Analysis records are persisted locally in SQLite (<code>sentiment_system.db</code>). No customer feedback is transmitted to third-party external APIs.</li>
                <li><strong>Data Integrity:</strong> Original uploaded files and repository dataset artifacts are never modified in-place.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    render_section_header("📥", "Export Active Database Records")
    all_b = Repository.get_batch_reviews(limit=1000)
    if all_b:
        csv_data = pd.DataFrame(all_b).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download All Stored Batch Reviews as CSV",
            data=csv_data,
            file_name="stored_batch_reviews_backup.csv",
            mime="text/csv",
            type="primary"
        )
    else:
        st.caption("No batch records available to export.")
