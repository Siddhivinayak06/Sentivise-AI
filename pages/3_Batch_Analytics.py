"""Page 3: Batch Review Analytics for uploading, validating, analyzing, and exporting bulk review datasets."""
import io
import streamlit as st
import pandas as pd
from datetime import datetime

from config import ML_MODEL_PATHS, X_TEST_CSV, THEME_COLORS
from services.batch_service import validate_batch_dataframe, process_batch_reviews
from components.ui_elements import (
    inject_custom_css,
    render_header,
    render_kpi,
    render_empty_state,
    render_section_header,
)
from components.charts import plot_sentiment_donut
from components.sidebar import render_sidebar

st.set_page_config(page_title="Batch Review Analytics", page_icon="📦", layout="wide")
inject_custom_css()
render_sidebar()

render_header(
    title="Batch Review Analytics & Bulk Inference",
    subtitle="Upload customer feedback datasets, perform automated schema validation, and run high-throughput sentiment classification",
    badge="Bulk Processing"
)

# Step 1: Upload or Load Sample Data
render_section_header("📤", "Upload Review Dataset")
up_c1, up_c2 = st.columns([8, 4])

with up_c1:
    uploaded_file = st.file_uploader("Upload CSV review file", type=["csv"], help="Upload any CSV file containing customer feedback columns.")

with up_c2:
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    load_sample = st.button("📂 Load 50 Samples from Holdout Test Set", use_container_width=True)

df_raw = None
source_name = "custom_upload.csv"

if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file)
        source_name = uploaded_file.name
        st.success(f"Loaded {len(df_raw):,} records from `{source_name}`.")
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
elif load_sample:
    try:
        df_sample = pd.read_csv(X_TEST_CSV, nrows=50)
        # Add synthetic product ID and date columns to demonstrate full enterprise mapping
        df_sample["product_id"] = [f"PRD-{(i % 5) + 101}" for i in range(len(df_sample))]
        df_sample["rating"] = [(5.0 if i % 2 == 0 else 1.0) for i in range(len(df_sample))]
        df_sample["date"] = pd.date_range(end=datetime.now(), periods=len(df_sample)).strftime("%Y-%m-%d")
        df_raw = df_sample
        source_name = "test_holdout_sample_50.csv"
        st.session_state["batch_loaded_sample"] = df_raw
        st.success("Loaded 50 representative reviews from test split.")
    except Exception as e:
        st.error(f"Error loading sample dataset: {e}")
elif "batch_loaded_sample" in st.session_state:
    df_raw = st.session_state["batch_loaded_sample"]
    source_name = "test_holdout_sample_50.csv"

if df_raw is not None:
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    render_section_header("🔍", "Schema Inspection & Column Mapping")
    
    with st.expander("👁️ Preview Raw Uploaded Dataset (First 5 Rows)", expanded=True):
        st.dataframe(df_raw.head(5), use_container_width=True)

    col_names = ["<None>"] + list(df_raw.columns)
    
    # Auto-detect text column
    default_text_idx = 0
    for cand in ["clean_review", "review_text", "review", "text", "comment", "feedback"]:
        if cand in df_raw.columns:
            default_text_idx = col_names.index(cand)
            break

    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        sel_text_col = st.selectbox("Review Text Column * (Required)", [c for c in col_names if c != "<None>"], index=max(0, default_text_idx - 1))
        sel_product_col = st.selectbox("Product ID Column (Optional)", col_names, index=col_names.index("product_id") if "product_id" in col_names else 0)
    with m_col2:
        sel_name_col = st.selectbox("Product Name Column (Optional)", col_names, index=col_names.index("product_name") if "product_name" in col_names else 0)
        sel_rating_col = st.selectbox("Rating Column (Optional)", col_names, index=col_names.index("rating") if "rating" in col_names else 0)
    with m_col3:
        sel_date_col = st.selectbox("Date Column (Optional)", col_names, index=col_names.index("date") if "date" in col_names else 0)
        remove_dups = st.checkbox("Remove Exact Text Duplicates", value=True)

    # Perform validation
    val_report = validate_batch_dataframe(
        df=df_raw,
        text_col=sel_text_col,
        rating_col=sel_rating_col if sel_rating_col != "<None>" else None,
        date_col=sel_date_col if sel_date_col != "<None>" else None
    )

    v1, v2, v3, v4 = st.columns(4)
    render_kpi(v1, "Total Rows", f"{val_report['total_rows']:,}", "Rows in file")
    render_kpi(v2, "Missing Text", f"{val_report['missing_text_count']}", "Empty reviews")
    render_kpi(v3, "Duplicate Reviews", f"{val_report['duplicate_count']}", "Identical text")
    render_kpi(v4, "Invalid Telemetry", f"{val_report['invalid_rating_count'] + val_report['invalid_date_count']}", "Malformed values")

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    render_section_header("🚀", "Model Selection & Batch Inference")

    ctrl1, ctrl2 = st.columns([8, 4])
    with ctrl1:
        chosen_model = st.selectbox("Select ML Model for Batch Inference", list(ML_MODEL_PATHS.keys()), index=0)
    with ctrl2:
        persist_db = st.checkbox("Persist Batch Results to SQLite Database", value=True)

    start_batch_btn = st.button("🚀 Start Batch Processing", type="primary", use_container_width=True)

    if start_batch_btn:
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_progress(pct, msg):
            progress_bar.progress(pct)
            status_text.text(msg)

        try:
            analyzed_df, summary = process_batch_reviews(
                df=df_raw,
                text_col=sel_text_col,
                product_id_col=sel_product_col if sel_product_col != "<None>" else None,
                product_name_col=sel_name_col if sel_name_col != "<None>" else None,
                date_col=sel_date_col if sel_date_col != "<None>" else None,
                rating_col=sel_rating_col if sel_rating_col != "<None>" else None,
                model_name=chosen_model,
                remove_duplicates=remove_dups,
                progress_callback=update_progress,
                persist_to_db=persist_db,
                filename=source_name
            )

            status_text.text("Batch Processing Complete!")
            st.session_state["analyzed_batch_df"] = analyzed_df
            st.session_state["analyzed_batch_summary"] = summary
            st.success(f"Successfully processed {summary['successful_predictions']:,} reviews! Removed {summary['duplicates_removed']} duplicate records.")

        except Exception as e:
            st.error(f"Batch processing error: {e}")

if "analyzed_batch_df" in st.session_state:
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    render_section_header("📊", "Batch Results & Interactive Filtering")
    
    analyzed_df = st.session_state["analyzed_batch_df"]
    summary = st.session_state["analyzed_batch_summary"]

    s1, s2, s3, s4 = st.columns(4)
    render_kpi(s1, "Analyzed Records", f"{summary['successful_predictions']:,}", "Successfully classified")
    render_kpi(s2, "Positive Count", f"{summary['positive_count']:,}", f"{summary['positive_percentage']}% of total")
    render_kpi(s3, "Negative Count", f"{summary['negative_count']:,}", f"{summary['negative_percentage']}% of total")
    render_kpi(s4, "Rejected / Skipped", f"{summary['rejected_rows']:,}", "Too short or invalid")

    # Filter Controls for results
    rf1, rf2 = st.columns(2)
    with rf1:
        res_sent_filter = st.selectbox("Filter Results by Sentiment", ["All", "Positive", "Negative"], index=0)
    with rf2:
        search_kw = st.text_input("Search Text Keywords in Batch", placeholder="e.g. battery, broke, sound")

    display_df = analyzed_df
    if res_sent_filter != "All":
        display_df = display_df[display_df["predicted_sentiment"] == res_sent_filter]
    if search_kw.strip():
        display_df = display_df[display_df.astype(str).apply(lambda row: search_kw.lower() in row.to_string().lower(), axis=1)]

    st.dataframe(display_df, use_container_width=True)

    # Export Download
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    render_section_header("📥", "Download Analyzed Results")
    csv_bytes = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Analyzed CSV (with Appended Predictions & Rules)",
        data=csv_bytes,
        file_name=f"analyzed_batch_reviews_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        type="primary"
    )
