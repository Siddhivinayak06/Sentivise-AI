"""Page 7: Machine Learning Model Evaluation displaying true benchmark metrics and comparison reports."""
import os
import streamlit as st
import pandas as pd
from PIL import Image

from config import (
    TASK6_EVAL_REPORT_PATH, TASK6_COMPARISON_REPORT_PATH, BASE_DIR,
    FIGURES_DIR, find_file, THEME_COLORS
)
from components.ui_elements import (
    inject_custom_css,
    render_header,
    render_kpi,
    render_section_header,
)
from components.charts import plot_model_comparison_bar
from components.sidebar import render_sidebar

st.set_page_config(page_title="ML Model Evaluation", page_icon="🔬", layout="wide")
inject_custom_css()
render_sidebar()

render_header(
    title="Classical ML Model Evaluation & Benchmarking",
    subtitle="Audit accuracy, precision, recall, F1, ROC-AUC, latency, and confusion matrices across Task 6 classifiers",
    badge="Model Governance"
)

# Load Stored Evaluation Report
if os.path.exists(TASK6_EVAL_REPORT_PATH):
    eval_df = pd.read_csv(TASK6_EVAL_REPORT_PATH)
else:
    eval_df = pd.DataFrame()

if eval_df.empty:
    st.error("Evaluation report artifact `task6_evaluation_report.csv` not found.")
else:
    render_section_header("🏆", "Model Performance Summary (Holdout Test Set &bull; 80,000 Samples)")
    
    st.markdown(
        f"""
        <div style="background: rgba(99, 102, 241, 0.08); border: 1px solid rgba(99, 102, 241, 0.25);
            border-radius: 12px; padding: 0.9rem 1.2rem; margin-bottom: 1.5rem; display: flex; align-items: center; gap: 0.8rem;">
            <span style="font-size: 1.3rem;">📌</span>
            <span style="font-size: 0.88rem; color: {THEME_COLORS['text_muted']};">
                <strong style="color: {THEME_COLORS['text']};">Audit Verification:</strong>
                Evaluated strictly on the 80,000-instance holdout test dataset (<code>X_test.csv</code>) using 100 optimized Task 5 features.
                Binary Label Mapping: <code>0: Negative</code>, <code>1: Positive</code>.
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # High-level KPIs for Best Model (Logistic Regression)
    best_row = eval_df[eval_df["Model"] == "Logistic Regression"].iloc[0]
    m1, m2, m3, m4, m5 = st.columns(5)
    render_kpi(m1, "Best Model Accuracy", f"{best_row['Accuracy'] * 100:.2f}%", "Logistic Regression")
    render_kpi(m2, "Best Model ROC-AUC", f"{best_row['AUC']:.4f}", "Highest discrimination")
    render_kpi(m3, "Precision (Weighted)", f"{best_row['Precision']:.4f}", "Positive predictive value")
    render_kpi(m4, "Recall (Weighted)", f"{best_row['Recall']:.4f}", "True positive rate")
    render_kpi(m5, "F1-Score", f"{best_row['F1-Score']:.4f}", "Harmonic mean")

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # Interactive Comparison Chart
    render_section_header("📊", "Multi-Metric Comparative Performance")
    fig_comp = plot_model_comparison_bar(eval_df)
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # Complete Tabular Evaluation
    render_section_header("📑", "Detailed Classification Benchmark Metrics")
    formatted_df = eval_df.copy()
    formatted_df["Accuracy"] = formatted_df["Accuracy"].apply(lambda x: f"{x * 100:.2f}%")
    formatted_df["Precision"] = formatted_df["Precision"].apply(lambda x: f"{x:.4f}")
    formatted_df["Recall"] = formatted_df["Recall"].apply(lambda x: f"{x:.4f}")
    formatted_df["F1-Score"] = formatted_df["F1-Score"].apply(lambda x: f"{x:.4f}")
    formatted_df["AUC"] = formatted_df["AUC"].apply(lambda x: f"{x:.4f}")
    formatted_df["Training Time"] = formatted_df["Training Time"].apply(lambda x: f"{x:.2f}s")
    formatted_df["Prediction Time"] = formatted_df["Prediction Time"].apply(lambda x: f"{x * 1000:.1f}ms")

    st.dataframe(formatted_df, use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # Confusion Matrices and ROC Curves
    render_section_header("🖼️", "Saved Confusion Matrices & ROC Curves")
    cm_tabs = st.tabs(["Logistic Regression", "Random Forest", "Support Vector Machine", "Decision Tree", "XGBoost", "ROC Curves"])

    cm_map = {
        0: "confusion_matrix_logistic_regression.png",
        1: "confusion_matrix_random_forest.png",
        2: "confusion_matrix_svm.png",
        3: "confusion_matrix_decision_tree.png",
        4: "confusion_matrix_xgboost.png",
    }

    for tab_idx, fname in cm_map.items():
        with cm_tabs[tab_idx]:
            img_path = find_file(FIGURES_DIR / fname, BASE_DIR / fname)
            if img_path.exists():
                img = Image.open(img_path)
                st.image(img, caption=f"Confusion Matrix ({fname})", use_container_width=True)
            else:
                st.info(f"Image `{fname}` not found.")

    with cm_tabs[5]:
        roc_path = find_file(FIGURES_DIR / "task6_roc_curves.png", BASE_DIR / "task6_roc_curves.png")
        if roc_path.exists():
            roc_img = Image.open(roc_path)
            st.image(roc_img, caption="Task 6 Combined ROC Curves (All Models)", use_container_width=True)
        else:
            st.info("ROC Curves figure not found.")
