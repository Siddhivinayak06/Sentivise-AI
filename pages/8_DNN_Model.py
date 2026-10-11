"""Page 8: Deep Neural Network (DNN) Model Architecture, Training History, and Evaluation."""
import os
import streamlit as st
import pandas as pd
from PIL import Image

from config import (
    TASK7_EVAL_REPORT_PATH, TASK7_HISTORY_PATH, TASK6_VS_TASK7_PATH,
    BASE_DIR, DNN_MODEL_PATH, FIGURES_DIR, find_file, THEME_COLORS
)
from services.model_loader import load_dnn_artifacts
from components.ui_elements import (
    inject_custom_css,
    render_header,
    render_kpi,
    render_section_header,
)
from components.charts import plot_training_history_curves
from components.sidebar import render_sidebar

st.set_page_config(page_title="Deep Neural Network", page_icon="🧬", layout="wide")
inject_custom_css()
render_sidebar()

render_header(
    title="Deep Neural Network (DNN) Architecture & Evaluation",
    subtitle="Audit Task 7 deep learning model parameters, loss/accuracy training curves, and comparison against classical ML",
    badge="Deep Learning"
)

# Load DNN artifacts
dnn_status = load_dnn_artifacts()
is_ready = dnn_status["is_ready"]

c1, c2, c3, c4 = st.columns(4)
render_kpi(c1, "Model Format", "Keras / HDF5", "task7_dnn_model.keras (314 KB)")
render_kpi(c2, "Input Dimension", "100 Dense Features", "Standardized by task7_scaler.pkl")
render_kpi(c3, "Output Format", "Sigmoid [0, 1]", "P(Positive Class)")
render_kpi(c4, "Artifact Status", "Loaded & Ready" if is_ready else "Artifact Missing", "StandardScaler + LabelEncoder ready")

st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

tab_curves, tab_metrics, tab_arch, tab_compare = st.tabs([
    "📈 Training & Validation Curves",
    "📊 Evaluation Metrics & Confusion Matrix",
    "🏗️ Architecture & Hyperparameters",
    "⚖️ Classical ML vs DNN Comparison"
])

# Tab 1: Curves
with tab_curves:
    render_section_header("📉", "Epoch Loss & Accuracy Trajectory")
    if os.path.exists(TASK7_HISTORY_PATH):
        hist_df = pd.read_csv(TASK7_HISTORY_PATH)
        fig_hist = plot_training_history_curves(hist_df)
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Training history CSV not found.")

    # Saved PNG Curves
    curve_col1, curve_col2 = st.columns(2)
    with curve_col1:
        loss_img_path = find_file(FIGURES_DIR / "task7_loss_curve.png", BASE_DIR / "task7_loss_curve.png")
        if loss_img_path.exists():
            st.image(Image.open(loss_img_path), caption="Task 7 Training vs Validation Loss Curve", use_container_width=True)
    with curve_col2:
        acc_img_path = find_file(FIGURES_DIR / "task7_accuracy_curve.png", BASE_DIR / "task7_accuracy_curve.png")
        if acc_img_path.exists():
            st.image(Image.open(acc_img_path), caption="Task 7 Training vs Validation Accuracy Curve", use_container_width=True)

# Tab 2: Metrics & Confusion Matrix
with tab_metrics:
    render_section_header("📊", "Task 7 Test Set Evaluation Metrics")
    if os.path.exists(TASK7_EVAL_REPORT_PATH):
        t7_eval = pd.read_csv(TASK7_EVAL_REPORT_PATH)
        st.dataframe(t7_eval, use_container_width=True, hide_index=True)
    else:
        st.info("Task 7 evaluation report not found.")

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        cm_path = find_file(FIGURES_DIR / "task7_confusion_matrix.png", BASE_DIR / "task7_confusion_matrix.png")
        if cm_path.exists():
            st.image(Image.open(cm_path), caption="Task 7 DNN Confusion Matrix", use_container_width=True)
    with m_col2:
        roc_path = find_file(FIGURES_DIR / "task7_roc_curve.png", BASE_DIR / "task7_roc_curve.png")
        if roc_path.exists():
            st.image(Image.open(roc_path), caption="Task 7 DNN ROC Curve (AUC: 0.8945)", use_container_width=True)

# Tab 3: Architecture
with tab_arch:
    render_section_header("🏗️", "Deep Neural Network Architecture Specification")
    
    st.markdown(
        f"""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin-bottom: 1.5rem;">
            <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid {THEME_COLORS['border']}; border-radius: 12px; padding: 1rem;">
                <div style="font-size: 0.75rem; text-transform: uppercase; color: {THEME_COLORS['text_muted']}; font-weight: 700;">Input Layer</div>
                <div style="font-size: 1.1rem; font-weight: 800; color: {THEME_COLORS['primary']}; margin: 0.3rem 0;">100 Dense Features</div>
                <div style="font-size: 0.8rem; color: {THEME_COLORS['text_muted']};">Task 5 Feature Matrix</div>
            </div>
            <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid {THEME_COLORS['border']}; border-radius: 12px; padding: 1rem;">
                <div style="font-size: 0.75rem; text-transform: uppercase; color: {THEME_COLORS['text_muted']}; font-weight: 700;">Hidden Block 1</div>
                <div style="font-size: 1.1rem; font-weight: 800; color: {THEME_COLORS['secondary']}; margin: 0.3rem 0;">Dense(64) + ReLU</div>
                <div style="font-size: 0.8rem; color: {THEME_COLORS['text_muted']};">Dropout (0.3) Regularization</div>
            </div>
            <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid {THEME_COLORS['border']}; border-radius: 12px; padding: 1rem;">
                <div style="font-size: 0.75rem; text-transform: uppercase; color: {THEME_COLORS['text_muted']}; font-weight: 700;">Hidden Block 2</div>
                <div style="font-size: 1.1rem; font-weight: 800; color: {THEME_COLORS['secondary']}; margin: 0.3rem 0;">Dense(32) + ReLU</div>
                <div style="font-size: 0.8rem; color: {THEME_COLORS['text_muted']};">Dropout (0.2) Regularization</div>
            </div>
            <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid {THEME_COLORS['border']}; border-radius: 12px; padding: 1rem;">
                <div style="font-size: 0.75rem; text-transform: uppercase; color: {THEME_COLORS['text_muted']}; font-weight: 700;">Output Layer</div>
                <div style="font-size: 1.1rem; font-weight: 800; color: {THEME_COLORS['positive']}; margin: 0.3rem 0;">Dense(1) + Sigmoid</div>
                <div style="font-size: 0.8rem; color: {THEME_COLORS['text_muted']};">Binary Calibrated Probability</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if is_ready and dnn_status["model"] is not None:
        st.markdown("#### Keras Summary Representation")
        stringlist = []
        dnn_status["model"].summary(print_fn=lambda x: stringlist.append(x))
        st.code("\n".join(stringlist), language="text")

# Tab 4: Comparison
with tab_compare:
    render_section_header("⚖️", "Task 6 Classical ML vs Task 7 DNN Benchmark")
    if os.path.exists(TASK6_VS_TASK7_PATH):
        comp_df = pd.read_csv(TASK6_VS_TASK7_PATH)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        st.markdown(
            f"""
            <div style="background: rgba(99, 102, 241, 0.08); border-left: 4px solid {THEME_COLORS['primary']};
                border-radius: 0 12px 12px 0; padding: 1.1rem; margin-top: 1rem;">
                <strong style="color: {THEME_COLORS['text']}; font-size: 0.95rem;">Architectural Finding:</strong>
                <p style="margin: 0.4rem 0 0; color: {THEME_COLORS['text_muted']}; font-size: 0.9rem; line-height: 1.5;">
                    While the Deep Neural Network achieves an exceptional precision of <strong>98.29%</strong> and a competitive ROC-AUC of <strong>0.8945</strong>,
                    the regularized <strong>Logistic Regression Pipeline</strong> (Task 6) remains the superior production model for overall balanced accuracy (79.01% vs 59.91%)
                    and execution latency (0.01ms vs 12ms per review).
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
