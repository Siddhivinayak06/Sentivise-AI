"""Page 9: Explainable AI (XAI) integrating SHAP, LIME, and Counterfactual Explanations."""
import os
import streamlit as st
import pandas as pd
from PIL import Image

from config import TASK8_DIR, THEME_COLORS
from services.explainability_service import ExplainabilityService
from components.ui_elements import (
    inject_custom_css,
    render_header,
    render_kpi,
    render_section_header,
)
from components.charts import plot_feature_attribution_bar
from components.sidebar import render_sidebar

st.set_page_config(page_title="Explainable AI (XAI)", page_icon="🔎", layout="wide")
inject_custom_css()
render_sidebar()

render_header(
    title="Explainable AI (XAI) Transparency Lab",
    subtitle="Audit model behavior using SHAP cooperative game theory, LIME surrogate models, and minimal Counterfactuals",
    badge="Algorithmic Transparency"
)

st.markdown(
    f"""
    <div style="background: rgba(251, 191, 36, 0.08); border: 1px solid rgba(251, 191, 36, 0.25);
        border-radius: 12px; padding: 0.9rem 1.2rem; margin-bottom: 1.5rem; display: flex; align-items: center; gap: 0.8rem;">
        <span style="font-size: 1.3rem;">⚠️</span>
        <span style="font-size: 0.88rem; color: {THEME_COLORS['text_muted']};">
            <strong style="color: {THEME_COLORS['warning']};">Responsible AI Notice:</strong>
            Feature attributions explain internal model mechanics; they do <strong>not</strong> establish causal proof or guarantee that the model prediction is objectively factual.
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

# Curated reviews from Task 8
CURATED_REVIEWS = {
    "Clearly Positive Review": "This product is absolutely amazing! The build quality is excellent, great performance, and I really love using it every day.",
    "Clearly Negative Review": "Terrible quality. It broke after two days of use. Completely useless, awful customer support, and a total waste of money.",
    "Ambiguous / Mixed Review": "The design looks good and attractive, but the battery life is poor and disappointing. Mixed feelings overall.",
    "Edge Case / Neutral Review": "Not bad at all for the price, though not the best quality either. It works as expected."
}

xai_col1, xai_col2 = st.columns([8, 4])
with xai_col2:
    st.markdown(
        f"""
        <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid {THEME_COLORS['border']};
            border-radius: 12px; padding: 1rem; margin-bottom: 0.8rem;">
            <div style="font-weight: 700; color: {THEME_COLORS['text']}; font-size: 0.95rem; margin-bottom: 0.5rem;">
                ⚙️ Select Benchmark Review
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    sel_preset = st.selectbox("Benchmark Review Sample", list(CURATED_REVIEWS.keys()))
    default_text = CURATED_REVIEWS[sel_preset]

with xai_col1:
    user_review_text = st.text_area("Review Text to Explain", value=default_text, height=110)

tab_shap, tab_lime, tab_cf, tab_global = st.tabs([
    "🧩 Local SHAP Attribution",
    "🍋 Local LIME Explanations",
    "🔄 Counterfactual Edits",
    "🌐 Global Corpus Importance"
])

# Tab 1: Local SHAP
with tab_shap:
    render_section_header("🧩", "SHAP (Shapley Additive exPlanations)")
    st.caption("Grounded in cooperative game theory: Features divide credit for the prediction relative to a background baseline.")

    if user_review_text.strip():
        with st.spinner("Computing local SHAP feature attributions..."):
            shap_res = ExplainabilityService.explain_with_shap(user_review_text, top_k=10)

        if shap_res.get("success"):
            top_f = shap_res["top_features"]
            names = [x["feature"] for x in top_f]
            attrs = [x["attribution"] for x in top_f]

            fig_shap = plot_feature_attribution_bar(names, attrs, title="Local SHAP Feature Contributions")
            st.plotly_chart(fig_shap, use_container_width=True)

            with st.expander("📊 View Attributions Data Table", expanded=False):
                st.dataframe(pd.DataFrame(top_f), use_container_width=True, hide_index=True)
        else:
            st.error(shap_res.get("error"))

# Tab 2: Local LIME
with tab_lime:
    render_section_header("🍋", "LIME (Local Interpretable Model-agnostic Explanations)")
    st.caption("Perturbation surrogate model: Tests variations of the text to estimate each token's local importance weight.")

    if st.button("🚀 Generate LIME Explanation", type="primary", key="btn_run_lime"):
        with st.spinner("Perturbing input text and fitting local surrogate model (300 samples)..."):
            lime_res = ExplainabilityService.explain_with_lime(user_review_text, num_features=8)

        if lime_res.get("success"):
            tokens = [x["token"] for x in lime_res["token_explanations"]]
            weights = [x["weight"] for x in lime_res["token_explanations"]]

            fig_lime = plot_feature_attribution_bar(tokens, weights, title="LIME Local Token Importance")
            st.plotly_chart(fig_lime, use_container_width=True)

            with st.expander("📊 View LIME Tokens Table", expanded=True):
                st.dataframe(pd.DataFrame(lime_res["token_explanations"]), use_container_width=True, hide_index=True)
        else:
            st.error(lime_res.get("error"))
    else:
        st.info("Click 'Generate LIME Explanation' above to run the perturbation sampler.")

# Tab 3: Counterfactuals
with tab_cf:
    render_section_header("🔄", "Counterfactual Explanation Generator")
    st.caption("Identifies the minimal token changes that flip the model's sentiment classification.")

    cf_res = ExplainabilityService.generate_counterfactual(user_review_text)
    if cf_res.get("success"):
        c1, c2 = st.columns(2)
        render_kpi(c1, "Original Sentiment", cf_res['original_sentiment'], f"Confidence: {cf_res['original_confidence'] * 100:.1f}%")
        render_kpi(c2, "Target Flipping Sentiment", cf_res['target_sentiment'], "Polarity Inversion Goal")

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

        cfs = cf_res["counterfactuals"]
        if cfs:
            for idx, c in enumerate(cfs):
                st.markdown(
                    f"""
                    <div class="rec-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                            <strong style="color: {THEME_COLORS['text']}; font-size: 1rem;">Counterfactual Variant #{idx + 1}</strong>
                            <span class="badge badge-info">Replaced &lsquo;{c['original_word']}&rsquo; &rarr; &lsquo;{c['replacement_word']}&rsquo;</span>
                        </div>
                        <p style="color: {THEME_COLORS['text']}; margin-bottom: 0.4rem; font-size: 0.95rem; font-style: italic;">
                            &ldquo;{c['counterfactual_text']}&rdquo;
                        </p>
                        <p style="font-size: 0.85rem; color: {THEME_COLORS['text_muted']}; margin: 0;">
                            Flipped Sentiment: <strong style="color: {THEME_COLORS['positive'] if c['new_sentiment'] == 'Positive' else THEME_COLORS['negative']};">{c['new_sentiment']}</strong>
                            ({c['new_confidence'] * 100:.1f}% confidence)
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info("No single-token replacement was sufficient to flip polarity for this review.")
    else:
        st.error(cf_res.get("error"))

# Tab 4: Global Corpus Importance
with tab_global:
    render_section_header("🌐", "Task 8 Global Corpus Feature Attributions")
    st.caption("Summary plot showing top global features driving model predictions across the entire corpus.")

    summary_img_path = TASK8_DIR / "graphs" / "03_shap_global_importance.png"
    if os.path.exists(summary_img_path):
        st.image(Image.open(summary_img_path), caption="Task 8 SHAP Global Feature Importance (Top Features Across Dataset)", use_container_width=True)
    else:
        st.info("Task 8 global importance figure not found.")

    comp_img_path = TASK8_DIR / "graphs" / "08_explanation_runtime_comparison.png"
    if os.path.exists(comp_img_path):
        st.image(Image.open(comp_img_path), caption="Runtime Efficiency Comparison: SHAP vs LIME vs Counterfactuals", use_container_width=True)
