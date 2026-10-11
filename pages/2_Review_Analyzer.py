"""Page 2: Single Review Analyzer for real-time sentiment prediction and rule-based decision support."""
import json
import streamlit as st
import pandas as pd
from datetime import datetime

from config import ML_MODEL_PATHS, APP_VERSION, THEME_COLORS
from services.prediction_service import predict_single_review
from services.explainability_service import ExplainabilityService
from database.repository import Repository
from components.ui_elements import (
    inject_custom_css,
    render_header,
    render_kpi,
    render_section_header,
)
from components.charts import plot_feature_attribution_bar
from components.sidebar import render_sidebar

st.set_page_config(page_title="Single Review Analyzer", page_icon="🔍", layout="wide")
inject_custom_css()
render_sidebar()

render_header(
    title="Single Review Sentiment Analyzer & Decision Engine",
    subtitle="Evaluate customer review text, inspect confidence probabilities, and trigger automated business actions",
    badge="Real-Time Inference"
)

# Available models
model_options = list(ML_MODEL_PATHS.keys()) + ["Deep Neural Network (DNN)"]

# Sample preset reviews
PRESET_REVIEWS = {
    "Custom Review (Type Below)": "",
    "Clearly Positive (Audio Product)": "This product is absolutely amazing! The sound quality is excellent, great performance, and I really love using it every day.",
    "Critical Defect / Durability (Negative)": "Terrible quality. The charging port broke after two days of use. Completely useless, awful customer support, and a total waste of money.",
    "Safety Hazard (Burn/Fire Risk)": "Warning: the power adapter became scorching hot, started smoking, and almost caught fire! Serious safety hazard and defective hardware.",
    "Logistics & Delivery Delay": "The product itself is okay, but shipping took over 3 weeks to arrive and the box was completely smashed by the carrier.",
    "Ambiguous / Mixed Sentiment": "The design looks good and attractive, but the battery life is poor and disappointing. Mixed feelings overall."
}

col_input, col_config = st.columns([8, 4])

with col_config:
    render_section_header("⚙️", "Configuration")
    selected_preset = st.selectbox("Select Benchmark Sample or Type Custom", list(PRESET_REVIEWS.keys()))
    selected_model = st.selectbox("Classification Model", model_options, index=0)
    enable_rules = st.checkbox("Execute Decision Rule Engine", value=True)
    save_to_db = st.checkbox("Persist Prediction to Audit History", value=True)

with col_input:
    render_section_header("✍️", "Customer Review Text")
    default_text = PRESET_REVIEWS[selected_preset]
    review_input = st.text_area(
        "Enter or paste customer feedback:",
        value=default_text,
        height=140,
        placeholder="Type or paste customer review text here (minimum 3 characters)..."
    )
    analyze_btn = st.button("🚀 Analyze Sentiment & Deduce Business Actions", type="primary", use_container_width=True)

if analyze_btn:
    if not review_input.strip():
        st.warning("⚠️ Please provide review text to analyze.")
    else:
        with st.spinner("Executing feature extraction, model inference, and decision rule reasoning..."):
            result = predict_single_review(
                review_text=review_input,
                model_name=selected_model,
                run_rules=enable_rules
            )

        if not result.get("success"):
            st.error(f"❌ Analysis Failed: {result.get('error', 'Unknown inference error')}")
        else:
            st.success("✅ Review Successfully Analyzed!")

            # Top KPI metrics
            k1, k2, k3, k4 = st.columns(4)
            sent = result["sentiment"]
            conf = result["confidence"]
            prob_pos = result["prob_pos"]
            prob_neg = result["prob_neg"]

            sent_color = THEME_COLORS["positive"] if sent == "Positive" else THEME_COLORS["negative"]
            render_kpi(k1, "Predicted Sentiment", f"<span style='color:{sent_color};'>{sent}</span>", f"Classified by {selected_model}")
            render_kpi(k2, "Prediction Confidence", f"{conf * 100:.1f}%", f"P(Positive)={prob_pos:.2f} | P(Negative)={prob_neg:.2f}")
            render_kpi(k3, "Classification Status", result["status"], "Uncertain flag if 0.45 <= P <= 0.55")
            render_kpi(k4, "Matched Rules", f"{len(result['rules_triggered'])} Fired", "Business action triggers")

            if result["uncertain"]:
                st.warning("⚠️ **Uncertainty Flag**: Model probability is close to decision boundary (45%-55%). Human verification recommended.")

            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

            res_col1, res_col2 = st.columns([6, 6])

            with res_col1:
                render_section_header("⚖️", "Business Recommendations")
                if result["recommendations"]:
                    for rec in result["recommendations"]:
                        st.markdown(
                            f"""
                            <div class="rec-card">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem;">
                                    <span style="font-weight: 700; color: {THEME_COLORS['primary']};">{rec['title']}</span>
                                    <span class="badge badge-warning">{rec['priority']} Priority</span>
                                </div>
                                <p style="font-weight: 600; color: {THEME_COLORS['text']}; margin-bottom: 0.3rem;">💡 Action: {rec['recommendation']}</p>
                                <p style="font-size: 0.82rem; color: {THEME_COLORS['text_muted']}; margin-bottom: 0.2rem;">🏢 Target Team: <strong>{rec['target_team']}</strong></p>
                                <p style="font-size: 0.82rem; color: {THEME_COLORS['text_muted']}; margin-bottom: 0;">📋 Evidence: {rec['explanation']}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.info("ℹ️ No specialized escalation rules triggered. Feedback aligns with standard operational baseline.")

                # Explanatory Facts Table
                with st.expander("🔍 View Deduce Facts & Semantic Markers", expanded=False):
                    facts_df = pd.DataFrame(list(result["facts"].items()), columns=["Predicate / Marker", "Value"])
                    st.dataframe(facts_df, use_container_width=True, hide_index=True)

            with res_col2:
                render_section_header("🔎", "Feature Attribution (SHAP)")
                shap_res = ExplainabilityService.explain_with_shap(review_input, model_name=selected_model, top_k=8)
                if shap_res.get("success"):
                    top_feats = shap_res["top_features"]
                    f_names = [f["feature"] for f in top_feats]
                    f_attrs = [f["attribution"] for f in top_feats]
                    fig_attr = plot_feature_attribution_bar(f_names, f_attrs, title="Top Token Marginal Contributions (SHAP)")
                    st.plotly_chart(fig_attr, use_container_width=True)
                else:
                    st.caption("SHAP feature attribution available for linear and tree models.")

            # Save to Database
            if save_to_db:
                rec_summary = result["recommendations"][0]["recommendation"] if result["recommendations"] else "Standard Feedback"
                saved_id = Repository.save_analysis(
                    review_text=review_input,
                    sentiment=sent,
                    confidence=conf,
                    prob_pos=prob_pos,
                    prob_neg=prob_neg,
                    model_used=selected_model,
                    rules_triggered=result["rules_triggered"],
                    recommendation=rec_summary
                )
                st.caption(f"💾 Record successfully persisted to database history (Analysis ID #{saved_id}).")

            # Export options
            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
            render_section_header("📥", "Export Prediction Record")
            exp_c1, exp_c2 = st.columns(2)
            export_payload = {
                "timestamp": datetime.now().isoformat(),
                "review_text": review_input,
                "predicted_sentiment": sent,
                "confidence": conf,
                "prob_positive": prob_pos,
                "prob_negative": prob_neg,
                "model_used": selected_model,
                "rules_triggered": result["rules_triggered"],
                "recommendations": result["recommendations"]
            }
            with exp_c1:
                st.download_button(
                    label="Download Result as JSON",
                    data=json.dumps(export_payload, indent=2),
                    file_name=f"sentiment_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            with exp_c2:
                csv_df = pd.DataFrame([{
                    "review_text": review_input,
                    "sentiment": sent,
                    "confidence": conf,
                    "model": selected_model,
                    "rules": ", ".join(result["rules_triggered"]),
                    "recommendation": rec_summary
                }])
                st.download_button(
                    label="Download Result as CSV",
                    data=csv_df.to_csv(index=False),
                    file_name=f"sentiment_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
