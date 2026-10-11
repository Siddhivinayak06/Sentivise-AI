"""Page 4: Customer Insights uncovering recurring complaint themes, aspect distributions, and term frequencies."""
import streamlit as st
import pandas as pd
from database.repository import Repository
from services.insights_service import analyze_term_frequencies, categorize_complaint_themes, detect_repeated_issues
from components.ui_elements import (
    inject_custom_css,
    render_header,
    render_kpi,
    render_empty_state,
    render_section_header,
)
from components.charts import plot_theme_breakdown_bar
from components.sidebar import render_sidebar
from config import ASPECT_TAXONOMY, THEME_COLORS

st.set_page_config(page_title="Customer Insights", page_icon="💡", layout="wide")
inject_custom_css()
render_sidebar()

render_header(
    title="Customer Feedback Insights & Issue Clustering",
    subtitle="Identify recurring complaints across Quality, Delivery, Usability, Pricing, and Customer Support",
    badge="Diagnostic Intelligence"
)

st.markdown(
    f"""
    <div style="background: rgba(99, 102, 241, 0.08); border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 12px; padding: 0.9rem 1.2rem; margin-bottom: 1.5rem; display: flex; align-items: center; gap: 0.8rem;">
        <span style="font-size: 1.3rem;">ℹ️</span>
        <span style="font-size: 0.88rem; color: {THEME_COLORS['text_muted']};">
            <strong style="color: {THEME_COLORS['text']};">Methodology Transparency:</strong>
            Aspect categorization employs transparent <strong>heuristic lexical pattern matching</strong> mapped directly to e-commerce customer experience taxonomies.
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

# Retrieve stored batch reviews from SQLite
batch_reviews = Repository.get_batch_reviews(limit=2000)

if not batch_reviews:
    render_empty_state(
        "No Review Data Available in Database",
        "Upload a review CSV on the 'Batch Review Analytics' page or load the test sample to populate insights."
    )
else:
    reviews_df = pd.DataFrame(batch_reviews)
    text_list = reviews_df["review_text"].dropna().tolist()

    # 1. High-Level Insight KPIs
    theme_res = categorize_complaint_themes(reviews_df, text_col="review_text", sentiment_col="sentiment")
    theme_counts = theme_res["theme_counts"]
    top_theme = max(theme_counts, key=theme_counts.get) if any(theme_counts.values()) else "N/A"

    c1, c2, c3, c4 = st.columns(4)
    render_kpi(c1, "Total Corpus Scanned", f"{len(reviews_df):,} Reviews", "Active SQLite records")
    render_kpi(c2, "Top Complaint Category", top_theme, f"{theme_counts.get(top_theme, 0)} total mentions")
    render_kpi(c3, "Quality Mentions", f"{theme_counts.get('Product Quality', 0)}", "Defects, durability, build")
    render_kpi(c4, "Logistics Mentions", f"{theme_counts.get('Delivery & Logistics', 0)}", "Transit delays, carrier shipping")

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # 2. Aspect Theme Distribution Chart
    render_section_header("📊", "Aspect Category Mentions & Sentiment Breakdown")
    fig_theme = plot_theme_breakdown_bar(theme_res["theme_breakdown_df"], title="Feedback Volume Across Core Aspect Categories")
    st.plotly_chart(fig_theme, use_container_width=True)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # 3. Frequent Polar Lexical Terms (Positive vs Negative)
    render_section_header("🔤", "Most Frequent Polar Lexical Indicators")
    term_data = analyze_term_frequencies(text_list, top_n=12)

    col_pos, col_neg = st.columns(2)
    with col_pos:
        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(52, 211, 153, 0.2);
                border-radius: 12px; padding: 1rem; margin-bottom: 0.8rem;">
                <div style="font-weight: 700; color: {THEME_COLORS['positive']}; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;">
                    <span>🟢</span> Top Favorable Keywords
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        pos_df = pd.DataFrame(term_data["top_positive_terms"], columns=["Word", "Corpus Frequency"])
        if not pos_df.empty:
            st.dataframe(pos_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No specific positive keywords detected in current corpus.")

    with col_neg:
        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(251, 113, 133, 0.2);
                border-radius: 12px; padding: 1rem; margin-bottom: 0.8rem;">
                <div style="font-weight: 700; color: {THEME_COLORS['negative']}; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;">
                    <span>🔴</span> Top Critical Complaint Keywords
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        neg_df = pd.DataFrame(term_data["top_negative_terms"], columns=["Word", "Corpus Frequency"])
        if not neg_df.empty:
            st.dataframe(neg_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No specific negative keywords detected in current corpus.")

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # 4. Repeated Complaint Clusters (2-Gram Analysis)
    render_section_header("🔁", "Repeated Complaint Phrases (N-Gram Patterns)")
    repeated_patterns = detect_repeated_issues(text_list, min_count=2)
    if repeated_patterns:
        rep_df = pd.DataFrame(repeated_patterns)
        rep_df.columns = ["Complaint Phrase", "Occurrences", "Aspect Category"]
        st.dataframe(rep_df.head(10), use_container_width=True, hide_index=True)
    else:
        st.caption("No repeated n-gram complaint clusters found meeting threshold.")

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # 5. Representative Evidence Reviews by Theme
    render_section_header("📝", "Supporting Evidence Reviews by Category")
    sel_theme_tab = st.selectbox("Select Theme to Inspect Sample Evidence", list(ASPECT_TAXONOMY.keys()))
    sample_recs = theme_res["theme_reviews"].get(sel_theme_tab, [])

    if sample_recs:
        for idx, s in enumerate(sample_recs):
            is_pos = s["sentiment"] == "Positive"
            badge_class = "badge-positive" if is_pos else "badge-negative"
            border_accent = THEME_COLORS["positive"] if is_pos else THEME_COLORS["negative"]
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(51, 65, 85, 0.4));
                    border: 1px solid {THEME_COLORS['border']}; border-left: 4px solid {border_accent};
                    border-radius: 0 12px 12px 0; padding: 1.1rem; margin-bottom: 0.75rem; transition: all 0.2s ease;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                        <span style="font-weight: 700; color: {THEME_COLORS['secondary']}; font-size: 0.92rem;">
                            Evidence #{idx + 1} &bull; Product: {s['product']}
                        </span>
                        <span class="badge {badge_class}">{s['sentiment']}</span>
                    </div>
                    <p style="color: {THEME_COLORS['text']}; margin-bottom: 0; font-size: 0.95rem; line-height: 1.5; font-style: italic;">
                        &ldquo;{s['text']}&rdquo;
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info(f"No specific evidence samples found for '{sel_theme_tab}'.")
