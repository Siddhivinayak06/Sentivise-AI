"""UI components package for the AI Review Sentiment Analysis platform.

Provides reusable UI elements, premium chart components, and sidebar rendering.
"""
from components.ui_elements import (
    inject_custom_css,
    render_header,
    render_kpi,
    render_empty_state,
    render_section_header,
    render_feature_card,
)
from components.charts import (
    plot_sentiment_donut,
    plot_theme_breakdown_bar,
    plot_model_comparison_bar,
    plot_confusion_matrix_heatmap,
    plot_feature_attribution_bar,
    plot_training_history_curves,
    plot_priority_distribution,
)
from components.sidebar import render_sidebar
