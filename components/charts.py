"""Interactive Plotly charts styled for the premium dark analytics design system."""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from config import THEME_COLORS

# Chart color palette for dark theme
CHART_PALETTE = [
    "#818CF8",  # Indigo 400
    "#38BDF8",  # Sky 400
    "#34D399",  # Emerald 400
    "#FBBF24",  # Amber 400
    "#FB7185",  # Rose 400
    "#A78BFA",  # Violet 400
    "#2DD4BF",  # Teal 400
    "#F472B6",  # Pink 400
    "#60A5FA",  # Blue 400
    "#4ADE80",  # Green 400
]

# Default layout styling for all Plotly charts (dark theme)
PLOT_LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(
        family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
        color=THEME_COLORS["text"],
        size=12
    ),
    margin=dict(l=40, r=40, t=55, b=40),
    hoverlabel=dict(
        bgcolor=THEME_COLORS["surface"],
        font_size=12,
        font_family="Inter, sans-serif",
        font_color=THEME_COLORS["text"],
        bordercolor=THEME_COLORS["border"]
    )
)

def plot_sentiment_donut(pos_count: int, neg_count: int, title: str = "Sentiment Distribution"):
    """Renders a modern donut chart for positive and negative sentiment split."""
    total = pos_count + neg_count
    if total == 0:
        labels = ["No Data Available"]
        values = [1]
        colors = [THEME_COLORS["border"]]
    else:
        labels = ["Positive", "Negative"]
        values = [pos_count, neg_count]
        colors = [THEME_COLORS["positive"], THEME_COLORS["negative"]]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.65,
        marker=dict(
            colors=colors,
            line=dict(color=THEME_COLORS["background"], width=3)
        ),
        textinfo="percent+label" if total > 0 else "label",
        textfont=dict(size=13, color=THEME_COLORS["text"]),
        hoverinfo="label+value+percent" if total > 0 else "none",
        pull=[0.02, 0.02] if total > 0 else [0]
    )])
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=15, color=THEME_COLORS["primary"], weight=700)
        ),
        showlegend=True,
        legend=dict(
            font=dict(color=THEME_COLORS["text_muted"], size=11),
            bgcolor="rgba(0,0,0,0)"
        ),
        annotations=[dict(
            text=f"<b>{total:,}</b><br>Total",
            x=0.5, y=0.5,
            font_size=16, font_color=THEME_COLORS["text_muted"],
            showarrow=False
        )] if total > 0 else [],
        **PLOT_LAYOUT_DEFAULTS
    )
    return fig

def plot_theme_breakdown_bar(theme_df: pd.DataFrame, title: str = "Customer Complaint Mentions by Aspect"):
    """Horizontal stacked bar chart showing theme feedback volumes."""
    if theme_df.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=theme_df["Theme"],
        x=theme_df["Negative Count"],
        name="Negative",
        orientation='h',
        marker=dict(
            color=THEME_COLORS["negative"],
            line=dict(color="rgba(0,0,0,0)", width=0)
        ),
        opacity=0.85
    ))
    fig.add_trace(go.Bar(
        y=theme_df["Theme"],
        x=theme_df["Positive/Neutral Count"],
        name="Positive/Neutral",
        orientation='h',
        marker=dict(
            color=THEME_COLORS["positive"],
            line=dict(color="rgba(0,0,0,0)", width=0)
        ),
        opacity=0.85
    ))
    fig.update_layout(
        barmode='stack',
        title=dict(text=title, font=dict(size=15, color=THEME_COLORS["primary"], weight=700)),
        xaxis=dict(title="Review Count", gridcolor=THEME_COLORS["border"], gridwidth=0.5, zeroline=False),
        yaxis=dict(title="", autorange="reversed"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(color=THEME_COLORS["text_muted"], size=11),
            bgcolor="rgba(0,0,0,0)"
        ),
        **PLOT_LAYOUT_DEFAULTS
    )
    return fig

def plot_model_comparison_bar(comparison_df: pd.DataFrame):
    """Bar chart comparing Accuracy, Precision, Recall, and F1 across evaluated models."""
    if comparison_df.empty:
        return go.Figure()

    metrics = [m for m in ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"] if m in comparison_df.columns]
    fig = go.Figure()

    for i, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            x=comparison_df["Model"],
            y=comparison_df[metric],
            name=metric,
            marker=dict(
                color=CHART_PALETTE[i % len(CHART_PALETTE)],
                line=dict(color="rgba(0,0,0,0)", width=0)
            ),
            opacity=0.9
        ))

    fig.update_layout(
        barmode='group',
        title=dict(text="Model Benchmark Performance Comparison", font=dict(size=15, color=THEME_COLORS["primary"], weight=700)),
        yaxis=dict(title="Score (0.0 to 1.0)", range=[0.5, 1.0], gridcolor=THEME_COLORS["border"], gridwidth=0.5, zeroline=False),
        xaxis=dict(title="", gridcolor=THEME_COLORS["border"]),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(color=THEME_COLORS["text_muted"], size=11),
            bgcolor="rgba(0,0,0,0)"
        ),
        **PLOT_LAYOUT_DEFAULTS
    )
    return fig

def plot_confusion_matrix_heatmap(cm: np.ndarray, labels: List[str] = None):
    """Confusion matrix heatmap styled for dark theme."""
    labels = labels or ["Negative (0)", "Positive (1)"]
    z = cm.tolist()
    annotations = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            annotations.append(dict(
                x=labels[j],
                y=labels[i],
                text=str(z[i][j]),
                showarrow=False,
                font=dict(
                    color="white" if z[i][j] > (np.max(cm) / 2) else THEME_COLORS["text"],
                    size=15, weight="bold"
                )
            ))

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        colorscale=[
            [0, THEME_COLORS["surface"]],
            [0.5, THEME_COLORS["primary"]],
            [1, "#4F46E5"]
        ],
        showscale=True
    ))
    fig.update_layout(
        title=dict(text="Confusion Matrix", font=dict(size=15, color=THEME_COLORS["primary"], weight=700)),
        xaxis=dict(title="Predicted Class"),
        yaxis=dict(title="Actual Class", autorange="reversed"),
        annotations=annotations,
        **PLOT_LAYOUT_DEFAULTS
    )
    return fig

def plot_feature_attribution_bar(features: List[str], attributions: List[float], title: str = "Feature Attributions"):
    """Horizontal divergence bar chart for SHAP or LIME attributions."""
    colors = [THEME_COLORS["positive"] if a > 0 else THEME_COLORS["negative"] for a in attributions]
    fig = go.Figure(go.Bar(
        x=attributions,
        y=features,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color="rgba(0,0,0,0)", width=0)
        ),
        opacity=0.9
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=THEME_COLORS["primary"], weight=700)),
        xaxis=dict(title="Attribution Weight", gridcolor=THEME_COLORS["border"], gridwidth=0.5, zeroline=False),
        yaxis=dict(autorange="reversed"),
        **PLOT_LAYOUT_DEFAULTS
    )
    return fig

def plot_training_history_curves(history_df: pd.DataFrame):
    """Plots dual training & validation Loss and Accuracy curves for the Deep Neural Network."""
    if history_df.empty:
        return go.Figure()

    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("📉 Loss Trajectory (Cross-Entropy)", "📈 Accuracy Trajectory"),
        horizontal_spacing=0.12
    )

    epochs = history_df.index + 1 if "epoch" not in history_df.columns else history_df["epoch"]

    # Left: Loss
    if "loss" in history_df.columns:
        fig.add_trace(go.Scatter(
            x=epochs, y=history_df["loss"],
            mode='lines+markers', name="Train Loss",
            line=dict(color="#F43F5E", width=2.5),
            marker=dict(size=6, color="#F43F5E")
        ), row=1, col=1)
    if "val_loss" in history_df.columns:
        fig.add_trace(go.Scatter(
            x=epochs, y=history_df["val_loss"],
            mode='lines+markers', name="Val Loss",
            line=dict(color="#F59E0B", dash="dash", width=2.5),
            marker=dict(size=6, color="#F59E0B")
        ), row=1, col=1)

    # Right: Accuracy
    if "accuracy" in history_df.columns:
        fig.add_trace(go.Scatter(
            x=epochs, y=history_df["accuracy"],
            mode='lines+markers', name="Train Accuracy",
            line=dict(color="#10B981", width=2.5),
            marker=dict(size=6, color="#10B981")
        ), row=1, col=2)
    if "val_accuracy" in history_df.columns:
        fig.add_trace(go.Scatter(
            x=epochs, y=history_df["val_accuracy"],
            mode='lines+markers', name="Val Accuracy",
            line=dict(color="#38BDF8", dash="dash", width=2.5),
            marker=dict(size=6, color="#38BDF8")
        ), row=1, col=2)

    fig.update_xaxes(title_text="Epoch", gridcolor="rgba(255, 255, 255, 0.08)", gridwidth=0.5, dtick=1)
    fig.update_yaxes(title_text="Loss Value", gridcolor="rgba(255, 255, 255, 0.08)", gridwidth=0.5, row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", gridcolor="rgba(255, 255, 255, 0.08)", gridwidth=0.5, tickformat=".1%", row=1, col=2)

    layout_opts = dict(PLOT_LAYOUT_DEFAULTS)
    layout_opts["margin"] = dict(l=40, r=40, t=65, b=40)
    fig.update_layout(
        title=dict(text="Deep Neural Network Training Telemetry", font=dict(size=15, color="#F8FAFC", weight=700)),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5,
            font=dict(color="#94A3B8", size=11),
            bgcolor="rgba(0,0,0,0)"
        ),
        **layout_opts
    )
    return fig

def plot_priority_distribution(issues: list):
    """Creates a priority score distribution histogram for the Decision Center."""
    if not issues:
        return go.Figure()

    scores = [i.get("priority_score", 0) for i in issues]
    fig = go.Figure(go.Histogram(
        x=scores,
        nbinsx=10,
        marker=dict(
            color=THEME_COLORS["primary"],
            line=dict(color=THEME_COLORS["background"], width=1)
        ),
        opacity=0.85
    ))
    fig.update_layout(
        title=dict(text="Issue Priority Score Distribution", font=dict(size=15, color=THEME_COLORS["primary"], weight=700)),
        xaxis=dict(title="Priority Score", gridcolor=THEME_COLORS["border"], gridwidth=0.5),
        yaxis=dict(title="Count", gridcolor=THEME_COLORS["border"], gridwidth=0.5),
        **PLOT_LAYOUT_DEFAULTS
    )
    return fig
