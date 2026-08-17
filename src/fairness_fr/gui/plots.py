"""Chart builders for the FairFaceEval dashboard.

Every function takes an already-loaded, real DataFrame (from
:mod:`fairness_fr.gui.data_loader`) and returns a Plotly figure. No
function here reads a file, computes a metric, or fabricates a value —
if the caller passes real data, the chart shows real data.
"""

from __future__ import annotations

import math

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

_TEMPLATE = "plotly_white"


def bar_metric_by_model(comparison_table: pd.DataFrame, metric: str, label: str) -> go.Figure:
    """Bar chart of one metric across models.

    Args:
        comparison_table: Output of ``model_comparison.csv``.
        metric: Column name to plot.
        label: Y-axis / title label.

    Returns:
        A Plotly bar figure.
    """
    figure = px.bar(
        comparison_table,
        x="model",
        y=metric,
        text_auto=".3f",
        template=_TEMPLATE,
        title=f"{label} by Model",
    )
    figure.update_layout(xaxis_title="Model", yaxis_title=label, showlegend=False)
    return figure


def bar_metric_by_group(group_table: pd.DataFrame, group_col: str, metric: str, label: str) -> go.Figure:
    """Bar chart of one metric across demographic groups.

    Args:
        group_table: Group-level DataFrame (e.g. fairness_metrics.csv,
            already filtered to one attribute, or an intersectional table).
        group_col: Column holding the group label.
        metric: Column name to plot.
        label: Y-axis / title label.

    Returns:
        A Plotly bar figure.
    """
    figure = px.bar(
        group_table,
        x=group_col,
        y=metric,
        text_auto=".3f",
        template=_TEMPLATE,
        title=f"{label} by {group_col.replace('_', ' ').title()}",
    )
    figure.update_layout(xaxis_title=group_col.replace("_", " ").title(), yaxis_title=label, showlegend=False)
    return figure


def score_distribution(scores: pd.DataFrame, score_column: str, label_column: str, threshold: float | None) -> go.Figure:
    """Overlaid genuine/impostor score histograms with an optional threshold line.

    Args:
        scores: A pair/score DataFrame with ``label_column`` (0/1) and
            ``score_column`` (numeric).
        score_column: Which score column to plot.
        label_column: Column holding the binary genuine/impostor label.
        threshold: Decision threshold to draw as a vertical line, or None.

    Returns:
        A Plotly figure with overlaid histograms.
    """
    plotting_frame = scores.copy()
    plotting_frame["Pair type"] = plotting_frame[label_column].map({1: "Genuine pairs", 0: "Impostor pairs"})

    figure = px.histogram(
        plotting_frame,
        x=score_column,
        color="Pair type",
        barmode="overlay",
        histnorm="probability density",
        opacity=0.6,
        template=_TEMPLATE,
        color_discrete_map={"Genuine pairs": "#2ca02c", "Impostor pairs": "#d62728"},
        title="Similarity Score Distribution",
    )
    if threshold is not None and not (isinstance(threshold, float) and math.isnan(threshold)):
        figure.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="black",
            annotation_text=f"Decision threshold ({threshold:.3f})",
            annotation_position="top",
        )
    figure.update_layout(xaxis_title=score_column, yaxis_title="Density")
    return figure


def roc_curve_chart(roc_points: pd.DataFrame, model_col: str | None = None) -> go.Figure:
    """ROC curve (FAR vs TAR), optionally split by model.

    Args:
        roc_points: DataFrame with ``fpr``/``tpr`` columns (from
            ``roc_points.csv``), optionally with a model/split column.
        model_col: Column to color/split lines by, or None for a single curve.

    Returns:
        A Plotly line figure.
    """
    figure = px.line(
        roc_points,
        x="fpr",
        y="tpr",
        color=model_col,
        template=_TEMPLATE,
        title="ROC Curve",
    )
    figure.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dot", color="gray"))
    figure.update_layout(xaxis_title="False Accept Rate (FAR)", yaxis_title="True Accept Rate (TAR)")
    return figure


def threshold_sweep_chart(threshold_analysis: pd.DataFrame, metric: str, label: str) -> go.Figure:
    """Line chart of one metric across the threshold sweep.

    Args:
        threshold_analysis: DataFrame from ``threshold_analysis.csv``
            (columns: threshold, far, frr, tar, tnr).
        metric: Which column to plot against threshold.
        label: Y-axis / title label.

    Returns:
        A Plotly line figure.
    """
    figure = px.line(
        threshold_analysis.sort_values("threshold"),
        x="threshold",
        y=metric,
        template=_TEMPLATE,
        title=f"{label} vs Threshold",
    )
    figure.update_layout(xaxis_title="Decision threshold", yaxis_title=label)
    return figure


def radar_chart(comparison_table: pd.DataFrame, metrics: list[str]) -> go.Figure:
    """Radar chart comparing models across several higher-is-better metrics.

    Args:
        comparison_table: Output of ``model_comparison.csv``.
        metrics: Metric column names to use as radar axes (all assumed
            "higher is better").

    Returns:
        A Plotly polar figure.
    """
    figure = go.Figure()
    for _, row in comparison_table.iterrows():
        values = [float(row.get(metric, 0.0)) if pd.notna(row.get(metric)) else 0.0 for metric in metrics]
        figure.add_trace(
            go.Scatterpolar(r=values + values[:1], theta=metrics + metrics[:1], fill="toself", name=str(row["model"]))
        )
    figure.update_layout(template=_TEMPLATE, title="Model Comparison Radar Chart", showlegend=True)
    return figure


def heatmap_chart(comparison_table: pd.DataFrame, metrics: list[str]) -> go.Figure:
    """Heatmap of every metric across every model.

    Args:
        comparison_table: Output of ``model_comparison.csv``.
        metrics: Metric column names to include as heatmap columns.

    Returns:
        A Plotly heatmap figure.
    """
    matrix = comparison_table[metrics].to_numpy()
    figure = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=metrics,
            y=comparison_table["model"].tolist(),
            colorscale="Viridis",
            text=[[f"{value:.3f}" for value in row] for row in matrix],
            texttemplate="%{text}",
        )
    )
    figure.update_layout(template=_TEMPLATE, title="Model Metric Heatmap")
    return figure


def ranking_chart(rankings: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of the composite overall model ranking.

    Args:
        rankings: DataFrame from ``model_rankings.csv``.

    Returns:
        A Plotly horizontal bar figure, best model at the top.
    """
    ordered = rankings.sort_values("overall_rank", ascending=True)
    figure = px.bar(
        ordered,
        x="overall_rank_score",
        y="model",
        orientation="h",
        template=_TEMPLATE,
        title="Overall Model Ranking (lower score = better)",
    )
    figure.update_yaxes(autorange="reversed")
    return figure


def disparity_bar(disparity_table: pd.DataFrame) -> go.Figure:
    """Bar chart of disparity range per metric, from ``fairness_disparity.csv``.

    Args:
        disparity_table: DataFrame with ``metric`` and ``range`` columns.

    Returns:
        A Plotly bar figure.
    """
    figure = px.bar(
        disparity_table,
        x="metric",
        y="range",
        text_auto=".3f",
        template=_TEMPLATE,
        title="Demographic Disparity (max - min) by Metric",
    )
    figure.update_layout(xaxis_title="Metric", yaxis_title="Disparity (range)", showlegend=False)
    return figure


def hard_impostor_distribution(hard_impostors: pd.DataFrame, score_column: str) -> go.Figure:
    """Histogram of the highest-similarity impostor pair scores.

    Args:
        hard_impostors: Already-filtered impostor-pair DataFrame.
        score_column: Score column to plot.

    Returns:
        A Plotly histogram figure.
    """
    figure = px.histogram(
        hard_impostors,
        x=score_column,
        template=_TEMPLATE,
        color_discrete_sequence=["#d62728"],
        title="Hard Impostor Score Distribution",
    )
    figure.update_layout(xaxis_title=score_column, yaxis_title="Count")
    return figure
