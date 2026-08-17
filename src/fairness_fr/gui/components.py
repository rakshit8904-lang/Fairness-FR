"""Reusable Streamlit UI components for the FairFaceEval dashboard.

Purely presentational: every function here renders something it is
given. None of these functions load files, compute metrics, or decide
what data is "correct" — that all happens in
:mod:`fairness_fr.gui.data_loader`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from fairness_fr.gui.data_loader import LoadResult


def page_header(title: str, subtitle: str) -> None:
    """Render the dashboard's title/subtitle header block."""
    st.markdown(f'<div class="ffe-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ffe-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def metric_card(label: str, value: str, sub: str = "") -> None:
    """Render a single metric card.

    Args:
        label: Short uppercase label.
        value: The headline value, already formatted as a string.
        sub: Optional small caption below the value.
    """
    sub_html = f'<div class="ffe-card-sub">{sub}</div>' if sub else ""
    st.markdown(
        f"""
        <div class="ffe-card">
            <div class="ffe-card-label">{label}</div>
            <div class="ffe-card-value">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_row(cards: list[tuple[str, str, str]]) -> None:
    """Render a horizontal row of metric cards.

    Args:
        cards: List of ``(label, value, sub)`` tuples.
    """
    columns = st.columns(len(cards))
    for column, (label, value, sub) in zip(columns, cards):
        with column:
            metric_card(label, value, sub)


def missing_file_notice(result: LoadResult, what: str) -> None:
    """Render a clear "this exact file is missing" notice.

    Args:
        result: The failed :class:`LoadResult`.
        what: Human-readable description of what was being loaded.
    """
    st.markdown(
        f"""
        <div class="ffe-missing-box">
            <strong>{what} not available.</strong><br/>
            Expected at: <code>{result.path}</code><br/>
            {result.reason}
        </div>
        """,
        unsafe_allow_html=True,
    )


def adapter_notice(text: str) -> None:
    """Render a note explaining that a value was computed by a GUI-side adapter.

    Args:
        text: Explanation of what was computed and from which existing
            pipeline functions, so the person never mistakes an adapter
            computation for a pipeline-produced artifact.
    """
    st.markdown(f'<div class="ffe-adapter-note">ℹ️ {text}</div>', unsafe_allow_html=True)


def interpretation_box(text: str) -> None:
    """Render a boxed interpretive statement, strictly derived from computed results."""
    st.markdown(f'<div class="ffe-interpretation-box">{text}</div>', unsafe_allow_html=True)


def status_indicator(dataset_available: bool, models_evaluated: int, models_configured: int) -> None:
    """Render the top-of-page pipeline status indicator.

    Args:
        dataset_available: Whether the selected dataset's metadata.csv exists.
        models_evaluated: Number of configured models with results present.
        models_configured: Total number of models configured for this experiment.
    """
    if not dataset_available:
        st.markdown('<span class="ffe-status-missing">● Missing results</span> — dataset not preprocessed yet.', unsafe_allow_html=True)
    elif models_evaluated == 0:
        st.markdown('<span class="ffe-status-warn">● Evaluation incomplete</span> — no model has produced results yet.', unsafe_allow_html=True)
    elif models_evaluated < models_configured:
        st.markdown(
            f'<span class="ffe-status-warn">● Results loaded</span> — {models_evaluated}/{models_configured} configured models evaluated.',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<span class="ffe-status-ok">● Pipeline ready</span> — all configured models evaluated.', unsafe_allow_html=True)


def pipeline_diagram() -> None:
    """Render the fixed nine-stage pipeline diagram as a vertical flow."""
    steps = [
        "Dataset",
        "Face Detection / Preprocessing",
        "Face Embeddings",
        "Genuine + Impostor Pairs",
        "Similarity Scores",
        "Threshold Selection",
        "FMR / FNMR",
        "Group-wise Fairness Analysis",
    ]
    for index, step in enumerate(steps):
        st.markdown(f'<div class="ffe-pipeline-step">{step}</div>', unsafe_allow_html=True)
        if index < len(steps) - 1:
            st.markdown(
                '<div style="text-align:center; color:#94a3b8;">↓</div>', unsafe_allow_html=True
            )


def dataframe_or_missing(result: LoadResult, what: str, **st_dataframe_kwargs: Any) -> None:
    """Render a DataFrame if available, otherwise a missing-file notice.

    Args:
        result: The :class:`LoadResult` to render.
        what: Human-readable description for the missing-file notice.
        **st_dataframe_kwargs: Forwarded to ``st.dataframe``.
    """
    if result.available and isinstance(result.data, pd.DataFrame):
        st.dataframe(result.data, **st_dataframe_kwargs)
    else:
        missing_file_notice(result, what)


def download_buttons(dataset_name: str, model_name: str | None, exports: dict[str, tuple[str, bytes]]) -> None:
    """Render one download button per named export.

    Args:
        dataset_name: Dataset identifier, used to disambiguate widget keys.
        model_name: Model identifier, or None for a dataset-level export.
        exports: Mapping of button label to ``(filename, file_bytes)``.
    """
    columns = st.columns(len(exports)) if exports else []
    for column, (label, (filename, file_bytes)) in zip(columns, exports.items()):
        with column:
            st.download_button(
                label=label,
                data=file_bytes,
                file_name=filename,
                key=f"download-{dataset_name}-{model_name}-{filename}",
            )
