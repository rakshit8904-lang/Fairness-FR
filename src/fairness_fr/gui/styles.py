"""Visual styling for the FairFaceEval dashboard.

Contains only CSS and small presentation constants — no data access or
evaluation logic. Injected once per page via :func:`inject_global_css`.
"""

from __future__ import annotations

PRIMARY_COLOR = "#1f4e8c"
ACCENT_COLOR = "#2ca02c"
WARNING_COLOR = "#d97706"
DANGER_COLOR = "#d62728"
MUTED_COLOR = "#6b7280"

_GLOBAL_CSS = """
<style>
.ffe-title {
    font-size: 2.1rem;
    font-weight: 700;
    color: #14213d;
    margin-bottom: 0;
}
.ffe-subtitle {
    font-size: 1.05rem;
    color: #6b7280;
    margin-top: 0;
    margin-bottom: 1.2rem;
}
.ffe-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
.ffe-card-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #6b7280;
    margin-bottom: 0.15rem;
}
.ffe-card-value {
    font-size: 1.55rem;
    font-weight: 700;
    color: #14213d;
}
.ffe-card-sub {
    font-size: 0.78rem;
    color: #9ca3af;
}
.ffe-status-ok { color: #2ca02c; font-weight: 600; }
.ffe-status-warn { color: #d97706; font-weight: 600; }
.ffe-status-missing { color: #d62728; font-weight: 600; }
.ffe-missing-box {
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    color: #7f1d1d;
    font-size: 0.9rem;
}
.ffe-interpretation-box {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    color: #1e3a8a;
    font-size: 0.95rem;
}
.ffe-pipeline-step {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.5rem 0.9rem;
    text-align: center;
    font-size: 0.85rem;
    color: #334155;
}
.ffe-adapter-note {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 8px;
    padding: 0.7rem 1rem;
    color: #78350f;
    font-size: 0.85rem;
}
</style>
"""


def inject_global_css() -> None:
    """Inject the dashboard's global CSS into the current Streamlit page."""
    import streamlit as st

    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)
