"""FairFaceEval — Streamlit dashboard for the fairness_fr evaluation pipeline.

This module is a thin presentation layer. It never computes a fairness
metric, generates a pair, extracts an embedding, or invents a number —
every figure and table here is either read directly from a file the
existing pipeline (:mod:`fairness_fr.data`, :mod:`fairness_fr.models`,
:mod:`fairness_fr.evaluation`) already produces, or computed by the
small, clearly-labeled adapter in
:func:`fairness_fr.gui.data_loader.compute_intersectional_metrics`
that itself reuses the pipeline's own formulas.

Launch with::

    streamlit run src/fairness_fr/gui/app.py

or::

    python -m fairness_fr.gui
"""

from __future__ import annotations

import json
import math

import pandas as pd
import streamlit as st

from fairness_fr.gui import components as ui
from fairness_fr.gui import data_loader as dl
from fairness_fr.gui import plots
from fairness_fr.gui.styles import inject_global_css

_PAGES = (
    "Overview",
    "Model Comparison",
    "Dataset",
    "Score Distribution",
    "Fairness / Group Analysis",
    "Intersectional Fairness",
    "Threshold Analysis",
    "Hard Impostor Analysis",
    "Results / Report",
)


# ------------------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------------------


def render_sidebar() -> dict:
    """Render the shared sidebar filters and page navigation.

    Returns:
        A dict of current selections: ``page``, ``dataset``, ``model``,
        ``models`` (multi-select for comparison), ``score_column``,
        ``group_attribute``.
    """
    st.sidebar.markdown("## FairFaceEval")
    page = st.sidebar.radio("Navigate", _PAGES, key="nav_page")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Filters")

    datasets = dl.discover_datasets()
    if not datasets:
        st.sidebar.error("No dataset configs found under configs/datasets/.")
        dataset = None
    else:
        dataset = st.sidebar.selectbox("Dataset", datasets, key="sel_dataset")

    evaluated_models = dl.evaluated_models_for_dataset(dataset) if dataset else []
    configured_models = dl.discover_models()
    model_choices = evaluated_models or configured_models

    if not model_choices:
        st.sidebar.warning("No model configs found under configs/models/.")
        model = None
        models = []
    else:
        model = st.sidebar.selectbox("Model", model_choices, key="sel_model")
        models = st.sidebar.multiselect(
            "Models to compare", model_choices, default=model_choices, key="sel_models"
        )

    score_column = st.sidebar.selectbox(
        "Threshold strategy / score column",
        ["cosine_similarity", "euclidean_distance", "cosine_distance"],
        key="sel_score_column",
        help="Which score column decisions are thresholded on for this session's charts.",
    )

    group_attribute = None
    if dataset:
        metadata_result = dl.load_metadata(dataset)
        if metadata_result.available:
            candidate_columns = [
                column for column in metadata_result.data.columns if column not in ("image_path", "identity")
            ]
            if candidate_columns:
                group_attribute = st.sidebar.selectbox("Group attribute", candidate_columns, key="sel_group_attribute")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Run Pipeline")
    _render_run_evaluation_control(dataset)

    return {
        "page": page,
        "dataset": dataset,
        "model": model,
        "models": models,
        "score_column": score_column,
        "group_attribute": group_attribute,
    }


def _render_run_evaluation_control(dataset: str | None) -> None:
    """Render the 'Run Evaluation' control that shells out to run_pipeline.py."""
    stage_options = ["preprocess", "pairs", "embeddings", "scores", "performance", "fairness", "compare"]
    selected_stages = st.sidebar.multiselect("Stages to run", stage_options, key="sel_run_stages")
    run_all = st.sidebar.checkbox("Run all stages (--all)", key="sel_run_all")

    if st.sidebar.button("▶ Run Evaluation", key="btn_run_evaluation", disabled=dataset is None):
        flags = ["--all"] if run_all else [f"--{stage}" for stage in selected_stages]
        if not flags:
            st.sidebar.warning("Select at least one stage, or check 'Run all stages'.")
        else:
            _execute_pipeline_run(flags)


def _execute_pipeline_run(flags: list[str]) -> None:
    """Execute run_pipeline.py with the given flags, streaming real progress/logs."""
    status_placeholder = st.sidebar.empty()
    log_expander = st.sidebar.expander("Pipeline logs", expanded=True)
    log_lines: list[str] = []
    log_box = log_expander.empty()

    status_placeholder.info(f"Running: python run_pipeline.py {' '.join(flags)}")
    exit_code: int | None = None

    for line in dl.run_pipeline_stages(flags):
        if line.startswith("__EXIT_CODE__:"):
            exit_code = int(line.split(":", 1)[1])
            break
        log_lines.append(line)
        log_box.code("\n".join(log_lines[-200:]), language="text")

    dl.clear_all_caches()

    if exit_code == 0:
        status_placeholder.success("Pipeline run completed successfully.")
    else:
        status_placeholder.error(f"Pipeline run finished with errors (exit code {exit_code}). See logs above.")


# ------------------------------------------------------------------------------
# Page 1 — Overview
# ------------------------------------------------------------------------------


def page_overview(selection: dict) -> None:
    """Render the Overview / Home page."""
    ui.page_header("FairFaceEval — Fairness Evaluation of Face Recognition", "Demographic Fairness Analysis of Face Recognition Systems")

    dataset = selection["dataset"]
    model = selection["model"]
    if dataset is None:
        st.warning("No datasets configured. Add a YAML file under configs/datasets/.")
        return

    metadata_result = dl.load_metadata(dataset)
    pairs_result = dl.load_pairs(dataset, split="test")
    evaluated_models = dl.evaluated_models_for_dataset(dataset)
    configured_models = dl.discover_models()

    ui.status_indicator(metadata_result.available, len(evaluated_models), len(configured_models))
    st.markdown("")

    identities = "—"
    images = "—"
    if metadata_result.available:
        frame = metadata_result.data
        identities = str(frame["identity"].nunique()) if "identity" in frame.columns else "—"
        images = str(len(frame))

    genuine_pairs = "—"
    impostor_pairs = "—"
    if pairs_result.available and "label" in pairs_result.data.columns:
        genuine_pairs = str(int((pairs_result.data["label"] == 1).sum()))
        impostor_pairs = str(int((pairs_result.data["label"] == 0).sum()))

    groups_evaluated = "—"
    if dataset:
        fairness_result = dl.load_fairness_metrics(dataset, model) if model else dl.load_fairness_metrics(dataset, "")
        if fairness_result.available:
            groups_evaluated = str(fairness_result.data["group_value"].nunique())

    ui.metric_row(
        [
            ("Dataset", dataset.upper(), ""),
            ("Identities", identities, "from metadata.csv"),
            ("Images", images, "from metadata.csv"),
            ("Genuine pairs", genuine_pairs, "test split"),
        ]
    )
    ui.metric_row(
        [
            ("Impostor pairs", impostor_pairs, "test split"),
            ("Models evaluated", f"{len(evaluated_models)}/{len(configured_models)}", "configured models"),
            ("Groups evaluated", groups_evaluated, f"model: {model or '—'}"),
            ("Score metric", selection["score_column"], "current session filter"),
        ]
    )

    st.markdown("### Pipeline")
    left, right = st.columns([1, 2])
    with left:
        ui.pipeline_diagram()

    with right:
        st.markdown("#### Global Performance")
        if model is None:
            st.info("No evaluated model selected.")
        else:
            performance_result = dl.load_performance_metrics(dataset, model)
            if not performance_result.available:
                ui.missing_file_notice(performance_result, "Global performance metrics")
            else:
                test_row = performance_result.data.loc[performance_result.data["split"] == "test"]
                if test_row.empty:
                    st.warning("performance_metrics.csv has no 'test' split row.")
                else:
                    row = test_row.iloc[0]
                    ui.metric_row(
                        [
                            ("Global FMR (FAR)", f"{row.get('far', float('nan')):.4f}", ""),
                            ("Global FNMR (FRR)", f"{row.get('frr', float('nan')):.4f}", ""),
                            ("EER", f"{row.get('eer', float('nan')):.4f}", ""),
                            ("Selected threshold", f"{row.get('eer_threshold', float('nan')):.4f}", "EER threshold"),
                        ]
                    )


# ------------------------------------------------------------------------------
# Page 2 — Model Comparison
# ------------------------------------------------------------------------------


def page_model_comparison(selection: dict) -> None:
    """Render the Model Comparison page."""
    ui.page_header("Model Comparison", "Side-by-side comparison of every evaluated model")
    dataset = selection["dataset"]
    if dataset is None:
        return

    comparison_result = dl.load_model_comparison(dataset)
    if not comparison_result.available:
        ui.missing_file_notice(comparison_result, "Model comparison table")
        st.info("Run the pipeline with --compare (requires 2+ evaluated models) to generate this.")
        return

    table = comparison_result.data.copy()
    embedding_dims = []
    for model_name in table["model"]:
        model_config = dl.load_model_config(model_name)
        embedding_dims.append(model_config.embedding_dim if model_config is not None else None)
    table.insert(1, "embedding_dim", embedding_dims)

    display_columns = [
        "model", "embedding_dim",
        "accuracy", "precision", "recall", "f1_score", "far", "frr", "fmr", "fnmr", "tar", "tnr", "eer", "mean_disparity",
    ]
    display_columns = [column for column in display_columns if column in table.columns]
    st.dataframe(table[display_columns], use_container_width=True)

    rankings_result = dl.load_model_rankings(dataset)
    if rankings_result.available:
        rankings = rankings_result.data
        best_row = rankings.sort_values("overall_rank").iloc[0]
        ui.interpretation_box(
            f"Best overall model (lowest composite rank): <strong>{best_row['model']}</strong> "
            f"— accuracy={best_row.get('accuracy', float('nan')):.4f}, "
            f"EER={best_row.get('eer', float('nan')):.4f}, "
            f"TAR={best_row.get('tar', float('nan')):.4f}."
        )
        st.plotly_chart(plots.ranking_chart(rankings), use_container_width=True)
    else:
        ui.missing_file_notice(rankings_result, "Model rankings")

    st.markdown("### Charts")
    tab1, tab2, tab3 = st.tabs(["FMR (FAR)", "FNMR (FRR)", "EER"])
    with tab1:
        st.plotly_chart(plots.bar_metric_by_model(table, "far", "FMR (FAR)"), use_container_width=True)
    with tab2:
        st.plotly_chart(plots.bar_metric_by_model(table, "frr", "FNMR (FRR)"), use_container_width=True)
    with tab3:
        st.plotly_chart(plots.bar_metric_by_model(table, "eer", "EER"), use_container_width=True)

    radar_metrics = [m for m in ("accuracy", "precision", "recall", "f1_score", "tar", "tnr") if m in table.columns]
    if len(radar_metrics) >= 3:
        st.plotly_chart(plots.radar_chart(table, radar_metrics), use_container_width=True)

    heatmap_metrics = [m for m in ("accuracy", "precision", "recall", "f1_score", "far", "frr", "tar", "tnr", "eer") if m in table.columns]
    if heatmap_metrics:
        st.plotly_chart(plots.heatmap_chart(table, heatmap_metrics), use_container_width=True)

    summary_result = dl.load_model_summary_text(dataset)
    if summary_result.available:
        with st.expander("Full automatic model comparison summary"):
            st.text(summary_result.data)


# ------------------------------------------------------------------------------
# Page 3 — Dataset
# ------------------------------------------------------------------------------


def page_dataset(selection: dict) -> None:
    """Render the Dataset page."""
    ui.page_header("Dataset", "Composition and demographic breakdown of the selected dataset")
    dataset = selection["dataset"]
    if dataset is None:
        return

    metadata_result = dl.load_metadata(dataset)
    if not metadata_result.available:
        ui.missing_file_notice(metadata_result, "Dataset metadata")
        st.info("Run the pipeline with --preprocess to generate metadata.csv.")
        return

    metadata = metadata_result.data
    identity_column = "identity" if "identity" in metadata.columns else None
    group_column = selection["group_attribute"] or ("group" if "group" in metadata.columns else None)

    ui.metric_row(
        [
            ("Total images", str(len(metadata)), ""),
            ("Identities", str(metadata[identity_column].nunique()) if identity_column else "—", ""),
            ("Demographic groups", str(metadata[group_column].nunique()) if group_column else "—", group_column or ""),
            ("Attributes available", str(len(metadata.columns) - 2), "columns beyond image_path/identity"),
        ]
    )

    if group_column:
        st.plotly_chart(
            plots.bar_metric_by_group(
                metadata.groupby(group_column).size().reset_index(name="images"), group_column, "images", "Images"
            ),
            use_container_width=True,
        )
        if identity_column:
            identities_per_group = metadata.groupby(group_column)[identity_column].nunique().reset_index(name="identities")
            st.plotly_chart(
                plots.bar_metric_by_group(identities_per_group, group_column, "identities", "Identities"),
                use_container_width=True,
            )

    st.markdown("### Pairs")
    pair_rows = []
    for split in ("train", "validation", "test"):
        pairs_result = dl.load_pairs(dataset, split=split)
        if pairs_result.available and "label" in pairs_result.data.columns:
            genuine = int((pairs_result.data["label"] == 1).sum())
            impostor = int((pairs_result.data["label"] == 0).sum())
            pair_rows.append({"split": split, "genuine_pairs": genuine, "impostor_pairs": impostor})
    if pair_rows:
        pair_table = pd.DataFrame(pair_rows)
        st.dataframe(pair_table, use_container_width=True)
        melted = pair_table.melt(id_vars="split", value_vars=["genuine_pairs", "impostor_pairs"], var_name="type", value_name="count")
        st.plotly_chart(
            plots.bar_metric_by_group(melted.rename(columns={"split": "group"}), "group", "count", "Pairs"),
            use_container_width=True,
        )
    else:
        st.info("No pair CSVs found yet. Run the pipeline with --pairs.")

    with st.expander("Raw metadata sample"):
        st.dataframe(metadata.head(200), use_container_width=True)


# ------------------------------------------------------------------------------
# Page 4 — Score Distribution
# ------------------------------------------------------------------------------


def page_score_distribution(selection: dict) -> None:
    """Render the Score Distribution page."""
    ui.page_header("Score Distribution", "Genuine vs impostor similarity scores")
    dataset, model = selection["dataset"], selection["model"]
    if dataset is None or model is None:
        st.info("Select a dataset and an evaluated model in the sidebar.")
        return

    scores_result = dl.load_scores(dataset, model, split="test")
    if not scores_result.available:
        ui.missing_file_notice(scores_result, "Test similarity scores")
        st.info("Run the pipeline with --scores to generate this.")
        return

    scores = scores_result.data
    score_column = selection["score_column"]
    if score_column not in scores.columns:
        st.error(f"Column '{score_column}' not present in test_scores.csv.")
        return

    performance_result = dl.load_performance_metrics(dataset, model)
    threshold = None
    if performance_result.available:
        test_row = performance_result.data.loc[performance_result.data["split"] == "test"]
        if not test_row.empty:
            threshold = float(test_row.iloc[0].get("eer_threshold", float("nan")))

    st.plotly_chart(plots.score_distribution(scores, score_column, "label", threshold), use_container_width=True)

    genuine = scores.loc[scores["label"] == 1, score_column]
    impostor = scores.loc[scores["label"] == 0, score_column]
    ui.metric_row(
        [
            ("Genuine mean", f"{genuine.mean():.4f}" if len(genuine) else "—", ""),
            ("Genuine median", f"{genuine.median():.4f}" if len(genuine) else "—", ""),
            ("Impostor mean", f"{impostor.mean():.4f}" if len(impostor) else "—", ""),
            ("Impostor median", f"{impostor.median():.4f}" if len(impostor) else "—", ""),
        ]
    )

    if len(genuine) and len(impostor):
        overlap_low = max(genuine.min(), impostor.min())
        overlap_high = min(genuine.max(), impostor.max())
        if overlap_high > overlap_low:
            st.caption(f"Score overlap region: [{overlap_low:.4f}, {overlap_high:.4f}]")
        else:
            st.caption("No overlap between genuine and impostor score ranges in this split.")


# ------------------------------------------------------------------------------
# Page 5 — Fairness / Group Analysis
# ------------------------------------------------------------------------------


def page_fairness(selection: dict) -> None:
    """Render the Fairness / Group Analysis page."""
    ui.page_header(
        "Fairness / Group Analysis",
        "Verification performance broken down by demographic group",
    )

    dataset, model = selection["dataset"], selection["model"]

    if dataset is None or model is None:
        st.info("Select a dataset and an evaluated model in the sidebar.")
        return

    # LFW does not have demographic attributes in this experiment.
    # Fairness analysis is therefore performed only for BFW.
    if dataset == "lfw":
        st.info(
            "Demographic fairness analysis is not applicable to LFW in this "
            "experiment because the LFW metadata does not contain demographic "
            "attributes. Fairness evaluation is provided using BFW."
        )
        return

    fairness_result = dl.load_fairness_metrics(dataset, model)

    if not fairness_result.available:
        ui.missing_file_notice(fairness_result, "Group-wise fairness metrics")
        st.info("Run the pipeline with --fairness to generate this.")
        return

    table = fairness_result.data
    attribute_options = sorted(table["attribute"].unique().tolist())
    chosen_attribute = st.selectbox(
        "Attribute",
        attribute_options,
        key="fairness_attribute",
    )
    attribute_table = table.loc[
        table["attribute"] == chosen_attribute
    ].copy()
    display_columns = [
        "group_value", "sample_size", "genuine_pairs", "impostor_pairs",
        "far", "frr", "fmr", "fnmr", "eer", "excluded", "exclusion_reason",
    ]
    display_columns = [column for column in display_columns if column in attribute_table.columns]
    st.dataframe(attribute_table[display_columns], use_container_width=True)

    included = attribute_table.loc[~attribute_table.get("excluded", False)]
    tab1, tab2, tab3 = st.tabs(["FMR (FAR)", "FNMR (FRR)", "EER"])
    with tab1:
        st.plotly_chart(plots.bar_metric_by_group(included, "group_value", "far", "FMR (FAR)"), use_container_width=True)
    with tab2:
        st.plotly_chart(plots.bar_metric_by_group(included, "group_value", "frr", "FNMR (FRR)"), use_container_width=True)
    with tab3:
        st.plotly_chart(plots.bar_metric_by_group(included, "group_value", "eer", "EER"), use_container_width=True)

    disparity_result = dl.load_fairness_disparity(dataset, model)
    if disparity_result.available:
        disparity_table = disparity_result.data.loc[disparity_result.data["attribute"] == chosen_attribute]
        if not disparity_table.empty:
            st.markdown("### Disparity")
            st.dataframe(disparity_table, use_container_width=True)
            st.plotly_chart(plots.disparity_bar(disparity_table), use_container_width=True)

            far_row = disparity_table.loc[disparity_table["metric"] == "far"]
            frr_row = disparity_table.loc[disparity_table["metric"] == "frr"]

            def _ratio(row: pd.DataFrame) -> str:
                if row.empty:
                    return "—"
                min_value = row.iloc[0]["min_value"]
                max_value = row.iloc[0]["max_value"]
                if pd.isna(min_value) or min_value == 0:
                    return "undefined (zero denominator)"
                return f"{max_value / min_value:.2f}x"

            ui.metric_row(
                [
                    ("FMR disparity", f"{far_row.iloc[0]['range']:.4f}" if not far_row.empty else "—", ""),
                    ("FMR ratio (max/min)", _ratio(far_row), ""),
                    ("FNMR disparity", f"{frr_row.iloc[0]['range']:.4f}" if not frr_row.empty else "—", ""),
                    ("FNMR ratio (max/min)", _ratio(frr_row), ""),
                ]
            )

            included_with_metrics = included.dropna(subset=["far", "eer"])
            if not included_with_metrics.empty:
                best = included_with_metrics.sort_values("eer").iloc[0]
                worst = included_with_metrics.sort_values("eer", ascending=False).iloc[0]
                max_disparity_row = disparity_table.dropna(subset=["range"]).sort_values("range", ascending=False)
                if not max_disparity_row.empty:
                    top = max_disparity_row.iloc[0]
                    detected = bool(top["range"] > 0.10)
                    verdict = "Demographic performance difference detected" if detected else "No substantial demographic difference detected under the selected evaluation setting"
                    ui.interpretation_box(
                        f"<strong>{verdict}.</strong><br/>"
                        f"Best-performing group (lowest EER): <strong>{best['group_value']}</strong> (EER={best['eer']:.4f}).<br/>"
                        f"Worst-performing group (highest EER): <strong>{worst['group_value']}</strong> (EER={worst['eer']:.4f}).<br/>"
                        f"Largest disparity: <strong>{top['metric'].upper()}</strong> (range={top['range']:.4f}, "
                        f"between '{top['max_group']}' and '{top['min_group']}')."
                    )
    else:
        ui.missing_file_notice(disparity_result, "Fairness disparity statistics")


# ------------------------------------------------------------------------------
# Page 6 — Intersectional Fairness
# ------------------------------------------------------------------------------


def page_intersectional(selection: dict) -> None:
    """Render the Intersectional Fairness page (GUI-side adapter, clearly labeled)."""
    ui.page_header("Intersectional Fairness", "Combined-attribute fairness breakdown (e.g. ethnicity x gender)")
    dataset, model = selection["dataset"], selection["model"]
    if dataset is None or model is None:
        st.info("Select a dataset and an evaluated model in the sidebar.")
        return

    metadata_result = dl.load_metadata(dataset)
    if not metadata_result.available:
        ui.missing_file_notice(metadata_result, "Dataset metadata")
        return

    candidate_columns = [c for c in metadata_result.data.columns if c not in ("image_path", "identity")]
    if len(candidate_columns) < 2:
        st.warning("metadata.csv has fewer than 2 demographic attribute columns; intersectional analysis needs at least 2.")
        return

    ui.adapter_notice(
        "The existing evaluation pipeline (evaluate_fairness.py) computes fairness metrics for one "
        "demographic attribute at a time and does not produce combined attributes. This page computes "
        "a real intersectional breakdown on demand, reusing the pipeline's own confusion-matrix and EER "
        "formulas (_confusion_counts_at_threshold, _safe_f1, _compute_eer_from_sweep) applied to a "
        "combined group label, evaluated at the same overall EER threshold from performance_metrics.csv. "
        "This is not a value pulled from a results file — it is computed live from real test_scores.csv "
        "and metadata.csv each time you load this page."
    )

    column1, column2 = st.columns(2)
    with column1:
        attribute_a = st.selectbox("Primary attribute", candidate_columns, index=0, key="intersect_a")
    with column2:
        remaining = [c for c in candidate_columns if c != attribute_a]
        attribute_b = st.selectbox("Intersect with", remaining, index=0, key="intersect_b")

    result = dl.compute_intersectional_metrics(dataset, model, attribute_a, attribute_b, selection["score_column"])
    if not result.available:
        ui.missing_file_notice(result, "Intersectional fairness table")
        return

    table = result.data
    st.dataframe(table, use_container_width=True)

    included = table.dropna(subset=["far", "eer"])
    tab1, tab2 = st.tabs(["FMR (FAR)", "FNMR (FRR)"])
    with tab1:
        st.plotly_chart(plots.bar_metric_by_group(included, "group", "far", "FMR (FAR)"), use_container_width=True)
    with tab2:
        st.plotly_chart(plots.bar_metric_by_group(included, "group", "frr", "FNMR (FRR)"), use_container_width=True)

    if not included.empty:
        worst = included.sort_values("eer", ascending=False).iloc[0]
        ui.interpretation_box(
            f"Worst-performing intersectional subgroup: <strong>{worst['group']}</strong> "
            f"(EER={worst['eer']:.4f}, n={int(worst['sample_size'])})."
        )


# ------------------------------------------------------------------------------
# Page 7 — Threshold Analysis
# ------------------------------------------------------------------------------


def page_threshold_analysis(selection: dict) -> None:
    """Render the Threshold Analysis page, using only cached scores."""
    ui.page_header("Threshold Analysis", "How decisions change across the decision threshold")
    dataset, model = selection["dataset"], selection["model"]
    if dataset is None or model is None:
        st.info("Select a dataset and an evaluated model in the sidebar.")
        return

    threshold_result = dl.load_threshold_analysis(dataset, model)
    if not threshold_result.available:
        ui.missing_file_notice(threshold_result, "Threshold analysis sweep")
        st.info("Run the pipeline with --performance to generate this (it is produced alongside performance metrics).")
        return

    sweep = threshold_result.data
    min_threshold, max_threshold = float(sweep["threshold"].min()), float(sweep["threshold"].max())
    if math.isfinite(min_threshold) and math.isfinite(max_threshold) and min_threshold < max_threshold:
        chosen_threshold = st.slider(
            "Threshold", min_value=min_threshold, max_value=max_threshold, value=(min_threshold + max_threshold) / 2
        )
        nearest_row = sweep.iloc[(sweep["threshold"] - chosen_threshold).abs().argsort()[:1]]
        if not nearest_row.empty:
            row = nearest_row.iloc[0]
            ui.metric_row(
                [
                    ("Threshold", f"{row['threshold']:.4f}", ""),
                    ("FMR (FAR)", f"{row.get('far', float('nan')):.4f}", ""),
                    ("FNMR (FRR)", f"{row.get('frr', float('nan')):.4f}", ""),
                    ("TAR", f"{row.get('tar', float('nan')):.4f}", ""),
                ]
            )

    tab1, tab2 = st.tabs(["Threshold vs FMR", "Threshold vs FNMR"])
    with tab1:
        st.plotly_chart(plots.threshold_sweep_chart(sweep, "far", "FMR (FAR)"), use_container_width=True)
    with tab2:
        st.plotly_chart(plots.threshold_sweep_chart(sweep, "frr", "FNMR (FRR)"), use_container_width=True)

    performance_result = dl.load_performance_metrics(dataset, model)
    if performance_result.available:
        test_row = performance_result.data.loc[performance_result.data["split"] == "test"]
        if not test_row.empty:
            row = test_row.iloc[0]
            st.markdown("### Configured Threshold Strategies")
            ui.metric_row(
                [
                    ("EER threshold", f"{row.get('eer_threshold', float('nan')):.4f}", "from performance_metrics.csv"),
                    ("EER", f"{row.get('eer', float('nan')):.4f}", ""),
                    ("", "", ""),
                    ("", "", ""),
                ]
            )

    st.markdown("### Global Threshold vs Group-Specific Thresholds")
    fairness_result = dl.load_fairness_metrics(dataset, model)
    if fairness_result.available and performance_result.available:
        overall_test_row = performance_result.data.loc[performance_result.data["split"] == "test"]
        overall_threshold = float(overall_test_row.iloc[0]["eer_threshold"]) if not overall_test_row.empty else float("nan")
        attribute_options = sorted(fairness_result.data["attribute"].unique().tolist())
        chosen_attribute = st.selectbox("Attribute", attribute_options, key="threshold_attribute")
        group_table = fairness_result.data.loc[fairness_result.data["attribute"] == chosen_attribute]
        comparison_table = group_table[["group_value", "eer_threshold", "far", "frr"]].copy()
        comparison_table.insert(1, "global_threshold", overall_threshold)
        comparison_table = comparison_table.rename(
            columns={"eer_threshold": "group_threshold", "far": "far_at_global_threshold", "frr": "frr_at_global_threshold"}
        )
        st.dataframe(comparison_table, use_container_width=True)
        st.caption(
            "far_at_global_threshold / frr_at_global_threshold are each group's error rates when the single "
            "global threshold is applied (from fairness_metrics.csv). group_threshold is that group's own "
            "EER threshold, shown for comparison — it is not applied to the error-rate columns above."
        )
    else:
        st.info("Requires both fairness_metrics.csv and performance_metrics.csv.")


# ------------------------------------------------------------------------------
# Page 8 — Hard Impostor Analysis
# ------------------------------------------------------------------------------


def page_hard_impostors(selection: dict) -> None:
    """Render the Hard Impostor Analysis page."""
    ui.page_header("Hard Impostor Analysis", "Impostor pairs most likely to be incorrectly accepted")
    dataset, model = selection["dataset"], selection["model"]
    if dataset is None or model is None:
        st.info("Select a dataset and an evaluated model in the sidebar.")
        return

    scores_result = dl.load_scores(dataset, model, split="test")
    if not scores_result.available:
        ui.missing_file_notice(scores_result, "Test similarity scores")
        return

    scores = scores_result.data
    score_column = selection["score_column"]
    impostors = scores.loc[scores["label"] == 0]
    if impostors.empty:
        st.info("No impostor pairs found in test_scores.csv.")
        return

    top_n = st.selectbox("Show top", [10, 25, 50, 100], index=0, key="hard_impostor_top_n")
    ascending = score_column == "euclidean_distance" or score_column == "cosine_distance"
    hardest = impostors.sort_values(score_column, ascending=ascending).head(top_n)

    display_columns = [c for c in ("image1", "image2", "identity1", "identity2", "demographic1", "demographic2", score_column) if c in hardest.columns]
    st.dataframe(hardest[display_columns], use_container_width=True)

    st.caption(
        "These impostor pairs (label=0, different identities) have the most similar embeddings in the "
        "dataset and are therefore the ones most likely to be incorrectly accepted as a match at a "
        "permissive threshold."
    )

    with st.expander("Show thumbnails (only for pairs whose image files still exist on disk)"):
        for _, row in hardest.head(min(top_n, 20)).iterrows():
            path1, path2 = row.get("image1"), row.get("image2")
            column1, column2 = st.columns(2)
            if path1 and __import__("pathlib").Path(path1).exists():
                column1.image(path1, caption=row.get("identity1", ""), width=150)
            else:
                column1.caption(f"Image not found on disk: {path1}")
            if path2 and __import__("pathlib").Path(path2).exists():
                column2.image(path2, caption=row.get("identity2", ""), width=150)
            else:
                column2.caption(f"Image not found on disk: {path2}")
            st.markdown(f"Score ({score_column}): **{row.get(score_column):.4f}**")
            st.markdown("---")

    st.plotly_chart(plots.hard_impostor_distribution(hardest, score_column), use_container_width=True)


# ------------------------------------------------------------------------------
# Page 9 — Results / Report
# ------------------------------------------------------------------------------


def page_report(selection: dict) -> None:
    """Render the Results / Report page with real-data export and a research-style summary."""
    ui.page_header("Results / Report", "Export results and generate a research-style summary")
    dataset, model = selection["dataset"], selection["model"]
    if dataset is None or model is None:
        st.info("Select a dataset and an evaluated model in the sidebar.")
        return

    performance_result = dl.load_performance_metrics(dataset, model)
    fairness_result = dl.load_fairness_metrics(dataset, model)
    disparity_result = dl.load_fairness_disparity(dataset, model)
    summary_result = dl.load_fairness_summary(dataset, model)
    comparison_result = dl.load_model_comparison(dataset)

    st.markdown("### Export")
    exports: dict[str, tuple[str, bytes]] = {}
    if performance_result.available:
        exports["Export performance CSV"] = ("performance_metrics.csv", performance_result.data.to_csv(index=False).encode("utf-8"))
    if fairness_result.available:
        exports["Export fairness CSV"] = ("fairness_metrics.csv", fairness_result.data.to_csv(index=False).encode("utf-8"))
    if performance_result.available:
        exports["Export performance JSON"] = (
            "performance_metrics.json",
            json.dumps(performance_result.data.to_dict(orient="records"), indent=2).encode("utf-8"),
        )
    if exports:
        ui.download_buttons(dataset, model, exports)
    else:
        st.info("No results available yet to export.")

    st.markdown("### Research-Style Summary")
    if not performance_result.available:
        ui.missing_file_notice(performance_result, "Performance metrics (required for the report)")
        return

    test_row = performance_result.data.loc[performance_result.data["split"] == "test"]
    perf = test_row.iloc[0].to_dict() if not test_row.empty else {}

    report_lines = [
        "# Fairness Evaluation Report",
        "",
        "## 1. Objective",
        "Evaluate demographic fairness of face recognition verification performance.",
        "",
        "## 2. Dataset",
        f"Dataset: {dataset}",
        "",
        "## 3. Model",
        f"Model: {model}",
        "",
        "## 4. Pair Generation",
        "Genuine and impostor pairs generated via the project's pair-generation pipeline (see pairs/<dataset>/).",
        "",
        "## 5. Similarity Metric",
        f"Score column used for this session: {selection['score_column']}",
        "",
        "## 6. Threshold Strategy",
        f"Overall decision threshold (EER): {perf.get('eer_threshold', 'N/A')}",
        "",
        "## 7. Global FMR/FNMR",
        f"FMR (FAR): {perf.get('far', 'N/A')}",
        f"FNMR (FRR): {perf.get('frr', 'N/A')}",
        f"EER: {perf.get('eer', 'N/A')}",
        "",
        "## 8. Group-wise FMR/FNMR",
    ]
    if fairness_result.available:
        report_lines.append(fairness_result.data.to_string(index=False))
    else:
        report_lines.append("Not available — fairness evaluation has not been run for this model.")

    report_lines += ["", "## 9. Intersectional Fairness", "See the Intersectional Fairness page (computed on demand, not a stored artifact)."]

    report_lines += ["", "## 10. Threshold Sensitivity"]
    threshold_result = dl.load_threshold_analysis(dataset, model)
    report_lines.append(
        "See threshold_analysis.csv." if threshold_result.available else "Not available."
    )

    report_lines += ["", "## 11. Hard Impostor Analysis", "See the Hard Impostor Analysis page for the highest-scoring impostor pairs in the test split."]

    report_lines += ["", "## 12. Model Comparison"]
    report_lines.append(comparison_result.data.to_string(index=False) if comparison_result.available else "Not available — fewer than 2 models evaluated.")

    report_lines += ["", "## 13. Key Findings"]
    if summary_result.available and "summary_text" in summary_result.data.columns:
        report_lines.append(str(summary_result.data.iloc[0]["summary_text"]))
    else:
        report_lines.append("Not available — fairness_summary.csv has not been generated for this model.")

    report_lines += [
        "",
        "## 14. Limitations",
        "- Intersectional fairness figures on this dashboard are computed on demand by a GUI-side adapter, "
        "not stored as a pipeline artifact.",
        "- Group-wise metrics below the configured minimum sample size are excluded from disparity statistics.",
        "",
        "## 15. Conclusion",
        "This report is generated entirely from the result files listed above; no values were fabricated.",
    ]

    report_text = "\n".join(report_lines)
    st.text_area("Report preview", report_text, height=400)
    st.download_button(
        "Generate Report (Markdown)",
        data=report_text.encode("utf-8"),
        file_name=f"fairness_report_{dataset}_{model}.md",
        key="btn_generate_report",
    )


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------

_PAGE_RENDERERS = {
    "Overview": page_overview,
    "Model Comparison": page_model_comparison,
    "Dataset": page_dataset,
    "Score Distribution": page_score_distribution,
    "Fairness / Group Analysis": page_fairness,
    "Intersectional Fairness": page_intersectional,
    "Threshold Analysis": page_threshold_analysis,
    "Hard Impostor Analysis": page_hard_impostors,
    "Results / Report": page_report,
}


def main() -> None:
    """Streamlit entry point: configure the page and dispatch to the selected view."""
    st.set_page_config(page_title="FairFaceEval", layout="wide", initial_sidebar_state="expanded")
    inject_global_css()

    selection = render_sidebar()
    renderer = _PAGE_RENDERERS[selection["page"]]
    renderer(selection)


if __name__ == "__main__":
    main()
