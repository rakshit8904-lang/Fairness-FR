"""Real-data access layer for the FairFaceEval dashboard.

Every function in this module either returns data read from an actual
file the existing pipeline produces (:mod:`fairness_fr.data.preprocess`,
:mod:`fairness_fr.evaluation.calculate_scores`,
:mod:`fairness_fr.evaluation.evaluate_performance`,
:mod:`fairness_fr.evaluation.evaluate_fairness`,
:mod:`fairness_fr.evaluation.model_comparator`) or ``None``. Nothing
here invents numbers: a missing file is reported as missing, with the
exact path the pipeline was expected to have written, never silently
substituted with a placeholder value.

Two things in this module go slightly beyond "just read a CSV":

- :func:`compute_intersectional_metrics` — the existing fairness
  pipeline (:mod:`fairness_fr.evaluation.evaluate_fairness`) evaluates
  one demographic attribute at a time; it does not compute combined
  attributes such as ethnicity x gender. This function is a thin
  adapter that reuses the pipeline's own confusion-matrix and EER
  helpers (imported, not reimplemented) to compute a genuine
  intersectional breakdown on demand from the same ``test_scores.csv``
  and ``metadata.csv`` the rest of the pipeline already produced.
- :func:`run_pipeline_stages` — invokes the existing
  ``run_pipeline.py`` CLI as a subprocess rather than duplicating any
  orchestration logic, streaming its real log output back to the caller.

All expensive loads are wrapped in ``st.cache_data`` so the GUI never
recomputes or re-reads a file just because the user switched charts;
:func:`clear_all_caches` is called once, explicitly, after a pipeline
run finishes.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import streamlit as st

from fairness_fr.config.config import ConfigLoader, DatasetConfig, ModelConfig, get_config_loader
from fairness_fr.config.constants import MetadataColumns
from fairness_fr.config.settings import Settings, get_settings
from fairness_fr.data.generate_pairs import PairOutputColumns
from fairness_fr.evaluation.calculate_scores import ScoreOutputColumns
from fairness_fr.evaluation.evaluate_performance import (
    _confusion_counts_at_threshold,
    _safe_f1,
)
from sklearn.metrics import roc_curve as _roc_curve

from fairness_fr.evaluation.evaluate_fairness import _compute_eer_from_sweep


@dataclass(frozen=True, slots=True)
class LoadResult:
    """Outcome of attempting to load one pipeline output file.

    Attributes:
        data: The loaded object (DataFrame, dict, or str), or None if unavailable.
        path: The exact path that was checked.
        available: Whether ``data`` is usable.
        reason: Human-readable explanation when ``available`` is False.
    """

    data: Any
    path: Path
    available: bool
    reason: str = ""


# ------------------------------------------------------------------------------
# Configuration / discovery
# ------------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_config_loader_cached() -> ConfigLoader:
    """Return the cached project :class:`ConfigLoader`."""
    return get_config_loader()


@st.cache_resource(show_spinner=False)
def get_settings_cached() -> Settings:
    """Return the cached project :class:`Settings`."""
    return get_settings()


def discover_datasets() -> list[str]:
    """List dataset names with a valid config file under ``configs/datasets/``.

    Returns:
        Sorted dataset identifiers that :class:`ConfigLoader` can load.
        Empty if the configs directory or its dataset subfolder is missing.
    """
    settings = get_settings_cached()
    datasets_dir = settings.configs_dir / "datasets"
    if not datasets_dir.exists():
        return []
    return sorted(path.stem for path in datasets_dir.glob("*.yaml"))


def discover_models() -> list[str]:
    """List model names with a valid config file under ``configs/models/``.

    Returns:
        Sorted model identifiers that :class:`ConfigLoader` can load.
        Empty if the configs directory or its model subfolder is missing.
    """
    settings = get_settings_cached()
    models_dir = settings.configs_dir / "models"
    if not models_dir.exists():
        return []
    return sorted(path.stem for path in models_dir.glob("*.yaml"))


@st.cache_data(show_spinner=False)
def load_dataset_config(dataset_name: str) -> DatasetConfig | None:
    """Load one dataset's configuration, or None if invalid/missing."""
    try:
        return get_config_loader_cached().load_dataset_config(dataset_name)
    except (FileNotFoundError, Exception):  # noqa: BLE001 - surfaced as "unavailable" in the GUI
        return None


@st.cache_data(show_spinner=False)
def load_model_config(model_name: str) -> ModelConfig | None:
    """Load one model's configuration, or None if invalid/missing."""
    try:
        return get_config_loader_cached().load_model_config(model_name)
    except (FileNotFoundError, Exception):  # noqa: BLE001 - surfaced as "unavailable" in the GUI
        return None


def evaluated_models_for_dataset(dataset_name: str) -> list[str]:
    """List configured models that have at least a performance metrics file.

    Args:
        dataset_name: Dataset identifier.

    Returns:
        Model names for which ``results/<dataset>/<model>/performance_metrics.csv``
        exists — i.e. models that have actually been evaluated, not just configured.
    """
    settings = get_settings_cached()
    evaluated = []
    for model_name in discover_models():
        candidate = settings.results_dir / dataset_name / model_name / "performance_metrics.csv"
        if candidate.exists():
            evaluated.append(model_name)
    return evaluated


# ------------------------------------------------------------------------------
# Path resolution (mirrors run_pipeline.py's own conventions exactly)
# ------------------------------------------------------------------------------


def model_results_dir(dataset_name: str, model_name: str) -> Path:
    """Return the results directory the pipeline writes for one dataset/model pair."""
    return get_settings_cached().results_dir / dataset_name / model_name


def model_embeddings_dir(dataset_name: str, model_name: str) -> Path:
    """Return the embeddings directory the pipeline writes for one dataset/model pair."""
    return get_settings_cached().embeddings_dir / dataset_name / model_name


def pairs_dir(dataset_name: str) -> Path:
    """Return the pair-CSV directory the pipeline writes for one dataset."""
    return get_settings_cached().pairs_dir / dataset_name


def comparison_dir(dataset_name: str) -> Path:
    """Return the model-comparison output directory."""
    return get_settings_cached().results_dir / "model_comparison"
# ------------------------------------------------------------------------------
# Generic loaders
# ------------------------------------------------------------------------------


def _load_csv(path: Path) -> LoadResult:
    """Load a CSV file if present, reporting a precise reason if not.

    Args:
        path: Exact path the pipeline is expected to have written.

    Returns:
        A :class:`LoadResult` wrapping the DataFrame or the failure reason.
    """
    if not path.exists():
        return LoadResult(None, path, False, "File not found. Run the corresponding pipeline stage.")
    try:
        frame = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        return LoadResult(None, path, False, f"File exists but could not be parsed: {exc}")
    if frame.empty:
        return LoadResult(frame, path, False, "File exists but contains no rows.")
    return LoadResult(frame, path, True)


def _load_json(path: Path) -> LoadResult:
    """Load a JSON file if present, reporting a precise reason if not."""
    if not path.exists():
        return LoadResult(None, path, False, "File not found. Run the corresponding pipeline stage.")
    try:
        with path.open("r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)
    except (json.JSONDecodeError, OSError) as exc:
        return LoadResult(None, path, False, f"File exists but could not be parsed: {exc}")
    return LoadResult(payload, path, True)


def _load_text(path: Path) -> LoadResult:
    """Load a plain-text file if present, reporting a precise reason if not."""
    if not path.exists():
        return LoadResult(None, path, False, "File not found. Run the corresponding pipeline stage.")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return LoadResult(None, path, False, f"File exists but could not be read: {exc}")
    return LoadResult(text, path, True)


# ------------------------------------------------------------------------------
# Per-model evaluation outputs
# ------------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_performance_metrics(dataset_name: str, model_name: str) -> LoadResult:
    """Load ``performance_metrics.csv`` for one dataset/model pair."""
    return _load_csv(model_results_dir(dataset_name, model_name) / "performance_metrics.csv")


@st.cache_data(show_spinner=False)
def load_fairness_metrics(dataset_name: str, model_name: str) -> LoadResult:
    """Load ``fairness_metrics.csv`` for one dataset/model pair."""
    return _load_csv(model_results_dir(dataset_name, model_name) / "fairness_metrics.csv")


@st.cache_data(show_spinner=False)
def load_fairness_disparity(dataset_name: str, model_name: str) -> LoadResult:
    """Load ``fairness_disparity.csv`` for one dataset/model pair."""
    return _load_csv(model_results_dir(dataset_name, model_name) / "fairness_disparity.csv")


@st.cache_data(show_spinner=False)
def load_fairness_summary(dataset_name: str, model_name: str) -> LoadResult:
    """Load ``fairness_summary.csv`` for one dataset/model pair."""
    return _load_csv(model_results_dir(dataset_name, model_name) / "fairness_summary.csv")


@st.cache_data(show_spinner=False)
def load_threshold_analysis(dataset_name: str, model_name: str) -> LoadResult:
    """Load ``threshold_analysis.csv`` for one dataset/model pair."""
    return _load_csv(model_results_dir(dataset_name, model_name) / "threshold_analysis.csv")


@st.cache_data(show_spinner=False)
def load_roc_points(dataset_name: str, model_name: str) -> LoadResult:
    """Load ``roc_points.csv`` for one dataset/model pair."""
    return _load_csv(model_results_dir(dataset_name, model_name) / "roc_points.csv")


@st.cache_data(show_spinner=False)
def load_scores(dataset_name: str, model_name: str, split: str = "test") -> LoadResult:
    """Load one split's similarity-score CSV for one dataset/model pair.

    Args:
        dataset_name: Dataset identifier.
        model_name: Model identifier.
        split: One of ``"train"``, ``"validation"``, ``"test"``.
    """
    return _load_csv(model_results_dir(dataset_name, model_name) / f"{split}_scores.csv")


@st.cache_data(show_spinner=False)
def load_metadata(dataset_name: str) -> LoadResult:
    """Load a dataset's ``metadata.csv`` via its own :class:`DatasetConfig` path."""
    dataset_config = load_dataset_config(dataset_name)
    if dataset_config is None:
        return LoadResult(None, get_settings_cached().data_dir, False, "Dataset configuration is invalid or missing.")
    return _load_csv(dataset_config.metadata_csv)


@st.cache_data(show_spinner=False)
def load_pairs(dataset_name: str, split: str = "test") -> LoadResult:
    """Load one split's pair CSV for a dataset."""
    return _load_csv(pairs_dir(dataset_name) / f"{split}_pairs.csv")


# ------------------------------------------------------------------------------
# Model comparison outputs
# ------------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_model_comparison(dataset_name: str) -> LoadResult:
    """Load ``model_comparison.csv`` for a dataset."""
    return _load_csv(comparison_dir(dataset_name) / "model_comparison.csv")


@st.cache_data(show_spinner=False)
def load_model_rankings(dataset_name: str) -> LoadResult:
    """Load ``model_rankings.csv`` for a dataset."""
    return _load_csv(comparison_dir(dataset_name) / "model_rankings.csv")


@st.cache_data(show_spinner=False)
def load_model_summary_text(dataset_name: str) -> LoadResult:
    """Load ``model_summary.txt`` for a dataset."""
    return _load_text(comparison_dir(dataset_name) / "model_summary.txt")


# ------------------------------------------------------------------------------
# Thin adapter: intersectional fairness (not computed by the existing pipeline)
# ------------------------------------------------------------------------------


def compute_intersectional_metrics(
    dataset_name: str,
    model_name: str,
    attribute_a: str,
    attribute_b: str,
    score_column: str = ScoreOutputColumns.COSINE_SIMILARITY,
    min_group_size: int = 20,
) -> LoadResult:
    """Compute a real ethnicity x gender (or any two-attribute) breakdown on demand.

    This is a thin adapter, not a reimplementation: it reuses
    :func:`fairness_fr.evaluation.evaluate_performance._confusion_counts_at_threshold`,
    :func:`fairness_fr.evaluation.evaluate_performance._safe_f1`, and
    :func:`fairness_fr.evaluation.evaluate_fairness._compute_eer_from_sweep`
    — the exact same formulas the fairness pipeline itself uses — applied
    to a combined ``attribute_a::attribute_b`` group label instead of a
    single attribute, since the existing pipeline evaluates one
    demographic attribute at a time.

    Args:
        dataset_name: Dataset identifier.
        model_name: Model identifier.
        attribute_a: First metadata.csv column to intersect (e.g. "group").
        attribute_b: Second metadata.csv column to intersect (e.g. "gender").
        score_column: Which score column to threshold on.
        min_group_size: Minimum same-group pair count for inclusion.

    Returns:
        A :class:`LoadResult` wrapping a DataFrame with columns
        ``group, sample_size, genuine_pairs, impostor_pairs, accuracy,
        far, frr, tar, eer``, or an explanation of why it could not be computed.
    """
    scores_result = load_scores(dataset_name, model_name, split="test")
    if not scores_result.available:
        return LoadResult(
            None,
            scores_result.path,
            False,
            f"test_scores.csv unavailable: {scores_result.reason}",
        )

    metadata_result = load_metadata(dataset_name)
    if not metadata_result.available:
        return LoadResult(
            None,
            metadata_result.path,
            False,
            f"metadata.csv unavailable: {metadata_result.reason}",
        )

    performance_result = load_performance_metrics(dataset_name, model_name)
    if not performance_result.available:
        return LoadResult(
            None,
            performance_result.path,
            False,
            "performance_metrics.csv unavailable; the overall EER threshold is required "
            "to evaluate intersectional groups at a fixed decision policy.",
        )

    metadata: pd.DataFrame = metadata_result.data
    if attribute_a not in metadata.columns or attribute_b not in metadata.columns:
        return LoadResult(
            None,
            metadata_result.path,
            False,
            f"metadata.csv does not contain both '{attribute_a}' and '{attribute_b}' columns. "
            f"Available columns: {list(metadata.columns)}",
        )

    performance: pd.DataFrame = performance_result.data
    test_row = performance.loc[performance["split"] == "test"]
    if test_row.empty or "eer_threshold" not in performance.columns:
        return LoadResult(
            None,
            performance_result.path,
            False,
            "No 'test' split row with an eer_threshold column in performance_metrics.csv.",
        )
    overall_threshold = float(test_row.iloc[0]["eer_threshold"])
    if pd.isna(overall_threshold):
        return LoadResult(None, performance_result.path, False, "Overall EER threshold is NaN.")

    scores: pd.DataFrame = scores_result.data
    required = {PairOutputColumns.IMAGE1, PairOutputColumns.IMAGE2, PairOutputColumns.LABEL, score_column}
    if not required.issubset(scores.columns):
        return LoadResult(
            None, scores_result.path, False, f"test_scores.csv missing required columns: {required - set(scores.columns)}"
        )

    combo_a = metadata.set_index(MetadataColumns.IMAGE_PATH)[attribute_a].to_dict()
    combo_b = metadata.set_index(MetadataColumns.IMAGE_PATH)[attribute_b].to_dict()

    def _combo_label(image_path: str) -> str | None:
        value_a = combo_a.get(image_path)
        value_b = combo_b.get(image_path)
        if not value_a or not value_b:
            return None
        return f"{value_a} + {value_b}"

    label1 = scores[PairOutputColumns.IMAGE1].map(_combo_label)
    label2 = scores[PairOutputColumns.IMAGE2].map(_combo_label)
    same_group_mask = label1.notna() & (label1 == label2)
    subset = scores.loc[same_group_mask].copy()
    subset["_combo_group"] = label1.loc[same_group_mask]

    if subset.empty:
        return LoadResult(
            None,
            scores_result.path,
            False,
            f"No pairs share the same '{attribute_a}' and '{attribute_b}' value on both images.",
        )

    higher_is_better = score_column != ScoreOutputColumns.EUCLIDEAN_DISTANCE and score_column != ScoreOutputColumns.COSINE_DISTANCE

    rows: list[dict[str, Any]] = []
    for group_value, group_frame in subset.groupby("_combo_group"):
        labels = group_frame[PairOutputColumns.LABEL].to_numpy()
        group_scores = group_frame[score_column].to_numpy()
        sample_size = len(labels)
        genuine = int(np.sum(labels == 1))
        impostor = int(np.sum(labels == 0))

        if sample_size < min_group_size or len({0, 1} & set(np.unique(labels).tolist())) < 2:
            rows.append(
                {
                    "group": group_value,
                    "sample_size": sample_size,
                    "genuine_pairs": genuine,
                    "impostor_pairs": impostor,
                    "accuracy": float("nan"),
                    "far": float("nan"),
                    "frr": float("nan"),
                    "tar": float("nan"),
                    "eer": float("nan"),
                }
            )
            continue

        true_accepts, false_accepts, true_rejects, false_rejects = _confusion_counts_at_threshold(
            labels, group_scores, overall_threshold, higher_is_better
        )
        total = true_accepts + false_accepts + true_rejects + false_rejects
        total_genuine = true_accepts + false_rejects
        total_impostor = false_accepts + true_rejects

        accuracy = (true_accepts + true_rejects) / total if total else float("nan")
        far = false_accepts / total_impostor if total_impostor else float("nan")
        frr = false_rejects / total_genuine if total_genuine else float("nan")
        tar = true_accepts / total_genuine if total_genuine else float("nan")

        decision_scores = group_scores if higher_is_better else -group_scores
        fpr, tpr, raw_thresholds = _roc_curve(labels, decision_scores)
        eer, _ = _compute_eer_from_sweep(fpr, 1.0 - tpr, raw_thresholds)

        rows.append(
            {
                "group": group_value,
                "sample_size": sample_size,
                "genuine_pairs": genuine,
                "impostor_pairs": impostor,
                "accuracy": accuracy,
                "far": far,
                "frr": frr,
                "tar": tar,
                "eer": eer,
            }
        )

    result_frame = pd.DataFrame(rows).sort_values("group").reset_index(drop=True)
    return LoadResult(result_frame, scores_result.path, True)


# ------------------------------------------------------------------------------
# Pipeline execution (subprocess call into the existing run_pipeline.py CLI)
# ------------------------------------------------------------------------------


def build_pipeline_command(stage_flags: list[str], experiment_config: Path | None = None) -> list[str]:
    """Build the ``run_pipeline.py`` command line for a set of stage flags.

    Args:
        stage_flags: e.g. ``["--preprocess", "--pairs"]``, or ``["--all"]``.
        experiment_config: Optional explicit experiment YAML path.

    Returns:
        The full argv list to execute, using the current Python interpreter.
    """
    settings = get_settings_cached()
    script_path = settings.project_root / "run_pipeline.py"
    command = [sys.executable, str(script_path), *stage_flags]
    if experiment_config is not None:
        command.extend(["--experiment-config", str(experiment_config)])
    return command


def run_pipeline_stages(stage_flags: list[str], experiment_config: Path | None = None) -> Iterator[str]:
    """Run ``run_pipeline.py`` as a subprocess, yielding its output line by line.

    This calls the existing CLI entry point exactly as a user would from
    a terminal; no orchestration logic is duplicated here.

    Args:
        stage_flags: e.g. ``["--preprocess", "--pairs"]``, or ``["--all"]``.
        experiment_config: Optional explicit experiment YAML path.

    Yields:
        Each line of combined stdout/stderr as it is produced.
    """
    settings = get_settings_cached()
    command = build_pipeline_command(stage_flags, experiment_config)
    process = subprocess.Popen(
        command,
        cwd=str(settings.project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    try:
        for line in process.stdout:
            yield line.rstrip("\n")
    finally:
        process.stdout.close()
        process.wait()
        yield f"__EXIT_CODE__:{process.returncode}"


def clear_all_caches() -> None:
    """Clear every cached loader so newly written pipeline outputs are picked up."""
    load_performance_metrics.clear()
    load_fairness_metrics.clear()
    load_fairness_disparity.clear()
    load_fairness_summary.clear()
    load_threshold_analysis.clear()
    load_roc_points.clear()
    load_scores.clear()
    load_metadata.clear()
    load_pairs.clear()
    load_model_comparison.clear()
    load_model_rankings.clear()
    load_model_summary_text.clear()
    load_dataset_config.clear()
    load_model_config.clear()
