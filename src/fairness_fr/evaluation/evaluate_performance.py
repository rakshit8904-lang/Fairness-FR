"""Biometric verification performance evaluation for the fairness pipeline.

Consumes the per-split similarity score CSVs produced by
:mod:`fairness_fr.evaluation.calculate_scores` (``train_scores.csv``,
``validation_scores.csv``, ``test_scores.csv``) and computes standard
biometric verification metrics — accuracy, precision, recall, F1,
FAR/FMR, FRR/FNMR, TAR, TNR, and the Equal Error Rate (EER) with its
corresponding decision threshold — along with the curve data (ROC, DET,
threshold sweeps) needed to visualize verification performance.

Outputs, combined across every evaluated split:

- ``performance_metrics.json`` / ``performance_metrics.csv``
- ``threshold_analysis.csv``
- ``roc_points.csv``
- ``det_points.csv``
- Per-split PNG plots under ``plots/``: ROC curve, DET curve, score
  distribution, confusion matrix (at the EER threshold), and
  threshold-vs-FAR/FRR/TAR curves.

Design notes:
    - Verification semantics: label ``1`` (genuine) is treated as the
      "positive"/match class; a pair is *accepted* (predicted match)
      when its score crosses the decision threshold in the direction of
      higher similarity. Distance-valued score columns (Euclidean
      distance, cosine distance) are handled transparently — internally
      the accept rule flips to "distance below threshold" — so callers
      can point this module at any of the three score columns produced
      by :mod:`fairness_fr.evaluation.calculate_scores` without special
      handling.
    - The ROC/FAR/FRR sweep is computed once via
      :func:`sklearn.metrics.roc_curve`, which is O(N log N) and reuses
      a well-tested implementation rather than a hand-rolled threshold
      grid, so this scales to large score files.
    - EER is found by locating the sign change of ``FAR - FRR`` across
      the sweep and linearly interpolating between the two bracketing
      thresholds; if no crossing exists (degenerate score
      distributions), the threshold minimizing ``|FAR - FRR|`` is used
      instead.
    - Single-class splits and other degenerate inputs are handled by
      emitting NaN-valued metrics and empty curve data rather than
      raising, since a fairness study may legitimately encounter a
      demographic slice with too little data.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # ensure headless rendering in any environment

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from tqdm import tqdm

from fairness_fr.config import ThresholdConfig
from ..config.constants import MetricName
from fairness_fr.data.generate_pairs import PairOutputColumns
from fairness_fr.evaluation.calculate_scores import ScoreOutputColumns
from fairness_fr.utils.logging import get_logger, setup_logging
from fairness_fr.utils import ensure_dir, safe_divide, timer

logger = get_logger(__name__)


class PerformanceEvaluationError(Exception):
    """Raised when performance evaluation cannot be completed for a split."""


#: Score columns that represent a *distance* (lower = more similar),
#: as opposed to a similarity (higher = more similar). Used to decide
#: the direction of the accept rule.
_DISTANCE_SCORE_COLUMNS: frozenset[str] = frozenset(
    {ScoreOutputColumns.EUCLIDEAN_DISTANCE, ScoreOutputColumns.COSINE_DISTANCE}
)


def _json_safe(value: Any) -> Any:
    """Fallback JSON encoder for numpy scalar types.

    Args:
        value: A value :func:`json.dump` could not serialize directly.

    Returns:
        A plain Python ``float`` or ``int`` equivalent.

    Raises:
        TypeError: If ``value`` is not a numpy scalar type.
    """
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    raise TypeError(f"Object of type {type(value)} is not JSON serializable.")


def _safe_f1(precision: float, recall: float) -> float:
    """Compute the F1 score from precision and recall, guarding against NaN/zero.

    Args:
        precision: Precision value, possibly NaN.
        recall: Recall value, possibly NaN.

    Returns:
        The harmonic mean of precision and recall, or NaN if either
        input is NaN or their sum is zero.
    """
    if math.isnan(precision) or math.isnan(recall):
        return float("nan")
    denominator = precision + recall
    if denominator == 0:
        return float("nan")
    return 2 * precision * recall / denominator


def _confusion_counts_at_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    higher_is_better: bool = True,
) -> tuple[int, int, int, int]:
    """Compute (true_accepts, false_accepts, true_rejects, false_rejects) at a threshold.

    Args:
        labels: ``(N,)`` array of binary labels (1=genuine, 0=impostor).
        scores: ``(N,)`` array of similarity or distance scores.
        threshold: Decision threshold, in the same units as ``scores``.
        higher_is_better: If True, a pair is accepted when
            ``score >= threshold`` (similarity semantics). If False, a
            pair is accepted when ``score <= threshold`` (distance
            semantics).

    Returns:
        A tuple ``(true_accepts, false_accepts, true_rejects, false_rejects)``.
    """
    accepted = scores >= threshold if higher_is_better else scores <= threshold
    genuine = labels == 1
    impostor = labels == 0

    true_accepts = int(np.sum(accepted & genuine))
    false_rejects = int(np.sum(~accepted & genuine))
    false_accepts = int(np.sum(accepted & impostor))
    true_rejects = int(np.sum(~accepted & impostor))
    return true_accepts, false_accepts, true_rejects, false_rejects


# ------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScoreDataset:
    """Validated score data for one split, ready for metric computation.

    Attributes:
        split_name: Name of the split (e.g. ``"train"``).
        labels: ``(N,)`` int array of ground-truth labels.
        scores: ``(N,)`` float array of the score column being evaluated.
        higher_is_better: Whether higher values of ``scores`` indicate
            greater similarity (True for similarity columns, False for
            distance columns).
    """

    split_name: str
    labels: np.ndarray
    scores: np.ndarray
    higher_is_better: bool


class ScoreDataLoader:
    """Loads and validates score CSVs produced by ``calculate_scores.py``.

    Validation drops (with logged counts) rows with non-binary labels
    or missing/NaN scores, rather than failing the whole split, so a
    handful of bad rows does not block evaluation of an otherwise
    healthy score file.
    """

    def __init__(self, score_column: str = ScoreOutputColumns.COSINE_SIMILARITY) -> None:
        """Initialize the loader.

        Args:
            score_column: Which score column to evaluate — one of
                :attr:`fairness_fr.evaluation.calculate_scores.ScoreOutputColumns.METRIC_COLUMNS`.
        """
        self.score_column = score_column
        self.higher_is_better = score_column not in _DISTANCE_SCORE_COLUMNS

    def load(self, csv_path: Path, split_name: str) -> ScoreDataset:
        """Load, validate, and return one split's score data.

        Args:
            csv_path: Path to the split's score CSV.
            split_name: Human-readable split name, used for logging.

        Returns:
            A validated :class:`ScoreDataset`.

        Raises:
            FileNotFoundError: If ``csv_path`` does not exist.
            PerformanceEvaluationError: If the file is empty, is missing
                required columns, or has no valid rows after validation.
        """
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Score file not found for split '{split_name}': {csv_path}. "
                f"Run calculate_scores.py before performance evaluation."
            )

        logger.info("Loading scores for split '%s' from %s", split_name, csv_path)
        frame = pd.read_csv(csv_path)

        if frame.empty:
            raise PerformanceEvaluationError(
                f"Score file for split '{split_name}' is empty: {csv_path}."
            )

        required_columns = {PairOutputColumns.LABEL, self.score_column}
        missing_columns = required_columns - set(frame.columns)
        if missing_columns:
            raise PerformanceEvaluationError(
                f"Score file for split '{split_name}' is missing required columns: "
                f"{sorted(missing_columns)}."
            )

        valid_label_mask = frame[PairOutputColumns.LABEL].isin([0, 1])
        invalid_label_count = int((~valid_label_mask).sum())
        if invalid_label_count:
            logger.warning(
                "Split '%s': dropping %d rows with invalid (non-binary) labels.",
                split_name,
                invalid_label_count,
            )
        frame = frame.loc[valid_label_mask]

        valid_score_mask = pd.to_numeric(frame[self.score_column], errors="coerce").notna()
        missing_score_count = int((~valid_score_mask).sum())
        if missing_score_count:
            logger.warning(
                "Split '%s': dropping %d rows with missing or invalid '%s' scores.",
                split_name,
                missing_score_count,
                self.score_column,
            )
        frame = frame.loc[valid_score_mask]

        if frame.empty:
            raise PerformanceEvaluationError(
                f"Split '{split_name}' has no valid rows remaining after validation."
            )

        labels = frame[PairOutputColumns.LABEL].astype(int).to_numpy()
        scores = frame[self.score_column].astype(float).to_numpy()

        unique_labels = set(np.unique(labels).tolist())
        if len(unique_labels) < 2:
            logger.warning(
                "Split '%s' contains only a single class (%s); FAR/FRR/ROC/EER are "
                "undefined and will be reported as NaN.",
                split_name,
                unique_labels,
            )

        logger.info(
            "Split '%s': %d valid pairs (%d genuine, %d impostor).",
            split_name,
            len(labels),
            int(np.sum(labels == 1)),
            int(np.sum(labels == 0)),
        )

        return ScoreDataset(
            split_name=split_name,
            labels=labels,
            scores=scores,
            higher_is_better=self.higher_is_better,
        )


# ------------------------------------------------------------------------------
# Metric computation
# ------------------------------------------------------------------------------


@dataclass
class SplitEvaluationResult:
    """All computed evaluation artifacts for one split.

    Attributes:
        split_name: Name of the split this result belongs to.
        metrics: Flat dict of scalar metrics (accuracy, precision,
            recall, f1_score, far, frr, fmr, fnmr, tar, tnr, eer,
            eer_threshold, and pair counts).
        roc_points: DataFrame with columns ``threshold, fpr, tpr``.
        det_points: DataFrame with columns ``threshold, far, frr``.
        threshold_analysis: DataFrame with columns
            ``threshold, far, frr, tar, tnr``.
        confusion_matrix: ``2x2`` array ``[[TN, FP], [FN, TP]]`` at the
            EER threshold.
        labels: The split's label array, retained for plotting.
        scores: The split's score array, retained for plotting.
    """

    split_name: str
    metrics: dict[str, float]
    roc_points: pd.DataFrame
    det_points: pd.DataFrame
    threshold_analysis: pd.DataFrame
    confusion_matrix: np.ndarray
    labels: np.ndarray
    scores: np.ndarray


class PerformanceMetricsCalculator:
    """Computes verification performance metrics and curve data for one split."""

    def __init__(self, threshold_config: ThresholdConfig | None = None) -> None:
        """Initialize the calculator.

        Args:
            threshold_config: Reserved for future threshold-strategy
                configuration (e.g. reporting metrics at configured
                target-FMR operating points in addition to the EER
                threshold). Currently metrics are always reported at
                the EER threshold, the standard biometric evaluation
                convention.
        """
        self.threshold_config = threshold_config or ThresholdConfig()

    def compute(self, dataset: ScoreDataset) -> SplitEvaluationResult:
        """Compute all metrics and curve data for a validated score dataset.

        Args:
            dataset: A validated :class:`ScoreDataset`.

        Returns:
            A :class:`SplitEvaluationResult` with metrics computed at
            the EER threshold and full ROC/DET/threshold curve data.
        """
        labels = dataset.labels
        unique_labels = set(np.unique(labels).tolist())

        if len(unique_labels) < 2:
            logger.warning(
                "Split '%s': single-class dataset; returning NaN metrics and empty curves.",
                dataset.split_name,
            )
            return self._degenerate_result(dataset)

        decision_scores = dataset.scores if dataset.higher_is_better else -dataset.scores
        fpr, tpr, raw_thresholds = roc_curve(labels, decision_scores)

        far = fpr
        tar = tpr
        frr = 1.0 - tpr
        tnr = 1.0 - fpr

        eer, eer_threshold_decision = self._compute_eer(far, frr, raw_thresholds)
        eer_threshold = (
            float(eer_threshold_decision) if dataset.higher_is_better else float(-eer_threshold_decision)
        )

        reported_thresholds = raw_thresholds if dataset.higher_is_better else -raw_thresholds

        roc_points = pd.DataFrame({"threshold": reported_thresholds, "fpr": far, "tpr": tar})
        det_points = pd.DataFrame({"threshold": reported_thresholds, "far": far, "frr": frr})
        threshold_analysis = pd.DataFrame(
            {"threshold": reported_thresholds, "far": far, "frr": frr, "tar": tar, "tnr": tnr}
        )

        true_accepts, false_accepts, true_rejects, false_rejects = _confusion_counts_at_threshold(
            labels, dataset.scores, eer_threshold, dataset.higher_is_better
        )

        total = true_accepts + false_accepts + true_rejects + false_rejects
        total_genuine = true_accepts + false_rejects
        total_impostor = false_accepts + true_rejects

        accuracy = safe_divide(true_accepts + true_rejects, total, default=float("nan"))
        precision = safe_divide(true_accepts, true_accepts + false_accepts, default=float("nan"))
        recall = safe_divide(true_accepts, total_genuine, default=float("nan"))
        f1 = _safe_f1(precision, recall)
        far_at_eer = safe_divide(false_accepts, total_impostor, default=float("nan"))
        frr_at_eer = safe_divide(false_rejects, total_genuine, default=float("nan"))
        tar_at_eer = safe_divide(true_accepts, total_genuine, default=float("nan"))
        tnr_at_eer = safe_divide(true_rejects, total_impostor, default=float("nan"))

        metrics: dict[str, float] = {
            MetricName.ACCURACY.value: accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            MetricName.FAR.value: far_at_eer,
            MetricName.FRR.value: frr_at_eer,
            MetricName.FMR.value: far_at_eer,
            MetricName.FNMR.value: frr_at_eer,
            MetricName.TAR.value: tar_at_eer,
            "tnr": tnr_at_eer,
            MetricName.EER.value: eer,
            "eer_threshold": eer_threshold,
            "total_pairs": total,
            "genuine_pairs": total_genuine,
            "impostor_pairs": total_impostor,
        }

        confusion = np.array(
            [[true_rejects, false_accepts], [false_rejects, true_accepts]], dtype=int
        )

        return SplitEvaluationResult(
            split_name=dataset.split_name,
            metrics=metrics,
            roc_points=roc_points,
            det_points=det_points,
            threshold_analysis=threshold_analysis,
            confusion_matrix=confusion,
            labels=labels,
            scores=dataset.scores,
        )

    def _degenerate_result(self, dataset: ScoreDataset) -> SplitEvaluationResult:
        """Build a NaN-valued result for a single-class (or otherwise degenerate) split.

        Args:
            dataset: The degenerate :class:`ScoreDataset`.

        Returns:
            A :class:`SplitEvaluationResult` with NaN metrics and empty
            curve DataFrames.
        """
        labels = dataset.labels
        total = len(labels)
        genuine = int(np.sum(labels == 1))
        impostor = int(np.sum(labels == 0))

        nan_keys = (
            MetricName.ACCURACY.value,
            "precision",
            "recall",
            "f1_score",
            MetricName.FAR.value,
            MetricName.FRR.value,
            MetricName.FMR.value,
            MetricName.FNMR.value,
            MetricName.TAR.value,
            "tnr",
            MetricName.EER.value,
            "eer_threshold",
        )
        metrics: dict[str, float] = {key: float("nan") for key in nan_keys}
        metrics.update({"total_pairs": total, "genuine_pairs": genuine, "impostor_pairs": impostor})

        empty_curve_columns = ["threshold", "far", "frr", "tar", "tnr"]
        empty_df = pd.DataFrame(columns=empty_curve_columns)

        return SplitEvaluationResult(
            split_name=dataset.split_name,
            metrics=metrics,
            roc_points=pd.DataFrame(columns=["threshold", "fpr", "tpr"]),
            det_points=pd.DataFrame(columns=["threshold", "far", "frr"]),
            threshold_analysis=empty_df,
            confusion_matrix=np.zeros((2, 2), dtype=int),
            labels=labels,
            scores=dataset.scores,
        )

    @staticmethod
    def _compute_eer(
        far: np.ndarray, frr: np.ndarray, thresholds: np.ndarray
    ) -> tuple[float, float]:
        """Locate the Equal Error Rate and its threshold from a FAR/FRR sweep.

        Operates entirely in the "decision score" space the sweep was
        computed in (i.e. before any sign flip for distance metrics);
        callers are responsible for converting the returned threshold
        back to the original score's units if needed.

        Args:
            far: FAR values across the sweep (same order as ``thresholds``).
            frr: FRR values across the sweep, same order.
            thresholds: Decision thresholds from
                :func:`sklearn.metrics.roc_curve`, monotonically
                decreasing.

        Returns:
            A tuple ``(eer, eer_threshold)``, both in decision-score units.
        """
        diff = far - frr
        sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]

        if len(sign_changes) == 0:
            idx = int(np.argmin(np.abs(diff)))
            eer = float((far[idx] + frr[idx]) / 2)
            return eer, float(thresholds[idx])

        idx = int(sign_changes[0])
        d1, d2 = float(diff[idx]), float(diff[idx + 1])
        t1, t2 = float(thresholds[idx]), float(thresholds[idx + 1])

        if d2 == d1:
            interp_threshold = t1
        else:
            interp_threshold = t1 + (t2 - t1) * (-d1) / (d2 - d1)

        # thresholds is monotonically decreasing; reverse for np.interp's
        # requirement that the x-coordinates be increasing.
        interp_far = float(np.interp(interp_threshold, thresholds[::-1], far[::-1]))
        interp_frr = float(np.interp(interp_threshold, thresholds[::-1], frr[::-1]))
        eer = (interp_far + interp_frr) / 2
        return eer, interp_threshold


# ------------------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlotConfig:
    """Configuration controlling plot generation and styling.

    Attributes:
        dpi: Resolution (dots per inch) for saved PNG plots.
        figure_size: ``(width, height)`` in inches for standard plots.
        style: A matplotlib style name applied globally. Falls back to
            the matplotlib default with a logged warning if unavailable.
        generate_roc: Whether to generate the ROC curve plot.
        generate_det: Whether to generate the DET curve plot.
        generate_score_distribution: Whether to generate the score
            distribution plot.
        generate_confusion_matrix: Whether to generate the confusion
            matrix plot.
        generate_threshold_curves: Whether to generate the
            threshold-vs-FAR/FRR/TAR plots.
    """

    dpi: int = 300
    figure_size: tuple[float, float] = (8.0, 6.0)
    style: str = "seaborn-v0_8-whitegrid"
    generate_roc: bool = True
    generate_det: bool = True
    generate_score_distribution: bool = True
    generate_confusion_matrix: bool = True
    generate_threshold_curves: bool = True


class PerformancePlotGenerator:
    """Generates publication-quality PNG plots for a split's evaluation results."""

    def __init__(self, plot_config: PlotConfig | None = None) -> None:
        """Initialize the plot generator and apply the configured matplotlib style.

        Args:
            plot_config: Plot styling and toggle configuration. Defaults
                to :class:`PlotConfig` defaults if not provided.
        """
        self.plot_config = plot_config or PlotConfig()
        self._apply_style()

    def _apply_style(self) -> None:
        """Apply the configured matplotlib style, falling back gracefully."""
        try:
            plt.style.use(self.plot_config.style)
        except (OSError, ValueError):
            logger.warning(
                "Matplotlib style '%s' not available; using the default style.",
                self.plot_config.style,
            )

    def generate_all(self, result: SplitEvaluationResult, output_dir: Path) -> list[Path]:
        """Generate every enabled plot type for one split's results.

        Args:
            result: The split's computed :class:`SplitEvaluationResult`.
            output_dir: Directory to save PNG files into.

        Returns:
            A list of paths to successfully generated plot files. A plot
            that fails to generate is logged and omitted rather than
            aborting the remaining plots.
        """
        ensure_dir(output_dir)
        plot_jobs: list[tuple[str, Any]] = []

        if self.plot_config.generate_roc:
            plot_jobs.append(("roc_curve", self._plot_roc))
        if self.plot_config.generate_det:
            plot_jobs.append(("det_curve", self._plot_det))
        if self.plot_config.generate_score_distribution:
            plot_jobs.append(("score_distribution", self._plot_score_distribution))
        if self.plot_config.generate_confusion_matrix:
            plot_jobs.append(("confusion_matrix", self._plot_confusion_matrix))
        if self.plot_config.generate_threshold_curves:
            plot_jobs.append(("threshold_vs_far", self._plot_threshold_vs_far))
            plot_jobs.append(("threshold_vs_frr", self._plot_threshold_vs_frr))
            plot_jobs.append(("threshold_vs_tar", self._plot_threshold_vs_tar))

        generated: list[Path] = []
        for plot_name, plot_fn in tqdm(
            plot_jobs, desc=f"Generating plots ({result.split_name})", unit="plot"
        ):
            output_path = output_dir / f"{result.split_name}_{plot_name}.png"
            try:
                plot_fn(result, output_path)
                if output_path.exists():
                    generated.append(output_path)
            except Exception as exc:  # noqa: BLE001 - one failed plot should not abort the rest
                logger.error(
                    "Failed to generate plot '%s' for split '%s': %s",
                    plot_name,
                    result.split_name,
                    exc,
                    exc_info=True,
                )
        return generated

    def _plot_roc(self, result: SplitEvaluationResult, output_path: Path) -> None:
        """Plot and save the ROC curve (FAR vs TAR)."""
        if result.roc_points.empty:
            logger.warning(
                "Skipping ROC plot for split '%s': no ROC points available.", result.split_name
            )
            return

        fig, ax = plt.subplots(figsize=self.plot_config.figure_size)
        ax.plot(
            result.roc_points["fpr"], result.roc_points["tpr"], color="#1f77b4", linewidth=2, label="ROC curve"
        )
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Chance")
        ax.set_xlabel("False Accept Rate (FAR)")
        ax.set_ylabel("True Accept Rate (TAR)")
        ax.set_title(f"ROC Curve — {result.split_name}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(output_path, dpi=self.plot_config.dpi)
        plt.close(fig)

    def _plot_det(self, result: SplitEvaluationResult, output_path: Path) -> None:
        """Plot and save the DET curve (log-scale FAR vs FRR)."""
        if result.det_points.empty:
            logger.warning(
                "Skipping DET plot for split '%s': no DET points available.", result.split_name
            )
            return

        far = np.clip(result.det_points["far"].to_numpy(), 1e-4, 1 - 1e-4)
        frr = np.clip(result.det_points["frr"].to_numpy(), 1e-4, 1 - 1e-4)

        fig, ax = plt.subplots(figsize=self.plot_config.figure_size)
        ax.plot(far, frr, color="#d62728", linewidth=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("False Accept Rate (FAR, log scale)")
        ax.set_ylabel("False Reject Rate (FRR, log scale)")
        ax.set_title(f"DET Curve — {result.split_name}")
        fig.tight_layout()
        fig.savefig(output_path, dpi=self.plot_config.dpi)
        plt.close(fig)

    def _plot_score_distribution(self, result: SplitEvaluationResult, output_path: Path) -> None:
        """Plot and save overlaid genuine/impostor score histograms."""
        genuine_scores = result.scores[result.labels == 1]
        impostor_scores = result.scores[result.labels == 0]

        fig, ax = plt.subplots(figsize=self.plot_config.figure_size)
        if genuine_scores.size:
            ax.hist(genuine_scores, bins=50, alpha=0.6, density=True, color="#2ca02c", label="Genuine")
        if impostor_scores.size:
            ax.hist(impostor_scores, bins=50, alpha=0.6, density=True, color="#d62728", label="Impostor")

        eer_threshold = result.metrics.get("eer_threshold", float("nan"))
        if not math.isnan(eer_threshold):
            ax.axvline(
                eer_threshold,
                color="black",
                linestyle="--",
                linewidth=1.5,
                label=f"EER threshold ({eer_threshold:.3f})",
            )

        ax.set_xlabel("Score")
        ax.set_ylabel("Density")
        ax.set_title(f"Score Distribution — {result.split_name}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path, dpi=self.plot_config.dpi)
        plt.close(fig)

    def _plot_confusion_matrix(self, result: SplitEvaluationResult, output_path: Path) -> None:
        """Plot and save the confusion matrix at the EER threshold."""
        confusion = result.confusion_matrix
        if confusion.sum() == 0:
            logger.warning(
                "Skipping confusion matrix plot for split '%s': matrix is empty.", result.split_name
            )
            return

        fig, ax = plt.subplots(figsize=(6.0, 5.0))
        image = ax.imshow(confusion, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Rejected", "Accepted"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Impostor (0)", "Genuine (1)"])
        ax.set_xlabel("Predicted decision")
        ax.set_ylabel("True label")
        ax.set_title(f"Confusion Matrix (at EER threshold) — {result.split_name}")

        for row in range(2):
            for col in range(2):
                ax.text(
                    col,
                    row,
                    str(int(confusion[row, col])),
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=12,
                )

        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(output_path, dpi=self.plot_config.dpi)
        plt.close(fig)

    def _plot_threshold_vs_far(self, result: SplitEvaluationResult, output_path: Path) -> None:
        """Plot and save threshold vs FAR."""
        self._plot_threshold_curve(result, output_path, "far", "False Accept Rate (FAR)", "#d62728")

    def _plot_threshold_vs_frr(self, result: SplitEvaluationResult, output_path: Path) -> None:
        """Plot and save threshold vs FRR."""
        self._plot_threshold_curve(result, output_path, "frr", "False Reject Rate (FRR)", "#1f77b4")

    def _plot_threshold_vs_tar(self, result: SplitEvaluationResult, output_path: Path) -> None:
        """Plot and save threshold vs TAR."""
        self._plot_threshold_curve(result, output_path, "tar", "True Accept Rate (TAR)", "#2ca02c")

    def _plot_threshold_curve(
        self,
        result: SplitEvaluationResult,
        output_path: Path,
        column: str,
        ylabel: str,
        color: str,
    ) -> None:
        """Shared implementation for the threshold-vs-metric line plots.

        Args:
            result: The split's computed evaluation result.
            output_path: Destination PNG path.
            column: Column name in ``result.threshold_analysis`` to plot.
            ylabel: Y-axis label and title fragment.
            color: Line color.
        """
        if result.threshold_analysis.empty or column not in result.threshold_analysis:
            logger.warning(
                "Skipping '%s vs threshold' plot for split '%s': no threshold data available.",
                ylabel,
                result.split_name,
            )
            return

        fig, ax = plt.subplots(figsize=self.plot_config.figure_size)
        ax.plot(
            result.threshold_analysis["threshold"],
            result.threshold_analysis[column],
            color=color,
            linewidth=2,
        )

        eer_threshold = result.metrics.get("eer_threshold", float("nan"))
        if not math.isnan(eer_threshold):
            ax.axvline(
                eer_threshold,
                color="black",
                linestyle="--",
                linewidth=1,
                label=f"EER threshold ({eer_threshold:.3f})",
            )
            ax.legend()

        ax.set_xlabel("Decision threshold")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs Threshold — {result.split_name}")
        fig.tight_layout()
        fig.savefig(output_path, dpi=self.plot_config.dpi)
        plt.close(fig)


# ------------------------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------------------------


class PerformanceEvaluationPipeline:
    """End-to-end orchestration for biometric performance evaluation.

    Loads per-split score CSVs, computes verification metrics and curve
    data, generates plots, and writes all output artifacts under a
    results directory.
    """

    _SPLIT_TO_SCORE_FILENAME: dict[str, str] = {
        "train": "train_scores.csv",
        "validation": "validation_scores.csv",
        "test": "test_scores.csv",
    }

    def __init__(
        self,
        scores_dir: Path,
        output_dir: Path,
        score_column: str = ScoreOutputColumns.COSINE_SIMILARITY,
        threshold_config: ThresholdConfig | None = None,
        plot_config: PlotConfig | None = None,
    ) -> None:
        """Initialize the pipeline.

        Args:
            scores_dir: Directory containing ``train_scores.csv``,
                ``validation_scores.csv``, and ``test_scores.csv``, as
                produced by :mod:`fairness_fr.evaluation.calculate_scores`.
            output_dir: Directory to write metrics, curve CSVs, and
                plots to, typically ``results/``.
            score_column: Which score column to evaluate.
            threshold_config: Reserved for future threshold-strategy
                configuration.
            plot_config: Plot styling and toggle configuration.
        """
        self.scores_dir = scores_dir
        self.output_dir = ensure_dir(output_dir)
        self.loader = ScoreDataLoader(score_column=score_column)
        self.calculator = PerformanceMetricsCalculator(threshold_config=threshold_config)
        self.plot_generator = PerformancePlotGenerator(plot_config=plot_config)
        self.results: dict[str, SplitEvaluationResult] = {}

    def run(self, splits: list[str] | None = None) -> dict[str, SplitEvaluationResult]:
        """Evaluate every requested split and write all output artifacts.

        Args:
            splits: Subset of ``{"train", "validation", "test"}`` to
                evaluate. Defaults to all three.

        Returns:
            A dict mapping split name to its :class:`SplitEvaluationResult`,
            for splits that were successfully evaluated.
        """
        split_names = splits or list(self._SPLIT_TO_SCORE_FILENAME.keys())
        logger.info("Starting performance evaluation for splits: %s", split_names)

        with timer("Performance evaluation (all splits)"):
            for split_name in tqdm(split_names, desc="Evaluating splits", unit="split"):
                self._evaluate_split(split_name)

        if not self.results:
            raise PerformanceEvaluationError(
                "No splits were successfully evaluated; check earlier log messages for details."
            )

        self._save_all_outputs()
        logger.info("Performance evaluation finished for %d split(s).", len(self.results))
        return self.results

    def _evaluate_split(self, split_name: str) -> None:
        """Load, evaluate, and generate plots for one split, logging any failure.

        Args:
            split_name: Name of the split to evaluate.
        """
        if split_name not in self._SPLIT_TO_SCORE_FILENAME:
            logger.error("Unknown split '%s'; skipping.", split_name)
            return

        score_csv_path = self.scores_dir / self._SPLIT_TO_SCORE_FILENAME[split_name]

        try:
            dataset = self.loader.load(score_csv_path, split_name)
        except (FileNotFoundError, PerformanceEvaluationError) as exc:
            logger.error("Skipping split '%s': %s", split_name, exc)
            return

        logger.info("Computing metrics for split '%s' (%d pairs).", split_name, len(dataset.labels))
        result = self.calculator.compute(dataset)
        self.results[split_name] = result

        plots_dir = self.output_dir / "plots"
        generated_plots = self.plot_generator.generate_all(result, plots_dir)
        logger.info("Generated %d plot(s) for split '%s'.", len(generated_plots), split_name)

    def _save_all_outputs(self) -> None:
        """Write all combined output files across every evaluated split."""
        self._save_performance_metrics_json()
        self._save_performance_metrics_csv()
        self._save_curve_csv("threshold_analysis.csv", "threshold_analysis", ["threshold", "far", "frr", "tar", "tnr"])
        self._save_curve_csv("roc_points.csv", "roc_points", ["threshold", "fpr", "tpr"])
        self._save_curve_csv("det_points.csv", "det_points", ["threshold", "far", "frr"])

    def _save_performance_metrics_json(self) -> None:
        """Write ``performance_metrics.json``, keyed by split name."""
        output_path = self.output_dir / "performance_metrics.json"
        payload = {split_name: result.metrics for split_name, result in self.results.items()}
        with output_path.open("w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=2, sort_keys=True, default=_json_safe)
        logger.info("Saved performance metrics JSON to %s", output_path)

    def _save_performance_metrics_csv(self) -> None:
        """Write ``performance_metrics.csv``, one row per split."""
        output_path = self.output_dir / "performance_metrics.csv"
        rows = [{"split": split_name, **result.metrics} for split_name, result in self.results.items()]
        pd.DataFrame(rows).to_csv(output_path, index=False)
        logger.info("Saved performance metrics CSV to %s", output_path)

    def _save_curve_csv(self, filename: str, attribute_name: str, empty_columns: list[str]) -> None:
        """Write one combined curve-data CSV across all evaluated splits.

        Args:
            filename: Output filename, e.g. ``"roc_points.csv"``.
            attribute_name: Name of the :class:`SplitEvaluationResult`
                attribute holding the per-split DataFrame to combine.
            empty_columns: Column names to use if no splits produced any rows.
        """
        output_path = self.output_dir / filename
        frames: list[pd.DataFrame] = []
        for split_name, result in self.results.items():
            frame = getattr(result, attribute_name).copy()
            frame.insert(0, "split", split_name)
            frames.append(frame)

        combined = (
            pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["split", *empty_columns])
        )
        combined.to_csv(output_path, index=False)
        logger.info("Saved %s to %s", filename, output_path)


# ------------------------------------------------------------------------------
# Command-line interface
# ------------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for this module.

    Returns:
        A configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate biometric verification performance from similarity scores."
    )
    parser.add_argument(
        "--scores-dir",
        type=Path,
        required=True,
        help="Directory containing train/validation/test score CSVs from calculate_scores.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write metrics, curve CSVs, and plots to.",
    )
    parser.add_argument(
        "--score-column",
        type=str,
        default=ScoreOutputColumns.COSINE_SIMILARITY,
        choices=list(ScoreOutputColumns.METRIC_COLUMNS),
        help="Which score column to threshold on.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=None,
        choices=["train", "validation", "test"],
        help="Subset of splits to evaluate (default: train validation test).",
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved plots.")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point for biometric performance evaluation.

    Args:
        argv: Optional explicit argument list (primarily for testing).
            Defaults to ``sys.argv`` when None.

    Returns:
        Process exit code: ``0`` on success, ``1`` on failure.
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    setup_logging(level=args.log_level, force=True)

    try:
        pipeline = PerformanceEvaluationPipeline(
            scores_dir=args.scores_dir,
            output_dir=args.output_dir,
            score_column=args.score_column,
            plot_config=PlotConfig(dpi=args.dpi),
        )
        pipeline.run(splits=args.splits)
    except Exception as exc:  # noqa: BLE001 - top-level CLI guard
        logger.error("Performance evaluation failed: %s", exc, exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
