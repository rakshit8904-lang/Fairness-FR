"""Demographic fairness evaluation for the face recognition pipeline.

Combines ``results/test_scores.csv`` (from
:mod:`fairness_fr.evaluation.calculate_scores`),
``results/performance_metrics.csv`` (from
:mod:`fairness_fr.evaluation.evaluate_performance`), and a dataset's
``metadata.csv`` (from :mod:`fairness_fr.data.preprocess`) to measure
how verification performance varies across demographic groups —
ethnicity, gender, age group, and any additional attribute column a
metadata file happens to contain (e.g. skin tone).

Outputs:

- ``fairness_metrics.csv`` / ``fairness_metrics.json`` — full per-group
  metric breakdown for every demographic attribute discovered.
- ``fairness_disparity.csv`` — max/min/range/mean/std across groups for
  each of the five core fairness metrics (FAR, FRR, TAR, accuracy, EER),
  per attribute.
- ``fairness_summary.csv`` — a single-row overall summary (best/worst
  group, largest/smallest disparity, mean disparity, a plain-language
  ``summary_text`` field, and a count of excluded groups).
- Per-attribute PNG plots under ``plots/fairness/``.

Design notes:
    - Demographic attributes are discovered dynamically from
      ``metadata.csv``: every column other than ``image_path`` and
      ``identity`` is treated as a demographic attribute, so ethnicity
      (``group``), ``gender``, ``age_group``, and any dataset-specific
      extra column (e.g. ``skin_tone``) are all evaluated without code
      changes.
    - A pair only has a well-defined "group" when both its images share
      the same attribute value; cross-group pairs (common among
      impostor pairs) are excluded from that attribute's per-group
      evaluation, which mirrors standard within-group verification
      protocols used in fairness benchmarks such as RFW.
    - Group-wise accuracy/precision/recall/F1/FAR/FRR/FMR/FNMR/TAR/TNR
      are computed at the single, fixed decision threshold recorded for
      the overall test set in ``performance_metrics.csv`` (its EER
      threshold) — this is what actually exposes demographic
      differentials, since a single deployed threshold is what real
      systems apply uniformly. Each group's *own* EER (and the
      threshold that would achieve it) is also reported separately as a
      diagnostic value, reusing the same sweep-and-interpolate approach
      as :mod:`fairness_fr.evaluation.evaluate_performance` for
      consistency.
    - Groups below a configurable minimum sample size, or containing
      only one label class, are excluded from cross-group comparison
      (with NaN metrics retained in the full output for transparency)
      rather than silently distorting disparity statistics with noisy
      small-sample estimates.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")  # ensure headless rendering in any environment

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve
from tqdm import tqdm

from fairness_fr.config.constants import MetadataColumns, MetricName
from fairness_fr.data.generate_pairs import PairOutputColumns
from fairness_fr.evaluation.calculate_scores import ScoreOutputColumns
from fairness_fr.evaluation.evaluate_performance import (
    _confusion_counts_at_threshold,
    _json_safe,
    _safe_f1,
)
from fairness_fr.utils.logging import get_logger
from fairness_fr.utils import ensure_dir, safe_divide, timer

logger = get_logger(__name__)


class FairnessEvaluationError(Exception):
    """Raised when demographic fairness evaluation cannot be completed."""


#: Score columns that represent a *distance* (lower = more similar).
_DISTANCE_SCORE_COLUMNS: frozenset[str] = frozenset(
    {ScoreOutputColumns.EUCLIDEAN_DISTANCE, ScoreOutputColumns.COSINE_DISTANCE}
)

#: The five metrics cross-group disparity is reported for.
_CORE_DISPARITY_METRICS: tuple[str, ...] = ("far", "frr", "tar", "accuracy", "eer")

#: Absolute disparity range above which a metric is flagged as a fairness concern.
_DISPARITY_CONCERN_THRESHOLD: float = 0.10


def _compute_eer_from_sweep(
    far: np.ndarray, frr: np.ndarray, thresholds: np.ndarray
) -> tuple[float, float]:
    """Locate the Equal Error Rate and its threshold from a FAR/FRR sweep.

    Mirrors the sign-change-and-interpolate approach used for the
    overall population EER, with one correction: candidate thresholds
    are restricted to finite values before searching for a crossing.
    :func:`sklearn.metrics.roc_curve` always prepends a non-finite
    sentinel threshold (``max(score) + 1``, surfaced as ``inf`` after
    any sign flip for distance metrics) so that FPR/TPR start at
    exactly zero; if the EER crossing happens to fall in the first
    bracket adjacent to that sentinel, naively interpolating against it
    produces a non-finite threshold. Dropping the sentinel before
    searching avoids that failure mode while leaving every real
    threshold and its FAR/FRR values untouched.

    Args:
        far: FAR values across the sweep (same order as ``thresholds``).
        frr: FRR values across the sweep, same order.
        thresholds: Decision thresholds from
            :func:`sklearn.metrics.roc_curve`, monotonically decreasing.

    Returns:
        A tuple ``(eer, eer_threshold)``, both in decision-score units.
    """
    finite_mask = np.isfinite(thresholds)
    if finite_mask.sum() < 2:
        index = int(np.argmin(np.abs(far - frr)))
        return float((far[index] + frr[index]) / 2), float(thresholds[index])

    far_f = far[finite_mask]
    frr_f = frr[finite_mask]
    thresholds_f = thresholds[finite_mask]

    diff = far_f - frr_f
    sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]

    if len(sign_changes) == 0:
        index = int(np.argmin(np.abs(diff)))
        return float((far_f[index] + frr_f[index]) / 2), float(thresholds_f[index])

    index = int(sign_changes[0])
    d1, d2 = float(diff[index]), float(diff[index + 1])
    t1, t2 = float(thresholds_f[index]), float(thresholds_f[index + 1])

    interp_threshold = t1 if d2 == d1 else t1 + (t2 - t1) * (-d1) / (d2 - d1)

    interp_far = float(np.interp(interp_threshold, thresholds_f[::-1], far_f[::-1]))
    interp_frr = float(np.interp(interp_threshold, thresholds_f[::-1], frr_f[::-1]))
    eer = (interp_far + interp_frr) / 2
    return eer, interp_threshold


# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FairnessEvaluationConfig:
    """Configuration controlling fairness evaluation behavior.

    Attributes:
        score_column: Which score column from ``test_scores.csv`` to
            evaluate. Must match the column the overall EER threshold in
            ``performance_metrics.csv`` was computed from.
        min_group_size: Minimum number of same-group pairs required for
            a demographic group to be included in comparison; smaller
            groups are still recorded (with NaN metrics) but excluded
            from disparity statistics.
        base_metadata_columns: Structural metadata columns that are
            never treated as demographic attributes. Every other column
            in metadata.csv is discovered dynamically as an attribute.
    """

    score_column: str = ScoreOutputColumns.COSINE_SIMILARITY
    min_group_size: int = 20
    base_metadata_columns: frozenset[str] = field(
        default_factory=lambda: frozenset({MetadataColumns.IMAGE_PATH, MetadataColumns.IDENTITY})
    )


@dataclass(frozen=True, slots=True)
class FairnessPlotConfig:
    """Configuration controlling fairness plot generation and styling.

    Attributes:
        dpi: Resolution (dots per inch) for saved PNG plots.
        figure_size: ``(width, height)`` in inches for standard plots.
        style: A matplotlib style name applied globally.
        generate_bar_charts: Whether to generate the five per-metric
            bar-by-group charts (FAR, FRR, TAR, Accuracy, EER).
        generate_boxplots: Whether to generate the score boxplot.
        generate_distributions: Whether to generate the score
            distribution (KDE) plot.
        generate_heatmap: Whether to generate the group-by-metric heatmap.
        generate_comparison_chart: Whether to generate the combined
            multi-metric fairness comparison chart.
    """

    dpi: int = 300
    figure_size: tuple[float, float] = (10.0, 6.0)
    style: str = "seaborn-v0_8-whitegrid"
    generate_bar_charts: bool = True
    generate_boxplots: bool = True
    generate_distributions: bool = True
    generate_heatmap: bool = True
    generate_comparison_chart: bool = True


# ------------------------------------------------------------------------------
# Metadata loading
# ------------------------------------------------------------------------------


class MetadataRepository:
    """Loads metadata.csv and exposes per-image demographic attribute lookups.

    Every column beyond the structural (``image_path``, ``identity``)
    columns is treated as a demographic attribute, so any additional
    column present in a given dataset's metadata (skin tone, dataset
    source, etc.) is automatically discovered without code changes.
    """

    def __init__(self, metadata_csv: Path, base_columns: frozenset[str]) -> None:
        """Load and index metadata.csv.

        Args:
            metadata_csv: Path to the dataset's metadata.csv, as
                produced by :mod:`fairness_fr.data.preprocess`.
            base_columns: Structural columns excluded from demographic
                attribute discovery.

        Raises:
            FileNotFoundError: If ``metadata_csv`` does not exist.
            FairnessEvaluationError: If the file is empty, is missing
                the image path column, or has no demographic attribute
                columns beyond ``base_columns``.
        """
        self.metadata_csv = metadata_csv
        self.base_columns = base_columns
        self.attribute_columns: list[str] = []
        self.attribute_maps: dict[str, dict[str, str]] = {}
        self._load()

    def _load(self) -> None:
        """Read metadata.csv and build per-attribute image-path lookup maps."""
        if not self.metadata_csv.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {self.metadata_csv}. "
                f"Run dataset preprocessing before fairness evaluation."
            )

        logger.info("Loading metadata from %s", self.metadata_csv)
        frame = pd.read_csv(self.metadata_csv, dtype=str, keep_default_na=False)

        if frame.empty:
            raise FairnessEvaluationError(f"Metadata file is empty: {self.metadata_csv}.")

        if MetadataColumns.IMAGE_PATH not in frame.columns:
            raise FairnessEvaluationError(
                f"Metadata file is missing required column "
                f"'{MetadataColumns.IMAGE_PATH}': {self.metadata_csv}."
            )

        self.attribute_columns = [
            column for column in frame.columns if column not in self.base_columns
        ]
        if not self.attribute_columns:
            raise FairnessEvaluationError(
                f"No demographic attribute columns found in metadata: {self.metadata_csv}."
            )

        logger.info("Discovered demographic attributes in metadata: %s", self.attribute_columns)

        indexed = frame.set_index(MetadataColumns.IMAGE_PATH)
        for attribute in self.attribute_columns:
            self.attribute_maps[attribute] = {
                path: (value if value else None) for path, value in indexed[attribute].items()
            }


# ------------------------------------------------------------------------------
# Score / threshold loading
# ------------------------------------------------------------------------------


class TestScoresLoader:
    """Loads and validates ``results/test_scores.csv`` for fairness evaluation."""

    def __init__(self, score_column: str) -> None:
        """Initialize the loader.

        Args:
            score_column: Which score column will be evaluated.
        """
        self.score_column = score_column

    def load(self, csv_path: Path) -> pd.DataFrame:
        """Load and validate the test score CSV.

        Args:
            csv_path: Path to ``test_scores.csv``.

        Returns:
            A validated DataFrame with binary integer labels and a
            float score column.

        Raises:
            FileNotFoundError: If ``csv_path`` does not exist.
            FairnessEvaluationError: If the file is empty, is missing
                required columns, or has no valid rows after validation.
        """
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Test scores file not found: {csv_path}. Run calculate_scores.py first."
            )

        logger.info("Loading test scores from %s", csv_path)
        frame = pd.read_csv(csv_path)

        if frame.empty:
            raise FairnessEvaluationError(f"Test scores file is empty: {csv_path}.")

        required_columns = {
            PairOutputColumns.IMAGE1,
            PairOutputColumns.IMAGE2,
            PairOutputColumns.LABEL,
            self.score_column,
        }
        missing_columns = required_columns - set(frame.columns)
        if missing_columns:
            raise FairnessEvaluationError(
                f"Test scores file is missing required columns: {sorted(missing_columns)}."
            )

        valid_label_mask = frame[PairOutputColumns.LABEL].isin([0, 1])
        valid_score_mask = pd.to_numeric(frame[self.score_column], errors="coerce").notna()
        valid_mask = valid_label_mask & valid_score_mask

        dropped = int((~valid_mask).sum())
        if dropped:
            logger.warning(
                "Dropping %d rows with invalid labels or scores from test scores.", dropped
            )
        frame = frame.loc[valid_mask].copy()

        if frame.empty:
            raise FairnessEvaluationError("No valid rows remain in test scores after validation.")

        frame[PairOutputColumns.LABEL] = frame[PairOutputColumns.LABEL].astype(int)
        frame[self.score_column] = frame[self.score_column].astype(float)

        logger.info("Loaded %d valid test pairs.", len(frame))
        return frame.reset_index(drop=True)


class OverallThresholdLoader:
    """Loads the overall (population-level) EER threshold from performance_metrics.csv."""

    def __init__(self, split_name: str = "test") -> None:
        """Initialize the loader.

        Args:
            split_name: Which split's row to read the threshold from.
        """
        self.split_name = split_name

    def load(self, csv_path: Path) -> float:
        """Load the overall EER threshold for the configured split.

        Args:
            csv_path: Path to ``performance_metrics.csv``.

        Returns:
            The overall EER decision threshold, in the score column's units.

        Raises:
            FileNotFoundError: If ``csv_path`` does not exist.
            FairnessEvaluationError: If required columns or the target
                split's row are missing, or the threshold is NaN.
        """
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Performance metrics file not found: {csv_path}. "
                f"Run evaluate_performance.py before fairness evaluation."
            )

        frame = pd.read_csv(csv_path)
        if "split" not in frame.columns or "eer_threshold" not in frame.columns:
            raise FairnessEvaluationError(
                f"Performance metrics file is missing required columns "
                f"'split'/'eer_threshold': {csv_path}."
            )

        matching_rows = frame.loc[frame["split"] == self.split_name]
        if matching_rows.empty:
            raise FairnessEvaluationError(
                f"No row for split '{self.split_name}' found in {csv_path}."
            )

        threshold = float(matching_rows.iloc[0]["eer_threshold"])
        if math.isnan(threshold):
            raise FairnessEvaluationError(
                f"Overall EER threshold for split '{self.split_name}' is NaN in {csv_path}; "
                f"cannot evaluate fairness at a fixed threshold."
            )

        logger.info("Loaded overall EER threshold for split '%s': %.6f", self.split_name, threshold)
        return threshold


# ------------------------------------------------------------------------------
# Group evaluation
# ------------------------------------------------------------------------------


@dataclass
class GroupEvaluationResult:
    """Verification metrics computed for one demographic group under one attribute.

    Attributes:
        attribute: Name of the demographic attribute (e.g. ``"gender"``).
        group_value: The specific group value (e.g. ``"female"``).
        sample_size: Number of same-group pairs evaluated.
        genuine_pairs: Number of genuine pairs in this group.
        impostor_pairs: Number of impostor pairs in this group.
        metrics: Dict of computed metrics (accuracy, precision, recall,
            f1_score, far, frr, fmr, fnmr, tar, tnr, eer, eer_threshold).
        labels: Label array for this group, retained for plotting.
        scores: Score array for this group, retained for plotting.
        excluded: Whether this group was excluded from cross-group
            comparison due to insufficient sample size or a single-class label set.
        exclusion_reason: Human-readable reason, populated when ``excluded`` is True.
    """

    attribute: str
    group_value: str
    sample_size: int
    genuine_pairs: int
    impostor_pairs: int
    metrics: dict[str, float]
    labels: np.ndarray
    scores: np.ndarray
    excluded: bool = False
    exclusion_reason: str = ""


class FairnessMetricsCalculator:
    """Computes per-demographic-group verification metrics at a fixed threshold."""

    def __init__(self, config: FairnessEvaluationConfig, overall_threshold: float) -> None:
        """Initialize the calculator.

        Args:
            config: Fairness evaluation configuration.
            overall_threshold: The fixed decision threshold (from the
                overall test-set EER) applied uniformly across every group.
        """
        self.config = config
        self.overall_threshold = overall_threshold
        self.higher_is_better = config.score_column not in _DISTANCE_SCORE_COLUMNS

    def evaluate_attribute(
        self,
        scores_frame: pd.DataFrame,
        metadata_repo: MetadataRepository,
        attribute: str,
    ) -> list[GroupEvaluationResult]:
        """Evaluate every group value of one demographic attribute.

        Args:
            scores_frame: Validated test score DataFrame.
            metadata_repo: Loaded metadata repository providing
                per-image attribute lookups.
            attribute: Name of the demographic attribute to evaluate.

        Returns:
            A list of :class:`GroupEvaluationResult`, one per group
            value with at least one same-group pair. Returns an empty
            list if no same-group pairs exist for this attribute.
        """
        attribute_map = metadata_repo.attribute_maps[attribute]
        attribute1 = scores_frame[PairOutputColumns.IMAGE1].map(attribute_map)
        attribute2 = scores_frame[PairOutputColumns.IMAGE2].map(attribute_map)

        both_known = attribute1.notna() & attribute2.notna()
        same_group_mask = both_known & (attribute1 == attribute2)

        missing_count = int((~both_known).sum())
        cross_group_count = int((both_known & (attribute1 != attribute2)).sum())
        if missing_count:
            logger.warning(
                "Attribute '%s': %d pairs missing this attribute for at least one image.",
                attribute,
                missing_count,
            )
        if cross_group_count:
            logger.info(
                "Attribute '%s': excluding %d cross-group pairs from group-wise evaluation.",
                attribute,
                cross_group_count,
            )

        subset = scores_frame.loc[same_group_mask].copy()
        if subset.empty:
            logger.warning(
                "Attribute '%s': no same-group pairs available; skipping this attribute.", attribute
            )
            return []

        subset["_group_value"] = attribute1.loc[same_group_mask]

        results: list[GroupEvaluationResult] = []
        group_values = sorted(subset["_group_value"].unique().tolist())
        for group_value in tqdm(group_values, desc=f"Evaluating '{attribute}' groups", unit="group"):
            group_frame = subset.loc[subset["_group_value"] == group_value]
            results.append(self._evaluate_group(attribute, str(group_value), group_frame))
        return results

    def _evaluate_group(
        self, attribute: str, group_value: str, group_frame: pd.DataFrame
    ) -> GroupEvaluationResult:
        """Compute metrics for a single demographic group's pairs.

        Args:
            attribute: Name of the demographic attribute.
            group_value: The group's value.
            group_frame: The subset of scored pairs belonging to this group.

        Returns:
            A :class:`GroupEvaluationResult`, with ``excluded=True`` and
            NaN metrics if the group is too small or single-class.
        """
        labels = group_frame[PairOutputColumns.LABEL].to_numpy()
        scores = group_frame[self.config.score_column].to_numpy()
        sample_size = len(labels)
        genuine_count = int(np.sum(labels == 1))
        impostor_count = int(np.sum(labels == 0))

        if sample_size < self.config.min_group_size:
            reason = f"sample size {sample_size} is below the minimum of {self.config.min_group_size}"
            logger.warning("Group '%s=%s': %s; excluding from comparison.", attribute, group_value, reason)
            return GroupEvaluationResult(
                attribute=attribute,
                group_value=group_value,
                sample_size=sample_size,
                genuine_pairs=genuine_count,
                impostor_pairs=impostor_count,
                metrics=self._nan_metrics(),
                labels=labels,
                scores=scores,
                excluded=True,
                exclusion_reason=reason,
            )

        if len({0, 1} & set(np.unique(labels).tolist())) < 2:
            reason = "single-class group (only genuine or only impostor pairs)"
            logger.warning("Group '%s=%s': %s; excluding from comparison.", attribute, group_value, reason)
            return GroupEvaluationResult(
                attribute=attribute,
                group_value=group_value,
                sample_size=sample_size,
                genuine_pairs=genuine_count,
                impostor_pairs=impostor_count,
                metrics=self._nan_metrics(),
                labels=labels,
                scores=scores,
                excluded=True,
                exclusion_reason=reason,
            )

        true_accepts, false_accepts, true_rejects, false_rejects = _confusion_counts_at_threshold(
            labels, scores, self.overall_threshold, self.higher_is_better
        )
        total = true_accepts + false_accepts + true_rejects + false_rejects
        total_genuine = true_accepts + false_rejects
        total_impostor = false_accepts + true_rejects

        accuracy = safe_divide(true_accepts + true_rejects, total, default=float("nan"))
        precision = safe_divide(true_accepts, true_accepts + false_accepts, default=float("nan"))
        recall = safe_divide(true_accepts, total_genuine, default=float("nan"))
        f1_score = _safe_f1(precision, recall)
        far = safe_divide(false_accepts, total_impostor, default=float("nan"))
        frr = safe_divide(false_rejects, total_genuine, default=float("nan"))
        tar = safe_divide(true_accepts, total_genuine, default=float("nan"))
        tnr = safe_divide(true_rejects, total_impostor, default=float("nan"))

        eer, eer_threshold = self._compute_group_eer(labels, scores)

        metrics: dict[str, float] = {
            MetricName.ACCURACY.value: accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            MetricName.FAR.value: far,
            MetricName.FRR.value: frr,
            MetricName.FMR.value: far,
            MetricName.FNMR.value: frr,
            MetricName.TAR.value: tar,
            "tnr": tnr,
            MetricName.EER.value: eer,
            "eer_threshold": eer_threshold,
        }

        return GroupEvaluationResult(
            attribute=attribute,
            group_value=group_value,
            sample_size=sample_size,
            genuine_pairs=genuine_count,
            impostor_pairs=impostor_count,
            metrics=metrics,
            labels=labels,
            scores=scores,
        )

    def _compute_group_eer(self, labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
        """Compute a group's own EER and threshold via its own ROC sweep.

        Uses the same sweep-and-interpolate approach as
        :mod:`fairness_fr.evaluation.evaluate_performance` (see
        :func:`_compute_eer_from_sweep`) so group-level and
        population-level EER are computed identically.

        Args:
            labels: The group's label array.
            scores: The group's score array.

        Returns:
            A tuple ``(eer, eer_threshold)`` in the original score units.
        """
        decision_scores = scores if self.higher_is_better else -scores
        fpr, tpr, raw_thresholds = roc_curve(labels, decision_scores)
        far = fpr
        frr = 1.0 - tpr

        eer, eer_threshold_decision = _compute_eer_from_sweep(far, frr, raw_thresholds)
        eer_threshold = (
            float(eer_threshold_decision) if self.higher_is_better else float(-eer_threshold_decision)
        )
        return eer, eer_threshold

    @staticmethod
    def _nan_metrics() -> dict[str, float]:
        """Return an all-NaN metrics dict for excluded groups.

        Returns:
            A dict with the same keys as a normal metrics result, all
            set to NaN.
        """
        keys = (
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "far",
            "frr",
            "fmr",
            "fnmr",
            "tar",
            "tnr",
            "eer",
            "eer_threshold",
        )
        return {key: float("nan") for key in keys}


# ------------------------------------------------------------------------------
# Disparity statistics
# ------------------------------------------------------------------------------


class FairnessDisparityCalculator:
    """Computes cross-group disparity statistics for the core fairness metrics."""

    @staticmethod
    def compute(results: list[GroupEvaluationResult]) -> pd.DataFrame:
        """Compute max/min/range/mean/std across groups for each core metric.

        Args:
            results: Per-group results for a single demographic attribute.

        Returns:
            A DataFrame with one row per metric in
            :data:`_CORE_DISPARITY_METRICS`, columns
            ``metric, max_value, max_group, min_value, min_group, range, mean, std``.
            Excluded groups and NaN values are ignored.
        """
        included = [result for result in results if not result.excluded]

        rows: list[dict[str, Any]] = []
        for metric_name in _CORE_DISPARITY_METRICS:
            pairs = [
                (result.group_value, result.metrics.get(metric_name, float("nan")))
                for result in included
            ]
            pairs = [(group, value) for group, value in pairs if not math.isnan(value)]

            if not pairs:
                rows.append(
                    {
                        "metric": metric_name,
                        "max_value": float("nan"),
                        "max_group": "",
                        "min_value": float("nan"),
                        "min_group": "",
                        "range": float("nan"),
                        "mean": float("nan"),
                        "std": float("nan"),
                    }
                )
                continue

            groups, values = zip(*pairs)
            values_array = np.asarray(values, dtype=float)
            max_index = int(np.argmax(values_array))
            min_index = int(np.argmin(values_array))

            rows.append(
                {
                    "metric": metric_name,
                    "max_value": float(values_array[max_index]),
                    "max_group": groups[max_index],
                    "min_value": float(values_array[min_index]),
                    "min_group": groups[min_index],
                    "range": float(values_array[max_index] - values_array[min_index]),
                    "mean": float(np.mean(values_array)),
                    "std": float(np.std(values_array)),
                }
            )

        return pd.DataFrame(rows)


class FairnessSummaryGenerator:
    """Produces a concise textual and tabular fairness summary."""

    @staticmethod
    def generate(
        all_results: list[GroupEvaluationResult], disparity_frame: pd.DataFrame
    ) -> tuple[pd.DataFrame, str]:
        """Build the single-row fairness summary and its plain-language text.

        Args:
            all_results: Every group result across every evaluated attribute.
            disparity_frame: Combined disparity statistics across all attributes.

        Returns:
            A tuple ``(summary_dataframe, summary_text)``.
        """
        included = [
            result
            for result in all_results
            if not result.excluded and not math.isnan(result.metrics.get("accuracy", float("nan")))
        ]
        excluded_count = sum(1 for result in all_results if result.excluded)

        if not included:
            summary_text = "No demographic groups had sufficient data for fairness comparison."
            summary_df = pd.DataFrame(
                [
                    {
                        "best_group": "",
                        "best_accuracy": float("nan"),
                        "worst_group": "",
                        "worst_accuracy": float("nan"),
                        "largest_disparity_metric": "",
                        "largest_disparity_value": float("nan"),
                        "smallest_disparity_metric": "",
                        "smallest_disparity_value": float("nan"),
                        "mean_disparity_across_metrics": float("nan"),
                        "excluded_group_count": excluded_count,
                        "summary_text": summary_text,
                    }
                ]
            )
            return summary_df, summary_text

        ranked_by_accuracy = sorted(included, key=lambda result: result.metrics["accuracy"], reverse=True)
        best = ranked_by_accuracy[0]
        worst = ranked_by_accuracy[-1]

        valid_disparities = disparity_frame.dropna(subset=["range"]) if not disparity_frame.empty else disparity_frame
        largest_row = None
        smallest_row = None
        mean_disparity = float("nan")

        if valid_disparities is not None and not valid_disparities.empty:
            largest_row = valid_disparities.loc[valid_disparities["range"].idxmax()]
            smallest_row = valid_disparities.loc[valid_disparities["range"].idxmin()]
            mean_disparity = float(valid_disparities["range"].mean())

        concerns: list[str] = []
        if valid_disparities is not None and not valid_disparities.empty:
            concerning = valid_disparities.loc[valid_disparities["range"] > _DISPARITY_CONCERN_THRESHOLD]
            for _, row in concerning.iterrows():
                concerns.append(
                    f"{str(row['metric']).upper()} disparity of {row['range']:.3f} between "
                    f"'{row['max_group']}' and '{row['min_group']}' exceeds the "
                    f"{_DISPARITY_CONCERN_THRESHOLD:.2f} concern threshold."
                )
        if not concerns:
            concerns.append(
                f"No metric disparity exceeded the {_DISPARITY_CONCERN_THRESHOLD:.2f} concern threshold."
            )

        text_parts = [
            f"Best-performing group: '{best.attribute}={best.group_value}' "
            f"(accuracy={best.metrics['accuracy']:.4f}, EER={best.metrics['eer']:.4f}).",
            f"Worst-performing group: '{worst.attribute}={worst.group_value}' "
            f"(accuracy={worst.metrics['accuracy']:.4f}, EER={worst.metrics['eer']:.4f}).",
        ]
        if largest_row is not None:
            text_parts.append(
                f"Largest disparity: {str(largest_row['metric']).upper()} "
                f"(range={largest_row['range']:.4f}, between '{largest_row['max_group']}' "
                f"and '{largest_row['min_group']}')."
            )
        if smallest_row is not None:
            text_parts.append(
                f"Smallest disparity: {str(smallest_row['metric']).upper()} "
                f"(range={smallest_row['range']:.4f})."
            )
        text_parts.append(f"Mean disparity across core metrics: {mean_disparity:.4f}.")
        text_parts.extend(concerns)
        if excluded_count:
            text_parts.append(
                f"{excluded_count} group(s) excluded from comparison due to insufficient data."
            )

        summary_text = " ".join(text_parts)

        summary_df = pd.DataFrame(
            [
                {
                    "best_group": f"{best.attribute}={best.group_value}",
                    "best_accuracy": best.metrics["accuracy"],
                    "worst_group": f"{worst.attribute}={worst.group_value}",
                    "worst_accuracy": worst.metrics["accuracy"],
                    "largest_disparity_metric": largest_row["metric"] if largest_row is not None else "",
                    "largest_disparity_value": (
                        largest_row["range"] if largest_row is not None else float("nan")
                    ),
                    "smallest_disparity_metric": (
                        smallest_row["metric"] if smallest_row is not None else ""
                    ),
                    "smallest_disparity_value": (
                        smallest_row["range"] if smallest_row is not None else float("nan")
                    ),
                    "mean_disparity_across_metrics": mean_disparity,
                    "excluded_group_count": excluded_count,
                    "summary_text": summary_text,
                }
            ]
        )

        return summary_df, summary_text


# ------------------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------------------


class FairnessPlotGenerator:
    """Generates publication-quality PNG plots comparing demographic groups."""

    _BAR_METRICS: tuple[tuple[str, str], ...] = (
        ("far", "FAR"),
        ("frr", "FRR"),
        ("tar", "TAR"),
        ("accuracy", "Accuracy"),
        ("eer", "EER"),
    )
    _HEATMAP_METRICS: tuple[str, ...] = ("accuracy", "far", "frr", "tar", "eer")

    def __init__(self, plot_config: FairnessPlotConfig | None = None) -> None:
        """Initialize the plot generator and apply the configured matplotlib style.

        Args:
            plot_config: Plot styling and toggle configuration. Defaults
                to :class:`FairnessPlotConfig` defaults if not provided.
        """
        self.plot_config = plot_config or FairnessPlotConfig()
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

    def generate_attribute_plots(
        self, attribute: str, results: list[GroupEvaluationResult], output_dir: Path
    ) -> list[Path]:
        """Generate every enabled plot type for one demographic attribute.

        Args:
            attribute: Name of the demographic attribute (e.g. ``"gender"``).
            results: Per-group results for this attribute (excluded groups
                are skipped for plotting purposes).
            output_dir: Directory to save PNG files into.

        Returns:
            A list of paths to successfully generated plot files.
        """
        ensure_dir(output_dir)
        included = [result for result in results if not result.excluded]

        if not included:
            logger.warning("Attribute '%s': no included groups; skipping plots.", attribute)
            return []

        plot_jobs: list[tuple[str, Callable[[list[GroupEvaluationResult], Path], None]]] = []

        if self.plot_config.generate_bar_charts:
            for metric_name, label in self._BAR_METRICS:
                plot_jobs.append(
                    (
                        f"{metric_name}_by_group",
                        lambda r, p, m=metric_name, l=label: self._plot_metric_bar(attribute, r, p, m, l),
                    )
                )
        if self.plot_config.generate_boxplots:
            plot_jobs.append(("score_boxplot", lambda r, p: self._plot_score_boxplot(attribute, r, p)))
        if self.plot_config.generate_distributions:
            plot_jobs.append(
                ("score_distribution", lambda r, p: self._plot_score_distribution(attribute, r, p))
            )
        if self.plot_config.generate_heatmap:
            plot_jobs.append(("metric_heatmap", lambda r, p: self._plot_metric_heatmap(attribute, r, p)))
        if self.plot_config.generate_comparison_chart:
            plot_jobs.append(
                ("fairness_comparison", lambda r, p: self._plot_fairness_comparison(attribute, r, p))
            )

        generated: list[Path] = []
        for plot_name, plot_fn in tqdm(
            plot_jobs, desc=f"Generating fairness plots ({attribute})", unit="plot"
        ):
            output_path = output_dir / f"{attribute}_{plot_name}.png"
            try:
                plot_fn(included, output_path)
                if output_path.exists():
                    generated.append(output_path)
            except Exception as exc:  # noqa: BLE001 - one failed plot should not abort the rest
                logger.error(
                    "Failed to generate plot '%s' for attribute '%s': %s",
                    plot_name,
                    attribute,
                    exc,
                    exc_info=True,
                )
        return generated

    def _plot_metric_bar(
        self,
        attribute: str,
        results: list[GroupEvaluationResult],
        output_path: Path,
        metric_name: str,
        label: str,
    ) -> None:
        """Plot and save a bar chart of one metric across groups."""
        groups = [result.group_value for result in results]
        values = [result.metrics.get(metric_name, float("nan")) for result in results]

        fig, ax = plt.subplots(figsize=self.plot_config.figure_size)
        bars = ax.bar(groups, values, color="#1f77b4")
        for bar, value in zip(bars, values):
            if not math.isnan(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        ax.set_xlabel(attribute.replace("_", " ").title())
        ax.set_ylabel(label)
        ax.set_title(f"{label} by {attribute.replace('_', ' ').title()}")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(output_path, dpi=self.plot_config.dpi)
        plt.close(fig)

    def _plot_score_boxplot(
        self, attribute: str, results: list[GroupEvaluationResult], output_path: Path
    ) -> None:
        """Plot and save a boxplot of similarity scores by group and label."""
        rows: list[dict[str, Any]] = []
        for result in results:
            for label_value, score_value in zip(result.labels, result.scores):
                rows.append(
                    {
                        "group": result.group_value,
                        "label": "Genuine" if label_value == 1 else "Impostor",
                        "score": score_value,
                    }
                )

        if not rows:
            logger.warning("Attribute '%s': no score data available for boxplot.", attribute)
            return

        frame = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=self.plot_config.figure_size)
        sns.boxplot(data=frame, x="group", y="score", hue="label", ax=ax)
        ax.set_xlabel(attribute.replace("_", " ").title())
        ax.set_ylabel("Similarity score")
        ax.set_title(f"Score Distribution by {attribute.replace('_', ' ').title()} (Boxplot)")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(output_path, dpi=self.plot_config.dpi)
        plt.close(fig)

    def _plot_score_distribution(
        self, attribute: str, results: list[GroupEvaluationResult], output_path: Path
    ) -> None:
        """Plot and save overlaid per-group score density curves."""
        fig, ax = plt.subplots(figsize=self.plot_config.figure_size)
        plotted_any = False

        for result in results:
            if result.scores.size < 2:
                continue
            sns.kdeplot(result.scores, ax=ax, label=result.group_value, linewidth=2)
            plotted_any = True

        if not plotted_any:
            logger.warning("Attribute '%s': no score data available for distribution plot.", attribute)
            plt.close(fig)
            return

        ax.set_xlabel("Similarity score")
        ax.set_ylabel("Density")
        ax.set_title(f"Score Distribution by {attribute.replace('_', ' ').title()}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path, dpi=self.plot_config.dpi)
        plt.close(fig)

    def _plot_metric_heatmap(
        self, attribute: str, results: list[GroupEvaluationResult], output_path: Path
    ) -> None:
        """Plot and save a heatmap of every core metric across groups."""
        groups = [result.group_value for result in results]
        matrix = np.array(
            [[result.metrics.get(metric, float("nan")) for metric in self._HEATMAP_METRICS] for result in results]
        )

        fig_width = max(6.0, len(self._HEATMAP_METRICS) * 1.4)
        fig_height = max(4.0, len(groups) * 0.6)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".3f",
            xticklabels=[metric.upper() for metric in self._HEATMAP_METRICS],
            yticklabels=groups,
            cmap="viridis",
            cbar_kws={"label": "Metric value"},
            ax=ax,
        )
        ax.set_title(f"Metric Heatmap by {attribute.replace('_', ' ').title()}")
        fig.tight_layout()
        fig.savefig(output_path, dpi=self.plot_config.dpi)
        plt.close(fig)

    def _plot_fairness_comparison(
        self, attribute: str, results: list[GroupEvaluationResult], output_path: Path
    ) -> None:
        """Plot and save a grouped bar chart comparing every core metric across groups."""
        groups = [result.group_value for result in results]
        x_positions = np.arange(len(groups))
        bar_width = 0.15

        fig, ax = plt.subplots(figsize=(max(8.0, len(groups) * 1.5), 6.0))
        for index, metric in enumerate(self._HEATMAP_METRICS):
            values = [result.metrics.get(metric, float("nan")) for result in results]
            ax.bar(x_positions + index * bar_width, values, width=bar_width, label=metric.upper())

        ax.set_xticks(x_positions + bar_width * (len(self._HEATMAP_METRICS) - 1) / 2)
        ax.set_xticklabels(groups, rotation=30)
        ax.set_xlabel(attribute.replace("_", " ").title())
        ax.set_ylabel("Metric value")
        ax.set_title(f"Fairness Comparison by {attribute.replace('_', ' ').title()}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path, dpi=self.plot_config.dpi)
        plt.close(fig)


# ------------------------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------------------------


class FairnessEvaluationPipeline:
    """End-to-end orchestration for demographic fairness evaluation.

    Loads the test score CSV, the overall EER threshold, and dataset
    metadata; evaluates every discovered demographic attribute; computes
    disparity statistics and a summary; generates plots; and writes all
    output artifacts under a results directory.
    """

    def __init__(
        self,
        test_scores_path: Path,
        performance_metrics_path: Path,
        metadata_csv: Path,
        output_dir: Path,
        config: FairnessEvaluationConfig | None = None,
        plot_config: FairnessPlotConfig | None = None,
        overall_split_name: str = "test",
    ) -> None:
        """Initialize the pipeline.

        Args:
            test_scores_path: Path to ``results/test_scores.csv``.
            performance_metrics_path: Path to ``results/performance_metrics.csv``.
            metadata_csv: Path to the dataset's ``metadata.csv``.
            output_dir: Directory to write fairness outputs and plots
                to, typically ``results/``.
            config: Fairness evaluation configuration.
            plot_config: Plot styling and toggle configuration.
            overall_split_name: Which split's row to read the overall
                EER threshold from (default ``"test"``, matching
                ``test_scores_path``).
        """
        self.test_scores_path = test_scores_path
        self.performance_metrics_path = performance_metrics_path
        self.metadata_csv = metadata_csv
        self.output_dir = ensure_dir(output_dir)
        self.config = config or FairnessEvaluationConfig()

        self.scores_loader = TestScoresLoader(score_column=self.config.score_column)
        self.threshold_loader = OverallThresholdLoader(split_name=overall_split_name)
        self.plot_generator = FairnessPlotGenerator(plot_config=plot_config)

        self.all_results: list[GroupEvaluationResult] = []
        self._disparity_frames: list[pd.DataFrame] = []

    def run(self) -> dict[str, Any]:
        """Execute the full fairness evaluation pipeline.

        Returns:
            A dict with keys ``"results"`` (list of
            :class:`GroupEvaluationResult`), ``"disparity"`` (combined
            disparity DataFrame), ``"summary"`` (single-row summary
            DataFrame), and ``"summary_text"`` (plain-language summary string).

        Raises:
            FileNotFoundError: If any required input file is missing.
            FairnessEvaluationError: If inputs are invalid or no
                demographic attribute yields any usable group.
        """
        logger.info("Starting demographic fairness evaluation.")

        with timer("Demographic fairness evaluation"):
            scores_frame = self.scores_loader.load(self.test_scores_path)
            overall_threshold = self.threshold_loader.load(self.performance_metrics_path)
            metadata_repo = MetadataRepository(self.metadata_csv, self.config.base_metadata_columns)
            calculator = FairnessMetricsCalculator(self.config, overall_threshold)

            for attribute in tqdm(
                metadata_repo.attribute_columns, desc="Evaluating demographic attributes", unit="attribute"
            ):
                attribute_results = calculator.evaluate_attribute(scores_frame, metadata_repo, attribute)
                if not attribute_results:
                    continue

                self.all_results.extend(attribute_results)

                disparity_frame = FairnessDisparityCalculator.compute(attribute_results)
                disparity_frame.insert(0, "attribute", attribute)
                self._disparity_frames.append(disparity_frame)

                plots_dir = self.output_dir / "plots" / "fairness"
                generated_plots = self.plot_generator.generate_attribute_plots(
                    attribute, attribute_results, plots_dir
                )
                logger.info(
                    "Generated %d fairness plot(s) for attribute '%s'.", len(generated_plots), attribute
                )

        if not self.all_results:
            raise FairnessEvaluationError(
                "No demographic attribute produced any usable group; check metadata.csv "
                "and the configured min_group_size."
            )

        disparity_frame = (
            pd.concat(self._disparity_frames, ignore_index=True) if self._disparity_frames else pd.DataFrame()
        )
        summary_df, summary_text = FairnessSummaryGenerator.generate(self.all_results, disparity_frame)

        self._save_outputs(disparity_frame, summary_df)

        logger.info("Fairness summary: %s", summary_text)
        logger.info("Demographic fairness evaluation finished.")

        return {
            "results": self.all_results,
            "disparity": disparity_frame,
            "summary": summary_df,
            "summary_text": summary_text,
        }

    def _save_outputs(self, disparity_frame: pd.DataFrame, summary_df: pd.DataFrame) -> None:
        """Write every output artifact for this pipeline run.

        Args:
            disparity_frame: Combined per-attribute disparity statistics.
            summary_df: Single-row overall fairness summary.
        """
        self._save_fairness_metrics_csv()
        self._save_fairness_metrics_json()

        disparity_path = self.output_dir / "fairness_disparity.csv"
        disparity_frame.to_csv(disparity_path, index=False)
        logger.info("Saved fairness disparity to %s", disparity_path)

        summary_path = self.output_dir / "fairness_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info("Saved fairness summary to %s", summary_path)

    def _save_fairness_metrics_csv(self) -> None:
        """Write ``fairness_metrics.csv``, one row per demographic group."""
        rows = [self._result_to_row(result) for result in self.all_results]
        output_path = self.output_dir / "fairness_metrics.csv"
        pd.DataFrame(rows).to_csv(output_path, index=False)
        logger.info("Saved fairness metrics CSV to %s", output_path)

    def _save_fairness_metrics_json(self) -> None:
        """Write ``fairness_metrics.json``, keyed by ``attribute::group_value``."""
        payload = {
            f"{result.attribute}::{result.group_value}": self._result_to_row(result)
            for result in self.all_results
        }
        output_path = self.output_dir / "fairness_metrics.json"
        with output_path.open("w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=2, sort_keys=True, default=_json_safe)
        logger.info("Saved fairness metrics JSON to %s", output_path)

    @staticmethod
    def _result_to_row(result: GroupEvaluationResult) -> dict[str, Any]:
        """Flatten one :class:`GroupEvaluationResult` into an output row.

        Args:
            result: The group result to flatten.

        Returns:
            A dict combining group metadata and its computed metrics.
        """
        return {
            "attribute": result.attribute,
            "group_value": result.group_value,
            "sample_size": result.sample_size,
            "genuine_pairs": result.genuine_pairs,
            "impostor_pairs": result.impostor_pairs,
            "excluded": result.excluded,
            "exclusion_reason": result.exclusion_reason,
            **result.metrics,
        }
