"""Cross-model comparison for the fairness evaluation pipeline.

Reads the evaluation artifacts already produced per model —
``performance_metrics.csv`` (from
:mod:`fairness_fr.evaluation.evaluate_performance`) and
``fairness_metrics.csv`` / ``fairness_disparity.csv`` (from
:mod:`fairness_fr.evaluation.evaluate_fairness`) — and builds a
side-by-side comparison across any number of pretrained face
recognition models: performance metrics, demographic disparity, several
ranking axes, publication-quality plots, and an automatic textual
summary.

Outputs:

- ``model_comparison.csv`` / ``model_comparison.json`` — one row/entry
  per model with every performance metric plus a mean demographic
  disparity score.
- ``model_rankings.csv`` — per-model rank on accuracy, EER, TAR,
  fairness, and a composite overall rank.
- ``model_summary.txt`` — plain-language summary: best performer,
  fairest model, lowest EER, lowest bias, and per-model
  strengths/weaknesses.
- Comparison PNG plots under ``plots/model_comparison/``.

Design notes:
    - This module never touches raw scores or embeddings; it only reads
      the already-aggregated metric CSVs each model's own performance
      and fairness evaluation runs produced. Differences in score
      distribution or scale between models are therefore irrelevant
      here — every model is compared on the same fixed metric schema.
    - A model missing its performance metrics file cannot be compared
      at all and is excluded (logged as an error). A model missing its
      fairness files is still compared on performance, with fairness
      fields reported as NaN and ``has_fairness_data=False`` — fairness
      evaluation is optional per model rather than a hard requirement.
    - Models may have been fairness-evaluated on different demographic
      attribute sets (e.g. one dataset has ``gender`` and ``ethnicity``,
      another only ``ethnicity``); the per-group fairness data for each
      model is preserved as-is in ``model_comparison.json`` rather than
      forced into a common schema, while the single aggregated
      "mean_disparity" score (mean of every attribute/metric range in
      that model's own ``fairness_disparity.csv``) is what drives
      cross-model fairness ranking and plots.
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
from tqdm import tqdm

from fairness_fr.evaluation.evaluate_performance import _json_safe
from fairness_fr.utils.logging import get_logger
from fairness_fr.utils import ensure_dir, timer

logger = get_logger(__name__)


class ModelComparisonError(Exception):
    """Raised when model comparison cannot be completed."""


def _is_numeric(value: Any) -> bool:
    """Check whether a value can be converted to ``float``.

    Args:
        value: Any value.

    Returns:
        True if ``float(value)`` would succeed.
    """
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelComparisonConfig:
    """Configuration controlling model comparison behavior.

    Attributes:
        split_name: Which split's row to read from each model's
            ``performance_metrics.csv`` (default ``"test"``).
        disparity_concern_threshold: Absolute disparity range above
            which a model's fairness is flagged as a concern in the
            textual summary.
    """

    split_name: str = "test"
    disparity_concern_threshold: float = 0.10


@dataclass(frozen=True, slots=True)
class ModelComparisonPlotConfig:
    """Configuration controlling comparison plot generation and styling.

    Attributes:
        dpi: Resolution (dots per inch) for saved PNG plots.
        figure_size: ``(width, height)`` in inches for standard plots.
        style: A matplotlib style name applied globally.
        generate_metric_bars: Whether to generate the per-metric
            (accuracy, EER, FAR, FRR, TAR) bar-by-model charts.
        generate_disparity_chart: Whether to generate the fairness
            disparity comparison chart.
        generate_radar_chart: Whether to generate the multi-metric
            radar chart.
        generate_heatmap: Whether to generate the model-by-metric heatmap.
        generate_ranking_chart: Whether to generate the overall ranking chart.
    """

    dpi: int = 300
    figure_size: tuple[float, float] = (10.0, 6.0)
    style: str = "seaborn-v0_8-whitegrid"
    generate_metric_bars: bool = True
    generate_disparity_chart: bool = True
    generate_radar_chart: bool = True
    generate_heatmap: bool = True
    generate_ranking_chart: bool = True


# ------------------------------------------------------------------------------
# Result loading
# ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelResultsPaths:
    """Filesystem locations of one model's evaluation outputs.

    Attributes:
        model_name: Human-readable model identifier (e.g. ``"arcface"``).
        performance_metrics_csv: Path to that model's ``performance_metrics.csv``.
        fairness_metrics_csv: Path to that model's ``fairness_metrics.csv``,
            or None if fairness evaluation was not run for this model.
        fairness_disparity_csv: Path to that model's
            ``fairness_disparity.csv``, or None if fairness evaluation
            was not run for this model.
    """

    model_name: str
    performance_metrics_csv: Path
    fairness_metrics_csv: Path | None = None
    fairness_disparity_csv: Path | None = None

    @classmethod
    def from_results_root(cls, model_name: str, results_root: Path) -> "ModelResultsPaths":
        """Build standard result file paths for a model under a results root.

        Assumes the conventional layout
        ``results_root/<model_name>/{performance_metrics.csv,
        fairness_metrics.csv, fairness_disparity.csv}``, which is how
        :mod:`fairness_fr.evaluation.evaluate_performance` and
        :mod:`fairness_fr.evaluation.evaluate_fairness` are typically
        invoked once per model.

        Args:
            model_name: Model identifier, matching its results subdirectory name.
            results_root: Root directory containing one subdirectory per model.

        Returns:
            A :class:`ModelResultsPaths` instance. Callers should be
            prepared for the referenced files to not exist; loading
            handles that gracefully.
        """
        model_dir = results_root / model_name
        return cls(
            model_name=model_name,
            performance_metrics_csv=model_dir / "performance_metrics.csv",
            fairness_metrics_csv=model_dir / "fairness_metrics.csv",
            fairness_disparity_csv=model_dir / "fairness_disparity.csv",
        )


@dataclass
class ModelResults:
    """Loaded, validated evaluation outputs for a single model.

    Attributes:
        model_name: Model identifier.
        performance: Dict of performance metrics for the configured split.
        fairness_groups: Per-group fairness metrics DataFrame, empty if unavailable.
        fairness_disparity: Per-attribute disparity statistics DataFrame,
            empty if unavailable.
        has_fairness_data: Whether usable fairness disparity data was loaded.
        load_warnings: Non-fatal issues encountered while loading this model.
    """

    model_name: str
    performance: dict[str, float]
    fairness_groups: pd.DataFrame
    fairness_disparity: pd.DataFrame
    has_fairness_data: bool
    load_warnings: list[str] = field(default_factory=list)


class ModelResultsLoader:
    """Loads and validates one model's evaluation output files.

    A missing or invalid performance metrics file makes a model
    entirely unusable for comparison (returns None, logged as an
    error). Missing or invalid fairness files degrade gracefully: the
    model is still returned, usable for performance comparison, with
    empty fairness DataFrames and ``has_fairness_data=False``.
    """

    def __init__(self, split_name: str = "test") -> None:
        """Initialize the loader.

        Args:
            split_name: Which split's row to read from
                ``performance_metrics.csv``.
        """
        self.split_name = split_name

    def load(self, paths: ModelResultsPaths) -> ModelResults | None:
        """Load one model's results.

        Args:
            paths: Filesystem locations of the model's evaluation outputs.

        Returns:
            A :class:`ModelResults` instance, or None if the model's
            performance metrics could not be loaded at all.
        """
        warnings: list[str] = []

        performance = self._load_performance(paths)
        if performance is None:
            return None

        fairness_groups, group_warning = self._load_optional_csv(paths.fairness_metrics_csv, "fairness_metrics.csv")
        if group_warning:
            warnings.append(group_warning)

        fairness_disparity, disparity_warning = self._load_optional_csv(
            paths.fairness_disparity_csv, "fairness_disparity.csv"
        )
        if disparity_warning:
            warnings.append(disparity_warning)

        has_fairness_data = not fairness_disparity.empty

        logger.info(
            "Model '%s': loaded performance metrics (split='%s'); fairness data available=%s.",
            paths.model_name,
            self.split_name,
            has_fairness_data,
        )

        return ModelResults(
            model_name=paths.model_name,
            performance=performance,
            fairness_groups=fairness_groups,
            fairness_disparity=fairness_disparity,
            has_fairness_data=has_fairness_data,
            load_warnings=warnings,
        )

    def _load_performance(self, paths: ModelResultsPaths) -> dict[str, float] | None:
        """Load and validate the required performance metrics for one model.

        Args:
            paths: Filesystem locations of the model's evaluation outputs.

        Returns:
            A dict of metric name to value, or None if unusable.
        """
        if not paths.performance_metrics_csv.exists():
            logger.error(
                "Model '%s': performance metrics file not found: %s; excluding from comparison.",
                paths.model_name,
                paths.performance_metrics_csv,
            )
            return None

        try:
            frame = pd.read_csv(paths.performance_metrics_csv)
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            logger.error(
                "Model '%s': failed to read performance metrics (%s); excluding from comparison.",
                paths.model_name,
                exc,
            )
            return None

        if frame.empty or "split" not in frame.columns:
            logger.error(
                "Model '%s': performance metrics file is empty or missing 'split' column; excluding.",
                paths.model_name,
            )
            return None

        matching_rows = frame.loc[frame["split"] == self.split_name]
        if matching_rows.empty:
            logger.error(
                "Model '%s': no '%s' split row found in performance metrics; excluding.",
                paths.model_name,
                self.split_name,
            )
            return None

        raw = matching_rows.iloc[0].to_dict()
        raw.pop("split", None)
        return {key: (float(value) if _is_numeric(value) else float("nan")) for key, value in raw.items()}

    @staticmethod
    def _load_optional_csv(path: Path | None, label: str) -> tuple[pd.DataFrame, str | None]:
        """Load an optional fairness CSV, tolerating absence or corruption.

        Args:
            path: Path to the optional CSV, or None if not configured.
            label: Human-readable filename, used in log/warning messages.

        Returns:
            A tuple ``(dataframe, warning_message_or_none)``. The
            DataFrame is empty if the file is missing, unreadable, or
            not configured.
        """
        if path is None or not path.exists():
            return pd.DataFrame(), f"{label} not found"

        try:
            frame = pd.read_csv(path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            return pd.DataFrame(), f"failed to read {label}: {exc}"

        return frame, None


# ------------------------------------------------------------------------------
# Comparison table construction
# ------------------------------------------------------------------------------


class FairnessDisparityAggregator:
    """Aggregates a model's per-attribute disparity table into a single score."""

    @staticmethod
    def mean_disparity(fairness_disparity: pd.DataFrame) -> float:
        """Compute the mean disparity range across every attribute/metric row.

        Args:
            fairness_disparity: A model's ``fairness_disparity.csv``
                contents (possibly empty).

        Returns:
            The mean of the ``range`` column, or NaN if unavailable.
        """
        if fairness_disparity.empty or "range" not in fairness_disparity.columns:
            return float("nan")

        values = pd.to_numeric(fairness_disparity["range"], errors="coerce").dropna()
        if values.empty:
            return float("nan")

        return float(values.mean())


class ModelComparator:
    """Loads model evaluation outputs and builds the cross-model comparison table."""

    #: The eleven performance metrics compared across models.
    PERFORMANCE_METRICS: tuple[str, ...] = (
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
    )

    def __init__(self, config: ModelComparisonConfig | None = None) -> None:
        """Initialize the comparator.

        Args:
            config: Comparison configuration. Defaults to
                :class:`ModelComparisonConfig` defaults if not provided.
        """
        self.config = config or ModelComparisonConfig()
        self.loader = ModelResultsLoader(split_name=self.config.split_name)

    def load_all(self, model_paths: list[ModelResultsPaths]) -> list[ModelResults]:
        """Load every model's results, excluding unusable ones.

        Args:
            model_paths: Filesystem locations for every model to compare.

        Returns:
            A list of successfully loaded :class:`ModelResults`.

        Raises:
            ModelComparisonError: If no model produced usable evaluation outputs.
        """
        results: list[ModelResults] = []
        for paths in tqdm(model_paths, desc="Loading model results", unit="model"):
            result = self.loader.load(paths)
            if result is not None:
                results.append(result)

        if not results:
            raise ModelComparisonError(
                "No model produced usable evaluation outputs; cannot build a comparison."
            )

        if len(results) < len(model_paths):
            logger.warning(
                "%d of %d requested models were excluded due to missing/invalid outputs.",
                len(model_paths) - len(results),
                len(model_paths),
            )

        return results

    def build_comparison_table(self, results: list[ModelResults]) -> pd.DataFrame:
        """Build the wide performance + fairness comparison table.

        Args:
            results: Successfully loaded model results.

        Returns:
            A DataFrame with one row per model: ``model``, every metric
            in :attr:`PERFORMANCE_METRICS`, ``mean_disparity``, and
            ``has_fairness_data``.
        """
        rows: list[dict[str, Any]] = []
        for result in results:
            row: dict[str, Any] = {"model": result.model_name}
            for metric in self.PERFORMANCE_METRICS:
                row[metric] = result.performance.get(metric, float("nan"))
            row["mean_disparity"] = FairnessDisparityAggregator.mean_disparity(result.fairness_disparity)
            row["has_fairness_data"] = result.has_fairness_data
            rows.append(row)

        return pd.DataFrame(rows)


# ------------------------------------------------------------------------------
# Ranking
# ------------------------------------------------------------------------------


class ModelRanker:
    """Ranks models along several axes and computes a composite overall rank."""

    @staticmethod
    def rank(comparison_table: pd.DataFrame) -> pd.DataFrame:
        """Compute per-axis ranks and a composite overall rank.

        Args:
            comparison_table: The wide comparison table from
                :meth:`ModelComparator.build_comparison_table`.

        Returns:
            A copy of ``comparison_table`` with added columns
            ``accuracy_rank, eer_rank, tar_rank, fairness_rank,
            overall_rank_score, overall_rank``, sorted best-to-worst by
            ``overall_rank``. Models missing a given metric receive NaN
            for that axis's rank and are excluded only from that axis's
            contribution to the composite score.
        """
        table = comparison_table.copy()

        table["accuracy_rank"] = table["accuracy"].rank(ascending=False, method="min")
        table["eer_rank"] = table["eer"].rank(ascending=True, method="min")
        table["tar_rank"] = table["tar"].rank(ascending=False, method="min")
        table["fairness_rank"] = table["mean_disparity"].rank(ascending=True, method="min")

        rank_columns = ["accuracy_rank", "eer_rank", "tar_rank", "fairness_rank"]
        table["overall_rank_score"] = table[rank_columns].mean(axis=1, skipna=True)
        table["overall_rank"] = table["overall_rank_score"].rank(ascending=True, method="min").astype(int)

        return table.sort_values("overall_rank").reset_index(drop=True)


# ------------------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------------------


class ModelComparisonPlotGenerator:
    """Generates publication-quality PNG plots comparing models."""

    _BAR_METRICS: tuple[tuple[str, str], ...] = (
        ("accuracy", "Accuracy"),
        ("eer", "EER"),
        ("far", "FAR"),
        ("frr", "FRR"),
        ("tar", "TAR"),
    )
    #: All naturally "higher is better", used unmodified for the radar chart.
    _RADAR_METRICS: tuple[str, ...] = ("accuracy", "precision", "recall", "f1_score", "tar", "tnr")
    _HEATMAP_METRICS: tuple[str, ...] = (
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "far",
        "frr",
        "tar",
        "tnr",
        "eer",
    )

    def __init__(self, plot_config: ModelComparisonPlotConfig | None = None) -> None:
        """Initialize the plot generator and apply the configured matplotlib style.

        Args:
            plot_config: Plot styling and toggle configuration. Defaults
                to :class:`ModelComparisonPlotConfig` defaults if not provided.
        """
        self.plot_config = plot_config or ModelComparisonPlotConfig()
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

    def generate_all(
        self, comparison_table: pd.DataFrame, rankings: pd.DataFrame, output_dir: Path
    ) -> list[Path]:
        """Generate every enabled comparison plot.

        Args:
            comparison_table: The wide model comparison table.
            rankings: The ranked comparison table from :class:`ModelRanker`.
            output_dir: Directory to save PNG files into.

        Returns:
            A list of paths to successfully generated plot files.
        """
        ensure_dir(output_dir)
        plot_jobs: list[tuple[str, Callable[[Path], None]]] = []

        if self.plot_config.generate_metric_bars:
            for metric, label in self._BAR_METRICS:
                plot_jobs.append(
                    (
                        f"{metric}_comparison",
                        lambda p, m=metric, l=label: self._plot_metric_bar(comparison_table, p, m, l),
                    )
                )
        if self.plot_config.generate_disparity_chart:
            plot_jobs.append(
                ("fairness_disparity_comparison", lambda p: self._plot_disparity_bar(comparison_table, p))
            )
        if self.plot_config.generate_radar_chart:
            plot_jobs.append(("radar_chart", lambda p: self._plot_radar(comparison_table, p)))
        if self.plot_config.generate_heatmap:
            plot_jobs.append(("metric_heatmap", lambda p: self._plot_heatmap(comparison_table, p)))
        if self.plot_config.generate_ranking_chart:
            plot_jobs.append(("overall_ranking", lambda p: self._plot_ranking(rankings, p)))

        generated: list[Path] = []
        for plot_name, plot_fn in tqdm(plot_jobs, desc="Generating model comparison plots", unit="plot"):
            output_path = output_dir / f"{plot_name}.png"
            try:
                plot_fn(output_path)
                if output_path.exists():
                    generated.append(output_path)
            except Exception as exc:  # noqa: BLE001 - one failed plot should not abort the rest
                logger.error("Failed to generate plot '%s': %s", plot_name, exc, exc_info=True)
        return generated

    def _plot_metric_bar(
        self, table: pd.DataFrame, output_path: Path, metric: str, label: str
    ) -> None:
        """Plot and save a bar chart of one metric across models."""
        if metric not in table.columns or table[metric].isna().all():
            logger.warning("Metric '%s' unavailable for any model; skipping plot.", metric)
            return

        fig, ax = plt.subplots(figsize=self.plot_config.figure_size)
        bars = ax.bar(table["model"], table[metric], color="#1f77b4")
        for bar, value in zip(bars, table[metric]):
            if not (isinstance(value, float) and math.isnan(value)):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        ax.set_xlabel("Model")
        ax.set_ylabel(label)
        ax.set_title(f"{label} Comparison Across Models")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(output_path, dpi=self.plot_config.dpi)
        plt.close(fig)

    def _plot_disparity_bar(self, table: pd.DataFrame, output_path: Path) -> None:
        """Plot and save a bar chart of mean demographic disparity across models."""
        subset = table.dropna(subset=["mean_disparity"])
        if subset.empty:
            logger.warning("No models have fairness disparity data; skipping disparity comparison plot.")
            return

        fig, ax = plt.subplots(figsize=self.plot_config.figure_size)
        bars = ax.bar(subset["model"], subset["mean_disparity"], color="#d62728")
        for bar, value in zip(bars, subset["mean_disparity"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_xlabel("Model")
        ax.set_ylabel("Mean Demographic Disparity (range)")
        ax.set_title("Fairness Disparity Comparison Across Models")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(output_path, dpi=self.plot_config.dpi)
        plt.close(fig)

    def _plot_radar(self, table: pd.DataFrame, output_path: Path) -> None:
        """Plot and save a radar chart comparing models across several metrics."""
        available_metrics = [metric for metric in self._RADAR_METRICS if metric in table.columns]
        if len(available_metrics) < 3:
            logger.warning("Fewer than 3 radar metrics available; skipping radar chart.")
            return

        subset = table.dropna(subset=available_metrics, how="all")
        if subset.empty:
            logger.warning("No models have data for radar chart metrics; skipping.")
            return

        angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8.0, 8.0), subplot_kw={"polar": True})
        for _, row in subset.iterrows():
            values = [
                0.0 if (isinstance(row.get(metric), float) and math.isnan(row.get(metric))) else float(row.get(metric, 0.0))
                for metric in available_metrics
            ]
            values += values[:1]
            ax.plot(angles, values, linewidth=2, label=str(row["model"]))
            ax.fill(angles, values, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([metric.upper() for metric in available_metrics])
        ax.set_title("Model Comparison Radar Chart")
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
        fig.tight_layout()
        fig.savefig(output_path, dpi=self.plot_config.dpi)
        plt.close(fig)

    def _plot_heatmap(self, table: pd.DataFrame, output_path: Path) -> None:
        """Plot and save a heatmap of every core metric across models."""
        available_metrics = [metric for metric in self._HEATMAP_METRICS if metric in table.columns]
        if not available_metrics:
            logger.warning("No metrics available for heatmap; skipping.")
            return

        matrix = table[available_metrics].to_numpy(dtype=float)
        fig_width = max(6.0, len(available_metrics) * 1.2)
        fig_height = max(4.0, len(table) * 0.6)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".3f",
            xticklabels=[metric.upper() for metric in available_metrics],
            yticklabels=table["model"].tolist(),
            cmap="viridis",
            cbar_kws={"label": "Metric value"},
            ax=ax,
        )
        ax.set_title("Model Metric Heatmap")
        fig.tight_layout()
        fig.savefig(output_path, dpi=self.plot_config.dpi)
        plt.close(fig)

    def _plot_ranking(self, rankings: pd.DataFrame, output_path: Path) -> None:
        """Plot and save a horizontal bar chart of the overall model ranking."""
        if rankings.empty:
            logger.warning("Rankings table is empty; skipping ranking chart.")
            return

        ordered = rankings.sort_values("overall_rank")
        fig, ax = plt.subplots(figsize=self.plot_config.figure_size)
        bars = ax.barh(ordered["model"], ordered["overall_rank_score"], color="#2ca02c")
        ax.invert_yaxis()  # best (lowest score) at the top
        for bar, value in zip(bars, ordered["overall_rank_score"]):
            ax.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                va="center",
                ha="left",
                fontsize=9,
            )
        ax.set_xlabel("Overall Rank Score (lower = better)")
        ax.set_ylabel("Model")
        ax.set_title("Overall Model Ranking")
        fig.tight_layout()
        fig.savefig(output_path, dpi=self.plot_config.dpi)
        plt.close(fig)


# ------------------------------------------------------------------------------
# Textual summary
# ------------------------------------------------------------------------------


class ModelSummaryGenerator:
    """Produces the automatic textual summary comparing all models."""

    @staticmethod
    def generate(
        comparison_table: pd.DataFrame, rankings: pd.DataFrame, disparity_threshold: float
    ) -> str:
        """Build the plain-language, multi-section comparison summary.

        Args:
            comparison_table: The wide model comparison table.
            rankings: The ranked comparison table from :class:`ModelRanker`.
            disparity_threshold: Disparity range above which a model is
                flagged as a fairness concern.

        Returns:
            A formatted multi-line summary string.
        """
        if comparison_table.empty:
            return "No models were available for comparison."

        lines: list[str] = ["=" * 70, "MODEL COMPARISON SUMMARY", "=" * 70, ""]

        if comparison_table["accuracy"].notna().any():
            best_accuracy_row = comparison_table.loc[comparison_table["accuracy"].idxmax()]
            lines.append(
                f"Best-performing model (accuracy): {best_accuracy_row['model']} "
                f"(accuracy={best_accuracy_row['accuracy']:.4f})."
            )

        eer_valid = comparison_table.dropna(subset=["eer"])
        if not eer_valid.empty:
            lowest_eer_row = eer_valid.loc[eer_valid["eer"].idxmin()]
            lines.append(f"Model with lowest EER: {lowest_eer_row['model']} (EER={lowest_eer_row['eer']:.4f}).")

        tar_valid = comparison_table.dropna(subset=["tar"])
        if not tar_valid.empty:
            highest_tar_row = tar_valid.loc[tar_valid["tar"].idxmax()]
            lines.append(f"Model with highest TAR: {highest_tar_row['model']} (TAR={highest_tar_row['tar']:.4f}).")

        disparity_valid = comparison_table.dropna(subset=["mean_disparity"])
        if not disparity_valid.empty:
            fairest_row = disparity_valid.loc[disparity_valid["mean_disparity"].idxmin()]
            lines.append(
                f"Fairest model (lowest mean demographic disparity): {fairest_row['model']} "
                f"(mean disparity={fairest_row['mean_disparity']:.4f})."
            )
        else:
            lines.append("Fairness disparity data was unavailable for all models.")

        if not rankings.empty:
            lines.append("")
            lines.append("Overall ranking (best to worst):")
            for _, row in rankings.sort_values("overall_rank").iterrows():
                lines.append(f"  {int(row['overall_rank'])}. {row['model']} (score={row['overall_rank_score']:.2f})")

        lines.append("")
        lines.append("Per-model notes:")
        for _, row in comparison_table.iterrows():
            notes = ModelSummaryGenerator._model_notes(row, comparison_table, disparity_threshold)
            lines.append(f"  - {row['model']}: {notes}")

        lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)

    @staticmethod
    def _model_notes(row: pd.Series, table: pd.DataFrame, disparity_threshold: float) -> str:
        """Build a single model's strengths/weaknesses sentence.

        Args:
            row: The model's row in the comparison table.
            table: The full comparison table, for relative comparisons.
            disparity_threshold: Disparity range above which fairness is
                flagged as a weakness.

        Returns:
            A formatted "strengths: ...; weaknesses: ..." sentence.
        """
        advantages: list[str] = []
        disadvantages: list[str] = []
        multiple_models = len(table) > 1

        if pd.notna(row.get("accuracy")) and table["accuracy"].notna().any():
            if row["accuracy"] >= table["accuracy"].max() - 1e-9:
                advantages.append("highest accuracy")

        if pd.notna(row.get("eer")) and table["eer"].notna().any():
            if row["eer"] <= table["eer"].min() + 1e-9:
                advantages.append("lowest EER")

        if pd.notna(row.get("tar")) and table["tar"].notna().any():
            if row["tar"] >= table["tar"].max() - 1e-9:
                advantages.append("highest TAR")

        if pd.notna(row.get("mean_disparity")):
            if table["mean_disparity"].notna().any() and row["mean_disparity"] <= table["mean_disparity"].min() + 1e-9:
                advantages.append("lowest demographic disparity")
            elif row["mean_disparity"] > disparity_threshold:
                disadvantages.append(
                    f"demographic disparity ({row['mean_disparity']:.3f}) exceeds the "
                    f"{disparity_threshold:.2f} concern threshold"
                )
        else:
            disadvantages.append("no fairness data available")

        if multiple_models and pd.notna(row.get("far")) and table["far"].notna().any():
            if row["far"] >= table["far"].max() - 1e-9:
                disadvantages.append("highest FAR")

        if multiple_models and pd.notna(row.get("frr")) and table["frr"].notna().any():
            if row["frr"] >= table["frr"].max() - 1e-9:
                disadvantages.append("highest FRR")

        advantage_text = ", ".join(advantages) if advantages else "no standout strengths"
        disadvantage_text = ", ".join(disadvantages) if disadvantages else "no notable weaknesses"
        return f"strengths: {advantage_text}; weaknesses: {disadvantage_text}."


# ------------------------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------------------------


class ModelComparisonPipeline:
    """End-to-end orchestration for comparing multiple face recognition models.

    Loads every model's evaluation outputs, builds the comparison
    table, computes rankings, generates plots, produces the textual
    summary, and writes all output artifacts under a results directory.
    """

    def __init__(
        self,
        model_paths: list[ModelResultsPaths],
        output_dir: Path,
        config: ModelComparisonConfig | None = None,
        plot_config: ModelComparisonPlotConfig | None = None,
    ) -> None:
        """Initialize the pipeline.

        Args:
            model_paths: Filesystem locations of every model's
                evaluation outputs to compare. Any number of models is
                supported.
            output_dir: Directory to write comparison outputs and plots
                to, typically ``results/``.
            config: Comparison behavior configuration.
            plot_config: Plot styling and toggle configuration.

        Raises:
            ModelComparisonError: If ``model_paths`` is empty.
        """
        if not model_paths:
            raise ModelComparisonError("At least one model must be provided for comparison.")

        self.model_paths = model_paths
        self.output_dir = ensure_dir(output_dir)
        self.config = config or ModelComparisonConfig()
        self.comparator = ModelComparator(config=self.config)
        self.plot_generator = ModelComparisonPlotGenerator(plot_config=plot_config)

    def run(self) -> dict[str, Any]:
        """Execute the full model comparison pipeline.

        Returns:
            A dict with keys ``"comparison_table"``, ``"rankings"``,
            ``"summary_text"``, and ``"results"`` (the loaded
            :class:`ModelResults` list).

        Raises:
            ModelComparisonError: If no model produced usable evaluation outputs.
        """
        logger.info("Starting model comparison for %d model(s).", len(self.model_paths))

        with timer("Model comparison"):
            results = self.comparator.load_all(self.model_paths)
            comparison_table = self.comparator.build_comparison_table(results)
            rankings = ModelRanker.rank(comparison_table)

            plots_dir = self.output_dir / "plots" / "model_comparison"
            generated_plots = self.plot_generator.generate_all(comparison_table, rankings, plots_dir)
            logger.info("Generated %d model comparison plot(s).", len(generated_plots))

            summary_text = ModelSummaryGenerator.generate(
                comparison_table, rankings, self.config.disparity_concern_threshold
            )

            self._save_outputs(comparison_table, rankings, results, summary_text)

        logger.info("Model comparison finished for %d model(s).", len(results))

        return {
            "comparison_table": comparison_table,
            "rankings": rankings,
            "summary_text": summary_text,
            "results": results,
        }

    def _save_outputs(
        self,
        comparison_table: pd.DataFrame,
        rankings: pd.DataFrame,
        results: list[ModelResults],
        summary_text: str,
    ) -> None:
        """Write every output artifact for this comparison run.

        Args:
            comparison_table: The wide model comparison table.
            rankings: The ranked comparison table.
            results: The loaded per-model results, for the JSON export's
                per-group fairness detail.
            summary_text: The generated plain-language summary.
        """
        comparison_csv_path = self.output_dir / "model_comparison.csv"
        comparison_table.to_csv(comparison_csv_path, index=False)
        logger.info("Saved model comparison CSV to %s", comparison_csv_path)

        comparison_json_path = self.output_dir / "model_comparison.json"
        payload = {
            "models": comparison_table.to_dict(orient="records"),
            "fairness_by_group": {
                result.model_name: (
                    result.fairness_groups.to_dict(orient="records") if not result.fairness_groups.empty else []
                )
                for result in results
            },
            "fairness_disparity": {
                result.model_name: (
                    result.fairness_disparity.to_dict(orient="records")
                    if not result.fairness_disparity.empty
                    else []
                )
                for result in results
            },
        }
        with comparison_json_path.open("w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=2, sort_keys=True, default=_json_safe)
        logger.info("Saved model comparison JSON to %s", comparison_json_path)

        rankings_csv_path = self.output_dir / "model_rankings.csv"
        rankings.to_csv(rankings_csv_path, index=False)
        logger.info("Saved model rankings CSV to %s", rankings_csv_path)

        summary_path = self.output_dir / "model_summary.txt"
        summary_path.write_text(summary_text, encoding="utf-8")
        logger.info("Saved model summary text to %s", summary_path)
