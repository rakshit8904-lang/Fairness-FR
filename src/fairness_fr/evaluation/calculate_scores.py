"""Similarity score computation for the fairness evaluation pipeline.

Combines the cached embeddings produced by
:mod:`fairness_fr.models.extract_embeddings` with the pair files
produced by :mod:`fairness_fr.data.generate_pairs` to compute, for
every genuine/impostor pair, three similarity/distance metrics:

- Cosine similarity
- Euclidean distance
- Cosine distance (``1 - cosine_similarity``)

Outputs are three per-split score CSVs (``train_scores.csv``,
``validation_scores.csv``, ``test_scores.csv``) plus a
``scoring_log.csv`` summarizing processing time, pair counts, skipped
pairs, and failures per split.

Design notes:
    - All embedding lookups and metric computations are fully
      vectorized with NumPy (no per-pair Python loops), and pair CSVs
      are streamed in configurable-size chunks via pandas'
      ``chunksize``, so the module scales to millions of pairs without
      loading an entire split into memory at once or falling back to
      row-by-row processing.
    - Missing embeddings are handled gracefully: pairs referencing an
      image absent from the embedding index are counted, logged in
      aggregate, and excluded from the corresponding output rows rather
      than raising or crashing the whole run.
    - All computations are pure NumPy linear algebra on fixed inputs, so
      results are deterministic and reproducible run-to-run.
    - Which distance metric a downstream consumer should treat as
      "primary" is configurable via :class:`ScoringConfig` (typically
      derived from :class:`fairness_fr.config.ModelConfig`), but all
      three metrics are always computed and written, per the fixed
      output schema.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from fairness_fr.config import ModelConfig
from ..config.constants import DistanceMetric
from fairness_fr.data.generate_pairs import PairOutputColumns
from fairness_fr.utils.logging import get_logger
from fairness_fr.models.extract_embeddings import EmbeddingIndexColumns
from fairness_fr.utils import ensure_dir, load_npy, timer

logger = get_logger(__name__)


class ScoreCalculationError(Exception):
    """Raised when similarity scores cannot be computed for a pair file."""


# ------------------------------------------------------------------------------
# Output schema
# ------------------------------------------------------------------------------


class ScoreOutputColumns:
    """Canonical column names for the metric columns added to score CSVs."""

    COSINE_SIMILARITY = "cosine_similarity"
    EUCLIDEAN_DISTANCE = "euclidean_distance"
    COSINE_DISTANCE = "cosine_distance"

    METRIC_COLUMNS: tuple[str, str, str] = (
        COSINE_SIMILARITY,
        EUCLIDEAN_DISTANCE,
        COSINE_DISTANCE,
    )

    #: Full output schema: every pair column, in original order, followed
    #: by the three similarity/distance metric columns.
    FULL_SCHEMA: tuple[str, ...] = PairOutputColumns.ALL + METRIC_COLUMNS


# ------------------------------------------------------------------------------
# Embedding lookup
# ------------------------------------------------------------------------------


class EmbeddingRepository:
    """Loads embeddings.npy and embedding_index.csv for efficient pair lookup.

    Provides vectorized retrieval of embeddings for entire batches of
    image pairs at once, which is what allows
    :class:`ScoreCalculator` to score millions of pairs without a
    per-pair Python loop.
    """

    def __init__(self, embeddings_path: Path, embedding_index_path: Path) -> None:
        """Load and validate the embeddings array and its index mapping.

        Args:
            embeddings_path: Path to ``embeddings.npy``, as produced by
                :mod:`fairness_fr.models.extract_embeddings`.
            embedding_index_path: Path to ``embedding_index.csv``.

        Raises:
            FileNotFoundError: If either input file is missing.
            ScoreCalculationError: If the embeddings array and index
                file have mismatched row counts.
        """
        self.embeddings_path = embeddings_path
        self.embedding_index_path = embedding_index_path
        self.embeddings: np.ndarray
        self.path_to_row: dict[str, int]
        self._load()

    def _load(self) -> None:
        """Load embeddings.npy and embedding_index.csv into memory."""
        if not self.embeddings_path.exists():
            raise FileNotFoundError(
                f"Embeddings file not found: {self.embeddings_path}. "
                f"Run embedding extraction before computing scores."
            )
        if not self.embedding_index_path.exists():
            raise FileNotFoundError(
                f"Embedding index file not found: {self.embedding_index_path}. "
                f"Run embedding extraction before computing scores."
            )

        logger.info("Loading embeddings from %s", self.embeddings_path)
        self.embeddings = load_npy(self.embeddings_path)

        index_df = pd.read_csv(
            self.embedding_index_path, dtype={EmbeddingIndexColumns.IMAGE_PATH: str}
        )
        self.path_to_row = {
            str(row[EmbeddingIndexColumns.IMAGE_PATH]): int(row[EmbeddingIndexColumns.INDEX])
            for _, row in index_df.iterrows()
        }

        if self.embeddings.shape[0] != len(self.path_to_row):
            raise ScoreCalculationError(
                f"Embeddings array has {self.embeddings.shape[0]} rows but the embedding "
                f"index has {len(self.path_to_row)} entries; the two files are out of sync. "
                f"Re-run embedding extraction to regenerate them together."
            )

        logger.info(
            "Loaded %d embeddings of dimension %d.", self.embeddings.shape[0], self.embedding_dim
        )

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of the loaded embeddings."""
        return int(self.embeddings.shape[1]) if self.embeddings.ndim == 2 else 0

    def gather_pair_embeddings(
        self, image1_paths: pd.Series, image2_paths: pd.Series
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized lookup of embeddings for a batch of image pairs.

        Args:
            image1_paths: Series of first-image paths for a batch of pairs.
            image2_paths: Series of second-image paths for a batch of pairs.

        Returns:
            A tuple ``(embeddings1, embeddings2, valid_mask)`` where
            ``embeddings1``/``embeddings2`` are ``(N, embedding_dim)``
            arrays (with a placeholder row of zeros at any invalid
            position) and ``valid_mask`` is a boolean array marking
            which rows had both embeddings available. Callers must
            filter by ``valid_mask`` before using the embeddings.
        """
        row1 = image1_paths.map(self.path_to_row)
        row2 = image2_paths.map(self.path_to_row)
        valid_mask = row1.notna().to_numpy() & row2.notna().to_numpy()

        safe_row1 = np.where(valid_mask, row1.fillna(0).astype(int).to_numpy(), 0)
        safe_row2 = np.where(valid_mask, row2.fillna(0).astype(int).to_numpy(), 0)

        embeddings1 = self.embeddings[safe_row1]
        embeddings2 = self.embeddings[safe_row2]
        return embeddings1, embeddings2, valid_mask


# ------------------------------------------------------------------------------
# Vectorized metric functions
# ------------------------------------------------------------------------------


def compute_cosine_similarity(
    embeddings_a: np.ndarray, embeddings_b: np.ndarray, epsilon: float = 1e-12
) -> np.ndarray:
    """Compute row-wise cosine similarity between two embedding batches.

    Args:
        embeddings_a: ``(N, D)`` array of embeddings.
        embeddings_b: ``(N, D)`` array of embeddings, paired row-wise
            with ``embeddings_a``.
        epsilon: Small value to avoid division by zero for near-zero
            norm embeddings.

    Returns:
        A ``(N,)`` array of cosine similarities in ``[-1, 1]``.
    """
    dot_products = np.einsum("ij,ij->i", embeddings_a, embeddings_b)
    norms_a = np.linalg.norm(embeddings_a, axis=1)
    norms_b = np.linalg.norm(embeddings_b, axis=1)
    denominator = np.clip(norms_a * norms_b, epsilon, None)
    return dot_products / denominator


def compute_euclidean_distance(embeddings_a: np.ndarray, embeddings_b: np.ndarray) -> np.ndarray:
    """Compute row-wise Euclidean (L2) distance between two embedding batches.

    Args:
        embeddings_a: ``(N, D)`` array of embeddings.
        embeddings_b: ``(N, D)`` array of embeddings, paired row-wise
            with ``embeddings_a``.

    Returns:
        A ``(N,)`` array of non-negative Euclidean distances.
    """
    difference = embeddings_a - embeddings_b
    return np.sqrt(np.einsum("ij,ij->i", difference, difference))


def compute_cosine_distance(cosine_similarity: np.ndarray) -> np.ndarray:
    """Compute cosine distance as the complement of cosine similarity.

    Args:
        cosine_similarity: A ``(N,)`` array of cosine similarities.

    Returns:
        A ``(N,)`` array equal to ``1 - cosine_similarity``.
    """
    return 1.0 - cosine_similarity


# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScoringConfig:
    """Configuration controlling scoring behavior.

    All three metrics (cosine similarity, Euclidean distance, cosine
    distance) are always computed and written to the output CSVs,
    matching the project's fixed score schema. ``primary_metric`` only
    records which one a downstream consumer (e.g. threshold selection)
    should default to.

    Attributes:
        primary_metric: The distance/similarity metric considered
            primary for this model, typically taken from
            ``model_config.distance_metric``.
        chunk_size: Number of pair rows processed per streaming chunk.
            Larger values trade memory for fewer, more vectorization-
            efficient batches.
    """

    primary_metric: DistanceMetric = DistanceMetric.COSINE
    chunk_size: int = 50_000

    @classmethod
    def from_model_config(cls, model_config: ModelConfig, chunk_size: int = 50_000) -> "ScoringConfig":
        """Build a :class:`ScoringConfig` from a model's configured distance metric.

        Args:
            model_config: Validated model configuration.
            chunk_size: Number of pair rows processed per streaming chunk.

        Returns:
            A :class:`ScoringConfig` with ``primary_metric`` taken from
            ``model_config.distance_metric``.
        """
        return cls(primary_metric=model_config.distance_metric, chunk_size=chunk_size)


# ------------------------------------------------------------------------------
# Stats
# ------------------------------------------------------------------------------


@dataclass
class ScoringStats:
    """Counters summarizing the outcome of scoring one pair split.

    Attributes:
        total_pairs: Total pair rows read from the split's pair CSV.
        scored_pairs: Pairs for which both embeddings were found and a
            score was successfully computed.
        skipped_missing_embedding: Pairs skipped because at least one
            referenced image had no available embedding.
        failed_pairs: Pairs that could not be scored due to an
            unexpected error during processing.
        processing_time_seconds: Wall-clock time spent scoring this split.
    """

    total_pairs: int = 0
    scored_pairs: int = 0
    skipped_missing_embedding: int = 0
    failed_pairs: int = 0
    processing_time_seconds: float = 0.0

    def summary(self) -> str:
        """Return a human-readable one-line summary of the run.

        Returns:
            A formatted string suitable for logging.
        """
        return (
            f"total_pairs={self.total_pairs}, scored_pairs={self.scored_pairs}, "
            f"skipped_missing_embedding={self.skipped_missing_embedding}, "
            f"failed_pairs={self.failed_pairs}, "
            f"processing_time={self.processing_time_seconds:.2f}s"
        )


# ------------------------------------------------------------------------------
# Score calculation
# ------------------------------------------------------------------------------


class ScoreCalculator:
    """Computes similarity scores for a pair CSV using cached embeddings.

    Streams the input pair file in chunks, vectorizes embedding lookup
    and metric computation per chunk, and streams results directly to
    the output CSV — at no point is either the full input or full
    output held in memory simultaneously.
    """

    def __init__(
        self,
        embedding_repository: EmbeddingRepository,
        scoring_config: ScoringConfig | None = None,
    ) -> None:
        """Initialize the calculator.

        Args:
            embedding_repository: Loaded embeddings and path-to-row index.
            scoring_config: Scoring behavior configuration. Defaults to
                :class:`ScoringConfig` defaults if not provided.
        """
        self.embedding_repository = embedding_repository
        self.scoring_config = scoring_config or ScoringConfig()

    def score_pair_file(
        self, pair_csv_path: Path, output_csv_path: Path, split_label: str
    ) -> ScoringStats:
        """Score every pair in one split's pair CSV and write the result CSV.

        Args:
            pair_csv_path: Path to the split's pair CSV (e.g. ``train_pairs.csv``).
            output_csv_path: Path to write the corresponding score CSV to
                (e.g. ``train_scores.csv``).
            split_label: Human-readable split name, used only for logging.

        Returns:
            :class:`ScoringStats` summarizing this split's run.

        Raises:
            FileNotFoundError: If ``pair_csv_path`` does not exist.
        """
        if not pair_csv_path.exists():
            raise FileNotFoundError(
                f"Pair file not found for split '{split_label}': {pair_csv_path}"
            )

        logger.info("Scoring pairs for split '%s' from %s", split_label, pair_csv_path)
        stats = ScoringStats()
        ensure_dir(output_csv_path.parent)

        start_time = time.perf_counter()
        first_chunk = True

        reader = pd.read_csv(
            pair_csv_path,
            chunksize=self.scoring_config.chunk_size,
            dtype=str,
            keep_default_na=False,
        )

        with output_csv_path.open("w", newline="", encoding="utf-8") as output_file:
            progress = tqdm(desc=f"Scoring {split_label} pairs", unit="pair")
            try:
                for chunk in reader:
                    stats.total_pairs += len(chunk)
                    scored_chunk, skipped_count, failed_count = self._score_chunk(chunk)

                    stats.skipped_missing_embedding += skipped_count
                    stats.failed_pairs += failed_count
                    stats.scored_pairs += len(scored_chunk)

                    scored_chunk.to_csv(output_file, index=False, header=first_chunk)
                    first_chunk = False
                    progress.update(len(chunk))
            finally:
                progress.close()

        stats.processing_time_seconds = time.perf_counter() - start_time

        if stats.skipped_missing_embedding:
            coverage_pct = 100.0 * (1 - stats.skipped_missing_embedding / max(stats.total_pairs, 1))
            logger.warning(
                "Split '%s': %d of %d pairs skipped due to missing embeddings (%.2f%% coverage).",
                split_label,
                stats.skipped_missing_embedding,
                stats.total_pairs,
                coverage_pct,
            )

        logger.info(
            "Finished scoring split '%s': %s", split_label, stats.summary()
        )
        return stats

    def _score_chunk(self, chunk: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
        """Score one chunk of pairs, filtering out rows with missing embeddings.

        Args:
            chunk: A raw chunk of pair rows, read as strings.

        Returns:
            A tuple ``(scored_chunk, skipped_count, failed_count)`` where
            ``scored_chunk`` contains only successfully scored rows with
            :attr:`ScoreOutputColumns.FULL_SCHEMA` columns.
        """
        try:
            embeddings1, embeddings2, valid_mask = self.embedding_repository.gather_pair_embeddings(
                chunk[PairOutputColumns.IMAGE1], chunk[PairOutputColumns.IMAGE2]
            )
        except Exception as exc:  # noqa: BLE001 - keep the run alive on unexpected lookup errors
            logger.error(
                "Failed to gather embeddings for a chunk of %d pairs: %s", len(chunk), exc, exc_info=True
            )
            empty = pd.DataFrame(columns=list(ScoreOutputColumns.FULL_SCHEMA))
            return empty, 0, len(chunk)

        skipped_count = int((~valid_mask).sum())

        valid_chunk = chunk.loc[valid_mask].copy()
        if valid_chunk.empty:
            empty = pd.DataFrame(columns=list(ScoreOutputColumns.FULL_SCHEMA))
            return empty, skipped_count, 0

        valid_embeddings1 = embeddings1[valid_mask]
        valid_embeddings2 = embeddings2[valid_mask]

        try:
            cosine_similarity = compute_cosine_similarity(valid_embeddings1, valid_embeddings2)
            euclidean_distance = compute_euclidean_distance(valid_embeddings1, valid_embeddings2)
            cosine_distance = compute_cosine_distance(cosine_similarity)

            valid_chunk[PairOutputColumns.LABEL] = valid_chunk[PairOutputColumns.LABEL].astype(int)
            valid_chunk[ScoreOutputColumns.COSINE_SIMILARITY] = cosine_similarity.astype(np.float32)
            valid_chunk[ScoreOutputColumns.EUCLIDEAN_DISTANCE] = euclidean_distance.astype(np.float32)
            valid_chunk[ScoreOutputColumns.COSINE_DISTANCE] = cosine_distance.astype(np.float32)
        except (ValueError, TypeError) as exc:
            logger.error(
                "Failed to compute metrics for a chunk of %d valid pairs: %s",
                len(valid_chunk),
                exc,
                exc_info=True,
            )
            empty = pd.DataFrame(columns=list(ScoreOutputColumns.FULL_SCHEMA))
            return empty, skipped_count, len(valid_chunk)

        valid_chunk = valid_chunk[list(ScoreOutputColumns.FULL_SCHEMA)]
        return valid_chunk, skipped_count, 0


# ------------------------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------------------------


class SimilarityScoringPipeline:
    """End-to-end orchestration: load embeddings, score every split, save outputs.

    This is the main entry point for the scoring stage: given a
    dataset's embeddings and pair files, it produces
    ``train_scores.csv``, ``validation_scores.csv``, ``test_scores.csv``,
    and ``scoring_log.csv`` under a configured results directory.
    """

    _SPLIT_TO_PAIR_FILENAME: dict[str, str] = {
        "train": "train_pairs.csv",
        "validation": "validation_pairs.csv",
        "test": "test_pairs.csv",
    }
    _SPLIT_TO_SCORE_FILENAME: dict[str, str] = {
        "train": "train_scores.csv",
        "validation": "validation_scores.csv",
        "test": "test_scores.csv",
    }
    _SCORING_LOG_FILENAME = "scoring_log.csv"

    def __init__(
        self,
        embeddings_path: Path,
        embedding_index_path: Path,
        pairs_dir: Path,
        output_dir: Path,
        scoring_config: ScoringConfig | None = None,
    ) -> None:
        """Initialize the pipeline.

        Args:
            embeddings_path: Path to ``embeddings.npy``.
            embedding_index_path: Path to ``embedding_index.csv``.
            pairs_dir: Directory containing ``train_pairs.csv``,
                ``validation_pairs.csv``, and ``test_pairs.csv``.
            output_dir: Directory to write score CSVs and
                ``scoring_log.csv`` to, typically ``results/``.
            scoring_config: Scoring behavior configuration. Defaults to
                :class:`ScoringConfig` defaults if not provided.
        """
        self.pairs_dir = pairs_dir
        self.output_dir = ensure_dir(output_dir)
        self.embedding_repository = EmbeddingRepository(embeddings_path, embedding_index_path)
        self.scoring_config = scoring_config or ScoringConfig()
        self.calculator = ScoreCalculator(self.embedding_repository, self.scoring_config)
        self.split_stats: dict[str, ScoringStats] = {}

    def run(self) -> dict[str, ScoringStats]:
        """Score every configured split and write all output files.

        Returns:
            A dict mapping split name to its :class:`ScoringStats`.
        """
        logger.info("Starting similarity score computation for all splits.")

        with timer("Similarity score computation (all splits)"):
            for split_name, pair_filename in self._SPLIT_TO_PAIR_FILENAME.items():
                pair_csv_path = self.pairs_dir / pair_filename
                output_csv_path = self.output_dir / self._SPLIT_TO_SCORE_FILENAME[split_name]
                stats = self.calculator.score_pair_file(
                    pair_csv_path=pair_csv_path,
                    output_csv_path=output_csv_path,
                    split_label=split_name,
                )
                self.split_stats[split_name] = stats

        self._save_scoring_log()
        logger.info("Similarity score computation finished for all splits.")
        return self.split_stats

    def _save_scoring_log(self) -> None:
        """Write the aggregated per-split scoring log to ``scoring_log.csv``."""
        log_path = self.output_dir / self._SCORING_LOG_FILENAME
        ensure_dir(log_path.parent)

        with log_path.open("w", newline="", encoding="utf-8") as log_file:
            writer = csv.writer(log_file)
            writer.writerow(
                [
                    "split",
                    "total_pairs",
                    "scored_pairs",
                    "skipped_missing_embedding",
                    "failed_pairs",
                    "processing_time_seconds",
                ]
            )
            for split_name, stats in self.split_stats.items():
                writer.writerow(
                    [
                        split_name,
                        stats.total_pairs,
                        stats.scored_pairs,
                        stats.skipped_missing_embedding,
                        stats.failed_pairs,
                        f"{stats.processing_time_seconds:.4f}",
                    ]
                )

        logger.info("Saved scoring log to %s", log_path)
