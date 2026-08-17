"""Genuine and impostor face pair generation for the fairness pipeline.

Consumes the standardized ``metadata.csv`` produced by
:mod:`fairness_fr.data.preprocess` and produces three CSV files —
``train_pairs.csv``, ``validation_pairs.csv``, ``test_pairs.csv`` — each
containing genuine (same-identity) and impostor (different-identity)
face verification pairs with their demographic attributes preserved,
ready for embedding extraction and similarity scoring.

Design notes:
    - Metadata rows are validated against the filesystem once, up front,
      so every downstream pair is guaranteed to reference an existing
      image file.
    - Pair sampling never materializes the full genuine or impostor
      combination space in memory. For identities with few images, exact
      combinations are enumerated and shuffled; for identities (or the
      impostor cross-identity space) large enough that enumerating every
      combination would be wasteful, pairs are drawn by reservoir-style
      random sampling with a seen-pair set, bounded by a generous
      attempt cap so generation always terminates.
    - Genuine pair count is controlled per identity
      (``max_pairs_per_identity``); impostor pair count is derived from
      it via the configured genuine:impostor ratio, keeping classes
      balanced by construction rather than by post-hoc downsampling.
    - Splitting is stratified by label (genuine/impostor) so the class
      balance achieved during generation is preserved in every split.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from fairness_fr.config import DatasetConfig, PairingConfig
from fairness_fr.config.constants import DEFAULT_RANDOM_SEED, MetadataColumns, PairLabel
from fairness_fr.utils.logging import get_logger
from fairness_fr.utils import ensure_dir, timer

logger = get_logger(__name__)


class MetadataValidationError(Exception):
    """Raised when metadata.csv is missing, empty, or has no valid rows."""


class PairGenerationError(Exception):
    """Raised when pairs cannot be generated or split as configured."""


# ------------------------------------------------------------------------------
# Output schema
# ------------------------------------------------------------------------------


class PairOutputColumns:
    """Canonical column names for generated pair CSV files.

    Kept local to this module (rather than in
    :mod:`fairness_fr.constants`) because this exact schema — with
    ``image1``/``image2`` rather than the ``*_path`` naming used
    elsewhere in the project — is the specific contract requested for
    pair-generation output files.
    """

    IMAGE1 = "image1"
    IMAGE2 = "image2"
    IDENTITY1 = "identity1"
    IDENTITY2 = "identity2"
    DEMOGRAPHIC1 = "demographic1"
    DEMOGRAPHIC2 = "demographic2"
    GENDER1 = "gender1"
    GENDER2 = "gender2"
    AGE1 = "age1"
    AGE2 = "age2"
    LABEL = "label"

    ALL: tuple[str, ...] = (
        IMAGE1,
        IMAGE2,
        IDENTITY1,
        IDENTITY2,
        DEMOGRAPHIC1,
        DEMOGRAPHIC2,
        GENDER1,
        GENDER2,
        AGE1,
        AGE2,
        LABEL,
    )


# ------------------------------------------------------------------------------
# Records
# ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ImageRecord:
    """A single validated image entry from metadata.csv.

    Attributes:
        image_path: Path to the (already preprocessed) image file.
        identity: Identity label the image belongs to.
        demographic_group: Primary demographic label for the image.
        gender: Gender label, if available.
        age: Age or age-group label, if available.
    """

    image_path: str
    identity: str
    demographic_group: str
    gender: str | None
    age: str | None


PairRecord = tuple[ImageRecord, ImageRecord, PairLabel]


# ------------------------------------------------------------------------------
# Metadata loading and validation
# ------------------------------------------------------------------------------


class MetadataLoader:
    """Loads metadata.csv and validates that every referenced image exists.

    Reads the file in chunks so validation scales to metadata files with
    millions of rows without ever holding more than one chunk's worth of
    unvalidated rows in memory at a time.
    """

    def __init__(
        self,
        metadata_csv: Path,
        chunk_size: int = 10_000,
        validate_paths: bool = True,
    ) -> None:
        """Initialize the loader.

        Args:
            metadata_csv: Path to the metadata CSV produced by
                :mod:`fairness_fr.data.preprocess`.
            chunk_size: Number of rows read per chunk during validation.
            validate_paths: If True, drop and log rows whose
                ``image_path`` does not exist on disk. If False, skip
                filesystem checks entirely (useful for fast dry runs).
        """
        self.metadata_csv = metadata_csv
        self.chunk_size = chunk_size
        self.validate_paths = validate_paths

    def load(self) -> pd.DataFrame:
        """Load and validate metadata into a single DataFrame.

        Returns:
            A DataFrame with the standardized metadata columns,
            containing only rows whose image file exists (when
            ``validate_paths`` is True).

        Raises:
            FileNotFoundError: If ``metadata_csv`` does not exist.
            MetadataValidationError: If no valid rows remain after
                validation.
        """
        if not self.metadata_csv.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {self.metadata_csv}. "
                f"Run dataset preprocessing before generating pairs."
            )

        logger.info("Loading and validating metadata from %s", self.metadata_csv)

        reader = pd.read_csv(
            self.metadata_csv,
            chunksize=self.chunk_size,
            dtype=str,
            keep_default_na=False,
        )

        validated_chunks: list[pd.DataFrame] = []
        total_rows = 0
        missing_rows = 0

        for chunk in tqdm(reader, desc="Validating metadata", unit="chunk"):
            total_rows += len(chunk)
            if self.validate_paths:
                exists_mask = chunk[MetadataColumns.IMAGE_PATH].map(lambda p: Path(p).exists())
                missing_rows += int((~exists_mask).sum())
                chunk = chunk.loc[exists_mask]
            validated_chunks.append(chunk)

        metadata = (
            pd.concat(validated_chunks, ignore_index=True)
            if validated_chunks
            else pd.DataFrame(columns=list(MetadataLoader._expected_columns()))
        )

        if missing_rows:
            logger.warning(
                "%d of %d metadata rows reference missing image files and were dropped.",
                missing_rows,
                total_rows,
            )

        if metadata.empty:
            raise MetadataValidationError(
                f"No valid metadata rows remain after validation for {self.metadata_csv}."
            )

        logger.info(
            "Loaded %d valid metadata rows spanning %d identities.",
            len(metadata),
            metadata[MetadataColumns.IDENTITY].nunique(),
        )
        return metadata.reset_index(drop=True)

    @staticmethod
    def _expected_columns() -> tuple[str, str, str, str, str]:
        """Return the expected metadata.csv column names, for empty-frame fallback."""
        return (
            MetadataColumns.IMAGE_PATH,
            MetadataColumns.IDENTITY,
            MetadataColumns.GROUP,
            MetadataColumns.GENDER,
            MetadataColumns.AGE_GROUP,
        )


# ------------------------------------------------------------------------------
# Stats
# ------------------------------------------------------------------------------


@dataclass
class PairGenerationStats:
    """Counters summarizing the outcome of a pair generation run.

    Attributes:
        total_identities: Total number of distinct identities in the
            validated metadata.
        eligible_identities: Number of identities with at least two
            images, i.e. capable of forming a genuine pair.
        genuine_pairs: Total genuine pairs generated.
        impostor_pairs: Total impostor pairs generated.
        duplicate_pairs_skipped: Number of randomly drawn candidate
            pairs discarded because they had already been generated.
    """

    total_identities: int = 0
    eligible_identities: int = 0
    genuine_pairs: int = 0
    impostor_pairs: int = 0
    duplicate_pairs_skipped: int = 0

    def summary(self) -> str:
        """Return a human-readable one-line summary of the run.

        Returns:
            A formatted string suitable for logging.
        """
        return (
            f"identities={self.total_identities}, eligible_identities={self.eligible_identities}, "
            f"genuine_pairs={self.genuine_pairs}, impostor_pairs={self.impostor_pairs}, "
            f"duplicate_pairs_skipped={self.duplicate_pairs_skipped}"
        )


# ------------------------------------------------------------------------------
# Pair generation
# ------------------------------------------------------------------------------


class PairGenerator:
    """Generates genuine and impostor pairs from validated metadata.

    Genuine pairs are sampled within each identity, capped at
    ``pairing_config.max_pairs_per_identity``. Impostor pairs are then
    sampled across distinct identities until the configured
    genuine:impostor ratio is met. Both sampling paths avoid duplicate
    pairs and avoid materializing combinatorially large candidate sets
    for identities or datasets with many images.
    """

    #: When the number of possible unique pairs for a group is at most
    #: this multiple of the number requested, enumerate exact
    #: combinations instead of random sampling — cheaper and gives an
    #: unbiased selection when the candidate space is small.
    _EXACT_ENUMERATION_MULTIPLIER = 5
    _MIN_EXACT_ENUMERATION_THRESHOLD = 200

    def __init__(self, pairing_config: PairingConfig, random_seed: int | None = None) -> None:
        """Initialize the generator.

        Args:
            pairing_config: Validated pairing configuration (ratio, max
                pairs per identity, etc.).
            random_seed: Explicit seed overriding
                ``pairing_config.random_seed``. Falls back to
                :data:`fairness_fr.constants.DEFAULT_RANDOM_SEED` if
                neither is provided, so runs are reproducible by default.
        """
        self.pairing_config = pairing_config
        self.random_seed = (
            random_seed
            if random_seed is not None
            else (pairing_config.random_seed or DEFAULT_RANDOM_SEED)
        )
        self._rng = random.Random(self.random_seed)
        self.stats = PairGenerationStats()

    def generate(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """Generate a full, shuffled genuine + impostor pair DataFrame.

        Args:
            metadata: Validated metadata DataFrame, as produced by
                :meth:`MetadataLoader.load`.

        Returns:
            A DataFrame with columns :attr:`PairOutputColumns.ALL`,
            containing both genuine and impostor pairs in random order.

        Raises:
            PairGenerationError: If fewer than two distinct identities
                are present, or if no genuine pairs could be formed.
        """
        records_by_identity = self._group_by_identity(metadata)
        self.stats.total_identities = len(records_by_identity)

        if len(records_by_identity) < 2:
            raise PairGenerationError(
                f"At least two distinct identities are required to generate pairs; "
                f"found {len(records_by_identity)}."
            )

        logger.info("Generating genuine pairs for %d identities.", len(records_by_identity))
        genuine_pairs = list(self._generate_genuine_pairs(records_by_identity))
        self.stats.genuine_pairs = len(genuine_pairs)

        if not genuine_pairs:
            raise PairGenerationError(
                "No genuine pairs could be generated. Every identity has fewer than "
                "two associated images."
            )

        target_impostor_count = self._compute_impostor_target(len(genuine_pairs))
        logger.info(
            "Generating %d impostor pairs (genuine:impostor ratio=%.2f).",
            target_impostor_count,
            self.pairing_config.genuine_impostor_ratio,
        )
        impostor_pairs = list(
            self._generate_impostor_pairs(records_by_identity, target_impostor_count)
        )
        self.stats.impostor_pairs = len(impostor_pairs)

        all_pairs: list[PairRecord] = genuine_pairs + impostor_pairs
        self._rng.shuffle(all_pairs)

        return self._pairs_to_dataframe(all_pairs)

    # -- grouping ----------------------------------------------------------------

    @staticmethod
    def _group_by_identity(metadata: pd.DataFrame) -> dict[str, list[ImageRecord]]:
        """Group validated metadata rows into per-identity image lists.

        Args:
            metadata: Validated metadata DataFrame.

        Returns:
            Mapping from identity label to the list of that identity's
            :class:`ImageRecord` entries.
        """
        grouped: dict[str, list[ImageRecord]] = {}
        for row in metadata.itertuples(index=False):
            gender = getattr(row, MetadataColumns.GENDER, "") or None
            age = getattr(row, MetadataColumns.AGE_GROUP, "") or None
            record = ImageRecord(
                image_path=getattr(row, MetadataColumns.IMAGE_PATH),
                identity=getattr(row, MetadataColumns.IDENTITY),
                demographic_group=getattr(row, MetadataColumns.GROUP),
                gender=gender,
                age=age,
            )
            grouped.setdefault(record.identity, []).append(record)
        return grouped

    # -- genuine pairs -------------------------------------------------------------

    def _generate_genuine_pairs(
        self, records_by_identity: dict[str, list[ImageRecord]]
    ) -> Iterator[PairRecord]:
        """Yield genuine (same-identity) pairs, capped per identity.

        Args:
            records_by_identity: Mapping from identity to its image records.

        Yields:
            ``(record_a, record_b, PairLabel.GENUINE)`` tuples.
        """
        max_pairs = self.pairing_config.max_pairs_per_identity

        for identity, records in tqdm(
            records_by_identity.items(), desc="Generating genuine pairs", unit="identity"
        ):
            n = len(records)
            if n < 2:
                continue  # a genuine pair requires at least two images of the identity

            self.stats.eligible_identities += 1
            for i, j in self._sample_unique_index_pairs(n, max_pairs):
                yield records[i], records[j], PairLabel.GENUINE

    # -- impostor pairs --------------------------------------------------------------

    def _compute_impostor_target(self, genuine_count: int) -> int:
        """Compute the number of impostor pairs to draw for a given genuine count.

        Args:
            genuine_count: Number of genuine pairs already generated.

        Returns:
            Target impostor pair count implied by
            ``pairing_config.genuine_impostor_ratio`` (genuine / impostor).
        """
        ratio = self.pairing_config.genuine_impostor_ratio
        if ratio <= 0:
            return genuine_count
        return max(round(genuine_count / ratio), 0)

    def _generate_impostor_pairs(
        self,
        records_by_identity: dict[str, list[ImageRecord]],
        target_count: int,
    ) -> Iterator[PairRecord]:
        """Yield impostor (different-identity) pairs via random sampling.

        Args:
            records_by_identity: Mapping from identity to its image records.
            target_count: Number of impostor pairs to generate.

        Yields:
            ``(record_a, record_b, PairLabel.IMPOSTOR)`` tuples.
        """
        identities = list(records_by_identity.keys())
        seen_pairs: set[tuple[str, str]] = set()
        max_attempts = target_count * 20 + 1000
        attempts = 0
        generated = 0

        progress = tqdm(total=target_count, desc="Generating impostor pairs", unit="pair")
        try:
            while generated < target_count and attempts < max_attempts:
                attempts += 1
                identity_a, identity_b = self._rng.sample(identities, 2)
                record_a = self._rng.choice(records_by_identity[identity_a])
                record_b = self._rng.choice(records_by_identity[identity_b])

                pair_key = self._normalized_pair_key(record_a.image_path, record_b.image_path)
                if pair_key in seen_pairs:
                    self.stats.duplicate_pairs_skipped += 1
                    continue

                seen_pairs.add(pair_key)
                generated += 1
                progress.update(1)
                yield record_a, record_b, PairLabel.IMPOSTOR
        finally:
            progress.close()

        if generated < target_count:
            logger.warning(
                "Only generated %d/%d requested impostor pairs after %d attempts; "
                "the dataset may have too few identities or images for the requested count.",
                generated,
                target_count,
                attempts,
            )

    # -- shared sampling helpers -------------------------------------------------

    def _sample_unique_index_pairs(self, n: int, max_pairs: int) -> list[tuple[int, int]]:
        """Sample up to ``max_pairs`` unique unordered index pairs from ``range(n)``.

        Uses exact enumeration for small candidate spaces (unbiased and
        cheap) and falls back to reservoir-style random sampling with a
        seen-set for large ones, so this scales to identities with
        thousands of images without ever materializing every combination.

        Args:
            n: Number of items to choose pairs from.
            max_pairs: Maximum number of pairs to return.

        Returns:
            A list of up to ``max_pairs`` unique ``(i, j)`` index tuples
            with ``i < j``.
        """
        total_possible = n * (n - 1) // 2
        enumeration_ceiling = max(
            max_pairs * self._EXACT_ENUMERATION_MULTIPLIER,
            self._MIN_EXACT_ENUMERATION_THRESHOLD,
        )

        if total_possible <= enumeration_ceiling:
            all_combos = list(combinations(range(n), 2))
            self._rng.shuffle(all_combos)
            return all_combos[:max_pairs]

        seen: set[tuple[int, int]] = set()
        selected: list[tuple[int, int]] = []
        max_attempts = max_pairs * 20 + 100
        attempts = 0

        while len(selected) < max_pairs and attempts < max_attempts:
            attempts += 1
            i, j = self._rng.sample(range(n), 2)
            key = (i, j) if i < j else (j, i)
            if key in seen:
                self.stats.duplicate_pairs_skipped += 1
                continue
            seen.add(key)
            selected.append(key)

        return selected

    @staticmethod
    def _normalized_pair_key(path_a: str, path_b: str) -> tuple[str, str]:
        """Return an order-independent key for a pair, so (a, b) and (b, a) match.

        Args:
            path_a: First image path.
            path_b: Second image path.

        Returns:
            A tuple sorted lexicographically, usable as a set key for
            duplicate detection regardless of argument order.
        """
        return (path_a, path_b) if path_a <= path_b else (path_b, path_a)

    # -- output assembly -------------------------------------------------------------

    @staticmethod
    def _pairs_to_dataframe(pairs: list[PairRecord]) -> pd.DataFrame:
        """Convert generated pair tuples into the canonical output DataFrame.

        Args:
            pairs: List of ``(record_a, record_b, label)`` tuples.

        Returns:
            A DataFrame with columns :attr:`PairOutputColumns.ALL`.
        """
        rows = [
            {
                PairOutputColumns.IMAGE1: a.image_path,
                PairOutputColumns.IMAGE2: b.image_path,
                PairOutputColumns.IDENTITY1: a.identity,
                PairOutputColumns.IDENTITY2: b.identity,
                PairOutputColumns.DEMOGRAPHIC1: a.demographic_group,
                PairOutputColumns.DEMOGRAPHIC2: b.demographic_group,
                PairOutputColumns.GENDER1: a.gender or "",
                PairOutputColumns.GENDER2: b.gender or "",
                PairOutputColumns.AGE1: a.age or "",
                PairOutputColumns.AGE2: b.age or "",
                PairOutputColumns.LABEL: int(label),
            }
            for a, b, label in pairs
        ]
        return pd.DataFrame(rows, columns=list(PairOutputColumns.ALL))


# ------------------------------------------------------------------------------
# Train / validation / test splitting
# ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SplitRatios:
    """Train/validation/test split proportions.

    Attributes:
        train: Fraction of pairs assigned to the training split.
        validation: Fraction of pairs assigned to the validation split.
        test: Fraction of pairs assigned to the test split.

    Raises:
        ValueError: If the three ratios do not sum to 1.0 (within a
            small floating-point tolerance) or if any ratio is negative.
    """

    train: float = 0.7
    validation: float = 0.15
    test: float = 0.15

    def __post_init__(self) -> None:
        """Validate that the ratios are non-negative and sum to 1.0."""
        for split_name, value in (("train", self.train), ("validation", self.validation), ("test", self.test)):
            if value < 0:
                raise ValueError(f"Split ratio '{split_name}' must be non-negative, got {value}.")

        total = self.train + self.validation + self.test
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}.")


class PairSplitter:
    """Splits a pair DataFrame into train/validation/test subsets.

    Splitting is stratified by ``label`` so the genuine/impostor balance
    established during generation is preserved in every split, then the
    combined split is reshuffled so rows are not grouped by label.
    """

    def __init__(self, split_ratios: SplitRatios, random_seed: int | None = None) -> None:
        """Initialize the splitter.

        Args:
            split_ratios: Train/validation/test proportions to apply.
            random_seed: Seed controlling both the per-label shuffle and
                the final reshuffle, for reproducible splits.
        """
        self.split_ratios = split_ratios
        self.random_seed = random_seed if random_seed is not None else DEFAULT_RANDOM_SEED

    def split(self, pairs: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Split ``pairs`` into train/validation/test DataFrames.

        Args:
            pairs: Full pair DataFrame with a ``label`` column, as
                produced by :meth:`PairGenerator.generate`.

        Returns:
            A dict with keys ``"train"``, ``"validation"``, ``"test"``,
            each mapping to a DataFrame with the same columns as ``pairs``.

        Raises:
            PairGenerationError: If ``pairs`` is empty.
        """
        if pairs.empty:
            raise PairGenerationError("Cannot split an empty pairs DataFrame.")

        logger.info(
            "Splitting %d pairs into train/validation/test with ratios %.2f/%.2f/%.2f (seed=%d).",
            len(pairs),
            self.split_ratios.train,
            self.split_ratios.validation,
            self.split_ratios.test,
            self.random_seed,
        )

        per_label_splits: dict[str, list[pd.DataFrame]] = {"train": [], "validation": [], "test": []}

        for _label, group in pairs.groupby(PairOutputColumns.LABEL, sort=False):
            shuffled = group.sample(frac=1.0, random_state=self.random_seed).reset_index(drop=True)
            n = len(shuffled)
            train_end = round(n * self.split_ratios.train)
            validation_end = train_end + round(n * self.split_ratios.validation)

            per_label_splits["train"].append(shuffled.iloc[:train_end])
            per_label_splits["validation"].append(shuffled.iloc[train_end:validation_end])
            per_label_splits["test"].append(shuffled.iloc[validation_end:])

        result: dict[str, pd.DataFrame] = {}
        for split_name, frames in per_label_splits.items():
            combined = (
                pd.concat(frames, ignore_index=True)
                if frames
                else pd.DataFrame(columns=pairs.columns)
            )
            result[split_name] = combined.sample(frac=1.0, random_state=self.random_seed).reset_index(
                drop=True
            )

        for split_name, split_df in result.items():
            genuine_count = int((split_df[PairOutputColumns.LABEL] == int(PairLabel.GENUINE)).sum())
            impostor_count = int((split_df[PairOutputColumns.LABEL] == int(PairLabel.IMPOSTOR)).sum())
            logger.info(
                "Split '%s': %d pairs (genuine=%d, impostor=%d).",
                split_name,
                len(split_df),
                genuine_count,
                impostor_count,
            )

        return result


# ------------------------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------------------------


class PairGenerationPipeline:
    """End-to-end orchestration: load metadata, generate pairs, split, and save.

    This is the main entry point for Week 3 of the pipeline: given a
    dataset's ``metadata.csv``, it produces ``train_pairs.csv``,
    ``validation_pairs.csv``, and ``test_pairs.csv`` under the
    configured output directory.
    """

    _SPLIT_FILENAMES: dict[str, str] = {
        "train": "train_pairs.csv",
        "validation": "validation_pairs.csv",
        "test": "test_pairs.csv",
    }

    def __init__(
        self,
        dataset_config: DatasetConfig,
        pairing_config: PairingConfig,
        output_dir: Path,
        split_ratios: SplitRatios | None = None,
        random_seed: int | None = None,
        validate_image_paths: bool = True,
        metadata_chunk_size: int = 10_000,
    ) -> None:
        """Initialize the pipeline.

        Args:
            dataset_config: Validated dataset configuration, used to
                locate ``metadata.csv``.
            pairing_config: Validated pairing configuration (ratio, max
                pairs per identity, seed).
            output_dir: Directory to write the three split CSV files to,
                typically ``pairs/<dataset_name>/``.
            split_ratios: Train/validation/test proportions. Defaults to
                a 70/15/15 split.
            random_seed: Overrides ``pairing_config.random_seed`` for
                both generation and splitting, if provided.
            validate_image_paths: Whether to check that every metadata
                image path exists on disk before generating pairs.
            metadata_chunk_size: Row chunk size used while validating
                the metadata CSV.
        """
        self.dataset_config = dataset_config
        self.pairing_config = pairing_config
        self.output_dir = output_dir
        self.split_ratios = split_ratios or SplitRatios()
        self.random_seed = (
            random_seed if random_seed is not None else (pairing_config.random_seed or DEFAULT_RANDOM_SEED)
        )

        self.metadata_loader = MetadataLoader(
            metadata_csv=dataset_config.metadata_csv,
            chunk_size=metadata_chunk_size,
            validate_paths=validate_image_paths,
        )
        self.pair_generator = PairGenerator(pairing_config=pairing_config, random_seed=self.random_seed)
        self.pair_splitter = PairSplitter(split_ratios=self.split_ratios, random_seed=self.random_seed)

    def run(self) -> dict[str, pd.DataFrame]:
        """Execute the full pair-generation pipeline for the configured dataset.

        Returns:
            A dict with keys ``"train"``, ``"validation"``, ``"test"``
            mapping to the corresponding pair DataFrames (the same data
            that gets written to disk).

        Raises:
            FileNotFoundError: If the dataset's metadata.csv is missing.
            MetadataValidationError: If no valid metadata rows remain
                after filesystem validation.
            PairGenerationError: If pairs cannot be generated (too few
                identities/images) or the resulting pair set is empty.
        """
        dataset_label = self.dataset_config.name.value
        logger.info("Starting pair generation for dataset '%s'.", dataset_label)

        with timer(f"Pair generation for dataset '{dataset_label}'"):
            metadata = self.metadata_loader.load()
            pairs = self.pair_generator.generate(metadata)
            splits = self.pair_splitter.split(pairs)
            self._save_splits(splits)

        logger.info(
            "Pair generation finished for '%s': %s",
            dataset_label,
            self.pair_generator.stats.summary(),
        )
        return splits

    def _save_splits(self, splits: dict[str, pd.DataFrame]) -> None:
        """Write each split DataFrame to its configured CSV path.

        Args:
            splits: Dict of split name to DataFrame, as returned by
                :meth:`PairSplitter.split`.
        """
        ensure_dir(self.output_dir)
        for split_name, split_df in splits.items():
            output_path = self.output_dir / self._SPLIT_FILENAMES[split_name]
            split_df.to_csv(output_path, index=False)
            logger.info("Saved %d pairs to %s", len(split_df), output_path)
