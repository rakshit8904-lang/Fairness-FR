"""Dataset preprocessing for the fairness face recognition pipeline.

This module is responsible for turning a raw, dataset-specific folder
structure (as downloaded for RFW, BFW, or DemogPairs) into two
standardized artifacts consumed by every later pipeline stage:

1. A directory of preprocessed images, resized to a fixed square
   resolution with aspect ratio preserved via letterbox padding, ready
   for face-embedding extraction.
2. A single ``metadata.csv`` file with the canonical columns
   ``image_path, identity, demographic_group, gender, age`` (as defined
   in :class:`fairness_fr.constants.MetadataColumns`), regardless of how
   wildly the raw folder layouts differ between datasets.

Design notes:
    - Dataset-specific folder-layout knowledge is isolated behind the
      :class:`DatasetScanner` strategy hierarchy, so adding a new dataset
      means adding one small scanner class, not touching the orchestration
      logic in :class:`DatasetPreprocessor`.
    - Everything is streamed: scanners are generators, images are
      processed one at a time, and metadata rows are written
      incrementally to disk. No stage holds the full dataset in memory,
      which matters at RFW/BFW scale (tens of thousands of images).
    - Corrupted or unreadable images are logged and skipped rather than
      raising, so a single bad file cannot abort a multi-hour run; a
      completely empty or failed run does raise, since that indicates a
      configuration problem worth surfacing immediately.
"""

from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError
from tqdm import tqdm

from fairness_fr.config import DatasetConfig
from ..config.constants import DatasetName, MetadataColumns
from fairness_fr.utils.logging import get_logger
from fairness_fr.utils import ensure_dir, timer

logger = get_logger(__name__)


class ImageProcessingError(Exception):
    """Raised when an image passes existence checks but cannot be processed.

    Distinct from a failed :class:`ImageValidator` check: this covers
    errors that occur during the actual resize/convert step (e.g. a
    truncated file that partially decodes), so callers can attribute
    skipped images to the correct failure stage in logs and stats.
    """


class DatasetScannerError(Exception):
    """Raised when a dataset's raw directory is missing, empty, or malformed."""


# ------------------------------------------------------------------------------
# Raw record representation
# ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RawImageRecord:
    """A single raw image discovered by a :class:`DatasetScanner`.

    Attributes:
        image_path: Absolute path to the source image in the raw dataset
            directory.
        identity: Identity label the image belongs to (e.g. a person ID
            or folder name), used later for genuine/impostor pairing.
        demographic_group: Primary demographic label for this image
            (e.g. an ethnicity such as ``"african"``), used for fairness
            evaluation.
        gender: Gender label, if the dataset provides one.
        age: Age or age-group label, if the dataset provides one.
    """

    image_path: Path
    identity: str
    demographic_group: str
    gender: str | None = None
    age: str | None = None


# ------------------------------------------------------------------------------
# Dataset-specific scanners
# ------------------------------------------------------------------------------


class DatasetScanner(ABC):
    """Abstract strategy for discovering raw images in a dataset-specific layout.

    Each concrete scanner knows only how to walk *one* dataset's raw
    folder structure and yield standardized :class:`RawImageRecord`
    instances; it does not open, validate, or transform image content.
    """

    def __init__(self, dataset_config: DatasetConfig) -> None:
        """Initialize the scanner.

        Args:
            dataset_config: Validated configuration describing where this
                dataset's raw files live and what demographic labels it has.
        """
        self.dataset_config = dataset_config

    @abstractmethod
    def scan(self) -> Iterator[RawImageRecord]:
        """Lazily yield every image found in the raw dataset directory.

        Implementations must be generators (or return an iterator) so the
        caller never needs the full file listing in memory at once.

        Yields:
            :class:`RawImageRecord` instances, one per discovered image.

        Raises:
            DatasetScannerError: If the raw directory is missing or no
                images matching the expected layout are found.
        """
        raise NotImplementedError

    def _require_raw_dir(self) -> Path:
        """Return the raw directory, raising if it does not exist.

        Returns:
            The dataset's raw directory path.

        Raises:
            DatasetScannerError: If the directory does not exist.
        """
        raw_dir = self.dataset_config.raw_dir
        if not raw_dir.exists() or not raw_dir.is_dir():
            raise DatasetScannerError(
                f"Raw dataset directory not found for '{self.dataset_config.name.value}': "
                f"{raw_dir}. Verify 'raw_dir' in the dataset config and that the dataset "
                f"has been downloaded."
            )
        return raw_dir

    @staticmethod
    def _iter_subdirs(directory: Path) -> Iterator[Path]:
        """Yield immediate subdirectories of ``directory`` in sorted order.

        Sorting makes scans deterministic across runs and machines, which
        matters for reproducible metadata ordering on large datasets.

        Args:
            directory: Directory to list.

        Yields:
            Subdirectory paths.
        """
        yield from sorted((p for p in directory.iterdir() if p.is_dir()), key=lambda p: p.name)


class RFWScanner(DatasetScanner):
    """Scanner for RFW (Racial Faces in-the-Wild).

    Expected raw layout::

        raw_dir/<ethnicity>/<identity>/<image>.jpg

    where ``<ethnicity>`` is one of Caucasian, Asian, Indian, African and
    is used directly as the demographic group.
    """

    def scan(self) -> Iterator[RawImageRecord]:
        """Yield one record per image under ``raw_dir/<ethnicity>/<identity>/``.

        Yields:
            :class:`RawImageRecord` with ``demographic_group`` set to the
            ethnicity subfolder name.

        Raises:
            DatasetScannerError: If the raw directory is missing or
                contains no matching images.
        """
        raw_dir = self._require_raw_dir()
        extension = self.dataset_config.image_extension
        found_any = False

        for ethnicity_dir in self._iter_subdirs(raw_dir):
            demographic_group = ethnicity_dir.name
            for identity_dir in self._iter_subdirs(ethnicity_dir):
                identity = identity_dir.name
                for image_path in sorted(identity_dir.glob(f"*{extension}")):
                    found_any = True
                    yield RawImageRecord(
                        image_path=image_path,
                        identity=identity,
                        demographic_group=demographic_group,
                    )

        if not found_any:
            raise DatasetScannerError(
                f"No images matching '*{extension}' found under {raw_dir} "
                f"using the expected RFW layout raw_dir/<ethnicity>/<identity>/*."
            )


class BFWScanner(DatasetScanner):
    """Scanner for BFW (Balanced Faces in the Wild).

    Expected raw layout::

        raw_dir/<ethnicity>_<gender>/<identity>/<image>.jpg

    e.g. ``raw_dir/asian_females/identity_001/*.jpg``. The top-level
    folder name is split on the last underscore into ethnicity and
    gender, so both attributes are captured from a single scan.
    """

    def scan(self) -> Iterator[RawImageRecord]:
        """Yield one record per image under ``raw_dir/<ethnicity>_<gender>/<identity>/``.

        Yields:
            :class:`RawImageRecord` with both ``demographic_group``
            (ethnicity) and ``gender`` populated.

        Raises:
            DatasetScannerError: If the raw directory is missing, no
                images are found, or a top-level folder name cannot be
                split into an ethnicity/gender pair.
        """
        raw_dir = self._require_raw_dir()
        extension = self.dataset_config.image_extension
        found_any = False

        for group_dir in self._iter_subdirs(raw_dir):
            ethnicity, gender = self._parse_group_folder(group_dir.name)
            for identity_dir in self._iter_subdirs(group_dir):
                identity = identity_dir.name
                for image_path in sorted(identity_dir.glob(f"*{extension}")):
                    found_any = True
                    yield RawImageRecord(
                        image_path=image_path,
                        identity=identity,
                        demographic_group=ethnicity,
                        gender=gender,
                    )

        if not found_any:
            raise DatasetScannerError(
                f"No images matching '*{extension}' found under {raw_dir} "
                f"using the expected BFW layout raw_dir/<ethnicity>_<gender>/<identity>/*."
            )

    @staticmethod
    def _parse_group_folder(folder_name: str) -> tuple[str, str]:
        """Split a BFW top-level folder name into (ethnicity, gender).

        Args:
            folder_name: Folder name such as ``"asian_females"``.

        Returns:
            A ``(ethnicity, gender)`` tuple.

        Raises:
            DatasetScannerError: If ``folder_name`` has no underscore to
                split on.
        """
        if "_" not in folder_name:
            raise DatasetScannerError(
                f"BFW group folder '{folder_name}' does not match the expected "
                f"'<ethnicity>_<gender>' naming convention."
            )
        ethnicity, _, gender = folder_name.rpartition("_")
        return ethnicity, gender


class DemogPairsScanner(DatasetScanner):
    """Scanner for DemogPairs.

    Expected raw layout::

        raw_dir/<identity>/<image>.jpg
        raw_dir/demographics.csv   (optional)

    DemogPairs does not organize images into demographic folders, so
    group/gender/age labels are looked up from an optional
    ``demographics.csv`` with columns ``identity, group, gender, age``.
    Identities absent from that file fall back to a ``"unknown"`` group
    rather than being dropped, so preprocessing never silently loses data.
    """

    _UNKNOWN_GROUP = "unknown"

    def scan(self) -> Iterator[RawImageRecord]:
        """Yield one record per image under ``raw_dir/<identity>/``.

        Yields:
            :class:`RawImageRecord`, enriched with group/gender/age from
            ``demographics.csv`` when available for that identity.

        Raises:
            DatasetScannerError: If the raw directory is missing or no
                images are found.
        """
        raw_dir = self._require_raw_dir()
        extension = self.dataset_config.image_extension
        demographics = self._load_demographics_lookup(raw_dir)
        found_any = False

        for identity_dir in self._iter_subdirs(raw_dir):
            identity = identity_dir.name
            demo = demographics.get(identity, {})
            for image_path in sorted(identity_dir.glob(f"*{extension}")):
                found_any = True
                yield RawImageRecord(
                    image_path=image_path,
                    identity=identity,
                    demographic_group=demo.get("group", self._UNKNOWN_GROUP),
                    gender=demo.get("gender"),
                    age=demo.get("age"),
                )

        if not found_any:
            raise DatasetScannerError(
                f"No images matching '*{extension}' found under {raw_dir} "
                f"using the expected DemogPairs layout raw_dir/<identity>/*."
            )

    def _load_demographics_lookup(self, raw_dir: Path) -> dict[str, dict[str, str]]:
        """Load an optional identity -> demographic-fields lookup table.

        Args:
            raw_dir: The dataset's raw directory, expected to optionally
                contain a ``demographics.csv`` file.

        Returns:
            A mapping from identity to a dict of available fields among
            ``group``, ``gender``, ``age``. Empty if no file is present.
        """
        demographics_path = raw_dir / "demographics.csv"
        if not demographics_path.exists():
            logger.warning(
                "No demographics.csv found for DemogPairs at %s; all identities "
                "will be labeled with demographic_group='%s'.",
                demographics_path,
                self._UNKNOWN_GROUP,
            )
            return {}

        lookup: dict[str, dict[str, str]] = {}
        with demographics_path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                identity = row.get("identity")
                if not identity:
                    continue
                lookup[identity] = {
                    key: value
                    for key, value in (
                        ("group", row.get("group")),
                        ("gender", row.get("gender")),
                        ("age", row.get("age")),
                    )
                    if value
                }
        return lookup


_SCANNER_REGISTRY: dict[DatasetName, type[DatasetScanner]] = {
    DatasetName.RFW: RFWScanner,
    DatasetName.BFW: BFWScanner,
    DatasetName.DEMOGPAIRS: DemogPairsScanner,
}


def get_scanner(dataset_config: DatasetConfig) -> DatasetScanner:
    """Resolve the appropriate :class:`DatasetScanner` for a dataset config.

    Args:
        dataset_config: Validated dataset configuration.

    Returns:
        A scanner instance ready to have :meth:`DatasetScanner.scan` called.

    Raises:
        ValueError: If no scanner is registered for the dataset's name.
    """
    scanner_cls = _SCANNER_REGISTRY.get(dataset_config.name)
    if scanner_cls is None:
        raise ValueError(
            f"No DatasetScanner registered for dataset '{dataset_config.name.value}'. "
            f"Known datasets: {[name.value for name in _SCANNER_REGISTRY]}."
        )
    return scanner_cls(dataset_config)


# ------------------------------------------------------------------------------
# Image validation and transformation
# ------------------------------------------------------------------------------


class ImageValidator:
    """Validates that an image file is readable and not corrupted.

    Validation is deliberately cheap (a single ``Image.verify()`` pass)
    so it can run on every file of a large dataset without becoming the
    bottleneck; full decoding happens once, later, inside
    :class:`ImagePreprocessor`.
    """

    @staticmethod
    def is_valid(image_path: Path) -> bool:
        """Check whether an image file can be opened and parsed by Pillow.

        Args:
            image_path: Path to the candidate image file.

        Returns:
            True if the file is a readable, non-corrupted image; False
            otherwise. Failures are logged as warnings, not raised, so
            the caller can skip-and-continue over bad files.
        """
        try:
            with Image.open(image_path) as image:
                image.verify()
            return True
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            logger.warning("Corrupted or unreadable image skipped: %s (%s)", image_path, exc)
            return False


class ImagePreprocessor:
    """Resizes images to a fixed square resolution, preserving aspect ratio.

    Uses a "letterbox" resize: the image is scaled so its longer side
    equals ``target_size``, then centered on a black square canvas of
    ``target_size x target_size``. This avoids the geometric distortion
    a naive stretch-to-square resize would introduce, which matters for
    face recognition models sensitive to facial proportions.
    """

    def __init__(self, target_size: int) -> None:
        """Initialize the preprocessor.

        Args:
            target_size: Target width and height in pixels, typically
                the input resolution required by the embedding model.

        Raises:
            ValueError: If ``target_size`` is not a positive integer.
        """
        if target_size <= 0:
            raise ValueError(f"target_size must be positive, got {target_size}.")
        self.target_size = target_size

    def process(self, image_path: Path) -> Image.Image:
        """Load, orient-correct, convert to RGB, and letterbox-resize an image.

        Args:
            image_path: Path to a source image that has already passed
                :class:`ImageValidator`.

        Returns:
            A processed ``PIL.Image.Image`` of size
            ``(target_size, target_size)`` in RGB mode.

        Raises:
            ImageProcessingError: If the image cannot be opened or
                transformed, e.g. due to truncation not caught by
                validation.
        """
        try:
            with Image.open(image_path) as image:
                image = ImageOps.exif_transpose(image)
                if image is None:
                    raise ImageProcessingError(
                        f"EXIF transpose returned no image for {image_path}."
                    )
                image = image.convert("RGB")
                return self._resize_with_letterbox(image)
        except (UnidentifiedImageError, OSError) as exc:
            raise ImageProcessingError(f"Failed to process image {image_path}: {exc}") from exc

    def _resize_with_letterbox(self, image: Image.Image) -> Image.Image:
        """Scale ``image`` to fit within a square canvas, preserving aspect ratio.

        Args:
            image: An RGB image already loaded into memory.

        Returns:
            A new RGB image of size ``(target_size, target_size)`` with
            the original content centered and padded with black.
        """
        original_width, original_height = image.size
        scale = self.target_size / max(original_width, original_height)
        new_width = max(1, round(original_width * scale))
        new_height = max(1, round(original_height * scale))

        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        canvas = Image.new("RGB", (self.target_size, self.target_size), color=(0, 0, 0))
        offset = (
            (self.target_size - new_width) // 2,
            (self.target_size - new_height) // 2,
        )
        canvas.paste(resized, offset)
        return canvas


# ------------------------------------------------------------------------------
# Metadata output
# ------------------------------------------------------------------------------


class MetadataWriter:
    """Streams standardized metadata rows to a CSV file, one row at a time.

    Writing incrementally (rather than accumulating rows in a DataFrame)
    keeps memory usage flat regardless of dataset size, and means a
    crash partway through preprocessing still leaves a usable partial
    metadata file instead of losing everything.
    """

    FIELDNAMES: tuple[str, str, str, str, str] = (
        MetadataColumns.IMAGE_PATH,
        MetadataColumns.IDENTITY,
        MetadataColumns.GROUP,
        MetadataColumns.GENDER,
        MetadataColumns.AGE_GROUP,
    )

    _FLUSH_INTERVAL = 500

    def __init__(self, output_path: Path) -> None:
        """Open the metadata CSV for writing and emit its header row.

        Args:
            output_path: Destination path for ``metadata.csv``. Parent
                directories are created if missing.
        """
        ensure_dir(output_path.parent)
        self.output_path = output_path
        self._file = output_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=list(self.FIELDNAMES))
        self._writer.writeheader()
        self._rows_written = 0

    def write_row(
        self,
        image_path: Path,
        identity: str,
        demographic_group: str,
        gender: str | None,
        age: str | None,
    ) -> None:
        """Write a single metadata row and periodically flush to disk.

        Args:
            image_path: Path to the *processed* image (not the raw source).
            identity: Identity label.
            demographic_group: Demographic group label.
            gender: Gender label, or None if unavailable.
            age: Age or age-group label, or None if unavailable.
        """
        self._writer.writerow(
            {
                MetadataColumns.IMAGE_PATH: str(image_path),
                MetadataColumns.IDENTITY: identity,
                MetadataColumns.GROUP: demographic_group,
                MetadataColumns.GENDER: gender or "",
                MetadataColumns.AGE_GROUP: age or "",
            }
        )
        self._rows_written += 1
        if self._rows_written % self._FLUSH_INTERVAL == 0:
            self._file.flush()

    def close(self) -> None:
        """Flush and close the underlying file handle."""
        self._file.flush()
        self._file.close()
        logger.info("Wrote %d metadata rows to %s", self._rows_written, self.output_path)

    def __enter__(self) -> "MetadataWriter":
        """Enable use as a context manager."""
        return self

    def __exit__(self, *_exc_info: object) -> None:
        """Ensure the file is closed when the context exits."""
        self.close()


# ------------------------------------------------------------------------------
# Stats
# ------------------------------------------------------------------------------


@dataclass
class PreprocessingStats:
    """Counters summarizing the outcome of a preprocessing run.

    Attributes:
        total_found: Total number of images discovered by the scanner.
        processed: Number of images successfully processed and saved.
        skipped_corrupted: Number of images skipped due to failing
            validation or raising during transformation.
        skipped_missing: Number of images listed by the scanner whose
            file path did not actually exist on disk.
    """

    total_found: int = 0
    processed: int = 0
    skipped_corrupted: int = 0
    skipped_missing: int = 0

    def summary(self) -> str:
        """Return a human-readable one-line summary of the run.

        Returns:
            A formatted string suitable for logging.
        """
        return (
            f"found={self.total_found}, processed={self.processed}, "
            f"skipped_corrupted={self.skipped_corrupted}, "
            f"skipped_missing={self.skipped_missing}"
        )


# ------------------------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------------------------


class DatasetPreprocessor:
    """Orchestrates scanning, validating, resizing, and saving a full dataset.

    This is the main entry point for Week 2 of the pipeline: given a
    validated :class:`DatasetConfig`, it produces a clean processed image
    directory and a standardized ``metadata.csv``, streaming throughout
    so it scales to large datasets without excessive memory use.
    """

    def __init__(
        self,
        dataset_config: DatasetConfig,
        target_size: int,
        overwrite_existing: bool = False,
    ) -> None:
        """Initialize the preprocessor for one dataset.

        Args:
            dataset_config: Validated configuration for the dataset to
                preprocess (raw/processed paths, image extension, etc.).
            target_size: Square resolution to resize processed images to,
                typically taken from the target model's ``input_size``.
            overwrite_existing: If False (default), images already present
                at the destination path are left untouched and their
                existing file is reused for the metadata row — this makes
                re-running preprocessing after an interruption resume
                cheaply instead of redoing all work. If True, every image
                is reprocessed and overwritten.
        """
        self.dataset_config = dataset_config
        self.image_preprocessor = ImagePreprocessor(target_size=target_size)
        self.validator = ImageValidator()
        self.overwrite_existing = overwrite_existing
        self.stats = PreprocessingStats()

    def run(self) -> PreprocessingStats:
        """Execute the full preprocessing pipeline for the configured dataset.

        Returns:
            :class:`PreprocessingStats` summarizing how many images were
            found, processed, and skipped.

        Raises:
            DatasetScannerError: If the raw dataset directory is missing
                or contains no images in the expected layout.
            RuntimeError: If scanning succeeds but zero images are
                successfully processed, which indicates a systemic
                problem (e.g. every file corrupted, wrong extension
                configured) rather than isolated bad files.
        """
        scanner = get_scanner(self.dataset_config)
        processed_dir = ensure_dir(self.dataset_config.processed_dir)
        metadata_path = self.dataset_config.metadata_csv

        dataset_label = self.dataset_config.name.value
        logger.info("Starting preprocessing for dataset '%s'.", dataset_label)

        with timer(f"Preprocessing dataset '{dataset_label}'"):
            with MetadataWriter(metadata_path) as writer:
                progress = tqdm(scanner.scan(), desc=f"Preprocessing {dataset_label}", unit="img")
                for record in progress:
                    self.stats.total_found += 1
                    self._process_record(record, processed_dir, writer)
                    progress.set_postfix(
                        processed=self.stats.processed,
                        skipped=self.stats.skipped_corrupted + self.stats.skipped_missing,
                    )

        if self.stats.processed == 0:
            raise RuntimeError(
                f"Preprocessing produced zero usable images for dataset '{dataset_label}'. "
                f"Stats: {self.stats.summary()}. Check raw_dir contents and image_extension."
            )

        logger.info("Finished preprocessing '%s': %s", dataset_label, self.stats.summary())
        return self.stats

    def _process_record(
        self,
        record: RawImageRecord,
        processed_dir: Path,
        writer: MetadataWriter,
    ) -> None:
        """Validate, transform, save, and record metadata for one raw image.

        All expected failure modes (missing file, corrupted image,
        processing error) are caught here and turned into stats/log
        entries rather than propagating, so one bad image never aborts
        the run. Unexpected exceptions are logged with a full traceback
        and also counted as skipped, keeping the run alive.

        Args:
            record: The raw image record to process.
            processed_dir: Root directory to save processed images under.
            writer: Open :class:`MetadataWriter` to append a row to on success.
        """
        try:
            if not record.image_path.exists():
                logger.warning("Listed image path does not exist, skipping: %s", record.image_path)
                self.stats.skipped_missing += 1
                return

            output_path = processed_dir / self._relative_output_path(record)

            if output_path.exists() and not self.overwrite_existing:
                writer.write_row(
                    image_path=output_path,
                    identity=record.identity,
                    demographic_group=record.demographic_group,
                    gender=record.gender,
                    age=record.age,
                )
                self.stats.processed += 1
                return

            if not self.validator.is_valid(record.image_path):
                self.stats.skipped_corrupted += 1
                return

            processed_image = self.image_preprocessor.process(record.image_path)
            ensure_dir(output_path.parent)
            processed_image.save(output_path, format="JPEG", quality=95)

            writer.write_row(
                image_path=output_path,
                identity=record.identity,
                demographic_group=record.demographic_group,
                gender=record.gender,
                age=record.age,
            )
            self.stats.processed += 1

        except ImageProcessingError as exc:
            logger.warning("Skipping image due to processing error: %s", exc)
            self.stats.skipped_corrupted += 1
        except Exception as exc:  # noqa: BLE001 - deliberately broad to keep large runs alive
            logger.error(
                "Unexpected error while processing %s: %s", record.image_path, exc, exc_info=True
            )
            self.stats.skipped_corrupted += 1

    @staticmethod
    def _relative_output_path(record: RawImageRecord) -> Path:
        """Compute the processed-image path for a raw record, relative to processed_dir.

        Args:
            record: The raw image record.

        Returns:
            A path of the form ``<demographic_group>/<identity>/<filename>``,
            keeping the processed directory organized the same way the
            raw directory typically is, which makes spot-checking output
            straightforward.
        """
        return Path(record.demographic_group) / record.identity / record.image_path.name
