"""Face embedding extraction for the fairness evaluation pipeline.

Reads the unique set of images referenced across a dataset's
``train_pairs.csv``, ``validation_pairs.csv``, and ``test_pairs.csv``
(as produced by :mod:`fairness_fr.data.generate_pairs`), runs each
image exactly once through the configured pretrained face recognition
model in batches, and persists three artifacts:

- ``embeddings.npy`` — a ``(N, embedding_dim)`` float32 array.
- ``embedding_index.csv`` — maps each row of ``embeddings.npy`` to the
  image path it corresponds to.
- ``inference_log.csv`` — a per-image audit trail of status (cached,
  success, skipped, failed), timing, and any error encountered.

Design notes:
    - Model loading is abstracted behind :class:`EmbeddingModel` with two
      concrete backends — TorchScript (``.pt``/``.pth``) and ONNX
      (``.onnx``) — selected automatically from the configured weights
      file extension via :func:`load_embedding_model`. This satisfies
      "support the model architecture defined in the configuration"
      without hardcoding any single architecture, while keeping the
      surface area to exactly one new file as instructed.
    - Deduplication happens twice: within a single run (each unique
      image path across all three pair files is embedded once,
      regardless of how many pairs reference it) and across runs (an
      existing ``embeddings.npy`` / ``embedding_index.csv`` pair is
      loaded as a cache, and only images missing from it are computed).
    - Corrupted, missing, or otherwise unprocessable images are logged
      and skipped rather than aborting the run; a run that produces zero
      embeddings overall (fresh and cached combined) raises, since that
      indicates a systemic configuration problem.
"""

from __future__ import annotations

import csv
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from fairness_fr.config import ModelConfig
from fairness_fr.config.constants import DEFAULT_RANDOM_SEED
from fairness_fr.data.generate_pairs import PairOutputColumns
from fairness_fr.data.preprocess import ImageValidator
from fairness_fr.utils.logging import get_logger
from fairness_fr.config.settings import get_settings
from fairness_fr.utils import chunked, ensure_dir, get_device, load_npy, set_seed, timer

logger = get_logger(__name__)


class EmbeddingModelLoadError(Exception):
    """Raised when a configured model's weights cannot be loaded."""


class EmbeddingExtractionError(Exception):
    """Raised when embedding extraction cannot proceed or produces no output."""


class ProcessingStatus(str, Enum):
    """Per-image outcome recorded in ``inference_log.csv``."""

    SUCCESS = "success"
    CACHED = "cached"
    SKIPPED_MISSING = "skipped_missing"
    SKIPPED_CORRUPTED = "skipped_corrupted"
    FAILED = "failed"


class EmbeddingIndexColumns:
    """Canonical column names for ``embedding_index.csv``."""

    INDEX = "index"
    IMAGE_PATH = "image_path"


def _save_embeddings_array_atomic(path: Path, array: np.ndarray) -> None:
    """Write the embeddings array to disk atomically.

    Writes to a temporary sibling file whose name already ends in
    ``.npy`` (so :func:`numpy.save` does not append a second ``.npy``
    suffix), then renames it into place. This avoids leaving a
    truncated ``embeddings.npy`` behind if the process is interrupted
    mid-write, which matters for large embedding matrices that can take
    a while to serialize.

    Args:
        path: Destination ``.npy`` file path.
        array: Embeddings array to serialize.
    """
    ensure_dir(path.parent)
    tmp_path = path.with_name(path.name + ".tmp.npy")
    np.save(tmp_path, array)
    os.replace(tmp_path, path)


# ------------------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------------------


class EmbeddingModel(ABC):
    """Abstract interface for a loaded face-embedding model.

    Concrete subclasses are responsible only for running inference on
    an already-preprocessed batch tensor; image loading and
    normalization are handled separately by
    :class:`ImageBatchPreprocessor` so any backend can be swapped in
    without touching preprocessing logic.
    """

    def __init__(self, model_config: ModelConfig, device: str) -> None:
        """Initialize the model wrapper.

        Args:
            model_config: Validated configuration for the model being loaded.
            device: Resolved compute device string (e.g. ``"cpu"``, ``"cuda"``).
        """
        self.model_config = model_config
        self.device = device

    @abstractmethod
    def embed_batch(self, batch: torch.Tensor) -> np.ndarray:
        """Run inference on a preprocessed batch of images.

        Args:
            batch: A ``(B, 3, H, W)`` float32 tensor of normalized images.

        Returns:
            A ``(B, embedding_dim)`` float32 numpy array of embeddings.
        """
        raise NotImplementedError


class TorchScriptEmbeddingModel(EmbeddingModel):
    """Embedding model backed by a TorchScript (``.pt``/``.pth``) checkpoint."""

    def __init__(self, model_config: ModelConfig, device: str) -> None:
        """Load a TorchScript module and prepare it for deterministic inference.

        Args:
            model_config: Validated model configuration; ``weights_path``
                must point to a TorchScript-serialized module.
            device: Resolved compute device string.

        Raises:
            EmbeddingModelLoadError: If the weights file cannot be loaded.
        """
        super().__init__(model_config, device)
        try:
            self._module = torch.jit.load(str(model_config.weights_path), map_location=device)
        except (FileNotFoundError, RuntimeError, OSError) as exc:
            raise EmbeddingModelLoadError(
                f"Failed to load TorchScript weights for model "
                f"'{model_config.name.value}' from {model_config.weights_path}: {exc}"
            ) from exc

        self._module.eval()
        self._module.to(device)

    @torch.no_grad()
    def embed_batch(self, batch: torch.Tensor) -> np.ndarray:
        """Run the TorchScript module on a batch and return embeddings.

        Args:
            batch: A ``(B, 3, H, W)`` float32 tensor.

        Returns:
            A ``(B, embedding_dim)`` float32 numpy array.
        """
        batch = batch.to(self.device)
        output = self._module(batch)
        if isinstance(output, (tuple, list)):
            output = output[0]
        return output.detach().cpu().numpy().astype(np.float32)


class ONNXEmbeddingModel(EmbeddingModel):
    """Embedding model backed by an ONNX (``.onnx``) graph via onnxruntime."""

    def __init__(self, model_config: ModelConfig, device: str) -> None:
        """Load an ONNX inference session with device-appropriate execution providers.

        Args:
            model_config: Validated model configuration; ``weights_path``
                must point to a ``.onnx`` file.
            device: Resolved compute device string; ``"cuda"``-prefixed
                values request the CUDA execution provider with a CPU
                fallback registered alongside it.

        Raises:
            EmbeddingModelLoadError: If onnxruntime is not installed or
                the model fails to load.
        """
        super().__init__(model_config, device)
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise EmbeddingModelLoadError(
                "onnxruntime is required to load .onnx model weights. "
                "Install it via requirements.txt."
            ) from exc

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device.startswith("cuda")
            else ["CPUExecutionProvider"]
        )
        try:
            self._session = ort.InferenceSession(str(model_config.weights_path), providers=providers)
        except Exception as exc:  # noqa: BLE001 - onnxruntime raises its own broad exception types
            raise EmbeddingModelLoadError(
                f"Failed to load ONNX weights for model '{model_config.name.value}' "
                f"from {model_config.weights_path}: {exc}"
            ) from exc

        self._input_name = self._session.get_inputs()[0].name

    def embed_batch(self, batch: torch.Tensor) -> np.ndarray:
        """Run the ONNX session on a batch and return embeddings.

        Args:
            batch: A ``(B, 3, H, W)`` float32 tensor.

        Returns:
            A ``(B, embedding_dim)`` float32 numpy array.
        """
        batch_np = batch.cpu().numpy().astype(np.float32)
        outputs = self._session.run(None, {self._input_name: batch_np})
        return np.asarray(outputs[0], dtype=np.float32)


def load_embedding_model(model_config: ModelConfig, device: str) -> EmbeddingModel:
    """Resolve and load the appropriate :class:`EmbeddingModel` backend.

    The backend is selected from the file extension of
    ``model_config.weights_path``: ``.onnx`` loads an
    :class:`ONNXEmbeddingModel`; ``.pt``, ``.pth``, or ``.torchscript``
    load a :class:`TorchScriptEmbeddingModel`.

    Args:
        model_config: Validated model configuration.
        device: Resolved compute device string.

    Returns:
        A ready-to-use :class:`EmbeddingModel` instance.

    Raises:
        EmbeddingModelLoadError: If the weights file extension is not
            recognized, or if loading fails.
    """
    weights_path = Path(model_config.weights_path)

    # DeepFace models are identified by the special
    # weights_path value "deepface".
    if str(model_config.weights_path).lower() == "deepface":
        logger.info(
            "Loading DeepFace embedding model '%s'",
            model_config.name.value,
        )
        return DeepFaceEmbeddingModel(model_config, device)

    suffix = weights_path.suffix.lower()

    if suffix == ".onnx":
        logger.info("Loading ONNX embedding model '%s' from %s", model_config.name.value, weights_path)
        return ONNXEmbeddingModel(model_config, device)

    if suffix in (".pt", ".pth", ".torchscript"):
        logger.info(
            "Loading TorchScript embedding model '%s' from %s", model_config.name.value, weights_path
        )
        return TorchScriptEmbeddingModel(model_config, device)

    raise EmbeddingModelLoadError(
        f"Unsupported weights file extension '{suffix}' for model '{model_config.name.value}'. "
        f"Expected one of: .onnx, .pt, .pth, .torchscript."
    )


# ------------------------------------------------------------------------------
# Image preprocessing for inference
# ------------------------------------------------------------------------------


class ImageBatchPreprocessor:
    """Loads an already-preprocessed image and applies model-specific normalization.

    Images are expected to already be resized to a square resolution by
    :mod:`fairness_fr.data.preprocess`; this class defensively resizes
    again if the on-disk size does not match ``model_config.input_size``,
    then applies the per-channel mean/std normalization the target model
    was trained with.
    """

    def __init__(self, model_config: ModelConfig) -> None:
        """Initialize the preprocessor.

        Args:
            model_config: Validated model configuration providing
                ``input_size``, ``normalization_mean``, and
                ``normalization_std``.
        """
        self.model_config = model_config
        self._mean = np.asarray(model_config.normalization_mean, dtype=np.float32).reshape(3, 1, 1)
        self._std = np.asarray(model_config.normalization_std, dtype=np.float32).reshape(3, 1, 1)

    def load_and_preprocess(self, image_path: Path) -> np.ndarray:
        """Load, resize if needed, and normalize a single image.

        Args:
            image_path: Path to an image that has already passed
                :class:`fairness_fr.data.preprocess.ImageValidator`.

        Returns:
            A ``(3, H, W)`` float32 array, normalized per the model's
            configured mean and standard deviation.

        Raises:
            UnidentifiedImageError: If Pillow cannot parse the file.
            OSError: If the file cannot be read.
        """
        target_size = self.model_config.input_size
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if image.size != (target_size, target_size):
                image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
            array = np.asarray(image, dtype=np.float32) / 255.0

        array = array.transpose(2, 0, 1)  # HWC -> CHW
        array = (array - self._mean) / self._std
        return array.astype(np.float32)


# ------------------------------------------------------------------------------
# Logging / stats
# ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InferenceLogEntry:
    """A single row of ``inference_log.csv``.

    Attributes:
        image_path: Path to the image this entry describes.
        status: Outcome of processing this image.
        duration_seconds: Time attributed to this image. For batched
            inference this is the batch's inference time divided evenly
            across its images; zero for cached/skipped images.
        error_message: Human-readable error detail, empty on success.
    """

    image_path: str
    status: ProcessingStatus
    duration_seconds: float
    error_message: str


@dataclass
class ExtractionStats:
    """Counters summarizing the outcome of an embedding extraction run.

    Attributes:
        total_unique_images: Total distinct images referenced across
            the provided pair CSV files.
        cached: Images already present in a loaded embedding cache.
        processed: Images newly embedded in this run.
        skipped_missing: Images referenced by a pair file whose path
            does not exist on disk.
        skipped_corrupted: Images that failed validation or could not
            be preprocessed.
        failed: Images whose batch failed during model inference.
        total_batches: Number of inference batches executed.
        total_inference_seconds: Cumulative wall-clock time spent inside
            :meth:`EmbeddingModel.embed_batch`.
    """

    total_unique_images: int = 0
    cached: int = 0
    processed: int = 0
    skipped_missing: int = 0
    skipped_corrupted: int = 0
    failed: int = 0
    total_batches: int = 0
    total_inference_seconds: float = 0.0

    def summary(self) -> str:
        """Return a human-readable one-line summary of the run.

        Returns:
            A formatted string suitable for logging.
        """
        return (
            f"total_unique_images={self.total_unique_images}, cached={self.cached}, "
            f"processed={self.processed}, skipped_missing={self.skipped_missing}, "
            f"skipped_corrupted={self.skipped_corrupted}, failed={self.failed}, "
            f"batches={self.total_batches}, inference_time={self.total_inference_seconds:.2f}s"
        )


# ------------------------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------------------------


class EmbeddingExtractor:
    """Orchestrates deduplicated, cached, batched embedding extraction.

    This is the main entry point for the embedding extraction stage:
    given a model configuration and the three pair CSVs for a dataset,
    it produces ``embeddings.npy``, ``embedding_index.csv``, and
    ``inference_log.csv`` under a configured output directory.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        output_dir: Path,
        device: str | None = None,
        batch_size: int | None = None,
        use_cache: bool = True,
        random_seed: int = DEFAULT_RANDOM_SEED,
    ) -> None:
        """Initialize the extractor.

        Args:
            model_config: Validated configuration for the model to run.
            output_dir: Directory to write ``embeddings.npy``,
                ``embedding_index.csv``, and ``inference_log.csv`` to,
                typically ``embeddings/<dataset>/<model>/``.
            device: Explicit device override (e.g. ``"cuda"``, ``"cpu"``).
                If None, CUDA is used automatically when available,
                otherwise CPU.
            batch_size: Inference batch size. Defaults to
                ``model_config.batch_size_override`` if set, otherwise
                the global settings batch size.
            use_cache: If True, an existing embedding cache at
                ``output_dir`` is loaded and only missing images are
                computed. If False, every image is recomputed.
            random_seed: Seed applied for deterministic inference.
        """
        set_seed(random_seed)
        self._configure_deterministic_backend()

        self.model_config = model_config
        self.output_dir = ensure_dir(output_dir)
        self.device = self._resolve_device(device)
        self.batch_size = batch_size or model_config.batch_size_override or get_settings().batch_size
        self.use_cache = use_cache

        self.embeddings_path = self.output_dir / "embeddings.npy"
        self.index_path = self.output_dir / "embedding_index.csv"
        self.log_path = self.output_dir / "inference_log.csv"

        self.preprocessor = ImageBatchPreprocessor(model_config)
        self.model = load_embedding_model(model_config, self.device)
        self.validator = ImageValidator()
        self.stats = ExtractionStats()

        logger.info(
            "Initialized EmbeddingExtractor: model='%s', device='%s', batch_size=%d, cache=%s",
            model_config.name.value,
            self.device,
            self.batch_size,
            use_cache,
        )

    @staticmethod
    def _configure_deterministic_backend() -> None:
        """Configure torch backend flags for reproducible inference where possible."""
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def _resolve_device(requested: str | None) -> str:
        """Resolve the compute device, auto-selecting CUDA when available.

        Args:
            requested: An explicit device string, or None to auto-select.

        Returns:
            ``"cuda"`` if requested (or auto-selected) and available,
            otherwise ``"cpu"``.
        """
        if requested is not None:
            resolved = get_device(requested)
        else:
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Resolved inference device: %s", resolved)
        return resolved

    def run(self, pair_csv_paths: list[Path]) -> ExtractionStats:
        """Execute embedding extraction across the given pair CSV files.

        Args:
            pair_csv_paths: Paths to ``train_pairs.csv``,
                ``validation_pairs.csv``, and ``test_pairs.csv`` (or any
                subset) whose ``image1``/``image2`` columns define the
                set of images to embed.

        Returns:
            :class:`ExtractionStats` summarizing the run.

        Raises:
            EmbeddingExtractionError: If no image paths are found across
                the provided CSVs, or if zero embeddings (cached and
                newly computed combined) exist at the end of the run.
        """
        model_label = self.model_config.name.value
        logger.info("Starting embedding extraction using model '%s'.", model_label)

        unique_paths = self._collect_unique_image_paths(pair_csv_paths)
        self.stats.total_unique_images = len(unique_paths)
        if not unique_paths:
            raise EmbeddingExtractionError(
                "No image paths found across the provided pair CSV files: "
                f"{[str(p) for p in pair_csv_paths]}."
            )

        existing_embeddings, path_to_row = self._load_existing_cache()

        log_entries: list[InferenceLogEntry] = []
        pending_paths: list[str] = []
        for path in unique_paths:
            if path in path_to_row:
                self.stats.cached += 1
                log_entries.append(
                    InferenceLogEntry(path, ProcessingStatus.CACHED, 0.0, "")
                )
            else:
                pending_paths.append(path)

        logger.info(
            "%d of %d unique images already cached; %d require extraction.",
            self.stats.cached,
            len(unique_paths),
            len(pending_paths),
        )

        new_embeddings: list[np.ndarray] = []
        new_paths: list[str] = []

        with timer(f"Embedding extraction for model '{model_label}'"):
            batches = list(chunked(pending_paths, self.batch_size))
            for batch_paths in tqdm(batches, desc=f"Extracting embeddings ({model_label})", unit="batch"):
                self._process_batch(batch_paths, log_entries, new_embeddings, new_paths)

        combined_embeddings, combined_paths = self._merge_with_cache(
            existing_embeddings, path_to_row, new_embeddings, new_paths
        )

        if combined_embeddings.shape[0] == 0:
            raise EmbeddingExtractionError(
                f"Embedding extraction produced zero usable embeddings for model "
                f"'{model_label}'. Stats: {self.stats.summary()}."
            )

        self._save_outputs(combined_embeddings, combined_paths, log_entries)

        logger.info("Embedding extraction finished for '%s': %s", model_label, self.stats.summary())
        return self.stats

    # -- collection ----------------------------------------------------------------

    @staticmethod
    def _collect_unique_image_paths(pair_csv_paths: list[Path]) -> list[str]:
        """Gather the deduplicated set of image paths referenced by pair CSVs.

        Args:
            pair_csv_paths: Paths to pair CSV files with ``image1`` and
                ``image2`` columns.

        Returns:
            A sorted list of unique image path strings.
        """
        unique_paths: set[str] = set()
        for csv_path in pair_csv_paths:
            if not csv_path.exists():
                logger.warning("Pair file not found, skipping: %s", csv_path)
                continue
            frame = pd.read_csv(
                csv_path,
                usecols=[PairOutputColumns.IMAGE1, PairOutputColumns.IMAGE2],
                dtype=str,
            )
            unique_paths.update(frame[PairOutputColumns.IMAGE1].dropna().tolist())
            unique_paths.update(frame[PairOutputColumns.IMAGE2].dropna().tolist())
        return sorted(unique_paths)

    # -- caching -------------------------------------------------------------------

    def _load_existing_cache(self) -> tuple[np.ndarray | None, dict[str, int]]:
        """Load a previously saved embedding cache, if enabled and present.

        Returns:
            A tuple of ``(embeddings_array_or_None, path_to_row_index)``.
            Both are empty/None if caching is disabled, no cache exists,
            or the cache fails to load.

        Raises:
            EmbeddingExtractionError: If a cache is found but its
                embedding dimensionality does not match the currently
                configured model.
        """
        if not self.use_cache or not self.embeddings_path.exists() or not self.index_path.exists():
            return None, {}

        try:
            existing_embeddings = load_npy(self.embeddings_path)
            index_df = pd.read_csv(self.index_path, dtype={EmbeddingIndexColumns.IMAGE_PATH: str})
        except (FileNotFoundError, ValueError, OSError) as exc:
            logger.warning("Failed to load existing embedding cache, recomputing from scratch: %s", exc)
            return None, {}

        if existing_embeddings.ndim != 2 or existing_embeddings.shape[1] != self.model_config.embedding_dim:
            raise EmbeddingExtractionError(
                f"Cached embeddings at {self.embeddings_path} have dimensionality "
                f"{existing_embeddings.shape}, which does not match the configured "
                f"model embedding_dim={self.model_config.embedding_dim}. Delete the "
                f"cache or point to a fresh output directory before re-running."
            )

        path_to_row = {
            str(row[EmbeddingIndexColumns.IMAGE_PATH]): int(row[EmbeddingIndexColumns.INDEX])
            for _, row in index_df.iterrows()
        }
        logger.info(
            "Loaded existing embedding cache with %d entries from %s", len(path_to_row), self.embeddings_path
        )
        return existing_embeddings, path_to_row

    @staticmethod
    def _merge_with_cache(
        existing_embeddings: np.ndarray | None,
        path_to_row: dict[str, int],
        new_embeddings: list[np.ndarray],
        new_paths: list[str],
    ) -> tuple[np.ndarray, list[str]]:
        """Combine cached embeddings with newly computed ones, preserving row order.

        Args:
            existing_embeddings: Previously cached embeddings, or None.
            path_to_row: Mapping from cached image path to its row index
                in ``existing_embeddings``.
            new_embeddings: Newly computed embedding vectors, in the
                order they were processed.
            new_paths: Image paths corresponding to ``new_embeddings``.

        Returns:
            A tuple of ``(combined_embeddings_array, combined_image_paths)``.
        """
        new_array = np.stack(new_embeddings, axis=0).astype(np.float32) if new_embeddings else None

        if existing_embeddings is not None and path_to_row:
            ordered_existing_paths: list[str] = [""] * len(path_to_row)
            for path, idx in path_to_row.items():
                ordered_existing_paths[idx] = path

            if new_array is not None:
                combined_embeddings = np.concatenate([existing_embeddings, new_array], axis=0)
            else:
                combined_embeddings = existing_embeddings
            combined_paths = ordered_existing_paths + new_paths
        else:
            combined_embeddings = new_array if new_array is not None else np.empty((0, 0), dtype=np.float32)
            combined_paths = new_paths

        return combined_embeddings, combined_paths

    # -- per-image / per-batch processing -----------------------------------------

    def _prepare_single_image(
        self, image_path: Path
    ) -> tuple[np.ndarray | None, InferenceLogEntry | None]:
        """Validate and preprocess a single image ahead of batched inference.

        Args:
            image_path: Path to the image to prepare.

        Returns:
            A tuple ``(array, failure_log_entry)``. On success, ``array``
            is the preprocessed ``(3, H, W)`` tensor and
            ``failure_log_entry`` is None (the success entry is logged
            by the caller once inference timing is known). On failure,
            ``array`` is None and ``failure_log_entry`` describes why.
        """
        if not image_path.exists():
            self.stats.skipped_missing += 1
            logger.warning("Referenced image does not exist, skipping: %s", image_path)
            return None, InferenceLogEntry(
                str(image_path), ProcessingStatus.SKIPPED_MISSING, 0.0, "File not found"
            )

        if not self.validator.is_valid(image_path):
            self.stats.skipped_corrupted += 1
            return None, InferenceLogEntry(
                str(image_path), ProcessingStatus.SKIPPED_CORRUPTED, 0.0, "Failed image validation"
            )

        try:
            array = self.preprocessor.load_and_preprocess(image_path)
            return array, None
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            self.stats.skipped_corrupted += 1
            logger.warning("Failed to preprocess image %s: %s", image_path, exc)
            return None, InferenceLogEntry(
                str(image_path), ProcessingStatus.SKIPPED_CORRUPTED, 0.0, str(exc)
            )

    def _process_batch(
        self,
        batch_paths: list[str],
        log_entries: list[InferenceLogEntry],
        new_embeddings: list[np.ndarray],
        new_paths: list[str],
    ) -> None:
        """Prepare, run inference on, and log the outcome of one batch of images.

        Args:
            batch_paths: Image path strings belonging to this batch.
            log_entries: Shared list to append per-image log entries to.
            new_embeddings: Shared list to append successful embedding
                vectors to.
            new_paths: Shared list to append the corresponding image
                paths of successful embeddings to.
        """
        self.stats.total_batches += 1
        batch_arrays: list[np.ndarray] = []
        batch_valid_paths: list[str] = []

        for image_path_str in batch_paths:
            array, failure_entry = self._prepare_single_image(Path(image_path_str))
            if failure_entry is not None:
                log_entries.append(failure_entry)
                continue
            if array is not None:
                batch_arrays.append(array)
                batch_valid_paths.append(image_path_str)

        if not batch_arrays:
            return

        batch_tensor = torch.from_numpy(np.stack(batch_arrays, axis=0))
        start_time = time.perf_counter()
        try:
            batch_embeddings = self.model.embed_batch(batch_tensor)
        except Exception as exc:  # noqa: BLE001 - keep the run alive on backend failures
            elapsed = time.perf_counter() - start_time
            per_image_time = elapsed / max(len(batch_valid_paths), 1)
            logger.error(
                "Batch inference failed for %d images: %s", len(batch_valid_paths), exc, exc_info=True
            )
            for path_str in batch_valid_paths:
                self.stats.failed += 1
                log_entries.append(
                    InferenceLogEntry(path_str, ProcessingStatus.FAILED, per_image_time, str(exc))
                )
            return

        elapsed = time.perf_counter() - start_time
        self.stats.total_inference_seconds += elapsed
        per_image_time = elapsed / max(len(batch_valid_paths), 1)

        for path_str, embedding_vector in zip(batch_valid_paths, batch_embeddings, strict=True):
            new_embeddings.append(embedding_vector)
            new_paths.append(path_str)
            self.stats.processed += 1
            log_entries.append(
                InferenceLogEntry(path_str, ProcessingStatus.SUCCESS, per_image_time, "")
            )

    # -- output saving -------------------------------------------------------------

    def _save_outputs(
        self,
        embeddings: np.ndarray,
        image_paths: list[str],
        log_entries: list[InferenceLogEntry],
    ) -> None:
        """Persist embeddings, the index mapping, and the inference log to disk.

        Args:
            embeddings: Final ``(N, embedding_dim)`` embedding array.
            image_paths: Image path for each row of ``embeddings``, in order.
            log_entries: Per-image log entries collected during the run.
        """
        _save_embeddings_array_atomic(self.embeddings_path, embeddings)

        ensure_dir(self.index_path.parent)
        with self.index_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([EmbeddingIndexColumns.INDEX, EmbeddingIndexColumns.IMAGE_PATH])
            writer.writerows((idx, path) for idx, path in enumerate(image_paths))
        logger.info("Saved embedding index with %d entries to %s", len(image_paths), self.index_path)

        ensure_dir(self.log_path.parent)
        with self.log_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["image_path", "status", "duration_seconds", "error_message"])
            writer.writerows(
                (entry.image_path, entry.status.value, f"{entry.duration_seconds:.6f}", entry.error_message)
                for entry in log_entries
            )
        logger.info("Saved inference log with %d entries to %s", len(log_entries), self.log_path)

class DeepFaceEmbeddingModel(EmbeddingModel):
    """DeepFace-backed embedding model."""

    def __init__(self, model_config: ModelConfig, device: str) -> None:
        super().__init__(model_config, device)

        try:
            from deepface import DeepFace
        except ImportError as exc:
            raise EmbeddingModelLoadError(
                "deepface is required for DeepFace models. "
                "Install it with: python -m pip install deepface"
            ) from exc

        model_name = model_config.name.value.lower()

        if model_name == "facenet512":
            deepface_name = "Facenet512"
        elif model_name == "ghostfacenet":
            deepface_name = "GhostFaceNet"
        elif model_name == "sface":
            deepface_name = "SFace"
        else:
            raise EmbeddingModelLoadError(
                f"Unsupported DeepFace model name: {model_config.name.value}"
            )

        try:
            self._model = DeepFace.build_model(deepface_name)
        except Exception as exc:
            raise EmbeddingModelLoadError(
                f"Failed to load DeepFace model '{deepface_name}': {exc}"
            ) from exc

        self._deepface_name = deepface_name

        logger.info(
            "Loaded DeepFace model '%s' for configuration '%s'",
            deepface_name,
            model_config.name.value,
        )

    def embed_batch(self, batch: torch.Tensor) -> np.ndarray:
        """Run DeepFace inference on a batch."""

        batch_np = batch.detach().cpu().numpy().astype(np.float32)

        # CHW -> HWC
        batch_np = np.transpose(batch_np, (0, 2, 3, 1))

        # The configured preprocessing produces [-1, 1].
        # DeepFace expects its image input in the same normalized
        # representation before its internal model preprocessing.
        batch_np = np.clip(batch_np, -1.0, 1.0)

        embeddings = []

        for image in batch_np:

            # DeepFace model wrappers generally expect RGB float images
            # in [0, 1] for their forward() implementations.
            image_rgb = (image + 1.0) / 2.0

            try:
                output = self._model.forward(
                    np.expand_dims(image_rgb.astype(np.float32), axis=0)
                )
            except Exception as exc:
                raise EmbeddingModelLoadError(
                    f"DeepFace inference failed for '{self._deepface_name}': {exc}"
                ) from exc

            output_array = np.asarray(output, dtype=np.float32)

            if output_array.ndim == 1:
                embedding = output_array
            else:
                embedding = output_array[0]

            embeddings.append(embedding)

        result = np.asarray(embeddings, dtype=np.float32)

        return result
