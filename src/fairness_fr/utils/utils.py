"""Generic utility functions shared across the pipeline.

These helpers are intentionally free of any dataset/model/metric-specific
logic — they cover reproducibility (seeding), filesystem safety (atomic
writes, directory creation), performance instrumentation (timing), and
memory-safe handling of large tabular files. Anything domain-specific
(pair generation, metric math) belongs in its own module under
``fairness_fr/pairing``, ``fairness_fr/metrics``, etc.
"""

from __future__ import annotations

import hashlib
import os
import random
import time
from collections.abc import Generator, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .logging import get_logger

logger = get_logger(__name__)


def set_seed(seed: int) -> None:
    """Seed all relevant random number generators for reproducibility.

    Seeds Python's ``random`` module and NumPy's global RNG. Torch is
    seeded lazily (imported inside the function) so that modules which
    never touch a model do not pay the cost of importing torch.

    Args:
        seed: The random seed to apply everywhere.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        logger.debug("torch not installed; skipping torch seeding.")

    logger.debug("Random seed set to %d.", seed)


def ensure_dir(path: Path) -> Path:
    """Create a directory (including parents) if it does not already exist.

    Args:
        path: Directory path to create.

    Returns:
        The same path, for convenient chaining, e.g.
        ``out_dir = ensure_dir(settings.results_dir / "plots")``.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def timer(label: str) -> Generator[None, None, None]:
    """Context manager that logs the wall-clock duration of a code block.

    Args:
        label: Human-readable description of the operation being timed,
            used in the log message.

    Yields:
        None.

    Example:
        >>> with timer("embedding extraction"):
        ...     extract_embeddings(images)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("%s completed in %.2fs", label, elapsed)


def save_npy_atomic(path: Path, array: np.ndarray) -> None:
    """Write a NumPy array to disk atomically.

    Writes to a temporary sibling file first, then renames it into place.
    This avoids leaving a truncated/corrupt ``.npy`` file behind if the
    process is interrupted mid-write — important for large embedding
    matrices that can take minutes to extract.

    Args:
        path: Destination ``.npy`` file path.
        array: Array to serialize.
    """
    ensure_dir(path.parent)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    np.save(tmp_path, array)
    os.replace(tmp_path, path)
    logger.debug("Saved array of shape %s to %s", array.shape, path)


def load_npy(path: Path) -> np.ndarray:
    """Load a NumPy array from disk, raising a clear error if missing.

    Args:
        path: Path to the ``.npy`` file.

    Returns:
        The loaded array.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Expected embeddings file not found: {path}. "
            "Run the embedding extraction stage first."
        )
    return np.load(path)


def compute_file_hash(path: Path, algorithm: str = "sha256", chunk_size: int = 1 << 20) -> str:
    """Compute a hex digest of a file's contents, reading it in chunks.

    Used to fingerprint large inputs (raw dataset archives, embedding
    files) for cache-invalidation checks without loading them fully
    into memory.

    Args:
        path: File to hash.
        algorithm: Any algorithm name accepted by :func:`hashlib.new`.
        chunk_size: Number of bytes to read per chunk (default 1 MiB).

    Returns:
        The hex digest string.
    """
    hasher = hashlib.new(algorithm)
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def chunked(iterable: Iterable[Any], size: int) -> Iterator[list[Any]]:
    """Yield successive chunks of a given size from an iterable.

    Used to batch large image lists for model inference or batch-write
    large pair/score tables without holding everything in memory at once.

    Args:
        iterable: Source iterable of arbitrary items.
        size: Maximum number of items per chunk.

    Yields:
        Lists of up to ``size`` items each.

    Raises:
        ValueError: If ``size`` is not positive.
    """
    if size <= 0:
        raise ValueError(f"Chunk size must be positive, got {size}.")

    chunk: list[Any] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def read_csv_in_chunks(
    path: Path,
    chunk_size: int,
    dtype: dict[str, Any] | None = None,
) -> Iterator[pd.DataFrame]:
    """Stream a large CSV file in row chunks instead of loading it fully.

    Intended for metadata, pair, or score CSVs that may grow into the
    millions of rows for large-scale dataset runs.

    Args:
        path: Path to the CSV file.
        chunk_size: Number of rows per chunk.
        dtype: Optional explicit dtype mapping to speed up parsing and
            avoid pandas' type-inference overhead on large files.

    Yields:
        DataFrame chunks of up to ``chunk_size`` rows.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    yield from pd.read_csv(path, chunksize=chunk_size, dtype=dtype)


def get_device(preferred: str = "cpu") -> str:
    """Resolve the best available compute device, falling back gracefully.

    Args:
        preferred: The device the caller wants, e.g. ``"cuda"``, ``"mps"``,
            ``"cpu"``. If the preferred device is unavailable, falls back
            to CPU rather than raising, so the pipeline remains runnable
            on any machine.

    Returns:
        A device string usable by torch (``"cpu"``, ``"cuda"``, ``"mps"``).
    """
    try:
        import torch
    except ImportError:
        logger.debug("torch not installed; defaulting to CPU device string.")
        return "cpu"

    if preferred.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU.")
        return "cpu"
    if preferred == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS requested but not available; falling back to CPU.")
        return "cpu"
    return preferred


def safe_divide(numerator: float, denominator: float, default: float = float("nan")) -> float:
    """Divide two numbers, returning a default instead of raising on zero division.

    Common in metric computation (e.g. FMR/FNMR denominators can be zero
    for a demographic group with too few pairs) where a NaN result should
    be surfaced and filtered downstream rather than crashing the run.

    Args:
        numerator: Dividend.
        denominator: Divisor.
        default: Value to return when ``denominator`` is zero.

    Returns:
        ``numerator / denominator``, or ``default`` if the denominator is zero.
    """
    if denominator == 0:
        return default
    return numerator / denominator
