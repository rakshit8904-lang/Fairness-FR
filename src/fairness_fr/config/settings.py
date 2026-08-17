"""Global, environment-driven settings for the fairness evaluation pipeline.

Settings here are things that vary by *machine* or *run environment*
(paths, worker counts, device selection, log level) as opposed to
:mod:`fairness_fr.config`, which loads *experiment-level* choices
(which dataset, which model, which pairing rules) from versioned YAML
files under ``configs/``.

All settings are overridable via environment variables prefixed with
``FAIRNESS_FR_``, e.g. ``FAIRNESS_FR_NUM_WORKERS=8``, which keeps the
pipeline reproducible in CI and on shared compute without editing code.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .constants import DEFAULT_RANDOM_SEED

#: Absolute path to the repository root, resolved relative to this file
#: (src/fairness_fr/settings.py -> repo root is two parents up).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]

class Settings(BaseSettings):
    """Machine- and run-level configuration for the pipeline.

    Attributes:
        project_root: Root directory of the repository.
        data_dir: Root of raw/processed/metadata datasets.
        pairs_dir: Root of generated genuine/impostor pair CSVs.
        embeddings_dir: Root of extracted embedding artifacts.
        scores_dir: Root of pairwise similarity score CSVs.
        results_dir: Root of metrics, plots, and reports.
        configs_dir: Root of experiment/dataset/model YAML configs.
        num_workers: Default number of parallel workers for data loading
            and batched embedding extraction.
        batch_size: Default batch size for model inference.
        device: Compute device identifier, e.g. ``"cpu"``, ``"cuda"``,
            ``"cuda:0"``, ``"mps"``.
        random_seed: Global random seed for reproducible pair sampling
            and subset selection.
        log_level: Root logging level, e.g. ``"INFO"``, ``"DEBUG"``.
        log_file: Optional path to a persistent log file for long-running
            batch jobs.
        cache_embeddings: Whether embedding extraction should skip images
            already present in an existing embeddings index (large-dataset
            resumability).
        chunk_size: Row chunk size used when streaming very large CSVs
            (metadata or pair files) instead of loading them fully into
            memory.
    """

    model_config = SettingsConfigDict(
        env_prefix="FAIRNESS_FR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_root: Path = Field(default=PROJECT_ROOT)

    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    pairs_dir: Path = Field(default=PROJECT_ROOT / "pairs")
    embeddings_dir: Path = Field(default=PROJECT_ROOT / "embeddings")
    scores_dir: Path = Field(default=PROJECT_ROOT / "scores")
    results_dir: Path = Field(default=PROJECT_ROOT / "results")
    configs_dir: Path = Field(default=PROJECT_ROOT / "configs")

    num_workers: int = Field(default=4, ge=1, le=128)
    batch_size: int = Field(default=64, ge=1)
    device: str = Field(default="cpu")

    random_seed: int = Field(default=DEFAULT_RANDOM_SEED)

    log_level: str = Field(default="INFO")
    log_file: Path | None = Field(default=None)

    cache_embeddings: bool = Field(default=True)
    chunk_size: int = Field(default=10_000, ge=1)

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        """Ensure log_level is one of the standard logging levels."""
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        normalized = value.upper()
        if normalized not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got {value!r}")
        return normalized

    def dataset_dir(self, dataset_name: str, stage: str = "raw") -> Path:
        """Return the directory for a given dataset at a given pipeline stage.

        Args:
            dataset_name: Dataset identifier, e.g. ``"rfw"``.
            stage: One of ``"raw"``, ``"processed"``, or ``"metadata"``.

        Returns:
            Path to ``data/<stage>/<dataset_name>`` (or the metadata CSV
            directory for ``stage="metadata"``).
        """
        return self.data_dir / stage / dataset_name

    def embeddings_path(self, dataset_name: str, model_name: str) -> Path:
        """Return the expected .npy embeddings file path for a dataset/model pair."""
        return self.embeddings_dir / dataset_name / f"{model_name}_embeddings.npy"

    def scores_path(self, dataset_name: str, model_name: str) -> Path:
        """Return the expected scores CSV path for a dataset/model pair."""
        return self.scores_dir / f"{dataset_name}_{model_name}_scores.csv"

    def ensure_directories(self) -> None:
        """Create all top-level pipeline directories if they do not exist."""
        for directory in (
            self.data_dir,
            self.pairs_dir,
            self.embeddings_dir,
            self.scores_dir,
            self.results_dir,
            self.configs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached :class:`Settings` instance.

    Using ``lru_cache`` gives us a cheap singleton: the environment is
    read once per process, and every module calls this function instead
    of constructing ``Settings()`` directly, so all components agree on
    the same paths and configuration.

    Returns:
        The cached :class:`Settings` instance.
    """
    return Settings()
