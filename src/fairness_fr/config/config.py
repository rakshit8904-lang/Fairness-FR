"""Experiment-level configuration schemas and loader.

Whereas :mod:`fairness_fr.settings` covers machine-level configuration
(paths, workers, device), this module covers *experiment-level* choices:
which dataset, which model, how pairs are sampled, which thresholds are
evaluated. These live as versioned YAML files under ``configs/`` so a
full experiment (dataset + model + pairing + threshold choices) is
reproducible from a single set of files, and new datasets/models can be
added without touching any Python code.

Typical usage:
    >>> loader = ConfigLoader()
    >>> dataset_cfg = loader.load_dataset_config("rfw")
    >>> model_cfg = loader.load_model_config("arcface")
    >>> experiment_cfg = loader.load_experiment_config()
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from .constants import (
    DEFAULT_TARGET_FMR_VALUES,
    DatasetName,
    DemographicAttribute,
    DistanceMetric,
    ModelName,
    ThresholdStrategy,
)
from fairness_fr.utils.logging import get_logger
from .settings import get_settings

logger = get_logger(__name__)


# ------------------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------------------


class DatasetConfig(BaseModel):
    """Schema for a ``configs/datasets/<name>.yaml`` file.

    Attributes:
        name: Dataset identifier, must match a :class:`DatasetName` value.
        raw_dir: Directory containing the untouched downloaded dataset.
        processed_dir: Directory where preprocessed/aligned images are written.
        metadata_csv: Path to the standardized metadata CSV for this dataset.
        has_official_pairs: Whether the dataset ships official verification
            pair files (e.g. RFW does) that should be used instead of
            synthetically generated pairs.
        official_pairs_dir: Directory of official pair files, if applicable.
        demographic_attributes: Which demographic axes this dataset has
            labels for (must be a subset of what the raw data provides).
        image_extension: File extension of images in this dataset.
    """

    name: DatasetName
    raw_dir: Path
    processed_dir: Path
    metadata_csv: Path
    has_official_pairs: bool = False
    official_pairs_dir: Path | None = None
    demographic_attributes: list[DemographicAttribute] = Field(
        default_factory=lambda: [DemographicAttribute.ETHNICITY]
    )
    image_extension: str = ".jpg"

    @field_validator("official_pairs_dir")
    @classmethod
    def _require_pairs_dir_if_official(
        cls, value: Path | None, info: Any
    ) -> Path | None:
        """Ensure official_pairs_dir is set whenever has_official_pairs is True."""
        has_official = info.data.get("has_official_pairs", False)
        if has_official and value is None:
            raise ValueError(
                "official_pairs_dir must be set when has_official_pairs=True."
            )
        return value


class ModelConfig(BaseModel):
    """Schema for a ``configs/models/<name>.yaml`` file.

    Attributes:
        name: Model identifier, must match a :class:`ModelName` value.
        weights_path: Local path or model-hub identifier for pretrained weights.
        input_size: Expected square input resolution in pixels (H == W).
        embedding_dim: Dimensionality of the output embedding vector.
        normalization_mean: Per-channel mean used to normalize input images.
        normalization_std: Per-channel std used to normalize input images.
        distance_metric: Similarity/distance function used to compare
            embeddings from this model.
        batch_size_override: Optional per-model override of the global
            batch size (some models are far more memory-hungry than others).
    """

    name: ModelName
    weights_path: str
    input_size: int = Field(gt=0)
    embedding_dim: int = Field(gt=0)
    normalization_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    normalization_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    distance_metric: DistanceMetric = DistanceMetric.COSINE
    batch_size_override: int | None = Field(default=None, gt=0)


class PairingConfig(BaseModel):
    """Schema for ``configs/pairing.yaml``.

    Attributes:
        genuine_impostor_ratio: Target ratio of genuine to impostor pairs
            (1.0 means balanced classes).
        max_pairs_per_identity: Cap on genuine pairs drawn per identity, to
            prevent a handful of over-represented identities from dominating.
        enforce_group_balance: Whether to equalize pair counts across
            demographic groups.
        min_pairs_per_group: Minimum number of pairs required per group
            for that group to be included in fairness evaluation — guards
            against statistically meaningless metrics on tiny subgroups.
        random_seed: Seed for pair sampling, overrides the global settings
            seed if provided so pairing is independently reproducible.
    """

    genuine_impostor_ratio: float = Field(default=1.0, gt=0)
    max_pairs_per_identity: int = Field(default=20, gt=0)
    enforce_group_balance: bool = True
    min_pairs_per_group: int = Field(default=100, ge=1)
    random_seed: int | None = None


class ThresholdConfig(BaseModel):
    """Schema for ``configs/thresholds.yaml``.

    Attributes:
        strategies: Which threshold selection strategies to evaluate.
        target_fmr_values: FMR operating points used by the
            :attr:`ThresholdStrategy.TARGET_FMR` strategy.
        fixed_threshold: Similarity cutoff used by the
            :attr:`ThresholdStrategy.FIXED` strategy.
    """

    strategies: list[ThresholdStrategy] = Field(
        default_factory=lambda: [
            ThresholdStrategy.FIXED,
            ThresholdStrategy.TARGET_FMR,
            ThresholdStrategy.EER,
        ]
    )
    target_fmr_values: tuple[float, ...] = DEFAULT_TARGET_FMR_VALUES
    fixed_threshold: float = Field(default=0.5)


class ExperimentConfig(BaseModel):
    """Schema for ``configs/experiment.yaml`` — the top-level run definition.

    Attributes:
        experiment_name: Human-readable identifier for this run, used to
            namespace outputs under ``results/``.
        datasets: Which datasets to run the pipeline over.
        models: Which models to run the pipeline over. When more than one
            model is listed, :mod:`fairness_fr.evaluation.model_comparator`
            is invoked automatically.
        demographic_attribute: Primary demographic axis for fairness
            evaluation in this run.
        run_improvement_stage: Whether to run demographic thresholding /
            balanced-subset evaluation (Week 7 Option B) after the base
            evaluation.
    """

    experiment_name: str
    datasets: list[DatasetName]
    models: list[ModelName]
    demographic_attribute: DemographicAttribute = DemographicAttribute.ETHNICITY
    run_improvement_stage: bool = False

    @field_validator("datasets", "models")
    @classmethod
    def _require_non_empty(cls, value: list[Any]) -> list[Any]:
        """Ensure at least one dataset and one model are specified."""
        if not value:
            raise ValueError("At least one entry is required.")
        return value


# ------------------------------------------------------------------------------
# Loader
# ------------------------------------------------------------------------------


class ConfigLoader:
    """Loads and validates YAML configuration files from the configs directory.

    Centralizing loading here means every stage script resolves config
    paths the same way and gets the same validation errors, instead of
    each script calling ``yaml.safe_load`` independently.

    Attributes:
        configs_dir: Root directory containing ``datasets/``, ``models/``,
            ``pairing.yaml``, ``thresholds.yaml``, and ``experiment.yaml``.
    """

    def __init__(self, configs_dir: Path | None = None) -> None:
        """Initialize the loader.

        Args:
            configs_dir: Override for the configs root directory. Defaults
                to ``settings.configs_dir``.
        """
        self.configs_dir = configs_dir or get_settings().configs_dir

    def _read_yaml(self, path: Path) -> dict[str, Any]:
        """Read and parse a YAML file, raising a clear error if missing.

        Args:
            path: Path to the YAML file.

        Returns:
            Parsed YAML content as a dictionary.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            yaml.YAMLError: If the file is not valid YAML.
        """
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            content = yaml.safe_load(fh)
        return content or {}

    def load_dataset_config(self, dataset_name: str) -> DatasetConfig:
        """Load and validate a dataset config by name.

        Args:
            dataset_name: Dataset identifier, e.g. ``"rfw"``.

        Returns:
            A validated :class:`DatasetConfig`.
        """
        path = self.configs_dir / "datasets" / f"{dataset_name}.yaml"
        raw = self._read_yaml(path)
        logger.debug("Loaded dataset config from %s", path)
        return DatasetConfig.model_validate(raw)

    def load_model_config(self, model_name: str) -> ModelConfig:
        """Load and validate a model config by name.

        Args:
            model_name: Model identifier, e.g. ``"arcface"``.

        Returns:
            A validated :class:`ModelConfig`.
        """
        path = self.configs_dir / "models" / f"{model_name}.yaml"
        raw = self._read_yaml(path)
        logger.debug("Loaded model config from %s", path)
        return ModelConfig.model_validate(raw)

    def load_pairing_config(self) -> PairingConfig:
        """Load and validate the pairing configuration.

        Returns:
            A validated :class:`PairingConfig`. Falls back to defaults
            if ``pairing.yaml`` is absent, since every field is optional.
        """
        path = self.configs_dir / "pairing.yaml"
        raw = self._read_yaml(path) if path.exists() else {}
        return PairingConfig.model_validate(raw)

    def load_threshold_config(self) -> ThresholdConfig:
        """Load and validate the threshold configuration.

        Returns:
            A validated :class:`ThresholdConfig`. Falls back to defaults
            if ``thresholds.yaml`` is absent.
        """
        path = self.configs_dir / "thresholds.yaml"
        raw = self._read_yaml(path) if path.exists() else {}
        return ThresholdConfig.model_validate(raw)

    def load_experiment_config(self, path: Path | None = None) -> ExperimentConfig:
        """Load and validate the top-level experiment configuration.

        Args:
            path: Optional explicit path to an experiment YAML file.
                Defaults to ``configs/experiment.yaml``.

        Returns:
            A validated :class:`ExperimentConfig`.
        """
        resolved_path = path or (self.configs_dir / "experiment.yaml")
        raw = self._read_yaml(resolved_path)
        return ExperimentConfig.model_validate(raw)


@lru_cache(maxsize=1)
def get_config_loader() -> ConfigLoader:
    """Return a process-wide cached :class:`ConfigLoader` instance.

    Returns:
        The cached :class:`ConfigLoader`.
    """
    return ConfigLoader()
