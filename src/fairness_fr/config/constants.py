"""Project-wide constants and enumerations.

This module centralizes every fixed value used across the pipeline —
dataset names, model names, demographic attributes, metric names,
threshold strategies, standard column names, and default numeric
targets. Nothing in this module reads from disk or the environment;
values that vary by machine or run belong in :mod:`fairness_fr.settings`
or :mod:`fairness_fr.config` instead.

Keeping these as enums (rather than bare strings scattered through the
codebase) means a typo in a dataset or model name fails fast at
validation time instead of silently producing an empty result set.
"""

from __future__ import annotations

from enum import Enum


class DatasetName(str, Enum):
    """Supported fairness-focused face datasets."""

    RFW = "rfw"
    BFW = "bfw"
    DEMOGPAIRS = "demogpairs"


class ModelName(str, Enum):
    """Supported pre-trained face recognition models."""

    FACENET512 = "facenet512"
    ARCFACE = "arcface"
    ADAFACE = "adaface"
    MOBILEFACENET = "mobilefacenet"
    ELASTICFACE = "elasticface"
    TRANSFACE = "transface"
    SWINFACE = "swinface"
    SFACE = "sface"
    GHOSTFACENET = "ghostfacenet"


class DemographicAttribute(str, Enum):
    """Demographic axes along which fairness is evaluated."""

    ETHNICITY = "ethnicity"
    GENDER = "gender"
    AGE_GROUP = "age_group"
    INTERSECTIONAL = "intersectional"  # e.g. ethnicity x gender


class PairLabel(int, Enum):
    """Binary label for a face pair."""

    IMPOSTOR = 0
    GENUINE = 1


class ThresholdStrategy(str, Enum):
    """Supported decision-threshold selection strategies."""

    FIXED = "fixed"
    TARGET_FMR = "target_fmr"
    EER = "eer"


class MetricName(str, Enum):
    """Names of the biometric and fairness metrics computed by the pipeline."""

    ACCURACY = "accuracy"
    FMR = "fmr"
    FNMR = "fnmr"
    FAR = "far"
    FRR = "frr"
    TAR = "tar"
    EER = "eer"
    FMR_GAP = "fmr_gap"
    FNMR_GAP = "fnmr_gap"
    ACCURACY_STD = "accuracy_std"
    WORST_GROUP_ACCURACY = "worst_group_accuracy"
    WORST_TAR = "worst_tar"
    EQUALIZED_ODDS_GAP = "equalized_odds_gap"


class DistanceMetric(str, Enum):
    """Supported embedding comparison functions."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    EUCLIDEAN_L2 = "euclidean_l2"


class FileFormat(str, Enum):
    """Serialization formats used for embeddings and intermediate artifacts."""

    NPY = "npy"
    PKL = "pkl"
    CSV = "csv"
    JSON = "json"


# --- Standard column names -----------------------------------------------------
# Using shared constants (instead of retyping literal strings) keeps every
# module that reads/writes these CSVs in sync if a column is ever renamed.

class MetadataColumns:
    """Canonical column names for the per-dataset metadata CSV."""

    IMAGE_PATH = "image_path"
    IDENTITY = "identity"
    GROUP = "group"
    GENDER = "gender"
    AGE_GROUP = "age_group"


class PairColumns:
    """Canonical column names for genuine/impostor pair CSVs."""

    IMAGE1_PATH = "image1_path"
    IMAGE2_PATH = "image2_path"
    LABEL = "label"
    IDENTITY1 = "identity1"
    IDENTITY2 = "identity2"
    GROUP1 = "group1"
    GROUP2 = "group2"


class ScoreColumns:
    """Canonical column names for the pairwise similarity score CSV."""

    SCORE = "similarity_score"
    LABEL = "label"
    GROUP1 = "group1"
    GROUP2 = "group2"


# --- Default numeric targets -----------------------------------------------------

#: Target FMR operating points required by the assignment spec (Section 2, 5).
DEFAULT_TARGET_FMR_VALUES: tuple[float, ...] = (0.1, 0.01, 0.001, 0.0001)

#: Minimum number of models required for the Week 7 model-comparison study.
MIN_MODELS_FOR_COMPARISON: int = 4

#: Random seed used wherever stochastic sampling occurs, unless overridden
#: by settings — kept here as the documented, spec-referenced default.
DEFAULT_RANDOM_SEED: int = 42

#: Sentinel value used when a similarity score could not be computed
#: (e.g. failed face detection) so it can be filtered rather than silently
#: treated as zero.
INVALID_SCORE_SENTINEL: float = float("nan")
