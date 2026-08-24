from pathlib import Path
import sys
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# ------------------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC))

from fairness_fr.config.config import get_config_loader
from fairness_fr.config.settings import get_settings
from fairness_fr.models.extract_embeddings import (
    ImageBatchPreprocessor,
    load_embedding_model,
)

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

DATASET = "bfw"
MODEL = "arcface"

PAIRS_FILE = PROJECT_ROOT / "pairs" / "bfw" / "test_pairs.csv"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "bfw"
    / "arcface"
    / "blur_arcface_stress_test"
)

EMBEDDING_DIR = OUTPUT_DIR / "embeddings"

THRESHOLD = 0.1546

BLUR_LEVELS = {
    "original": 0,
    "blur_light": 3,
    "blur_medium": 7,
    "blur_heavy": 15,
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# LOAD PROJECT CONFIGURATION
# ------------------------------------------------------------

loader = get_config_loader()

model_config = loader.load_model_config(MODEL)
# Resolve relative model path from project root
model_config.weights_path = (
    PROJECT_ROOT / model_config.weights_path
)

print("=" * 70)
print("FAIRNESS-FR — ACTUAL IMAGE QUALITY FAIRNESS STRESS TEST")
print("STEP 4B — GAUSSIAN BLUR + PROJECT ARCFACE")
print("=" * 70)

print("\nModel:")
print(model_config)

# ------------------------------------------------------------
# LOAD TEST PAIRS
# ------------------------------------------------------------

if not PAIRS_FILE.exists():
    raise FileNotFoundError(
        f"Cannot find:\n{PAIRS_FILE.resolve()}"
    )

df = pd.read_csv(PAIRS_FILE)

required = [
    "image1",
    "image2",
    "identity1",
    "identity2",
    "gender1",
    "gender2",
    "label",
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )

print(f"\nTotal pairs: {len(df)}")

# ------------------------------------------------------------
# COLLECT UNIQUE IMAGES
# ------------------------------------------------------------

image_paths = sorted(
    set(df["image1"].astype(str))
    | set(df["image2"].astype(str))
)

print(f"Unique images: {len(image_paths)}")

# ------------------------------------------------------------
# RESOLVE IMAGE PATHS
# ------------------------------------------------------------

def resolve_image(path_string):

    p = Path(path_string)

    candidates = [
        PROJECT_ROOT / p,
        PROJECT_ROOT.parent / p,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Image not found:\n{path_string}"
    )


resolved_paths = {}

print("\nResolving images...")

for p in tqdm(image_paths):

    resolved_paths[p] = resolve_image(p)

print(
    f"Resolved {len(resolved_paths)} images."
)

# ------------------------------------------------------------
# LOAD ACTUAL FAIRNESS-FR ARCFACE
# ------------------------------------------------------------

print("\nLoading project's ArcFace ONNX model...")

model = load_embedding_model(
    model_config,
    "cpu",
)

preprocessor = ImageBatchPreprocessor(
    model_config
)

print("ArcFace ONNX model loaded.")

# ------------------------------------------------------------
# EMBEDDING FUNCTION
# ------------------------------------------------------------

def get_embedding(image):

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image_rgb = cv2.resize(
        image_rgb,
        (
            model_config.input_size,
            model_config.input_size,
        ),
        interpolation=cv2.INTER_LANCZOS4,
    )

    temp_path = OUTPUT_DIR / "_temp.jpg"

    cv2.imwrite(
        str(temp_path),
        cv2.cvtColor(
            image_rgb,
            cv2.COLOR_RGB2BGR
        ),
    )

    # Fairness-FR's exact preprocessing
    array = preprocessor.load_and_preprocess(
        temp_path
    )

    # NumPy -> PyTorch tensor
    tensor = torch.from_numpy(
        array
    ).unsqueeze(0)

    # Project's actual ONNX ArcFace
    embedding = model.embed_batch(
        tensor
    )[0]

    embedding = np.asarray(
        embedding,
        dtype=np.float32
    )

    # L2 normalize embedding
    norm = np.linalg.norm(
        embedding
    )

    if norm > 0:
        embedding = embedding / norm

    return embedding                                               
# ------------------------------------------------------------
# BLUR FUNCTION
# ------------------------------------------------------------

def apply_blur(image, kernel):

    if kernel == 0:
        return image.copy()

    return cv2.GaussianBlur(
        image,
        (kernel, kernel),
        0
    )


# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------

def calculate_metrics(scores, labels):

    scores = np.asarray(scores)
    labels = np.asarray(labels)

    predictions = (
        scores >= THRESHOLD
    ).astype(int)

    accuracy = np.mean(
        predictions == labels
    )

    genuine = labels == 1
    impostor = labels == 0

    tar = (
        np.mean(
            predictions[genuine] == 1
        )
        if np.any(genuine)
        else np.nan
    )

    fmr = (
        np.mean(
            predictions[impostor] == 1
        )
        if np.any(impostor)
        else np.nan
    )

    fnmr = 1.0 - tar

    return {
        "accuracy": accuracy,
        "tar": tar,
        "fmr": fmr,
        "fnmr": fnmr,
    }


def gender_metrics(scores, labels, gender1, gender2):

    scores = np.asarray(scores)
    labels = np.asarray(labels)

    groups = {}

    for group in ["females", "males"]:

        mask = (
            (gender1 == group)
            & (gender2 == group)
        )

        if mask.sum() == 0:
            continue

        groups[group] = calculate_metrics(
            scores[mask],
            labels[mask],
        )

        groups[group]["pairs"] = int(
            mask.sum()
        )

    return groups


def fairness_disparity(group_metrics):

    if len(group_metrics) < 2:
        return np.nan

    values = []

    for metric in [
        "fmr",
        "fnmr",
        "tar",
        "accuracy",
    ]:

        vals = [
            m[metric]
            for m in group_metrics.values()
        ]

        values.append(
            max(vals) - min(vals)
        )

    return float(
        np.mean(values)
    )


# ------------------------------------------------------------
# MAIN EXPERIMENT
# ------------------------------------------------------------

all_results = []

gender1 = df["gender1"].fillna("").astype(str).values
gender2 = df["gender2"].fillna("").astype(str).values
labels = df["label"].astype(int).values

for condition, kernel in BLUR_LEVELS.items():

    print("\n" + "=" * 70)
    print(f"PROCESSING: {condition}")
    print(f"Gaussian kernel: {kernel}")
    print("=" * 70)

    embeddings = {}

    # --------------------------------------------------------
    # GENERATE EMBEDDINGS
    # --------------------------------------------------------

    for image_key in tqdm(
        image_paths,
        desc=f"ArcFace [{condition}]",
    ):

        path = resolved_paths[image_key]

        image = cv2.imread(
            str(path)
        )

        if image is None:
            raise RuntimeError(
                f"Cannot read image:\n{path}"
            )

        processed = apply_blur(
            image,
            kernel,
        )

        embeddings[image_key] = (
            get_embedding(processed)
        )

    # --------------------------------------------------------
    # SCORE PAIRS
    # --------------------------------------------------------

    scores = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"Scoring [{condition}]",
    ):

        e1 = embeddings[
            str(row["image1"])
        ]

        e2 = embeddings[
            str(row["image2"])
        ]

        similarity = float(
            np.dot(e1, e2)
        )

        scores.append(
            similarity
        )

    scores = np.asarray(
        scores,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # GLOBAL METRICS
    # --------------------------------------------------------

    metrics = calculate_metrics(
        scores,
        labels,
    )

    # --------------------------------------------------------
    # GENDER FAIRNESS
    # --------------------------------------------------------

    groups = gender_metrics(
        scores,
        labels,
        gender1,
        gender2,
    )

    disparity = fairness_disparity(
        groups
    )

    result = {
        "condition": condition,
        "blur_kernel": kernel,
        "accuracy": metrics["accuracy"],
        "tar": metrics["tar"],
        "fmr": metrics["fmr"],
        "fnmr": metrics["fnmr"],
        "mean_disparity": disparity,
    }

    all_results.append(result)

    print("\nRESULT")

    print(
        f"Accuracy: {metrics['accuracy']:.6f}"
    )

    print(
        f"TAR:      {metrics['tar']:.6f}"
    )

    print(
        f"FMR:      {metrics['fmr']:.6f}"
    )

    print(
        f"FNMR:     {metrics['fnmr']:.6f}"
    )

    print(
        f"Mean disparity: {disparity:.6f}"
    )

# ------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------

results_df = pd.DataFrame(
    all_results
)

results_file = (
    OUTPUT_DIR
    / "blur_fairness_results.csv"
)

results_df.to_csv(
    results_file,
    index=False
)

print("\n" + "=" * 70)
print("BLUR FAIRNESS STRESS TEST COMPLETE")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)

print("\nOutput:")
print(results_file)