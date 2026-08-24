from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from deepface import DeepFace


# ============================================================
# FAIRNESS-FR
# ACTUAL IMAGE-QUALITY FAIRNESS STRESS TEST
# STEP 4B — GAUSSIAN BLUR + ARCFACE
# ============================================================

DATASET = "bfw"
MODEL = "ArcFace"

PROJECT_ROOT = Path("..").resolve()

PAIRS_FILE = (
    Path("results")
    / DATASET
    / "arcface"
    / "test_scores.csv"
)

OUTPUT_DIR = (
    Path("results")
    / DATASET
    / "arcface"
    / "blur_arcface_stress_test"
)

EMBEDDING_DIR = OUTPUT_DIR / "embeddings"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# OPERATING THRESHOLD
# ============================================================

THRESHOLD = 0.1546


# ============================================================
# BLUR CONDITIONS
# ============================================================

BLUR_LEVELS = {
    "original": 0,
    "blur_light": 3,
    "blur_medium": 7,
    "blur_heavy": 15,
}


# ============================================================
# LOAD PAIRS
# ============================================================

print("=" * 70)
print("FAIRNESS-FR — ACTUAL IMAGE QUALITY FAIRNESS STRESS TEST")
print("STEP 4B — GAUSSIAN BLUR + ARCFACE")
print("=" * 70)

if not PAIRS_FILE.exists():
    raise FileNotFoundError(
        f"Cannot find test scores:\n{PAIRS_FILE.resolve()}"
    )

df = pd.read_csv(PAIRS_FILE)

required_columns = [
    "image1",
    "image2",
    "identity1",
    "identity2",
    "gender1",
    "gender2",
    "label",
]

missing = [
    c for c in required_columns
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


df = df.dropna(
    subset=[
        "image1",
        "image2",
        "gender1",
        "gender2",
        "label",
    ]
).copy()

df["label"] = df["label"].astype(int)

df["gender1"] = (
    df["gender1"]
    .astype(str)
    .str.lower()
)

df["gender2"] = (
    df["gender2"]
    .astype(str)
    .str.lower()
)

df = df.reset_index(drop=True)


print()
print(f"Total pairs: {len(df)}")


# ============================================================
# IMAGE PATH RESOLUTION
# ============================================================

def resolve_image_path(path_string):

    path_string = str(path_string)

    # Normalize Windows separators
    path_string = path_string.replace("\\", "/")

    path = Path(path_string)

    candidates = [
        PROJECT_ROOT / path,
        PROJECT_ROOT / "data" / "processed" / "bfw" / path.name,
        Path(path_string),
    ]

    for candidate in candidates:

        candidate = candidate.resolve()

        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not locate image:\n{path_string}"
    )


print()
print("Checking image paths...")

unique_images = pd.unique(
    pd.concat(
        [
            df["image1"],
            df["image2"]
        ],
        ignore_index=True
    )
)

image_paths = {}

for image in tqdm(
    unique_images,
    desc="Resolving images"
):

    image_paths[image] = resolve_image_path(image)


print(
    f"Resolved {len(image_paths)} unique images."
)


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(a, b):

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    denominator = (
        np.linalg.norm(a)
        *
        np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


# ============================================================
# EMBEDDING CACHE
# ============================================================

def embedding_file(condition, image_key):

    safe_name = (
        str(abs(hash(str(image_key))))
        + ".npy"
    )

    return (
        EMBEDDING_DIR
        / condition
        / safe_name
    )


def get_embedding(
    image_key,
    condition,
    kernel
):

    cache_file = embedding_file(
        condition,
        image_key
    )

    cache_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load cached embedding
    # --------------------------------------------------------

    if cache_file.exists():

        return np.load(
            cache_file
        )


    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image_path = image_paths[image_key]

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise ValueError(
            f"Could not read image:\n{image_path}"
        )


    # --------------------------------------------------------
    # Apply blur
    # --------------------------------------------------------

    if kernel > 0:

        image = cv2.GaussianBlur(
            image,
            (kernel, kernel),
            0
        )


    # --------------------------------------------------------
    # ArcFace embedding
    # --------------------------------------------------------

    result = DeepFace.represent(
        img_path=image,
        model_name=MODEL,
        detector_backend="opencv",
        enforce_detection=False,
        align=True,
        normalization="ArcFace",
    )

    if not result:
        raise RuntimeError(
            f"ArcFace returned no embedding for:\n"
            f"{image_path}"
        )

    embedding = np.asarray(
        result[0]["embedding"],
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    norm = np.linalg.norm(
        embedding
    )

    if norm > 0:
        embedding = embedding / norm


    # --------------------------------------------------------
    # Save cache
    # --------------------------------------------------------

    np.save(
        cache_file,
        embedding
    )

    return embedding


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    data,
    threshold
):

    if len(data) == 0:

        return {
            "accuracy": np.nan,
            "tar": np.nan,
            "fmr": np.nan,
            "fnmr": np.nan,
        }


    prediction = (
        data["similarity"]
        >= threshold
    ).astype(int)


    genuine = (
        data["label"] == 1
    )

    impostor = (
        data["label"] == 0
    )


    accuracy = (
        prediction == data["label"]
    ).mean()


    genuine_count = genuine.sum()

    impostor_count = impostor.sum()


    false_rejection = (
        genuine
        &
        (prediction == 0)
    ).sum()


    false_acceptance = (
        impostor
        &
        (prediction == 1)
    ).sum()


    fnmr = (
        false_rejection
        /
        genuine_count
        if genuine_count > 0
        else np.nan
    )


    fmr = (
        false_acceptance
        /
        impostor_count
        if impostor_count > 0
        else np.nan
    )


    tar = (
        1.0 - fnmr
        if not np.isnan(fnmr)
        else np.nan
    )


    return {
        "accuracy": accuracy,
        "tar": tar,
        "fmr": fmr,
        "fnmr": fnmr,
    }


# ============================================================
# PROCESS EACH CONDITION
# ============================================================

all_results = []
all_scores = []


for condition, kernel in BLUR_LEVELS.items():

    print()
    print("=" * 70)
    print(
        f"PROCESSING CONDITION: "
        f"{condition}"
    )
    print(
        f"Gaussian kernel: {kernel}"
    )
    print("=" * 70)


    # --------------------------------------------------------
    # Generate embeddings for all required images
    # --------------------------------------------------------

    embeddings = {}

    for image_key in tqdm(
        image_paths.keys(),
        desc=f"ArcFace embeddings [{condition}]"
    ):

        embeddings[image_key] = get_embedding(
            image_key,
            condition,
            kernel
        )


    # --------------------------------------------------------
    # Calculate pair similarities
    # --------------------------------------------------------

    similarities = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"Scoring pairs [{condition}]"
    ):

        emb1 = embeddings[
            row["image1"]
        ]

        emb2 = embeddings[
            row["image2"]
        ]

        similarity = cosine_similarity(
            emb1,
            emb2
        )

        similarities.append(
            similarity
        )


    condition_df = df.copy()

    condition_df["similarity"] = (
        similarities
    )

    condition_df["condition"] = (
        condition
    )

    condition_df["blur_kernel"] = (
        kernel
    )


    # --------------------------------------------------------
    # Save pair scores
    # --------------------------------------------------------

    score_file = (
        OUTPUT_DIR
        / f"{condition}_scores.csv"
    )

    condition_df.to_csv(
        score_file,
        index=False
    )


    # --------------------------------------------------------
    # Gender groups
    # --------------------------------------------------------

    female_mask = (
        (condition_df["gender1"] == "females")
        &
        (condition_df["gender2"] == "females")
    )

    male_mask = (
        (condition_df["gender1"] == "males")
        &
        (condition_df["gender2"] == "males")
    )


    female_df = condition_df[
        female_mask
    ]

    male_df = condition_df[
        male_mask
    ]


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    overall = calculate_metrics(
        condition_df,
        THRESHOLD
    )

    female = calculate_metrics(
        female_df,
        THRESHOLD
    )

    male = calculate_metrics(
        male_df,
        THRESHOLD
    )


    # --------------------------------------------------------
    # Fairness disparities
    # --------------------------------------------------------

    fmr_disparity = abs(
        female["fmr"]
        -
        male["fmr"]
    )

    fnmr_disparity = abs(
        female["fnmr"]
        -
        male["fnmr"]
    )

    tar_disparity = abs(
        female["tar"]
        -
        male["tar"]
    )

    accuracy_disparity = abs(
        female["accuracy"]
        -
        male["accuracy"]
    )


    mean_disparity = np.mean(
        [
            fmr_disparity,
            fnmr_disparity,
            tar_disparity,
            accuracy_disparity,
        ]
    )


    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    result = {

        "condition": condition,

        "blur_kernel": kernel,

        "threshold": THRESHOLD,

        "accuracy": overall["accuracy"],
        "tar": overall["tar"],
        "fmr": overall["fmr"],
        "fnmr": overall["fnmr"],

        "female_accuracy":
            female["accuracy"],

        "male_accuracy":
            male["accuracy"],

        "female_tar":
            female["tar"],

        "male_tar":
            male["tar"],

        "female_fmr":
            female["fmr"],

        "male_fmr":
            male["fmr"],

        "female_fnmr":
            female["fnmr"],

        "male_fnmr":
            male["fnmr"],

        "fmr_disparity":
            fmr_disparity,

        "fnmr_disparity":
            fnmr_disparity,

        "tar_disparity":
            tar_disparity,

        "accuracy_disparity":
            accuracy_disparity,

        "mean_disparity":
            mean_disparity,

        "num_female_pairs":
            len(female_df),

        "num_male_pairs":
            len(male_df),
    }


    all_results.append(
        result
    )

    all_scores.append(
        condition_df
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

results_df = pd.DataFrame(
    all_results
)

summary_file = (
    OUTPUT_DIR
    / "blur_fairness_results.csv"
)

results_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# CHANGE FROM ORIGINAL
# ============================================================

baseline = results_df[
    results_df["condition"] == "original"
].iloc[0]


comparison_rows = []

for _, row in results_df.iterrows():

    comparison_rows.append({

        "condition":
            row["condition"],

        "blur_kernel":
            row["blur_kernel"],

        "accuracy_change":
            row["accuracy"]
            -
            baseline["accuracy"],

        "tar_change":
            row["tar"]
            -
            baseline["tar"],

        "fmr_change":
            row["fmr"]
            -
            baseline["fmr"],

        "fnmr_change":
            row["fnmr"]
            -
            baseline["fnmr"],

        "mean_disparity_change":
            row["mean_disparity"]
            -
            baseline["mean_disparity"],

        "accuracy_disparity_change":
            row["accuracy_disparity"]
            -
            baseline["accuracy_disparity"],

        "fmr_disparity_change":
            row["fmr_disparity"]
            -
            baseline["fmr_disparity"],

        "fnmr_disparity_change":
            row["fnmr_disparity"]
            -
            baseline["fnmr_disparity"],
    })


comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_file = (
    OUTPUT_DIR
    / "blur_before_after.csv"
)

comparison_df.to_csv(
    comparison_file,
    index=False
)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print()
print("=" * 70)
print("BLUR FAIRNESS STRESS TEST COMPLETE")
print("=" * 70)

print()

print(
    results_df[
        [
            "condition",
            "blur_kernel",
            "accuracy",
            "tar",
            "fmr",
            "fnmr",
            "mean_disparity",
        ]
    ].to_string(
        index=False
    )
)

print()
print("=" * 70)
print("OUTPUT FILES")
print("=" * 70)

print(
    summary_file
)

print(
    comparison_file
)

print(
    OUTPUT_DIR
)

print("=" * 70)