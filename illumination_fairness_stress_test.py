from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# ============================================================
# FAIRNESS-FR — ACTUAL IMAGE QUALITY FAIRNESS STRESS TEST
# STEP 4A — GAUSSIAN BLUR + PROJECT ARCFACE
# ============================================================


# ============================================================
# PROJECT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fairness_fr.config.config import ConfigLoader
from fairness_fr.models.extract_embeddings import load_embedding_model


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "arcface"

PAIRS_FILE = (
    PROJECT_ROOT
    / "results"
    / "bfw"
    / "arcface"
    / "test_scores.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "bfw"
    / "arcface"
    / "image_quality_stress_test"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Same operating threshold used in previous experiments
THRESHOLD = 0.1546


# ============================================================
# BLUR CONDITIONS
# Kernel size must be odd.
# 0 means no blur.
# ============================================================

BLUR_LEVELS = {
    "original": 0,
    "blur_light": 3,
    "blur_medium": 7,
    "blur_heavy": 15,
}


# ============================================================
# LOAD MODEL CONFIGURATION
# ============================================================

print("=" * 70)
print("FAIRNESS-FR — ACTUAL IMAGE QUALITY FAIRNESS STRESS TEST")
print("STEP 4A — GAUSSIAN BLUR + PROJECT ARCFACE")
print("=" * 70)

loader = ConfigLoader(
    PROJECT_ROOT / "configs"
)

model_config = loader.load_model_config(MODEL_NAME)


# Convert relative model path to absolute path
weights_path = Path(model_config.weights_path)

if not weights_path.is_absolute():

    model_config = model_config.model_copy(
        update={
            "weights_path": PROJECT_ROOT / weights_path
        }
    )


print("\nModel:")
print(model_config)


# ============================================================
# LOAD PAIRS
# ============================================================

if not PAIRS_FILE.exists():

    raise FileNotFoundError(
        f"Cannot find:\n{PAIRS_FILE}"
    )


df = pd.read_csv(PAIRS_FILE)


required_columns = [
    "image1",
    "image2",
    "gender1",
    "gender2",
    "label",
]


missing = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing:

    raise ValueError(
        f"Missing columns: {missing}"
    )


print(f"\nTotal pairs: {len(df)}")


# ============================================================
# IMAGE PATH RESOLUTION
# ============================================================

def resolve_image_path(image_path):
    """
    Resolve image paths stored in test_scores.csv.
    """

    image_path = Path(str(image_path))

    candidates = [
        image_path,
        PROJECT_ROOT / image_path,
    ]

    for candidate in candidates:

        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Cannot resolve image:\n{image_path}"
    )


# ============================================================
# GET UNIQUE IMAGES
# ============================================================

unique_images = sorted(
    set(df["image1"].astype(str))
    | set(df["image2"].astype(str))
)


print(f"Unique images: {len(unique_images)}")
print("\nResolving images...")


image_paths = {}


for image_name in tqdm(
    unique_images,
    desc="Resolving images"
):

    image_paths[image_name] = resolve_image_path(
        image_name
    )


print(
    f"Resolved {len(image_paths)} unique images."
)


# ============================================================
# LOAD PROJECT ARCFACE MODEL
# ============================================================

print(
    "\nLoading project's ArcFace ONNX model..."
)


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


model = load_embedding_model(
    model_config,
    DEVICE
)


# ============================================================
# IMAGE PREPROCESSING + GAUSSIAN BLUR
# ============================================================

def preprocess_image(
    image_path,
    blur_kernel
):
    """
    Load image, apply Gaussian blur, then perform
    the same ArcFace preprocessing used by the project.
    """

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR
    )


    if image is None:

        raise ValueError(
            f"Could not read image:\n{image_path}"
        )


    # OpenCV BGR -> RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    # ========================================================
    # APPLY ACTUAL GAUSSIAN BLUR
    # ========================================================

    if blur_kernel > 0:

        image = cv2.GaussianBlur(
            image,
            (blur_kernel, blur_kernel),
            0
        )


    # ========================================================
    # ARC FACE PREPROCESSING
    # ========================================================

    target_size = model_config.input_size


    if (
        image.shape[0] != target_size
        or image.shape[1] != target_size
    ):

        image = cv2.resize(
            image,
            (target_size, target_size),
            interpolation=cv2.INTER_LANCZOS4
        )


    # Convert [0,255] -> [0,1]
    image = image.astype(np.float32) / 255.0


    # HWC -> CHW
    image = image.transpose(
        2,
        0,
        1
    )


    mean = np.asarray(
        model_config.normalization_mean,
        dtype=np.float32
    ).reshape(
        3,
        1,
        1
    )


    std = np.asarray(
        model_config.normalization_std,
        dtype=np.float32
    ).reshape(
        3,
        1,
        1
    )


    # Project ArcFace normalization
    image = (
        image - mean
    ) / std


    return image.astype(
        np.float32
    )


# ============================================================
# EMBEDDING GENERATION
# ============================================================

def get_embeddings(
    blur_kernel,
    condition_name
):
    """
    Generate fresh ArcFace embeddings from the
    Gaussian-blurred images.
    """

    embeddings = {}


    batch_size = (
        model_config.batch_size_override
        or 32
    )


    image_items = list(
        image_paths.items()
    )


    for start in tqdm(
        range(
            0,
            len(image_items),
            batch_size
        ),
        desc=f"ArcFace embeddings [{condition_name}]"
    ):

        batch_items = image_items[
            start:start + batch_size
        ]


        batch_images = []


        for (
            image_name,
            image_path
        ) in batch_items:

            processed = preprocess_image(
                image_path,
                blur_kernel
            )

            batch_images.append(
                processed
            )


        batch_np = np.stack(
            batch_images,
            axis=0
        ).astype(
            np.float32
        )


        # Project embed_batch expects torch.Tensor
        batch_tensor = torch.from_numpy(
            batch_np
        )


        if DEVICE.startswith("cuda"):

            batch_tensor = batch_tensor.to(
                DEVICE
            )


        batch_embeddings = model.embed_batch(
            batch_tensor
        )


        batch_embeddings = np.asarray(
            batch_embeddings,
            dtype=np.float32
        )


        # L2 normalization
        norms = np.linalg.norm(
            batch_embeddings,
            axis=1,
            keepdims=True
        )


        norms = np.maximum(
            norms,
            1e-12
        )


        batch_embeddings = (
            batch_embeddings / norms
        )


        for (
            image_name,
            _
        ), embedding in zip(
            batch_items,
            batch_embeddings
        ):

            embeddings[
                image_name
            ] = embedding


    return embeddings


# ============================================================
# SCORE PAIRS
# ============================================================

def score_pairs(
    embeddings,
    source_df
):
    """
    Calculate fresh cosine similarities
    from newly extracted embeddings.
    """

    scored_rows = []


    for _, row in tqdm(
        source_df.iterrows(),
        total=len(source_df),
        desc="Scoring pairs"
    ):

        image1 = str(
            row["image1"]
        )

        image2 = str(
            row["image2"]
        )


        embedding1 = embeddings[
            image1
        ]

        embedding2 = embeddings[
            image2
        ]


        cosine_similarity = float(
            np.dot(
                embedding1,
                embedding2
            )
        )


        prediction = int(
            cosine_similarity >= THRESHOLD
        )


        scored_rows.append(
            {
                "image1": image1,
                "image2": image2,
                "gender1": row["gender1"],
                "gender2": row["gender2"],
                "label": int(row["label"]),
                "cosine_similarity": cosine_similarity,
                "prediction": prediction,
            }
        )


    return pd.DataFrame(
        scored_rows
    )


# ============================================================
# METRIC CALCULATION
# ============================================================

def calculate_metrics(pair_df):

    labels = pair_df["label"].to_numpy()

    predictions = pair_df[
        "prediction"
    ].to_numpy()


    tp = int(
        np.sum(
            (labels == 1)
            & (predictions == 1)
        )
    )


    tn = int(
        np.sum(
            (labels == 0)
            & (predictions == 0)
        )
    )


    fp = int(
        np.sum(
            (labels == 0)
            & (predictions == 1)
        )
    )


    fn = int(
        np.sum(
            (labels == 1)
            & (predictions == 0)
        )
    )


    accuracy = (
        (tp + tn) / len(pair_df)
        if len(pair_df) > 0
        else np.nan
    )


    tar = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else np.nan
    )


    fmr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else np.nan
    )


    fnmr = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else np.nan
    )


    return {
        "accuracy": accuracy,
        "tar": tar,
        "fmr": fmr,
        "fnmr": fnmr,
        "pairs": len(pair_df),
    }


# ============================================================
# FAIRNESS ANALYSIS
# ============================================================

def calculate_fairness(pair_df):

    fairness_df = pair_df.copy()


    gender_map = {
        "female": "female",
        "females": "female",
        "f": "female",
        "male": "male",
        "males": "male",
        "m": "male",
    }


    fairness_df[
        "gender1_clean"
    ] = (
        fairness_df["gender1"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(gender_map)
    )


    fairness_df[
        "gender2_clean"
    ] = (
        fairness_df["gender2"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(gender_map)
    )


    female_female = fairness_df[
        (
            fairness_df[
                "gender1_clean"
            ] == "female"
        )
        &
        (
            fairness_df[
                "gender2_clean"
            ] == "female"
        )
    ]


    male_male = fairness_df[
        (
            fairness_df[
                "gender1_clean"
            ] == "male"
        )
        &
        (
            fairness_df[
                "gender2_clean"
            ] == "male"
        )
    ]


    female_metrics = calculate_metrics(
        female_female
    )

    male_metrics = calculate_metrics(
        male_male
    )


    accuracy_disparity = abs(
        female_metrics["accuracy"]
        - male_metrics["accuracy"]
    )

    tar_disparity = abs(
        female_metrics["tar"]
        - male_metrics["tar"]
    )

    fmr_disparity = abs(
        female_metrics["fmr"]
        - male_metrics["fmr"]
    )

    fnmr_disparity = abs(
        female_metrics["fnmr"]
        - male_metrics["fnmr"]
    )


    mean_disparity = np.mean(
        [
            accuracy_disparity,
            tar_disparity,
            fmr_disparity,
            fnmr_disparity,
        ]
    )


    return {
        "female_pairs": len(female_female),
        "male_pairs": len(male_male),

        "female_metrics": female_metrics,
        "male_metrics": male_metrics,

        "accuracy_disparity": accuracy_disparity,
        "tar_disparity": tar_disparity,
        "fmr_disparity": fmr_disparity,
        "fnmr_disparity": fnmr_disparity,
        "mean_disparity": mean_disparity,
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

all_results = []


for (
    condition,
    blur_kernel
) in BLUR_LEVELS.items():

    print(
        "\n" + "=" * 70
    )

    print(
        f"PROCESSING CONDITION: {condition}"
    )

    print(
        f"Gaussian blur kernel: {blur_kernel}"
    )

    print(
        "=" * 70
    )


    # ========================================================
    # GENERATE FRESH EMBEDDINGS
    # ========================================================

    embeddings = get_embeddings(
        blur_kernel,
        condition
    )


    # ========================================================
    # SCORE PAIRS AGAIN
    # ========================================================

    scored_df = score_pairs(
        embeddings,
        df
    )


    # Save pair-level scores
    condition_file = (
        OUTPUT_DIR
        / f"{condition}_pair_scores.csv"
    )


    scored_df.to_csv(
        condition_file,
        index=False
    )


    # ========================================================
    # METRICS
    # ========================================================

    overall_metrics = calculate_metrics(
        scored_df
    )


    fairness = calculate_fairness(
        scored_df
    )


    # ========================================================
    # CONSOLE RESULTS
    # ========================================================

    print("\nOVERALL METRICS")

    print(
        f"Accuracy: {overall_metrics['accuracy']:.6f}"
    )

    print(
        f"TAR:      {overall_metrics['tar']:.6f}"
    )

    print(
        f"FMR:      {overall_metrics['fmr']:.6f}"
    )

    print(
        f"FNMR:     {overall_metrics['fnmr']:.6f}"
    )


    print("\nGROUP COUNTS")

    print(
        f"Female-Female pairs: "
        f"{fairness['female_pairs']}"
    )

    print(
        f"Male-Male pairs:     "
        f"{fairness['male_pairs']}"
    )


    print("\nFAIRNESS")

    print(
        f"Accuracy disparity: "
        f"{fairness['accuracy_disparity']:.6f}"
    )

    print(
        f"TAR disparity:      "
        f"{fairness['tar_disparity']:.6f}"
    )

    print(
        f"FMR disparity:      "
        f"{fairness['fmr_disparity']:.6f}"
    )

    print(
        f"FNMR disparity:     "
        f"{fairness['fnmr_disparity']:.6f}"
    )

    print(
        f"Mean disparity:     "
        f"{fairness['mean_disparity']:.6f}"
    )


    # ========================================================
    # STORE CONDITION RESULT
    # ========================================================

    all_results.append(
        {
            "condition": condition,
            "blur_kernel": blur_kernel,

            "accuracy": overall_metrics[
                "accuracy"
            ],

            "tar": overall_metrics[
                "tar"
            ],

            "fmr": overall_metrics[
                "fmr"
            ],

            "fnmr": overall_metrics[
                "fnmr"
            ],

            "female_pairs": fairness[
                "female_pairs"
            ],

            "male_pairs": fairness[
                "male_pairs"
            ],

            "accuracy_disparity": fairness[
                "accuracy_disparity"
            ],

            "tar_disparity": fairness[
                "tar_disparity"
            ],

            "fmr_disparity": fairness[
                "fmr_disparity"
            ],

            "fnmr_disparity": fairness[
                "fnmr_disparity"
            ],

            "mean_disparity": fairness[
                "mean_disparity"
            ],
        }
    )


# ============================================================
# SAVE FINAL RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)


output_file = (
    OUTPUT_DIR
    / "blur_fairness_results.csv"
)


results_df.to_csv(
    output_file,
    index=False
)


summary_file = (
    OUTPUT_DIR
    / "blur_fairness_summary.txt"
)


with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "FAIRNESS-FR — ACTUAL IMAGE QUALITY FAIRNESS STRESS TEST\n"
    )

    f.write(
        "STEP 4A — GAUSSIAN BLUR + PROJECT ARCFACE\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        results_df.to_string(
            index=False
        )
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "GAUSSIAN BLUR FAIRNESS STRESS TEST COMPLETE"
)

print(
    "=" * 70
)


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


print("\nOutput files:")

print(
    output_file.resolve()
)

print(
    summary_file.resolve()
)

print(
    "=" * 70
)