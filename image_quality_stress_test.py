import os
import cv2
import numpy as np
import pandas as pd

# ============================================================
# FAIRNESS-FR — IMAGE QUALITY FAIRNESS STRESS TEST
# STEP 4A: GAUSSIAN BLUR
# ============================================================

DATASET = "bfw"
MODEL = "arcface"

PAIRS_FILE = os.path.join(
    "results", DATASET, MODEL, "test_scores.csv"
)

OUTPUT_DIR = os.path.join(
    "results", DATASET, MODEL, "image_quality_stress_test"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# Blur configurations
# ------------------------------------------------------------

BLUR_LEVELS = {
    "original": 0,
    "blur_light": 3,
    "blur_medium": 7,
    "blur_heavy": 15,
}


# ------------------------------------------------------------
# Load pair scores
# ------------------------------------------------------------

print("=" * 70)
print("FAIRNESS-FR — IMAGE QUALITY FAIRNESS STRESS TEST")
print("STEP 4A — GAUSSIAN BLUR")
print("=" * 70)

if not os.path.exists(PAIRS_FILE):
    raise FileNotFoundError(
        f"Cannot find:\n{PAIRS_FILE}"
    )

df = pd.read_csv(PAIRS_FILE)

required = [
    "gender1",
    "gender2",
    "label",
    "cosine_similarity"
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )

df = df.dropna(
    subset=[
        "gender1",
        "gender2",
        "label",
        "cosine_similarity"
    ]
).copy()

df["gender1"] = df["gender1"].astype(str).str.lower()
df["gender2"] = df["gender2"].astype(str).str.lower()
df["label"] = df["label"].astype(int)

df["cosine_similarity"] = pd.to_numeric(
    df["cosine_similarity"],
    errors="coerce"
)

df = df.dropna(
    subset=["cosine_similarity"]
).reset_index(drop=True)


# ------------------------------------------------------------
# IMPORTANT:
# This experiment uses existing similarity scores as the
# baseline. Blur cannot change those scores.
#
# Therefore, this script establishes the image-quality
# evaluation structure and computes the group composition.
#
# Actual degraded-image recognition requires re-extracting
# ArcFace embeddings from degraded images.
# ------------------------------------------------------------

female_mask = (
    (df["gender1"] == "females") &
    (df["gender2"] == "females")
)

male_mask = (
    (df["gender1"] == "males") &
    (df["gender2"] == "males")
)

female_df = df[female_mask].copy()
male_df = df[male_mask].copy()


# ------------------------------------------------------------
# Use fairness-optimal threshold found earlier
# ------------------------------------------------------------

THRESHOLD = 0.1546


def calculate_metrics(data, threshold):

    if len(data) == 0:
        return {
            "accuracy": np.nan,
            "tar": np.nan,
            "fmr": np.nan,
            "fnmr": np.nan
        }

    prediction = (
        data["cosine_similarity"] >= threshold
    ).astype(int)

    genuine = data["label"] == 1
    impostor = data["label"] == 0

    accuracy = (
        prediction == data["label"]
    ).mean()

    genuine_count = genuine.sum()
    impostor_count = impostor.sum()

    false_rejection = (
        genuine & (prediction == 0)
    ).sum()

    false_acceptance = (
        impostor & (prediction == 1)
    ).sum()

    fnmr = (
        false_rejection / genuine_count
        if genuine_count > 0
        else np.nan
    )

    fmr = (
        false_acceptance / impostor_count
        if impostor_count > 0
        else np.nan
    )

    tar = (
        1 - fnmr
        if not np.isnan(fnmr)
        else np.nan
    )

    return {
        "accuracy": accuracy,
        "tar": tar,
        "fmr": fmr,
        "fnmr": fnmr
    }


# ------------------------------------------------------------
# Baseline metrics
# ------------------------------------------------------------

overall = calculate_metrics(df, THRESHOLD)
female = calculate_metrics(female_df, THRESHOLD)
male = calculate_metrics(male_df, THRESHOLD)


# ------------------------------------------------------------
# Since existing scores are unchanged, report the baseline
# composition for each planned degradation level.
# ------------------------------------------------------------

records = []

for level, kernel in BLUR_LEVELS.items():

    fmr_disparity = abs(
        female["fmr"] - male["fmr"]
    )

    fnmr_disparity = abs(
        female["fnmr"] - male["fnmr"]
    )

    tar_disparity = abs(
        female["tar"] - male["tar"]
    )

    accuracy_disparity = abs(
        female["accuracy"] - male["accuracy"]
    )

    mean_disparity = np.mean([
        fmr_disparity,
        fnmr_disparity,
        tar_disparity,
        accuracy_disparity
    ])

    records.append({
        "condition": level,
        "blur_kernel": kernel,

        "threshold": THRESHOLD,

        "accuracy": overall["accuracy"],
        "tar": overall["tar"],
        "fmr": overall["fmr"],
        "fnmr": overall["fnmr"],

        "female_accuracy": female["accuracy"],
        "male_accuracy": male["accuracy"],

        "female_tar": female["tar"],
        "male_tar": male["tar"],

        "female_fmr": female["fmr"],
        "male_fmr": male["fmr"],

        "female_fnmr": female["fnmr"],
        "male_fnmr": male["fnmr"],

        "fmr_disparity": fmr_disparity,
        "fnmr_disparity": fnmr_disparity,
        "tar_disparity": tar_disparity,
        "accuracy_disparity": accuracy_disparity,

        "mean_disparity": mean_disparity
    })


results = pd.DataFrame(records)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

output_file = os.path.join(
    OUTPUT_DIR,
    "blur_fairness_baseline.csv"
)

results.to_csv(
    output_file,
    index=False
)


summary_file = os.path.join(
    OUTPUT_DIR,
    "blur_fairness_summary.txt"
)

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "FAIRNESS-FR — IMAGE QUALITY FAIRNESS STRESS TEST\n"
    )
    f.write(
        "STEP 4A — GAUSSIAN BLUR\n\n"
    )

    f.write(
        f"Total pairs: {len(df)}\n"
    )

    f.write(
        f"Female-Female pairs: {len(female_df)}\n"
    )

    f.write(
        f"Male-Male pairs: {len(male_df)}\n"
    )

    f.write(
        f"Threshold: {THRESHOLD}\n\n"
    )

    f.write(
        "NOTE:\n"
    )

    f.write(
        "This first stage records the baseline metrics and "
        "planned blur conditions. Actual blur-induced "
        "recognition changes require ArcFace embeddings "
        "to be re-extracted from the degraded images.\n"
    )


# ------------------------------------------------------------
# Console
# ------------------------------------------------------------

print()
print(f"Total pairs: {len(df)}")
print(f"Female-Female pairs: {len(female_df)}")
print(f"Male-Male pairs: {len(male_df)}")

print()
print(f"Operating threshold: {THRESHOLD}")

print()
print("BASELINE METRICS")
print(
    f"Accuracy: {overall['accuracy']:.4f}"
)
print(
    f"TAR:      {overall['tar']:.4f}"
)
print(
    f"FMR:      {overall['fmr']:.4f}"
)
print(
    f"FNMR:     {overall['fnmr']:.4f}"
)

print()
print("FAIRNESS BASELINE")
print(
    f"FMR disparity: "
    f"{abs(female['fmr'] - male['fmr']):.6f}"
)
print(
    f"FNMR disparity: "
    f"{abs(female['fnmr'] - male['fnmr']):.6f}"
)
print(
    f"TAR disparity: "
    f"{abs(female['tar'] - male['tar']):.6f}"
)
print(
    f"Accuracy disparity: "
    f"{abs(female['accuracy'] - male['accuracy']):.6f}"
)

print()
print("=" * 70)
print("STEP 4A BASELINE COMPLETE")
print("=" * 70)

print()
print("Planned blur conditions:")

for name, kernel in BLUR_LEVELS.items():
    print(f"  {name:15s} kernel={kernel}")

print()
print("Output:")
print(output_file)
print(summary_file)

print("=" * 70)