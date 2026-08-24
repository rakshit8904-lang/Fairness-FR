from pathlib import Path
import sys
import pandas as pd
import numpy as np

# ============================================================
# FAIRNESS-FR — ARCFACE BASELINE REPRODUCTION CHECK
# ============================================================

print("=" * 70)
print("FAIRNESS-FR — ARCFACE BASELINE REPRODUCTION CHECK")
print("=" * 70)

# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

PROJECT_ROOT = Path("..").resolve()

SRC = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC))

# ------------------------------------------------------------
# Load the project's own configuration
# ------------------------------------------------------------

try:
    from fairness_fr.config.config import get_config_loader

    loader = get_config_loader()

    print("\n[1] Configuration loader:")
    print(loader)

except Exception as e:

    print("\nCould not load project configuration:")
    print(type(e).__name__, e)

# ------------------------------------------------------------
# Check ArcFace configuration
# ------------------------------------------------------------

arcface_config = PROJECT_ROOT / "configs" / "models" / "arcface.yaml"

print("\n[2] ArcFace configuration:")
print(arcface_config)

if arcface_config.exists():

    print("\n--- arcface.yaml ---")

    print(
        arcface_config.read_text(
            encoding="utf-8"
        )
    )

else:

    print("ERROR: arcface.yaml not found")


# ------------------------------------------------------------
# Check BFW configuration
# ------------------------------------------------------------

bfw_config = PROJECT_ROOT / "configs" / "datasets" / "bfw.yaml"

print("\n[3] BFW configuration:")
print(bfw_config)

if bfw_config.exists():

    print("\n--- bfw.yaml ---")

    print(
        bfw_config.read_text(
            encoding="utf-8"
        )
    )

else:

    print("ERROR: bfw.yaml not found")


# ------------------------------------------------------------
# Existing project scores
# ------------------------------------------------------------

scores_file = (
    PROJECT_ROOT
    / "results"
    / "bfw"
    / "arcface"
    / "test_scores.csv"
)

print("\n[4] Existing ArcFace test scores:")
print(scores_file)

if not scores_file.exists():

    print("ERROR: test_scores.csv not found")

    sys.exit(1)


df = pd.read_csv(scores_file)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ------------------------------------------------------------
# Existing score statistics
# ------------------------------------------------------------

if "cosine_similarity" in df.columns:

    scores = pd.to_numeric(
        df["cosine_similarity"],
        errors="coerce"
    ).dropna()

    print("\n[5] Existing cosine similarity statistics")

    print(
        scores.describe()
    )


# ------------------------------------------------------------
# Existing labels
# ------------------------------------------------------------

if "label" in df.columns:

    print("\n[6] Pair labels")

    print(
        df["label"]
        .value_counts()
        .sort_index()
    )


# ------------------------------------------------------------
# Existing gender composition
# ------------------------------------------------------------

if (
    "gender1" in df.columns
    and
    "gender2" in df.columns
):

    print("\n[7] Gender pair composition")

    female = (
        (df["gender1"].astype(str).str.lower() == "females")
        &
        (df["gender2"].astype(str).str.lower() == "females")
    )

    male = (
        (df["gender1"].astype(str).str.lower() == "males")
        &
        (df["gender2"].astype(str).str.lower() == "males")
    )

    print(
        "Female-Female:",
        female.sum()
    )

    print(
        "Male-Male:",
        male.sum()
    )


# ------------------------------------------------------------
# Recalculate existing pipeline metrics
# ------------------------------------------------------------

THRESHOLD = 0.1546

print("\n[8] Recalculating existing scores")
print(
    f"Threshold = {THRESHOLD}"
)

df2 = df.copy()

df2["score"] = pd.to_numeric(
    df2["cosine_similarity"],
    errors="coerce"
)

df2["label"] = pd.to_numeric(
    df2["label"],
    errors="coerce"
)

df2 = df2.dropna(
    subset=["score", "label"]
)

df2["prediction"] = (
    df2["score"] >= THRESHOLD
).astype(int)


# ------------------------------------------------------------
# Overall metrics
# ------------------------------------------------------------

accuracy = (
    df2["prediction"]
    ==
    df2["label"]
).mean()


genuine = (
    df2["label"] == 1
)

impostor = (
    df2["label"] == 0
)


false_rejection = (
    genuine
    &
    (df2["prediction"] == 0)
).sum()


false_acceptance = (
    impostor
    &
    (df2["prediction"] == 1)
).sum()


genuine_count = genuine.sum()

impostor_count = impostor.sum()


fnmr = (
    false_rejection / genuine_count
    if genuine_count
    else np.nan
)

fmr = (
    false_acceptance / impostor_count
    if impostor_count
    else np.nan
)

tar = (
    1 - fnmr
    if not np.isnan(fnmr)
    else np.nan
)


print("\nOVERALL")

print(
    f"Accuracy : {accuracy:.6f}"
)

print(
    f"TAR      : {tar:.6f}"
)

print(
    f"FMR      : {fmr:.6f}"
)

print(
    f"FNMR     : {fnmr:.6f}"
)


# ------------------------------------------------------------
# Gender metrics
# ------------------------------------------------------------

def gender_metrics(data):

    if len(data) == 0:
        return None

    prediction = (
        data["score"] >= THRESHOLD
    ).astype(int)

    genuine = (
        data["label"] == 1
    )

    impostor = (
        data["label"] == 0
    )

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
        false_rejection / genuine_count
        if genuine_count
        else np.nan
    )

    fmr = (
        false_acceptance / impostor_count
        if impostor_count
        else np.nan
    )

    tar = (
        1 - fnmr
        if not np.isnan(fnmr)
        else np.nan
    )

    accuracy = (
        prediction == data["label"]
    ).mean()

    return {
        "accuracy": accuracy,
        "tar": tar,
        "fmr": fmr,
        "fnmr": fnmr,
        "pairs": len(data),
    }


female_df = df2[
    (df2["gender1"].astype(str).str.lower() == "females")
    &
    (df2["gender2"].astype(str).str.lower() == "females")
]

male_df = df2[
    (df2["gender1"].astype(str).str.lower() == "males")
    &
    (df2["gender2"].astype(str).str.lower() == "males")
]


female_metrics = gender_metrics(
    female_df
)

male_metrics = gender_metrics(
    male_df
)


print("\n[9] GENDER METRICS")

print("\nFemale-Female")

print(female_metrics)

print("\nMale-Male")

print(male_metrics)


# ------------------------------------------------------------
# Fairness disparity
# ------------------------------------------------------------

if female_metrics and male_metrics:

    fmr_disparity = abs(
        female_metrics["fmr"]
        -
        male_metrics["fmr"]
    )

    fnmr_disparity = abs(
        female_metrics["fnmr"]
        -
        male_metrics["fnmr"]
    )

    tar_disparity = abs(
        female_metrics["tar"]
        -
        male_metrics["tar"]
    )

    accuracy_disparity = abs(
        female_metrics["accuracy"]
        -
        male_metrics["accuracy"]
    )

    mean_disparity = np.mean(
        [
            fmr_disparity,
            fnmr_disparity,
            tar_disparity,
            accuracy_disparity,
        ]
    )

    print("\n[10] FAIRNESS DISPARITY")

    print(
        f"FMR disparity      : {fmr_disparity:.6f}"
    )

    print(
        f"FNMR disparity     : {fnmr_disparity:.6f}"
    )

    print(
        f"TAR disparity      : {tar_disparity:.6f}"
    )

    print(
        f"Accuracy disparity : {accuracy_disparity:.6f}"
    )

    print(
        f"Mean disparity     : {mean_disparity:.6f}"
    )


# ------------------------------------------------------------
# Final
# ------------------------------------------------------------

print("\n" + "=" * 70)

print(
    "BASELINE REPRODUCTION CHECK COMPLETE"
)

print("=" * 70)

print(
    "\nIMPORTANT:"
)

print(
    "This script does NOT generate new embeddings."
)

print(
    "It only verifies the project's existing ArcFace "
    "test_scores.csv and configuration."
)

print(
    "\nPaste the COMPLETE terminal output here."
)

print("=" * 70)