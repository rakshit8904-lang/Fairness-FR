import os
import numpy as np
import pandas as pd


# ============================================================
# FAIRNESS-FR
# FAIRNESS-PERFORMANCE THRESHOLD TRADE-OFF ANALYSIS
#
# Purpose:
# Study how changing the global verification threshold affects:
#   1. Overall recognition performance
#   2. Female-Female vs Male-Male performance
#   3. Demographic disparities
#   4. The fairness-performance trade-off
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

DATASET = "bfw"
MODEL = "arcface"

INPUT_FILE = os.path.join(
    "results",
    DATASET,
    MODEL,
    "test_scores.csv"
)

OUTPUT_DIR = os.path.join(
    "results",
    DATASET,
    MODEL,
    "fairness_performance_tradeoff"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Maximum acceptable mean demographic disparity for the
# fairness-constrained operating point.
FAIRNESS_LIMIT = 0.01

# Number of thresholds evaluated across the score range.
NUM_THRESHOLDS = 181


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FAIRNESS-FR - FAIRNESS-PERFORMANCE THRESHOLD TRADE-OFF ANALYSIS")
print("=" * 70)

print()
print(f"Dataset: {DATASET}")
print(f"Model:   {MODEL}")
print(f"Input:   {INPUT_FILE}")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"\nInput file not found:\n{INPUT_FILE}\n\n"
        "Run this script from the project root directory:\n"
        "C:\\Users\\Rakshit\\Downloads\\drdo"
    )

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "gender1",
    "gender2",
    "label",
    "cosine_similarity"
]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}\n"
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# CLEAN DATA
# ============================================================

df = df.dropna(
    subset=[
        "gender1",
        "gender2",
        "label",
        "cosine_similarity"
    ]
).copy()

df["label"] = pd.to_numeric(
    df["label"],
    errors="coerce"
)

df["cosine_similarity"] = pd.to_numeric(
    df["cosine_similarity"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "label",
        "cosine_similarity"
    ]
).copy()

df["label"] = df["label"].astype(int)

# Keep only valid binary labels.
df = df[
    df["label"].isin([0, 1])
].copy()


# ============================================================
# NORMALIZE GENDER LABELS
# ============================================================

gender_map = {
    "female": "female",
    "females": "female",
    "f": "female",

    "male": "male",
    "males": "male",
    "m": "male"
}


def normalize_gender(value):
    value = str(value).strip().lower()
    return gender_map.get(value, value)


df["gender1_clean"] = df["gender1"].apply(normalize_gender)
df["gender2_clean"] = df["gender2"].apply(normalize_gender)


# ============================================================
# CREATE SAME-GENDER DEMOGRAPHIC GROUPS
# ============================================================

female_mask = (
    (df["gender1_clean"] == "female")
    &
    (df["gender2_clean"] == "female")
)

male_mask = (
    (df["gender1_clean"] == "male")
    &
    (df["gender2_clean"] == "male")
)

female_df = df[female_mask].copy()
male_df = df[male_mask].copy()


# ============================================================
# DATA VALIDATION
# ============================================================

print()
print("DATA SUMMARY")
print("-" * 70)
print(f"Total valid pairs:       {len(df)}")
print(f"Female-Female pairs:     {len(female_df)}")
print(f"Male-Male pairs:         {len(male_df)}")

print()
print("Female-Female label counts:")
print(female_df["label"].value_counts().sort_index().to_string())

print()
print("Male-Male label counts:")
print(male_df["label"].value_counts().sort_index().to_string())

if len(female_df) == 0:
    raise ValueError(
        "\nNo Female-Female pairs found after gender normalization."
    )

if len(male_df) == 0:
    raise ValueError(
        "\nNo Male-Male pairs found after gender normalization."
    )

if df["cosine_similarity"].nunique() < 2:
    raise ValueError(
        "\nNot enough unique similarity scores for threshold analysis."
    )


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(data, threshold):
    """
    Calculate verification metrics at a given threshold.

    Positive prediction:
        cosine_similarity >= threshold

    Labels:
        1 = genuine pair
        0 = impostor pair
    """

    if len(data) == 0:
        return {
            "accuracy": np.nan,
            "tar": np.nan,
            "fmr": np.nan,
            "fnmr": np.nan,
            "frr": np.nan,
            "genuine_count": 0,
            "impostor_count": 0
        }

    labels = data["label"].to_numpy()

    predictions = (
        data["cosine_similarity"].to_numpy() >= threshold
    ).astype(int)

    genuine_mask = labels == 1
    impostor_mask = labels == 0

    genuine_count = int(genuine_mask.sum())
    impostor_count = int(impostor_mask.sum())

    false_rejection = int(
        (
            genuine_mask
            &
            (predictions == 0)
        ).sum()
    )

    false_acceptance = int(
        (
            impostor_mask
            &
            (predictions == 1)
        ).sum()
    )

    accuracy = float(
        (predictions == labels).mean()
    )

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
        1.0 - fnmr
        if not np.isnan(fnmr)
        else np.nan
    )

    return {
        "accuracy": accuracy,
        "tar": tar,
        "fmr": fmr,
        "fnmr": fnmr,
        "frr": fnmr,
        "genuine_count": genuine_count,
        "impostor_count": impostor_count
    }


# ============================================================
# THRESHOLD RANGE
# ============================================================

min_score = float(df["cosine_similarity"].min())
max_score = float(df["cosine_similarity"].max())

thresholds = np.linspace(
    min_score,
    max_score,
    NUM_THRESHOLDS
)

print()
print("THRESHOLD RANGE")
print("-" * 70)
print(f"Minimum score: {min_score:.6f}")
print(f"Maximum score: {max_score:.6f}")
print(f"Thresholds:    {len(thresholds)}")


# ============================================================
# TRADE-OFF ANALYSIS
# ============================================================

records = []

for threshold in thresholds:

    overall = calculate_metrics(
        df,
        threshold
    )

    female = calculate_metrics(
        female_df,
        threshold
    )

    male = calculate_metrics(
        male_df,
        threshold
    )

    # --------------------------------------------------------
    # Demographic disparities
    # --------------------------------------------------------

    accuracy_disparity = abs(
        female["accuracy"]
        -
        male["accuracy"]
    )

    tar_disparity = abs(
        female["tar"]
        -
        male["tar"]
    )

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

    disparities = np.array([
        accuracy_disparity,
        tar_disparity,
        fmr_disparity,
        fnmr_disparity
    ])

    mean_disparity = float(
        np.nanmean(disparities)
    )

    max_disparity = float(
        np.nanmax(disparities)
    )

    # --------------------------------------------------------
    # Fairness score
    #
    # 1.0 = no disparity
    # 0.0 = maximum disparity
    # --------------------------------------------------------

    fairness_score = max(
        0.0,
        1.0 - mean_disparity
    )

    # --------------------------------------------------------
    # Balanced score
    #
    # Equal weighting:
    # 50% recognition performance
    # 50% demographic fairness
    # --------------------------------------------------------

    balanced_score = (
        0.5 * overall["accuracy"]
        +
        0.5 * fairness_score
    )

    records.append({
        "threshold": threshold,

        # Overall metrics
        "accuracy": overall["accuracy"],
        "tar": overall["tar"],
        "fmr": overall["fmr"],
        "fnmr": overall["fnmr"],

        # Female-Female metrics
        "female_accuracy": female["accuracy"],
        "female_tar": female["tar"],
        "female_fmr": female["fmr"],
        "female_fnmr": female["fnmr"],

        # Male-Male metrics
        "male_accuracy": male["accuracy"],
        "male_tar": male["tar"],
        "male_fmr": male["fmr"],
        "male_fnmr": male["fnmr"],

        # Disparities
        "accuracy_disparity": accuracy_disparity,
        "tar_disparity": tar_disparity,
        "fmr_disparity": fmr_disparity,
        "fnmr_disparity": fnmr_disparity,
        "mean_disparity": mean_disparity,
        "max_disparity": max_disparity,

        # Combined scores
        "fairness_score": fairness_score,
        "balanced_score": balanced_score
    })


results = pd.DataFrame(records)


# ============================================================
# OPERATING POINT 1:
# FAIRNESS-OPTIMAL THRESHOLD
#
# To avoid selecting a trivial extreme threshold,
# only consider thresholds with reasonable accuracy.
# ============================================================

performance_floor = (
    results["accuracy"].max() - 0.05
)

fairness_candidates = results[
    results["accuracy"] >= performance_floor
].copy()

fairness_optimal = fairness_candidates.loc[
    fairness_candidates["mean_disparity"].idxmin()
]


# ============================================================
# OPERATING POINT 2:
# ACCURACY-OPTIMAL THRESHOLD
# ============================================================

accuracy_optimal = results.loc[
    results["accuracy"].idxmax()
]


# ============================================================
# OPERATING POINT 3:
# BALANCED FAIRNESS-PERFORMANCE THRESHOLD
# ============================================================

balanced_optimal = results.loc[
    results["balanced_score"].idxmax()
]


# ============================================================
# OPERATING POINT 4:
# FAIRNESS-CONSTRAINED ACCURACY-OPTIMAL THRESHOLD
# ============================================================

fair_candidates = results[
    results["mean_disparity"] <= FAIRNESS_LIMIT
].copy()

if len(fair_candidates) > 0:

    fairness_constrained = fair_candidates.loc[
        fair_candidates["accuracy"].idxmax()
    ]

    fairness_constraint_status = (
        f"Constraint satisfied: mean disparity <= {FAIRNESS_LIMIT}"
    )

else:

    fairness_constrained = balanced_optimal

    fairness_constraint_status = (
        f"No threshold satisfied mean disparity <= {FAIRNESS_LIMIT}. "
        "Balanced threshold selected instead."
    )


# ============================================================
# SAVE COMPLETE THRESHOLD TABLE
# ============================================================

results_file = os.path.join(
    OUTPUT_DIR,
    "threshold_tradeoff.csv"
)

results.to_csv(
    results_file,
    index=False
)


# ============================================================
# SAVE OPERATING POINTS
# ============================================================

operating_points = pd.DataFrame([
    {
        "selection": "fairness_optimal",
        **fairness_optimal.to_dict()
    },
    {
        "selection": "accuracy_optimal",
        **accuracy_optimal.to_dict()
    },
    {
        "selection": "balanced_optimal",
        **balanced_optimal.to_dict()
    },
    {
        "selection": "fairness_constrained_accuracy_optimal",
        **fairness_constrained.to_dict()
    }
])

operating_file = os.path.join(
    OUTPUT_DIR,
    "operating_points.csv"
)

operating_points.to_csv(
    operating_file,
    index=False
)


# ============================================================
# PARETO FRONT
#
# A point is dominated if another threshold has:
#   >= accuracy
#   <= mean disparity
#
# and at least one strict improvement.
# ============================================================

pareto_indices = []

for i, row in results.iterrows():

    dominated = False

    for j, other in results.iterrows():

        if i == j:
            continue

        better_or_equal_accuracy = (
            other["accuracy"] >= row["accuracy"]
        )

        better_or_equal_fairness = (
            other["mean_disparity"]
            <= row["mean_disparity"]
        )

        strictly_better = (
            (
                other["accuracy"]
                > row["accuracy"]
            )
            or
            (
                other["mean_disparity"]
                < row["mean_disparity"]
            )
        )

        if (
            better_or_equal_accuracy
            and better_or_equal_fairness
            and strictly_better
        ):
            dominated = True
            break

    if not dominated:
        pareto_indices.append(i)


pareto_front = results.loc[
    pareto_indices
].sort_values(
    by="threshold"
)

pareto_file = os.path.join(
    OUTPUT_DIR,
    "pareto_front.csv"
)

pareto_front.to_csv(
    pareto_file,
    index=False
)


# ============================================================
# SUMMARY FILE
# ============================================================

summary_file = os.path.join(
    OUTPUT_DIR,
    "tradeoff_summary.txt"
)

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as file:

    file.write("=" * 70 + "\n")
    file.write(
        "FAIRNESS-FR - FAIRNESS-PERFORMANCE TRADE-OFF SUMMARY\n"
    )
    file.write("=" * 70 + "\n\n")

    file.write(f"Dataset: {DATASET}\n")
    file.write(f"Model: {MODEL}\n")
    file.write(f"Total valid pairs: {len(df)}\n")
    file.write(f"Female-Female pairs: {len(female_df)}\n")
    file.write(f"Male-Male pairs: {len(male_df)}\n")
    file.write(
        f"Fairness limit: {FAIRNESS_LIMIT}\n"
    )
    file.write("\n")

    file.write(
        "FAIRNESS-OPTIMAL THRESHOLD\n"
    )
    file.write(
        f"Threshold: {fairness_optimal['threshold']:.6f}\n"
    )
    file.write(
        f"Accuracy: {fairness_optimal['accuracy']:.6f}\n"
    )
    file.write(
        f"TAR: {fairness_optimal['tar']:.6f}\n"
    )
    file.write(
        f"FMR: {fairness_optimal['fmr']:.6f}\n"
    )
    file.write(
        f"FNMR: {fairness_optimal['fnmr']:.6f}\n"
    )
    file.write(
        f"Mean disparity: "
        f"{fairness_optimal['mean_disparity']:.6f}\n\n"
    )

    file.write(
        "ACCURACY-OPTIMAL THRESHOLD\n"
    )
    file.write(
        f"Threshold: {accuracy_optimal['threshold']:.6f}\n"
    )
    file.write(
        f"Accuracy: {accuracy_optimal['accuracy']:.6f}\n"
    )
    file.write(
        f"Mean disparity: "
        f"{accuracy_optimal['mean_disparity']:.6f}\n\n"
    )

    file.write(
        "BALANCED FAIRNESS-PERFORMANCE THRESHOLD\n"
    )
    file.write(
        f"Threshold: {balanced_optimal['threshold']:.6f}\n"
    )
    file.write(
        f"Accuracy: {balanced_optimal['accuracy']:.6f}\n"
    )
    file.write(
        f"Mean disparity: "
        f"{balanced_optimal['mean_disparity']:.6f}\n"
    )
    file.write(
        f"Balanced score: "
        f"{balanced_optimal['balanced_score']:.6f}\n\n"
    )

    file.write(
        "FAIRNESS-CONSTRAINED ACCURACY-OPTIMAL THRESHOLD\n"
    )
    file.write(
        f"{fairness_constraint_status}\n"
    )
    file.write(
        f"Threshold: {fairness_constrained['threshold']:.6f}\n"
    )
    file.write(
        f"Accuracy: {fairness_constrained['accuracy']:.6f}\n"
    )
    file.write(
        f"Mean disparity: "
        f"{fairness_constrained['mean_disparity']:.6f}\n\n"
    )

    file.write(
        f"Pareto-optimal candidates: "
        f"{len(pareto_front)}\n"
    )


# ============================================================
# CONSOLE OUTPUT
# ============================================================

print()
print("=" * 70)
print("TRADE-OFF ANALYSIS COMPLETE")
print("=" * 70)

print()
print("1. FAIRNESS-OPTIMAL")
print(f"Threshold:      {fairness_optimal['threshold']:.6f}")
print(f"Accuracy:       {fairness_optimal['accuracy']:.6f}")
print(f"TAR:            {fairness_optimal['tar']:.6f}")
print(f"FMR:            {fairness_optimal['fmr']:.6f}")
print(f"FNMR:           {fairness_optimal['fnmr']:.6f}")
print(
    f"Mean disparity: "
    f"{fairness_optimal['mean_disparity']:.6f}"
)

print()
print("2. ACCURACY-OPTIMAL")
print(f"Threshold:      {accuracy_optimal['threshold']:.6f}")
print(f"Accuracy:       {accuracy_optimal['accuracy']:.6f}")
print(
    f"Mean disparity: "
    f"{accuracy_optimal['mean_disparity']:.6f}"
)

print()
print("3. BALANCED FAIRNESS-PERFORMANCE")
print(f"Threshold:      {balanced_optimal['threshold']:.6f}")
print(f"Accuracy:       {balanced_optimal['accuracy']:.6f}")
print(
    f"Mean disparity: "
    f"{balanced_optimal['mean_disparity']:.6f}"
)
print(
    f"Balanced score: "
    f"{balanced_optimal['balanced_score']:.6f}"
)

print()
print("4. FAIRNESS-CONSTRAINED ACCURACY-OPTIMAL")
print(fairness_constraint_status)
print(
    f"Threshold:      "
    f"{fairness_constrained['threshold']:.6f}"
)
print(
    f"Accuracy:       "
    f"{fairness_constrained['accuracy']:.6f}"
)
print(
    f"Mean disparity: "
    f"{fairness_constrained['mean_disparity']:.6f}"
)

print()
print(f"Pareto-optimal candidates: {len(pareto_front)}")

print()
print("=" * 70)
print("OUTPUT FILES")
print("=" * 70)
print(results_file)
print(operating_file)
print(pareto_file)
print(summary_file)
print("=" * 70)