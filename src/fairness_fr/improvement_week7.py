from pathlib import Path
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RESULTS = ROOT / "results" / "bfw" / "arcface"

VALIDATION_FILE = RESULTS / "validation_scores.csv"
TEST_FILE = RESULTS / "test_scores.csv"

OUT_DIR = RESULTS / "improvement"
PLOT_DIR = RESULTS / "plots" / "improvement"

OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(df, threshold):
    """
    Calculate verification metrics using cosine similarity.

    label = 1 -> genuine
    label = 0 -> impostor
    similarity >= threshold -> accept
    """

    y_true = df["label"].astype(int).to_numpy()
    scores = df["cosine_similarity"].astype(float).to_numpy()

    predicted = (scores >= threshold).astype(int)

    genuine = y_true == 1
    impostor = y_true == 0

    genuine_count = genuine.sum()
    impostor_count = impostor.sum()

    false_rejects = np.sum(genuine & (predicted == 0))
    false_accepts = np.sum(impostor & (predicted == 1))

    correct = np.sum(predicted == y_true)

    fnmr = (
        false_rejects / genuine_count
        if genuine_count > 0 else np.nan
    )

    fmr = (
        false_accepts / impostor_count
        if impostor_count > 0 else np.nan
    )

    tar = 1.0 - fnmr
    tnr = 1.0 - fmr
    accuracy = correct / len(df) if len(df) else np.nan

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "fmr": float(fmr),
        "fnmr": float(fnmr),
        "tar": float(tar),
        "tnr": float(tnr),
        "total_pairs": int(len(df)),
        "genuine_pairs": int(genuine_count),
        "impostor_pairs": int(impostor_count),
    }


# ============================================================
# EER THRESHOLD
# ============================================================

def find_eer_threshold(df):
    """
    Find the threshold where FMR and FNMR are closest.
    """

    scores = np.sort(
        df["cosine_similarity"].astype(float).unique()
    )

    best_threshold = None
    best_difference = float("inf")

    for threshold in scores:

        metrics = calculate_metrics(df, threshold)

        difference = abs(
            metrics["fmr"] - metrics["fnmr"]
        )

        if difference < best_difference:
            best_difference = difference
            best_threshold = threshold

    return float(best_threshold)


# ============================================================
# LOAD DATA
# ============================================================

validation = pd.read_csv(VALIDATION_FILE)
test = pd.read_csv(TEST_FILE)

print("Validation pairs:", len(validation))
print("Test pairs:", len(test))


# ============================================================
# GLOBAL THRESHOLD
# ============================================================

global_threshold = find_eer_threshold(validation)

print()
print("Global validation EER threshold:", global_threshold)


baseline = calculate_metrics(
    test,
    global_threshold
)

baseline["evaluation"] = "global_threshold"


# ============================================================
# GENDER-SPECIFIC THRESHOLDS
# ============================================================

gender_thresholds = {}

for gender in sorted(
    validation["gender1"].dropna().unique()
):

    group_validation = validation[
        validation["gender1"] == gender
    ].copy()

    if group_validation["label"].nunique() < 2:
        print(
            f"Skipping {gender}: "
            "does not contain both genuine and impostor pairs."
        )
        continue

    threshold = find_eer_threshold(
        group_validation
    )

    gender_thresholds[gender] = threshold

    print(
        f"Gender threshold - {gender}: "
        f"{threshold:.6f}"
    )


# ============================================================
# APPLY GENDER-SPECIFIC THRESHOLDS
# ============================================================

after_rows = []

for gender in sorted(
    test["gender1"].dropna().unique()
):

    group_test = test[
        test["gender1"] == gender
    ].copy()

    if gender not in gender_thresholds:
        continue

    threshold = gender_thresholds[gender]

    metrics = calculate_metrics(
        group_test,
        threshold
    )

    metrics["gender"] = gender
    metrics["evaluation"] = "gender_specific_threshold"

    after_rows.append(metrics)


after_df = pd.DataFrame(after_rows)


# ============================================================
# OVERALL GENDER-SPECIFIC RESULT
# ============================================================

threshold_map = test["gender1"].map(
    gender_thresholds
)

valid_mask = threshold_map.notna()

adjusted_test = test.loc[
    valid_mask
].copy()

adjusted_test["threshold"] = threshold_map[
    valid_mask
].astype(float)

adjusted_test["prediction"] = (
    adjusted_test["cosine_similarity"]
    >= adjusted_test["threshold"]
).astype(int)

y = adjusted_test["label"].astype(int)

accuracy_after = (
    adjusted_test["prediction"] == y
).mean()

genuine = adjusted_test["label"] == 1
impostor = adjusted_test["label"] == 0

false_rejects = (
    genuine &
    (adjusted_test["prediction"] == 0)
).sum()

false_accepts = (
    impostor &
    (adjusted_test["prediction"] == 1)
).sum()

genuine_count = genuine.sum()
impostor_count = impostor.sum()

fnmr_after = (
    false_rejects / genuine_count
)

fmr_after = (
    false_accepts / impostor_count
)

overall_after = {
    "evaluation": "gender_specific_threshold",
    "threshold": np.nan,
    "accuracy": float(accuracy_after),
    "fmr": float(fmr_after),
    "fnmr": float(fnmr_after),
    "tar": float(1 - fnmr_after),
    "tnr": float(1 - fmr_after),
    "total_pairs": int(len(adjusted_test)),
    "genuine_pairs": int(genuine_count),
    "impostor_pairs": int(impostor_count),
}


# ============================================================
# BEFORE / AFTER TABLE
# ============================================================

comparison = pd.DataFrame([
    baseline,
    overall_after
])

comparison.to_csv(
    OUT_DIR / "before_after_overall.csv",
    index=False
)


# ============================================================
# GROUP-WISE BEFORE / AFTER
# ============================================================

group_comparison = []

for gender in sorted(
    test["gender1"].dropna().unique()
):

    group_test = test[
        test["gender1"] == gender
    ].copy()

    # Before
    before = calculate_metrics(
        group_test,
        global_threshold
    )

    # After
    if gender in gender_thresholds:

        after = calculate_metrics(
            group_test,
            gender_thresholds[gender]
        )

        group_comparison.append({
            "gender": gender,

            "global_threshold":
                global_threshold,

            "gender_threshold":
                gender_thresholds[gender],

            "before_accuracy":
                before["accuracy"],

            "after_accuracy":
                after["accuracy"],

            "before_fmr":
                before["fmr"],

            "after_fmr":
                after["fmr"],

            "before_fnmr":
                before["fnmr"],

            "after_fnmr":
                after["fnmr"],

            "before_tar":
                before["tar"],

            "after_tar":
                after["tar"],
        })


group_df = pd.DataFrame(
    group_comparison
)

group_df.to_csv(
    OUT_DIR / "before_after_gender.csv",
    index=False
)


# ============================================================
# FAIRNESS GAPS
# ============================================================

before_fmr_gap = (
    group_df["before_fmr"].max()
    - group_df["before_fmr"].min()
)

after_fmr_gap = (
    group_df["after_fmr"].max()
    - group_df["after_fmr"].min()
)

before_fnmr_gap = (
    group_df["before_fnmr"].max()
    - group_df["before_fnmr"].min()
)

after_fnmr_gap = (
    group_df["after_fnmr"].max()
    - group_df["after_fnmr"].min()
)

before_accuracy_gap = (
    group_df["before_accuracy"].max()
    - group_df["before_accuracy"].min()
)

after_accuracy_gap = (
    group_df["after_accuracy"].max()
    - group_df["after_accuracy"].min()
)


fairness_comparison = pd.DataFrame([
    {
        "metric": "FMR gap",
        "before": before_fmr_gap,
        "after": after_fmr_gap,
        "change": after_fmr_gap - before_fmr_gap,
    },
    {
        "metric": "FNMR gap",
        "before": before_fnmr_gap,
        "after": after_fnmr_gap,
        "change": after_fnmr_gap - before_fnmr_gap,
    },
    {
        "metric": "Accuracy gap",
        "before": before_accuracy_gap,
        "after": after_accuracy_gap,
        "change": after_accuracy_gap - before_accuracy_gap,
    },
])

fairness_comparison.to_csv(
    OUT_DIR / "fairness_before_after.csv",
    index=False
)


# ============================================================
# BALANCED SUBSET EVALUATION
# ============================================================

# Balance both gender and class.
# The smallest gender/class cell determines the sample size.

cell_counts = (
    test
    .groupby(["gender1", "label"])
    .size()
)

balanced_n = int(
    cell_counts.min()
)

print()
print(
    "Balanced subset per gender/class:",
    balanced_n
)

balanced_parts = []

for gender in sorted(
    test["gender1"].dropna().unique()
):

    for label in [0, 1]:

        subset = test[
            (test["gender1"] == gender)
            &
            (test["label"] == label)
        ].copy()

        subset = subset.sample(
            n=balanced_n,
            random_state=42
        )

        balanced_parts.append(subset)


balanced_test = pd.concat(
    balanced_parts,
    ignore_index=True
)

balanced_test.to_csv(
    OUT_DIR / "balanced_test_subset.csv",
    index=False
)


balanced_before = calculate_metrics(
    balanced_test,
    global_threshold
)

balanced_test["threshold"] = (
    balanced_test["gender1"]
    .map(gender_thresholds)
)

balanced_test["prediction"] = (
    balanced_test["cosine_similarity"]
    >= balanced_test["threshold"]
).astype(int)

balanced_after_accuracy = (
    balanced_test["prediction"]
    == balanced_test["label"]
).mean()

balanced_genuine = (
    balanced_test["label"] == 1
)

balanced_impostor = (
    balanced_test["label"] == 0
)

balanced_fnmr = (
    (
        balanced_genuine
        &
        (balanced_test["prediction"] == 0)
    ).sum()
    /
    balanced_genuine.sum()
)

balanced_fmr = (
    (
        balanced_impostor
        &
        (balanced_test["prediction"] == 1)
    ).sum()
    /
    balanced_impostor.sum()
)

balanced_after = {
    "evaluation": "balanced_gender_specific",
    "accuracy": float(
        balanced_after_accuracy
    ),
    "fmr": float(
        balanced_fmr
    ),
    "fnmr": float(
        balanced_fnmr
    ),
    "tar": float(
        1 - balanced_fnmr
    ),
    "tnr": float(
        1 - balanced_fmr
    ),
    "total_pairs": len(
        balanced_test
    ),
}

balanced_comparison = pd.DataFrame([
    {
        "evaluation": "balanced_before",
        **{
            k: v for k, v in
            balanced_before.items()
            if k != "evaluation"
        }
    },
    balanced_after
])

balanced_comparison.to_csv(
    OUT_DIR / "balanced_before_after.csv",
    index=False
)


# ============================================================
# PLOTS
# ============================================================

# FMR
plt.figure(figsize=(8, 5))

x = np.arange(len(group_df))
width = 0.35

plt.bar(
    x - width / 2,
    group_df["before_fmr"],
    width,
    label="Before"
)

plt.bar(
    x + width / 2,
    group_df["after_fmr"],
    width,
    label="After"
)

plt.xticks(
    x,
    group_df["gender"]
)

plt.ylabel("FMR")
plt.xlabel("Gender")
plt.title("Gender-wise FMR Before vs After Thresholding")
plt.legend()
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "fmr_before_after.png",
    dpi=200
)

plt.close()


# FNMR
plt.figure(figsize=(8, 5))

plt.bar(
    x - width / 2,
    group_df["before_fnmr"],
    width,
    label="Before"
)

plt.bar(
    x + width / 2,
    group_df["after_fnmr"],
    width,
    label="After"
)

plt.xticks(
    x,
    group_df["gender"]
)

plt.ylabel("FNMR")
plt.xlabel("Gender")
plt.title("Gender-wise FNMR Before vs After Thresholding")
plt.legend()
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "fnmr_before_after.png",
    dpi=200
)

plt.close()


# Accuracy
plt.figure(figsize=(8, 5))

plt.bar(
    x - width / 2,
    group_df["before_accuracy"],
    width,
    label="Before"
)

plt.bar(
    x + width / 2,
    group_df["after_accuracy"],
    width,
    label="After"
)

plt.xticks(
    x,
    group_df["gender"]
)

plt.ylabel("Accuracy")
plt.xlabel("Gender")
plt.title("Gender-wise Accuracy Before vs After Thresholding")
plt.legend()
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "accuracy_before_after.png",
    dpi=200
)

plt.close()


# ============================================================
# SUMMARY JSON
# ============================================================

summary = {
    "global_validation_threshold":
        global_threshold,

    "gender_specific_thresholds":
        gender_thresholds,

    "overall_before":
        baseline,

    "overall_after":
        overall_after,

    "fairness_gaps": {
        "before_fmr_gap":
            float(before_fmr_gap),

        "after_fmr_gap":
            float(after_fmr_gap),

        "before_fnmr_gap":
            float(before_fnmr_gap),

        "after_fnmr_gap":
            float(after_fnmr_gap),

        "before_accuracy_gap":
            float(before_accuracy_gap),

        "after_accuracy_gap":
            float(after_accuracy_gap),
    },

    "balanced_subset_size":
        int(len(balanced_test)),

    "balanced_subset_per_gender_label":
        balanced_n,

    "note":
        "Ethnicity-specific thresholding was not "
        "performed because the available BFW test "
        "split contains only the asian ethnicity. "
        "Gender-specific thresholding was evaluated "
        "because both females and males are present."
}

with open(
    OUT_DIR / "improvement_summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


print()
print("=" * 70)
print("WEEK 7 IMPROVEMENT ANALYSIS COMPLETE")
print("=" * 70)

print()
print("Global threshold:", global_threshold)

print()
print("Gender thresholds:")

for gender, threshold in gender_thresholds.items():

    print(
        f"  {gender}: {threshold:.6f}"
    )

print()
print(
    "Before FMR gap:",
    before_fmr_gap
)

print(
    "After FMR gap:",
    after_fmr_gap
)

print()
print(
    "Before FNMR gap:",
    before_fnmr_gap
)

print(
    "After FNMR gap:",
    after_fnmr_gap
)

print()
print(
    "Before accuracy:",
    baseline["accuracy"]
)

print(
    "After accuracy:",
    overall_after["accuracy"]
)

print()
print(
    "Balanced subset:",
    len(balanced_test),
    "pairs"
)

print()
print("Outputs written to:")
print(OUT_DIR)