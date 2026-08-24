from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET = "bfw"
MODEL = "arcface"

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / DATASET
    / MODEL
    / "test_scores.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / DATASET
    / MODEL
    / "intersectional_fairness_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Use the fairness-optimal threshold found earlier
THRESHOLD = 0.154624


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_metrics(df):
    """Calculate verification performance metrics."""

    if len(df) == 0:
        return None

    labels = df["label"].astype(int).to_numpy()
    scores = df["cosine_similarity"].astype(float).to_numpy()

    # Higher cosine similarity means more likely genuine
    predictions = (scores >= THRESHOLD).astype(int)

    tp = np.sum((predictions == 1) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))

    total = len(labels)

    genuine = tp + fn
    impostor = tn + fp

    accuracy = (tp + tn) / total if total > 0 else np.nan
    tar = tp / genuine if genuine > 0 else np.nan
    fnmr = fn / genuine if genuine > 0 else np.nan
    fmr = fp / impostor if impostor > 0 else np.nan

    return {
        "total_pairs": total,
        "genuine_pairs": genuine,
        "impostor_pairs": impostor,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": accuracy,
        "tar": tar,
        "fmr": fmr,
        "fnmr": fnmr,
    }


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FAIRNESS-FR — INTERSECTIONAL FAIRNESS ANALYSIS")
print("=" * 70)

print(f"\nDataset:   {DATASET.upper()}")
print(f"Model:     {MODEL.upper()}")
print(f"Threshold: {THRESHOLD:.6f}")
print(f"Input:     {INPUT_FILE}")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nInput file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(f"\nTotal pairs loaded: {len(df)}")

required_columns = [
    "demographic1",
    "demographic2",
    "gender1",
    "gender2",
    "label",
    "cosine_similarity",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"\nMissing required columns: {missing_columns}"
    )


# ============================================================
# CLEAN DATA
# ============================================================

df = df.dropna(subset=required_columns).copy()

df["label"] = df["label"].astype(int)
df["cosine_similarity"] = df["cosine_similarity"].astype(float)

for column in [
    "demographic1",
    "demographic2",
    "gender1",
    "gender2",
]:
    df[column] = (
        df[column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

print(f"Valid pairs after cleaning: {len(df)}")


# ============================================================
# CREATE INTERSECTIONAL GROUP
# ============================================================

df["group1"] = (
    df["demographic1"]
    + "_"
    + df["gender1"]
)

df["group2"] = (
    df["demographic2"]
    + "_"
    + df["gender2"]
)


# ============================================================
# SELECT WITHIN-GROUP PAIRS
# ============================================================

within_group = df[
    df["group1"] == df["group2"]
].copy()

print(f"\nWithin-intersection pairs: {len(within_group)}")

if len(within_group) == 0:
    raise ValueError(
        "\nNo within-intersection pairs found. "
        "Check demographic and gender labels."
    )

print("\nIntersectional groups found:")

groups = sorted(within_group["group1"].unique())

for group in groups:
    count = len(
        within_group[
            within_group["group1"] == group
        ]
    )
    print(f"  {group}: {count} pairs")


# ============================================================
# CALCULATE GROUP-WISE METRICS
# ============================================================

results = []

for group in groups:

    group_df = within_group[
        within_group["group1"] == group
    ].copy()

    metrics = calculate_metrics(group_df)

    if metrics is None:
        continue

    demographic, gender = group.rsplit("_", 1)

    results.append({
        "intersectional_group": group,
        "demographic": demographic,
        "gender": gender,
        **metrics,
    })


results_df = pd.DataFrame(results)

if results_df.empty:
    raise ValueError(
        "No valid intersectional metrics could be calculated."
    )


# ============================================================
# SORT RESULTS
# ============================================================

results_df = results_df.sort_values(
    by=["accuracy", "intersectional_group"],
    ascending=[False, True],
).reset_index(drop=True)


# ============================================================
# CALCULATE DISPARITIES
# ============================================================

metric_columns = [
    "accuracy",
    "tar",
    "fmr",
    "fnmr",
]

disparities = []

for metric in metric_columns:

    valid_values = results_df[metric].dropna()

    if len(valid_values) == 0:
        continue

    max_value = valid_values.max()
    min_value = valid_values.min()

    max_group = results_df.loc[
        results_df[metric].idxmax(),
        "intersectional_group"
    ]

    min_group = results_df.loc[
        results_df[metric].idxmin(),
        "intersectional_group"
    ]

    disparities.append({
        "metric": metric,
        "maximum_value": max_value,
        "maximum_group": max_group,
        "minimum_value": min_value,
        "minimum_group": min_group,
        "disparity": max_value - min_value,
    })

disparity_df = pd.DataFrame(disparities)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("INTERSECTIONAL PERFORMANCE RESULTS")
print("=" * 70)

display_columns = [
    "intersectional_group",
    "total_pairs",
    "genuine_pairs",
    "impostor_pairs",
    "accuracy",
    "tar",
    "fmr",
    "fnmr",
]

print(
    results_df[display_columns].to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}"
    )
)


print("\n" + "=" * 70)
print("INTERSECTIONAL FAIRNESS DISPARITIES")
print("=" * 70)

print(
    disparity_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}"
    )
)


# ============================================================
# OVERALL SUMMARY
# ============================================================

mean_disparity = disparity_df["disparity"].mean()

max_disparity_row = disparity_df.loc[
    disparity_df["disparity"].idxmax()
]

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"Intersectional groups evaluated: {len(results_df)}")
print(f"Operating threshold:             {THRESHOLD:.6f}")
print(f"Mean metric disparity:           {mean_disparity:.6f}")
print(
    f"Maximum disparity:               "
    f"{max_disparity_row['disparity']:.6f}"
)
print(
    f"Most disparate metric:           "
    f"{max_disparity_row['metric']}"
)
print(
    f"Highest group:                   "
    f"{max_disparity_row['maximum_group']}"
)
print(
    f"Lowest group:                    "
    f"{max_disparity_row['minimum_group']}"
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

results_file = (
    OUTPUT_DIR
    / "intersectional_metrics.csv"
)

disparity_file = (
    OUTPUT_DIR
    / "intersectional_disparities.csv"
)

summary_file = (
    OUTPUT_DIR
    / "intersectional_summary.txt"
)

results_df.to_csv(
    results_file,
    index=False
)

disparity_df.to_csv(
    disparity_file,
    index=False
)

with open(summary_file, "w", encoding="utf-8") as file:

    file.write(
        "FAIRNESS-FR — INTERSECTIONAL FAIRNESS ANALYSIS\n"
    )
    file.write("=" * 70 + "\n\n")

    file.write(f"Dataset: {DATASET}\n")
    file.write(f"Model: {MODEL}\n")
    file.write(f"Threshold: {THRESHOLD:.6f}\n")
    file.write(
        f"Intersectional groups: {len(results_df)}\n\n"
    )

    file.write("GROUP METRICS\n")
    file.write("-" * 70 + "\n")
    file.write(
        results_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}"
        )
    )

    file.write("\n\nDISPARITIES\n")
    file.write("-" * 70 + "\n")
    file.write(
        disparity_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}"
        )
    )

    file.write("\n\nSUMMARY\n")
    file.write("-" * 70 + "\n")
    file.write(
        f"Mean metric disparity: {mean_disparity:.6f}\n"
    )
    file.write(
        f"Maximum disparity: "
        f"{max_disparity_row['disparity']:.6f}\n"
    )
    file.write(
        f"Most disparate metric: "
        f"{max_disparity_row['metric']}\n"
    )


print("\n" + "=" * 70)
print("INTERSECTIONAL FAIRNESS ANALYSIS COMPLETE")
print("=" * 70)

print("\nOutput files:")
print(results_file)
print(disparity_file)
print(summary_file)
print("=" * 70)