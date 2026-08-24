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
    PROJECT_ROOT / "results" / DATASET / MODEL / "test_scores.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / DATASET
    / MODEL
    / "global_vs_group_threshold"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Global threshold obtained from fairness-performance analysis
GLOBAL_THRESHOLD = 0.154624

NUM_THRESHOLDS = 200


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(data, threshold):

    labels = data["label"].astype(int).to_numpy()
    scores = data["cosine_similarity"].astype(float).to_numpy()

    predictions = (scores >= threshold).astype(int)

    tp = np.sum((predictions == 1) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))

    genuine = tp + fn
    impostor = tn + fp
    total = len(labels)

    accuracy = (tp + tn) / total if total else np.nan
    tar = tp / genuine if genuine else np.nan
    fmr = fp / impostor if impostor else np.nan
    fnmr = fn / genuine if genuine else np.nan

    # Equal Error Rate approximation
    eer_gap = abs(fmr - fnmr)

    return {
        "threshold": threshold,
        "total_pairs": total,
        "accuracy": accuracy,
        "tar": tar,
        "fmr": fmr,
        "fnmr": fnmr,
        "eer_gap": eer_gap,
    }


# ============================================================
# FIND GROUP-SPECIFIC EER THRESHOLD
# ============================================================

def find_optimal_threshold(data):

    scores = data["cosine_similarity"].astype(float)

    thresholds = np.linspace(
        scores.min(),
        scores.max(),
        NUM_THRESHOLDS
    )

    candidates = []

    for threshold in thresholds:
        metrics = calculate_metrics(data, threshold)
        candidates.append(metrics)

    candidates_df = pd.DataFrame(candidates)

    # Primary objective: minimize |FMR - FNMR|
    # Secondary objective: maximize accuracy
    candidates_df = candidates_df.sort_values(
        by=["eer_gap", "accuracy"],
        ascending=[True, False]
    )

    best = candidates_df.iloc[0].to_dict()

    return best


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FAIRNESS-FR — GLOBAL VS GROUP-SPECIFIC THRESHOLD ANALYSIS")
print("=" * 70)

print(f"\nDataset: {DATASET.upper()}")
print(f"Model:   {MODEL.upper()}")
print(f"Input:   {INPUT_FILE}")
print(f"Global threshold: {GLOBAL_THRESHOLD:.6f}")

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found:\n{INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "demographic1",
    "demographic2",
    "gender1",
    "gender2",
    "label",
    "cosine_similarity",
]

missing = [
    column for column in required_columns
    if column not in df.columns
]

if missing:
    raise ValueError(f"Missing columns: {missing}")

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

df["group1"] = (
    df["demographic1"] + "_" + df["gender1"]
)

df["group2"] = (
    df["demographic2"] + "_" + df["gender2"]
)

# Only compare within the same intersectional group
within_group = df[df["group1"] == df["group2"]].copy()

groups = sorted(within_group["group1"].unique())

print(f"\nTotal valid pairs: {len(df)}")
print(f"Within-group pairs: {len(within_group)}")
print(f"Groups: {', '.join(groups)}")


# ============================================================
# ANALYZE EACH GROUP
# ============================================================

results = []

for group in groups:

    group_data = within_group[
        within_group["group1"] == group
    ].copy()

    print("\n" + "-" * 70)
    print(f"GROUP: {group}")
    print(f"Pairs: {len(group_data)}")

    # Global threshold performance
    global_result = calculate_metrics(
        group_data,
        GLOBAL_THRESHOLD
    )

    # Group-specific optimal threshold
    group_result = find_optimal_threshold(group_data)

    threshold_change = (
        group_result["threshold"]
        - GLOBAL_THRESHOLD
    )

    accuracy_change = (
        group_result["accuracy"]
        - global_result["accuracy"]
    )

    fmr_change = (
        group_result["fmr"]
        - global_result["fmr"]
    )

    fnmr_change = (
        group_result["fnmr"]
        - global_result["fnmr"]
    )

    print(f"\nGlobal threshold: {GLOBAL_THRESHOLD:.6f}")
    print(f"Group threshold:  {group_result['threshold']:.6f}")

    print("\nGLOBAL PERFORMANCE")
    print(f"Accuracy: {global_result['accuracy']:.6f}")
    print(f"TAR:      {global_result['tar']:.6f}")
    print(f"FMR:      {global_result['fmr']:.6f}")
    print(f"FNMR:     {global_result['fnmr']:.6f}")

    print("\nGROUP-SPECIFIC PERFORMANCE")
    print(f"Accuracy: {group_result['accuracy']:.6f}")
    print(f"TAR:      {group_result['tar']:.6f}")
    print(f"FMR:      {group_result['fmr']:.6f}")
    print(f"FNMR:     {group_result['fnmr']:.6f}")

    results.append({
        "group": group,
        "total_pairs": len(group_data),

        "global_threshold": GLOBAL_THRESHOLD,
        "global_accuracy": global_result["accuracy"],
        "global_tar": global_result["tar"],
        "global_fmr": global_result["fmr"],
        "global_fnmr": global_result["fnmr"],

        "group_threshold": group_result["threshold"],
        "group_accuracy": group_result["accuracy"],
        "group_tar": group_result["tar"],
        "group_fmr": group_result["fmr"],
        "group_fnmr": group_result["fnmr"],

        "threshold_change": threshold_change,
        "accuracy_change": accuracy_change,
        "fmr_change": fmr_change,
        "fnmr_change": fnmr_change,
    })


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("GLOBAL VS GROUP-SPECIFIC THRESHOLD SUMMARY")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# DISPARITY COMPARISON
# ============================================================

def disparity(column):
    return results_df[column].max() - results_df[column].min()


disparity_results = pd.DataFrame([
    {
        "metric": "accuracy",
        "global_threshold_disparity":
            disparity("global_accuracy"),
        "group_specific_disparity":
            disparity("group_accuracy"),
    },
    {
        "metric": "tar",
        "global_threshold_disparity":
            disparity("global_tar"),
        "group_specific_disparity":
            disparity("group_tar"),
    },
    {
        "metric": "fmr",
        "global_threshold_disparity":
            disparity("global_fmr"),
        "group_specific_disparity":
            disparity("group_fmr"),
    },
    {
        "metric": "fnmr",
        "global_threshold_disparity":
            disparity("global_fnmr"),
        "group_specific_disparity":
            disparity("group_fnmr"),
    },
])

disparity_results["disparity_change"] = (
    disparity_results["group_specific_disparity"]
    - disparity_results["global_threshold_disparity"]
)

print("\n" + "=" * 70)
print("FAIRNESS DISPARITY COMPARISON")
print("=" * 70)

print(
    disparity_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

results_file = (
    OUTPUT_DIR / "global_vs_group_threshold_metrics.csv"
)

disparity_file = (
    OUTPUT_DIR / "threshold_disparity_comparison.csv"
)

summary_file = (
    OUTPUT_DIR / "global_vs_group_threshold_summary.txt"
)

results_df.to_csv(results_file, index=False)
disparity_results.to_csv(disparity_file, index=False)

with open(summary_file, "w", encoding="utf-8") as f:

    f.write(
        "FAIRNESS-FR — GLOBAL VS GROUP-SPECIFIC THRESHOLD ANALYSIS\n"
    )
    f.write("=" * 70 + "\n\n")

    f.write(f"Dataset: {DATASET}\n")
    f.write(f"Model: {MODEL}\n")
    f.write(f"Global threshold: {GLOBAL_THRESHOLD:.6f}\n\n")

    f.write("GROUP RESULTS\n")
    f.write("-" * 70 + "\n")
    f.write(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    f.write("\n\nDISPARITY COMPARISON\n")
    f.write("-" * 70 + "\n")
    f.write(
        disparity_results.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print("\nOutput files:")
print(results_file)
print(disparity_file)
print(summary_file)