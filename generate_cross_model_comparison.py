import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

RESULTS_DIR = os.path.join(BASE_DIR, "results", "bfw")
OUTPUT_DIR = os.path.join(BASE_DIR, "all_graphs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Only models with confirmed test_scores.csv files
MODELS = ["arcface", "facenet", "ghostfacenet", "sface"]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(df, possible_names):
    """Find a column ignoring case and small naming differences."""
    normalized = {
        col.lower().replace(" ", "").replace("_", "").replace("-", ""): col
        for col in df.columns
    }

    for name in possible_names:
        key = name.lower().replace(" ", "").replace("_", "").replace("-", "")
        if key in normalized:
            return normalized[key]

    return None


def compute_metrics(df):
    """
    Compute global verification metrics and gender fairness metrics.
    Expected test_scores.csv columns include:
    score, label, and a demographic/group column.
    """

    score_col = find_column(
        df,
        ["score", "similarity", "cosine_similarity"]
    )

    label_col = find_column(
        df,
        ["label", "target", "same_identity", "is_same"]
    )

    if score_col is None or label_col is None:
        raise ValueError(
            f"Could not find score/label columns. "
            f"Available columns: {list(df.columns)}"
        )

    scores = pd.to_numeric(df[score_col], errors="coerce")
    labels = pd.to_numeric(df[label_col], errors="coerce")

    valid = scores.notna() & labels.notna()
    scores = scores[valid]
    labels = labels[valid].astype(int)

    # --------------------------------------------------------
    # Select accuracy-optimal threshold
    # --------------------------------------------------------
    thresholds = np.linspace(scores.min(), scores.max(), 500)

    best_accuracy = -1
    best_threshold = None
    best_metrics = None

    for threshold in thresholds:

        predictions = (scores >= threshold).astype(int)

        tp = ((predictions == 1) & (labels == 1)).sum()
        tn = ((predictions == 0) & (labels == 0)).sum()
        fp = ((predictions == 1) & (labels == 0)).sum()
        fn = ((predictions == 0) & (labels == 1)).sum()

        accuracy = (tp + tn) / len(labels)

        tar = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fmr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnmr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
            best_metrics = {
                "Accuracy": accuracy,
                "TAR": tar,
                "FMR": fmr,
                "FNMR": fnmr
            }

    # --------------------------------------------------------
    # Try to locate demographic group column
    # --------------------------------------------------------
    group_col = find_column(
        df,
        [
            "group",
            "gender_group",
            "demographic_group",
            "pair_group",
            "gender"
        ]
    )

    mean_disparity = np.nan

    if group_col is not None:

        predictions = (scores >= best_threshold).astype(int)

        temp = pd.DataFrame({
            "label": labels.values,
            "prediction": predictions.values,
            "group": df.loc[valid, group_col].values
        })

        group_metrics = []

        for group_name, group_df in temp.groupby("group"):

            y_true = group_df["label"]
            y_pred = group_df["prediction"]

            tp = ((y_pred == 1) & (y_true == 1)).sum()
            tn = ((y_pred == 0) & (y_true == 0)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()

            accuracy = (tp + tn) / len(group_df)
            tar = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fmr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnmr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

            group_metrics.append({
                "Accuracy": accuracy,
                "TAR": tar,
                "FMR": fmr,
                "FNMR": fnmr
            })

        # Compute max-min disparity across groups
        if len(group_metrics) >= 2:

            disparities = []

            for metric in ["Accuracy", "TAR", "FMR", "FNMR"]:

                values = [x[metric] for x in group_metrics]
                disparities.append(max(values) - min(values))

            mean_disparity = np.mean(disparities)

    return {
        "Threshold": best_threshold,
        "Accuracy": best_metrics["Accuracy"],
        "TAR": best_metrics["TAR"],
        "FMR": best_metrics["FMR"],
        "FNMR": best_metrics["FNMR"],
        "Mean_Disparity": mean_disparity
    }


# ============================================================
# PROCESS ALL MODELS
# ============================================================

all_results = []

print("=" * 70)
print("FAIRFACEEVAL - CROSS-MODEL COMPARISON")
print("=" * 70)

for model in MODELS:

    pattern = os.path.join(
        RESULTS_DIR,
        model,
        "**",
        "test_scores.csv"
    )

    matches = glob.glob(pattern, recursive=True)

    if not matches:
        print(f"\n[SKIP] {model}: test_scores.csv not found")
        continue

    csv_path = matches[0]

    print(f"\nProcessing: {model}")
    print(f"File: {csv_path}")

    try:
        df = pd.read_csv(csv_path)

        print("Columns:", list(df.columns))

        metrics = compute_metrics(df)
        metrics["Model"] = model

        all_results.append(metrics)

        print(
            f"Accuracy={metrics['Accuracy']:.4f}, "
            f"TAR={metrics['TAR']:.4f}, "
            f"FMR={metrics['FMR']:.4f}, "
            f"FNMR={metrics['FNMR']:.4f}, "
            f"Mean Disparity={metrics['Mean_Disparity']:.6f}"
            if not np.isnan(metrics["Mean_Disparity"])
            else
            f"Accuracy={metrics['Accuracy']:.4f}, "
            f"TAR={metrics['TAR']:.4f}, "
            f"FMR={metrics['FMR']:.4f}, "
            f"FNMR={metrics['FNMR']:.4f}, "
            f"Mean Disparity=N/A"
        )

    except Exception as e:
        print(f"[ERROR] {model}: {e}")


# ============================================================
# SAVE RESULTS
# ============================================================

if len(all_results) < 2:
    raise RuntimeError(
        "Fewer than two models were successfully processed."
    )

results_df = pd.DataFrame(all_results)

column_order = [
    "Model",
    "Threshold",
    "Accuracy",
    "TAR",
    "FMR",
    "FNMR",
    "Mean_Disparity"
]

results_df = results_df[column_order]

csv_output = os.path.join(
    OUTPUT_DIR,
    "cross_model_comparison.csv"
)

results_df.to_csv(csv_output, index=False)

print("\n" + "=" * 70)
print("CROSS-MODEL RESULTS")
print("=" * 70)
print(results_df.to_string(index=False))


# ============================================================
# GRAPH 1: ACCURACY VS FAIRNESS
# ============================================================

plt.figure(figsize=(9, 6))

valid_fairness = results_df.dropna(subset=["Mean_Disparity"])

plt.scatter(
    valid_fairness["Mean_Disparity"],
    valid_fairness["Accuracy"],
    s=120
)

for _, row in valid_fairness.iterrows():
    plt.annotate(
        row["Model"],
        (row["Mean_Disparity"], row["Accuracy"]),
        xytext=(5, 5),
        textcoords="offset points"
    )

plt.xlabel("Mean Demographic Disparity (Lower is Better)")
plt.ylabel("Verification Accuracy (Higher is Better)")
plt.title("Cross-Model Fairness vs Recognition Performance")
plt.grid(True, alpha=0.3)
plt.tight_layout()

output_1 = os.path.join(
    OUTPUT_DIR,
    "overall_ranking.png"
)

plt.savefig(output_1, dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# GRAPH 2: METRIC HEATMAP
# ============================================================

heatmap_cols = [
    "Accuracy",
    "TAR",
    "FMR",
    "FNMR",
    "Mean_Disparity"
]

heatmap_data = results_df.set_index("Model")[heatmap_cols]

# Normalize each column for visual comparison only
normalized = heatmap_data.copy()

for col in normalized.columns:

    values = normalized[col]

    if values.notna().sum() > 0:

        min_val = values.min()
        max_val = values.max()

        if max_val > min_val:
            normalized[col] = (values - min_val) / (max_val - min_val)
        else:
            normalized[col] = 0.5

plt.figure(figsize=(10, 5))

image = plt.imshow(
    normalized.fillna(0).values,
    aspect="auto"
)

plt.colorbar(image, label="Normalized Metric Value")

plt.xticks(
    range(len(normalized.columns)),
    normalized.columns,
    rotation=30,
    ha="right"
)

plt.yticks(
    range(len(normalized.index)),
    normalized.index
)

for i in range(len(heatmap_data.index)):
    for j in range(len(heatmap_data.columns)):

        value = heatmap_data.iloc[i, j]

        if pd.notna(value):
            plt.text(
                j,
                i,
                f"{value:.3f}",
                ha="center",
                va="center"
            )

plt.title("Cross-Model Comparison of Recognition and Fairness Metrics")
plt.tight_layout()

output_2 = os.path.join(
    OUTPUT_DIR,
    "metric_heatmap.png"
)

plt.savefig(output_2, dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# GRAPH 3: PERFORMANCE METRICS BAR CHART
# ============================================================

plot_cols = ["Accuracy", "TAR"]

x = np.arange(len(results_df))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    x - width / 2,
    results_df["Accuracy"],
    width,
    label="Accuracy"
)

plt.bar(
    x + width / 2,
    results_df["TAR"],
    width,
    label="TAR"
)

plt.xticks(x, results_df["Model"])
plt.ylabel("Score")
plt.ylim(0, 1)
plt.title("Cross-Model Recognition Performance")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

output_3 = os.path.join(
    OUTPUT_DIR,
    "cross_model_performance.png"
)

plt.savefig(output_3, dpi=300, bbox_inches="tight")
plt.close()


print("\n" + "=" * 70)
print("GRAPHS GENERATED SUCCESSFULLY")
print("=" * 70)
print(f"CSV: {csv_output}")
print(f"Graph 1: {output_1}")
print(f"Graph 2: {output_2}")
print(f"Graph 3: {output_3}")
print("=" * 70)