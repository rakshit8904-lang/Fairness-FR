import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================================
# LOAD THRESHOLD FAIRNESS RESULTS
# =========================================================

file_path = Path(
    "results/bfw/arcface/threshold_fairness/threshold_fairness.csv"
)

if not file_path.exists():
    raise FileNotFoundError(
        f"File not found:\n{file_path.resolve()}"
    )

df = pd.read_csv(file_path)

print("\nTHRESHOLD FAIRNESS DATA")
print("=" * 80)
print(df.head())
print("\nColumns:")
print(df.columns.tolist())

# =========================================================
# CALCULATE MEAN DEMOGRAPHIC DISPARITY
# =========================================================

df["mean_disparity"] = (
    df["fmr_disparity"]
    + df["fnmr_disparity"]
    + df["tar_disparity"]
    + df["accuracy_disparity"]
) / 4

# =========================================================
# CALCULATE OVERALL AVERAGE PERFORMANCE
# =========================================================
# Average Female and Male accuracy at each threshold

df["overall_accuracy"] = (
    df["female_accuracy"] + df["male_accuracy"]
) / 2

# =========================================================
# FIND IMPORTANT OPERATING POINTS
# =========================================================

# Fairness-optimal: minimum demographic disparity
fairness_idx = df["mean_disparity"].idxmin()
fairness_point = df.loc[fairness_idx]

# Accuracy-optimal: maximum overall accuracy
accuracy_idx = df["overall_accuracy"].idxmax()
accuracy_point = df.loc[accuracy_idx]

# =========================================================
# PRINT RESULTS
# =========================================================

print("\n" + "=" * 80)
print("FAIRNESS-PERFORMANCE THRESHOLD ANALYSIS")
print("=" * 80)

print("\nFairness-Optimal Operating Point")
print(f"Threshold       : {fairness_point['threshold']:.6f}")
print(f"Accuracy        : {fairness_point['overall_accuracy']:.6f}")
print(f"Mean Disparity  : {fairness_point['mean_disparity']:.6f}")

print("\nAccuracy-Optimal Operating Point")
print(f"Threshold       : {accuracy_point['threshold']:.6f}")
print(f"Accuracy        : {accuracy_point['overall_accuracy']:.6f}")
print(f"Mean Disparity  : {accuracy_point['mean_disparity']:.6f}")

# =========================================================
# CREATE OUTPUT DIRECTORY
# =========================================================

output_dir = Path("all_graphs")
output_dir.mkdir(exist_ok=True)

# =========================================================
# GRAPH 1:
# FAIRNESS VS PERFORMANCE ACROSS THRESHOLDS
# =========================================================

plt.figure(figsize=(10, 6))

plt.plot(
    df["threshold"],
    df["overall_accuracy"],
    marker="o",
    markersize=3,
    linewidth=2,
    label="Overall Accuracy"
)

plt.plot(
    df["threshold"],
    df["mean_disparity"],
    marker="s",
    markersize=3,
    linewidth=2,
    label="Mean Demographic Disparity"
)

# Mark fairness-optimal threshold
plt.scatter(
    fairness_point["threshold"],
    fairness_point["overall_accuracy"],
    s=100,
    marker="o",
    label="Fairness-Optimal Threshold"
)

# Mark accuracy-optimal threshold
plt.scatter(
    accuracy_point["threshold"],
    accuracy_point["overall_accuracy"],
    s=100,
    marker="X",
    label="Accuracy-Optimal Threshold"
)

plt.xlabel("Verification Threshold")
plt.ylabel("Metric Value")
plt.title("Fairness-Performance Trade-off Across Thresholds")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

output_path = output_dir / "fairness_performance_tradeoff.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nGraph saved successfully:")
print(output_path.resolve())

# =========================================================
# SAVE PROCESSED RESULTS
# =========================================================

processed_csv = output_dir / "threshold_tradeoff_results.csv"

df.to_csv(processed_csv, index=False)

print("\nProcessed CSV saved:")
print(processed_csv.resolve())