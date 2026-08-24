import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ---------------------------------------------------------
# Load cross-model comparison results
# ---------------------------------------------------------
file_path = Path("all_graphs/cross_model_comparison.csv")
df = pd.read_csv(file_path)

print("\nOriginal Results:")
print(df)

# ---------------------------------------------------------
# Keep models with valid recognition metrics
# ---------------------------------------------------------
metrics = ["Accuracy", "TAR", "FMR", "FNMR"]

df = df.dropna(subset=metrics).copy()

# ---------------------------------------------------------
# Normalize metrics between 0 and 1
# Higher Accuracy and TAR are better
# Lower FMR and FNMR are better
# ---------------------------------------------------------
def normalize_higher(values):
    min_val = values.min()
    max_val = values.max()

    if max_val == min_val:
        return pd.Series(np.ones(len(values)), index=values.index)

    return (values - min_val) / (max_val - min_val)


def normalize_lower(values):
    min_val = values.min()
    max_val = values.max()

    if max_val == min_val:
        return pd.Series(np.ones(len(values)), index=values.index)

    return (max_val - values) / (max_val - min_val)


df["Accuracy_Score"] = normalize_higher(df["Accuracy"])
df["TAR_Score"] = normalize_higher(df["TAR"])
df["FMR_Score"] = normalize_lower(df["FMR"])
df["FNMR_Score"] = normalize_lower(df["FNMR"])

# ---------------------------------------------------------
# Overall recognition score
# Equal contribution from all four metrics
# ---------------------------------------------------------
df["Recognition_Score"] = (
    df["Accuracy_Score"]
    + df["TAR_Score"]
    + df["FMR_Score"]
    + df["FNMR_Score"]
) / 4

# Sort from best to worst
ranking = df.sort_values(
    "Recognition_Score",
    ascending=False
).reset_index(drop=True)

ranking["Rank"] = ranking.index + 1

# ---------------------------------------------------------
# Print final ranking
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("CROSS-MODEL RECOGNITION RANKING")
print("=" * 70)

print(
    ranking[
        [
            "Rank",
            "Model",
            "Accuracy",
            "TAR",
            "FMR",
            "FNMR",
            "Recognition_Score"
        ]
    ].to_string(index=False)
)

# ---------------------------------------------------------
# Create graph
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

bars = plt.bar(
    ranking["Model"],
    ranking["Recognition_Score"]
)

plt.title("Cross-Model Overall Recognition Ranking")
plt.xlabel("Face Recognition Model")
plt.ylabel("Composite Recognition Score")
plt.ylim(0, 1.05)
plt.grid(axis="y", alpha=0.3)

# Add rank and score above bars
for bar, rank, score in zip(
    bars,
    ranking["Rank"],
    ranking["Recognition_Score"]
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"Rank {rank}\n{score:.3f}",
        ha="center",
        va="bottom",
        fontsize=11
    )

plt.tight_layout()

# Save graph
output_path = Path("all_graphs/cross_model_ranking.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")

plt.show()

print("\nGraph saved successfully:")
print(output_path)