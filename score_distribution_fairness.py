import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("FAIRNESS-FR — SCORE DISTRIBUTION & DEMOGRAPHIC OVERLAP ANALYSIS")
print("=" * 70)

# --------------------------------------------------
# PATHS
# --------------------------------------------------

INPUT_FILE = r"results\bfw\arcface\test_scores.csv"
OUTPUT_DIR = r"results\bfw\arcface\score_distribution_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print(f"\nInput: {os.path.abspath(INPUT_FILE)}")
print(f"Total pairs loaded: {len(df)}")

required_columns = [
    "demographic1",
    "demographic2",
    "label",
    "cosine_similarity"
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

# --------------------------------------------------
# CLEAN DATA
# --------------------------------------------------

df = df.dropna(
    subset=[
        "demographic1",
        "demographic2",
        "label",
        "cosine_similarity"
    ]
).copy()

df["label"] = pd.to_numeric(df["label"], errors="coerce")
df["cosine_similarity"] = pd.to_numeric(
    df["cosine_similarity"],
    errors="coerce"
)

df = df.dropna(subset=["label", "cosine_similarity"])

df["label"] = df["label"].astype(int)

print(f"Valid pairs after cleaning: {len(df)}")

# --------------------------------------------------
# CREATE GROUP COLUMN
# --------------------------------------------------

# For within-demographic pairs, both demographics are same
df["group"] = df["demographic1"].astype(str)

within_df = df[
    df["demographic1"] == df["demographic2"]
].copy()

print(f"Within-demographic pairs: {len(within_df)}")

groups = sorted(within_df["group"].unique())

print("\nGroups found:")
for group in groups:
    count = len(within_df[within_df["group"] == group])
    print(f"  {group}: {count} pairs")

# --------------------------------------------------
# ANALYZE EACH GROUP
# --------------------------------------------------

results = []

print("\n" + "=" * 70)
print("GROUP-WISE SCORE STATISTICS")
print("=" * 70)

for group in groups:

    group_df = within_df[
        within_df["group"] == group
    ]

    genuine = group_df[
        group_df["label"] == 1
    ]["cosine_similarity"]

    impostor = group_df[
        group_df["label"] == 0
    ]["cosine_similarity"]

    if len(genuine) == 0 or len(impostor) == 0:
        print(f"\nSkipping {group}: insufficient genuine/impostor pairs")
        continue

    genuine_mean = genuine.mean()
    genuine_std = genuine.std()

    impostor_mean = impostor.mean()
    impostor_std = impostor.std()

    separation = genuine_mean - impostor_mean

    # Cohen-like normalized separation
    pooled_std = np.sqrt(
        (
            genuine_std ** 2 +
            impostor_std ** 2
        ) / 2
    )

    if pooled_std > 0:
        normalized_separation = separation / pooled_std
    else:
        normalized_separation = np.nan

    print(f"\nGROUP: {group}")
    print("-" * 50)
    print(f"Total pairs:              {len(group_df)}")
    print(f"Genuine pairs:            {len(genuine)}")
    print(f"Impostor pairs:           {len(impostor)}")
    print()
    print(f"Genuine mean score:       {genuine_mean:.6f}")
    print(f"Genuine std:              {genuine_std:.6f}")
    print(f"Impostor mean score:      {impostor_mean:.6f}")
    print(f"Impostor std:             {impostor_std:.6f}")
    print()
    print(f"Mean separation:          {separation:.6f}")
    print(f"Normalized separation:    {normalized_separation:.6f}")

    results.append({
        "group": group,
        "total_pairs": len(group_df),
        "genuine_pairs": len(genuine),
        "impostor_pairs": len(impostor),
        "genuine_mean": genuine_mean,
        "genuine_std": genuine_std,
        "impostor_mean": impostor_mean,
        "impostor_std": impostor_std,
        "mean_separation": separation,
        "normalized_separation": normalized_separation
    })

# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

csv_path = os.path.join(
    OUTPUT_DIR,
    "group_score_statistics.csv"
)

results_df.to_csv(csv_path, index=False)

# --------------------------------------------------
# FAIRNESS DISPARITY
# --------------------------------------------------

print("\n" + "=" * 70)
print("DEMOGRAPHIC SEPARATION DISPARITY")
print("=" * 70)

if len(results_df) >= 2:

    max_sep = results_df["mean_separation"].max()
    min_sep = results_df["mean_separation"].min()

    max_group = results_df.loc[
        results_df["mean_separation"].idxmax(),
        "group"
    ]

    min_group = results_df.loc[
        results_df["mean_separation"].idxmin(),
        "group"
    ]

    separation_disparity = max_sep - min_sep

    print(f"\nHighest separation: {max_sep:.6f}")
    print(f"Highest group:      {max_group}")
    print()
    print(f"Lowest separation:  {min_sep:.6f}")
    print(f"Lowest group:       {min_group}")
    print()
    print(f"Separation disparity: {separation_disparity:.6f}")

    disparity_df = pd.DataFrame([{
        "maximum_separation": max_sep,
        "maximum_group": max_group,
        "minimum_separation": min_sep,
        "minimum_group": min_group,
        "separation_disparity": separation_disparity
    }])

else:
    disparity_df = pd.DataFrame()

# Save disparity
disparity_path = os.path.join(
    OUTPUT_DIR,
    "separation_disparity.csv"
)

disparity_df.to_csv(disparity_path, index=False)

# --------------------------------------------------
# PLOT 1: SCORE DISTRIBUTION
# --------------------------------------------------

plt.figure(figsize=(12, 7))

for group in groups:

    group_df = within_df[
        within_df["group"] == group
    ]

    genuine = group_df[
        group_df["label"] == 1
    ]["cosine_similarity"]

    impostor = group_df[
        group_df["label"] == 0
    ]["cosine_similarity"]

    if len(genuine) > 0:
        plt.hist(
            genuine,
            bins=30,
            alpha=0.35,
            label=f"{group} genuine"
        )

    if len(impostor) > 0:
        plt.hist(
            impostor,
            bins=30,
            alpha=0.35,
            label=f"{group} impostor"
        )

plt.xlabel("Cosine Similarity Score")
plt.ylabel("Number of Pairs")
plt.title(
    "Genuine vs Impostor Score Distribution Across Demographic Groups"
)

plt.legend()
plt.tight_layout()

plot1_path = os.path.join(
    OUTPUT_DIR,
    "score_distribution.png"
)

plt.savefig(
    plot1_path,
    dpi=300
)

plt.close()

# --------------------------------------------------
# PLOT 2: GENUINE / IMPOSTOR MEANS
# --------------------------------------------------

if len(results_df) > 0:

    x = np.arange(len(results_df))
    width = 0.35

    plt.figure(figsize=(10, 6))

    plt.bar(
        x - width / 2,
        results_df["genuine_mean"],
        width,
        label="Genuine Mean"
    )

    plt.bar(
        x + width / 2,
        results_df["impostor_mean"],
        width,
        label="Impostor Mean"
    )

    plt.xticks(
        x,
        results_df["group"],
        rotation=20
    )

    plt.ylabel("Mean Cosine Similarity")
    plt.title(
        "Genuine vs Impostor Mean Scores by Demographic Group"
    )

    plt.legend()
    plt.tight_layout()

    plot2_path = os.path.join(
        OUTPUT_DIR,
        "genuine_vs_impostor_means.png"
    )

    plt.savefig(
        plot2_path,
        dpi=300
    )

    plt.close()

# --------------------------------------------------
# PLOT 3: SEPARATION BY GROUP
# --------------------------------------------------

if len(results_df) > 0:

    plt.figure(figsize=(10, 6))

    plt.bar(
        results_df["group"],
        results_df["mean_separation"]
    )

    plt.xlabel("Demographic Group")
    plt.ylabel("Genuine Mean − Impostor Mean")
    plt.title(
        "Score Separation Across Demographic Groups"
    )

    plt.xticks(rotation=20)
    plt.tight_layout()

    plot3_path = os.path.join(
        OUTPUT_DIR,
        "score_separation_by_group.png"
    )

    plt.savefig(
        plot3_path,
        dpi=300
    )

    plt.close()

# --------------------------------------------------
# SAVE TEXT SUMMARY
# --------------------------------------------------

summary_path = os.path.join(
    OUTPUT_DIR,
    "score_distribution_summary.txt"
)

with open(summary_path, "w", encoding="utf-8") as f:

    f.write(
        "FAIRNESS-FR — SCORE DISTRIBUTION & DEMOGRAPHIC OVERLAP ANALYSIS\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(f"Total valid pairs: {len(df)}\n")
    f.write(f"Within-demographic pairs: {len(within_df)}\n")
    f.write(f"Groups analyzed: {len(results_df)}\n\n")

    if len(results_df) > 0:

        f.write(
            results_df.to_string(index=False)
        )

        f.write("\n\n")

    if not disparity_df.empty:

        f.write("DEMOGRAPHIC SEPARATION DISPARITY\n")
        f.write("-" * 50 + "\n")

        f.write(
            disparity_df.to_string(index=False)
        )

        f.write("\n")

print("\n" + "=" * 70)
print("SCORE DISTRIBUTION ANALYSIS COMPLETE")
print("=" * 70)

print("\nOutput directory:")
print(os.path.abspath(OUTPUT_DIR))

print("\nOutput files:")
print(os.path.abspath(csv_path))
print(os.path.abspath(disparity_path))
print(os.path.abspath(summary_path))
print(os.path.abspath(plot1_path))

if len(results_df) > 0:
    print(os.path.abspath(plot2_path))
    print(os.path.abspath(plot3_path))