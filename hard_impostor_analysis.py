from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "bfw"
    / "arcface"
    / "test_scores.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "bfw"
    / "arcface"
    / "hard_impostor_analysis"
)

TOP_K = 20

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("FAIRNESS-FR — HARD IMPOSTOR PAIR ANALYSIS")
print("=" * 70)

print(f"\nInput: {INPUT_FILE}")

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

print(f"Total pairs loaded: {len(df)}")
print("\nAvailable columns:")
print(", ".join(df.columns))

# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = ["label", "cosine_similarity"]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}\n"
        f"Available columns: {list(df.columns)}"
    )

df = df.dropna(subset=required_columns).copy()

df["label"] = pd.to_numeric(df["label"], errors="coerce")
df["cosine_similarity"] = pd.to_numeric(
    df["cosine_similarity"],
    errors="coerce"
)

df = df.dropna(subset=required_columns).copy()
df["label"] = df["label"].astype(int)

# ============================================================
# SELECT IMPOSTOR PAIRS
# label = 0 means different identities / impostor pair
# ============================================================

impostors = df[df["label"] == 0].copy()

if impostors.empty:
    raise ValueError("No impostor pairs found where label == 0.")

print(f"\nTotal impostor pairs: {len(impostors)}")

# Hard impostors = highest similarity among impostor pairs
hard_impostors = (
    impostors
    .sort_values("cosine_similarity", ascending=False)
    .reset_index(drop=True)
)

hard_impostors.insert(
    0,
    "hardness_rank",
    np.arange(1, len(hard_impostors) + 1)
)

top_hard = hard_impostors.head(TOP_K).copy()

# ============================================================
# SAVE FULL AND TOP RESULTS
# ============================================================

full_file = OUTPUT_DIR / "all_impostor_pairs_ranked.csv"
top_file = OUTPUT_DIR / f"top_{TOP_K}_hard_impostor_pairs.csv"

hard_impostors.to_csv(full_file, index=False)
top_hard.to_csv(top_file, index=False)

# ============================================================
# SUMMARY STATISTICS
# ============================================================

max_score = hard_impostors["cosine_similarity"].max()
min_score = hard_impostors["cosine_similarity"].min()
mean_score = hard_impostors["cosine_similarity"].mean()
median_score = hard_impostors["cosine_similarity"].median()

summary = pd.DataFrame(
    {
        "metric": [
            "total_impostor_pairs",
            "hardest_impostor_score",
            "mean_impostor_score",
            "median_impostor_score",
            "lowest_impostor_score",
            f"top_{TOP_K}_mean_score",
        ],
        "value": [
            len(hard_impostors),
            max_score,
            mean_score,
            median_score,
            min_score,
            top_hard["cosine_similarity"].mean(),
        ],
    }
)

summary_file = OUTPUT_DIR / "hard_impostor_summary.csv"
summary.to_csv(summary_file, index=False)

# ============================================================
# DEMOGRAPHIC ANALYSIS IF AVAILABLE
# ============================================================

demographic_columns = [
    c for c in [
        "gender1",
        "gender2",
        "ethnicity1",
        "ethnicity2",
        "race1",
        "race2",
    ]
    if c in hard_impostors.columns
]

if demographic_columns:
    demo_file = OUTPUT_DIR / "hard_impostor_demographics.csv"

    demo_df = top_hard[demographic_columns + ["cosine_similarity"]].copy()

    if "gender1" in demo_df.columns and "gender2" in demo_df.columns:
        demo_df["gender_pair"] = (
            demo_df["gender1"].astype(str)
            + "-"
            + demo_df["gender2"].astype(str)
        )

    if "ethnicity1" in demo_df.columns and "ethnicity2" in demo_df.columns:
        demo_df["ethnicity_pair"] = (
            demo_df["ethnicity1"].astype(str)
            + "-"
            + demo_df["ethnicity2"].astype(str)
        )

    demo_df.to_csv(demo_file, index=False)

# ============================================================
# PLOT 1 — TOP HARD IMPOSTOR SCORES
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    top_hard["hardness_rank"].astype(str),
    top_hard["cosine_similarity"]
)

plt.xlabel("Hardness Rank")
plt.ylabel("Cosine Similarity")
plt.title(f"Top {TOP_K} Hardest Impostor Pairs")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "top_hard_impostor_pairs.png",
    dpi=300
)

plt.close()

# ============================================================
# PLOT 2 — IMPOSTOR SCORE DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    hard_impostors["cosine_similarity"],
    bins=30
)

plt.axvline(
    max_score,
    linestyle="--",
    label=f"Hardest = {max_score:.4f}"
)

plt.xlabel("Cosine Similarity")
plt.ylabel("Number of Impostor Pairs")
plt.title("Distribution of Impostor Pair Similarity Scores")
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "impostor_score_distribution.png",
    dpi=300
)

plt.close()

# ============================================================
# TEXT SUMMARY
# ============================================================

text_file = OUTPUT_DIR / "hard_impostor_summary.txt"

with open(text_file, "w", encoding="utf-8") as f:
    f.write("FAIRNESS-FR — HARD IMPOSTOR PAIR ANALYSIS\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Dataset: BFW\n")
    f.write(f"Model: ArcFace\n")
    f.write(f"Total pairs: {len(df)}\n")
    f.write(f"Total impostor pairs: {len(hard_impostors)}\n")
    f.write(f"Top K analyzed: {TOP_K}\n\n")
    f.write(f"Hardest impostor score: {max_score:.6f}\n")
    f.write(f"Mean impostor score: {mean_score:.6f}\n")
    f.write(f"Median impostor score: {median_score:.6f}\n")
    f.write(f"Lowest impostor score: {min_score:.6f}\n")
    f.write(
        f"Top {TOP_K} mean score: "
        f"{top_hard['cosine_similarity'].mean():.6f}\n"
    )

print("\n" + "=" * 70)
print("HARD IMPOSTOR ANALYSIS COMPLETE")
print("=" * 70)

print(f"\nTotal impostor pairs: {len(hard_impostors)}")
print(f"Hardest impostor score: {max_score:.6f}")
print(f"Top {TOP_K} mean score: {top_hard['cosine_similarity'].mean():.6f}")

print("\nTop 10 hardest impostor pairs:")
print(
    top_hard[
        ["hardness_rank", "cosine_similarity"]
    ].head(10).to_string(index=False)
)

print("\nOutput directory:")
print(OUTPUT_DIR)