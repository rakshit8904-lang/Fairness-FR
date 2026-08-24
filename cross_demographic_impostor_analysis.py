import os
import pandas as pd
import numpy as np


# ============================================================
# FAIRNESS-FR — CROSS-DEMOGRAPHIC IMPOSTOR ANALYSIS
# ============================================================

print("=" * 70)
print("FAIRNESS-FR — CROSS-DEMOGRAPHIC IMPOSTOR ANALYSIS")
print("=" * 70)


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = r"results\bfw\arcface\test_scores.csv"
OUTPUT_DIR = r"results\bfw\arcface\cross_demographic_impostor_analysis"

# Use the same global threshold obtained from threshold analysis
THRESHOLD = 0.154624

TOP_HARD_PAIRS = 20


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"\nERROR: Input file not found:\n{os.path.abspath(INPUT_FILE)}"
    )

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
        f"\nERROR: Missing required columns: {missing}\n"
        f"Available columns:\n{', '.join(df.columns)}"
    )


# ============================================================
# CLEAN DATA
# ============================================================

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

df = df.dropna(subset=["label", "cosine_similarity"]).copy()

df["label"] = df["label"].astype(int)

print(f"Valid pairs after cleaning: {len(df)}")


# ============================================================
# FILTER IMPOSTOR PAIRS
# label = 0 means different identities
# ============================================================

impostors = df[df["label"] == 0].copy()

print(f"Total impostor pairs: {len(impostors)}")

if len(impostors) == 0:
    raise ValueError("No impostor pairs found in the dataset.")


# ============================================================
# KEEP ONLY CROSS-DEMOGRAPHIC PAIRS
# ============================================================

cross_demo = impostors[
    impostors["demographic1"] != impostors["demographic2"]
].copy()

print(f"Cross-demographic impostor pairs: {len(cross_demo)}")

if len(cross_demo) == 0:
    print("\nWARNING: No cross-demographic impostor pairs found.")
    print(
        "Your current test_scores.csv may contain impostor pairs "
        "only within the same demographic groups."
    )
    print("\nAnalysis finished.")
    exit()


# ============================================================
# CREATE ORDER-INDEPENDENT DEMOGRAPHIC PAIR
#
# Example:
# asian_females + asian_males
# and
# asian_males + asian_females
#
# become the same group:
# asian_females vs asian_males
# ============================================================

def make_pair(row):
    group1 = str(row["demographic1"])
    group2 = str(row["demographic2"])

    groups = sorted([group1, group2])

    return f"{groups[0]} vs {groups[1]}"


cross_demo["demographic_pair"] = cross_demo.apply(
    make_pair,
    axis=1
)


# ============================================================
# FALSE MATCH CALCULATION
#
# An impostor is falsely accepted if:
# cosine_similarity >= threshold
# ============================================================

cross_demo["false_match"] = (
    cross_demo["cosine_similarity"] >= THRESHOLD
)


# ============================================================
# GROUP-WISE ANALYSIS
# ============================================================

results = []

for pair_name, group in cross_demo.groupby("demographic_pair"):

    total_pairs = len(group)

    false_matches = int(group["false_match"].sum())

    fmr = (
        false_matches / total_pairs
        if total_pairs > 0
        else np.nan
    )

    mean_similarity = group["cosine_similarity"].mean()
    median_similarity = group["cosine_similarity"].median()
    max_similarity = group["cosine_similarity"].max()
    min_similarity = group["cosine_similarity"].min()
    std_similarity = group["cosine_similarity"].std()

    results.append({
        "demographic_pair": pair_name,
        "total_impostor_pairs": total_pairs,
        "false_matches": false_matches,
        "fmr": fmr,
        "mean_cosine_similarity": mean_similarity,
        "median_cosine_similarity": median_similarity,
        "std_cosine_similarity": std_similarity,
        "min_cosine_similarity": min_similarity,
        "max_cosine_similarity": max_similarity
    })


results_df = pd.DataFrame(results)

if len(results_df) == 0:
    raise ValueError(
        "No demographic-pair results could be generated."
    )


# Sort by highest FMR, then highest mean similarity
results_df = results_df.sort_values(
    by=["fmr", "mean_cosine_similarity"],
    ascending=[False, False]
).reset_index(drop=True)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("CROSS-DEMOGRAPHIC IMPOSTOR RESULTS")
print("=" * 70)

print(f"\nOperating threshold: {THRESHOLD:.6f}")
print(f"Cross-demographic pairs analysed: {len(cross_demo)}")
print(f"Unique demographic combinations: {len(results_df)}")

print("\n" + results_df.to_string(index=False))


# ============================================================
# FMR DISPARITY
# ============================================================

max_fmr = results_df["fmr"].max()
min_fmr = results_df["fmr"].min()

max_fmr_pair = results_df.loc[
    results_df["fmr"].idxmax(),
    "demographic_pair"
]

min_fmr_pair = results_df.loc[
    results_df["fmr"].idxmin(),
    "demographic_pair"
]

fmr_disparity = max_fmr - min_fmr


# ============================================================
# MEAN SIMILARITY DISPARITY
# ============================================================

max_mean_similarity = results_df["mean_cosine_similarity"].max()
min_mean_similarity = results_df["mean_cosine_similarity"].min()

max_similarity_pair = results_df.loc[
    results_df["mean_cosine_similarity"].idxmax(),
    "demographic_pair"
]

min_similarity_pair = results_df.loc[
    results_df["mean_cosine_similarity"].idxmin(),
    "demographic_pair"
]

similarity_disparity = (
    max_mean_similarity - min_mean_similarity
)


# ============================================================
# SUMMARY
# ============================================================

summary = []

summary.append("=" * 70)
summary.append("CROSS-DEMOGRAPHIC IMPOSTOR ANALYSIS SUMMARY")
summary.append("=" * 70)

summary.append(f"Input file: {os.path.abspath(INPUT_FILE)}")
summary.append(f"Operating threshold: {THRESHOLD:.6f}")
summary.append(f"Total impostor pairs: {len(impostors)}")
summary.append(
    f"Cross-demographic impostor pairs: {len(cross_demo)}"
)
summary.append(
    f"Unique demographic combinations: {len(results_df)}"
)

summary.append("")
summary.append("FMR DISPARITY")
summary.append("-" * 70)
summary.append(f"Highest FMR: {max_fmr:.6f}")
summary.append(f"Highest FMR pair: {max_fmr_pair}")
summary.append(f"Lowest FMR: {min_fmr:.6f}")
summary.append(f"Lowest FMR pair: {min_fmr_pair}")
summary.append(f"FMR disparity: {fmr_disparity:.6f}")

summary.append("")
summary.append("SIMILARITY DISPARITY")
summary.append("-" * 70)
summary.append(
    f"Highest mean similarity: {max_mean_similarity:.6f}"
)
summary.append(
    f"Highest similarity pair: {max_similarity_pair}"
)
summary.append(
    f"Lowest mean similarity: {min_mean_similarity:.6f}"
)
summary.append(
    f"Lowest similarity pair: {min_similarity_pair}"
)
summary.append(
    f"Mean similarity disparity: {similarity_disparity:.6f}"
)

summary.append("")
summary.append("INTERPRETATION")
summary.append("-" * 70)
summary.append(
    "Higher FMR indicates that impostor pairs from that "
    "demographic combination are more likely to be incorrectly "
    "accepted as a match at the selected operating threshold."
)

summary.append(
    "Higher mean cosine similarity indicates that the model "
    "produces more similar embeddings on average for impostor "
    "pairs from that demographic combination."
)

summary.append("=" * 70)


summary_text = "\n".join(summary)

print("\n")
print(summary_text)


# ============================================================
# TOP HARD CROSS-DEMOGRAPHIC IMPOSTORS
# ============================================================

hard_cross_demo = cross_demo.sort_values(
    by="cosine_similarity",
    ascending=False
).head(TOP_HARD_PAIRS).copy()

hard_cross_demo.insert(
    0,
    "hardness_rank",
    range(1, len(hard_cross_demo) + 1)
)

columns_to_save = [
    "hardness_rank",
    "demographic_pair",
    "image1",
    "image2",
    "identity1",
    "identity2",
    "demographic1",
    "demographic2",
    "gender1",
    "gender2",
    "age1",
    "age2",
    "cosine_similarity",
    "euclidean_distance",
    "cosine_distance",
    "false_match"
]

# Keep only columns that actually exist
columns_to_save = [
    col for col in columns_to_save
    if col in hard_cross_demo.columns
]

hard_cross_demo = hard_cross_demo[columns_to_save]


# ============================================================
# SAVE OUTPUT FILES
# ============================================================

metrics_file = os.path.join(
    OUTPUT_DIR,
    "cross_demographic_impostor_metrics.csv"
)

hard_pairs_file = os.path.join(
    OUTPUT_DIR,
    "top_hard_cross_demographic_impostors.csv"
)

summary_file = os.path.join(
    OUTPUT_DIR,
    "cross_demographic_impostor_summary.txt"
)


results_df.to_csv(
    metrics_file,
    index=False
)

hard_cross_demo.to_csv(
    hard_pairs_file,
    index=False
)

with open(summary_file, "w", encoding="utf-8") as f:
    f.write(summary_text)


# ============================================================
# TOP HARD PAIRS DISPLAY
# ============================================================

print("\n" + "=" * 70)
print(f"TOP {len(hard_cross_demo)} HARDEST CROSS-DEMOGRAPHIC IMPOSTORS")
print("=" * 70)

display_columns = [
    col for col in [
        "hardness_rank",
        "demographic_pair",
        "cosine_similarity",
        "false_match"
    ]
    if col in hard_cross_demo.columns
]

print(
    hard_cross_demo[display_columns].to_string(index=False)
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("CROSS-DEMOGRAPHIC IMPOSTOR ANALYSIS COMPLETE")
print("=" * 70)

print("\nOutput files:")
print(os.path.abspath(metrics_file))
print(os.path.abspath(hard_pairs_file))
print(os.path.abspath(summary_file))