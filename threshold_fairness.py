from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

SCORES_FILE = Path("results/bfw/arcface/test_scores.csv")

OUTPUT_DIR = Path("results/bfw/arcface/threshold_fairness")

THRESHOLDS = np.arange(0.05, 0.951, 0.01)


# ============================================================
# Metric calculation
# ============================================================

def calculate_metrics(df, threshold):
    """
    Calculate verification metrics for one demographic group
    at a particular similarity threshold.
    """

    genuine = df[df["label"] == 1]
    impostor = df[df["label"] == 0]

    if len(genuine) == 0 or len(impostor) == 0:
        return {
            "fmr": np.nan,
            "fnmr": np.nan,
            "tar": np.nan,
            "accuracy": np.nan,
        }

    # Genuine pair accepted if score >= threshold
    genuine_accepted = (
        genuine["cosine_similarity"] >= threshold
    )

    # Impostor pair incorrectly accepted
    impostor_accepted = (
        impostor["cosine_similarity"] >= threshold
    )

    tar = genuine_accepted.mean()
    fnmr = 1.0 - tar
    fmr = impostor_accepted.mean()

    total_correct = (
        genuine_accepted.sum()
        + (~impostor_accepted).sum()
    )

    total = len(genuine) + len(impostor)

    accuracy = total_correct / total

    return {
        "fmr": float(fmr),
        "fnmr": float(fnmr),
        "tar": float(tar),
        "accuracy": float(accuracy),
    }


# ============================================================
# Main experiment
# ============================================================

def main():

    if not SCORES_FILE.exists():
        raise FileNotFoundError(
            f"Score file not found: {SCORES_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FAIRNESS-FR — THRESHOLD FAIRNESS STRESS TEST")
    print("=" * 70)

    df = pd.read_csv(SCORES_FILE)

    required_columns = {
        "label",
        "cosine_similarity",
        "gender1",
        "gender2",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    # --------------------------------------------------------
    # Normalize gender names
    # --------------------------------------------------------

    df["gender1"] = (
        df["gender1"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df["gender2"] = (
        df["gender2"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # --------------------------------------------------------
    # Same-gender evaluation groups
    # --------------------------------------------------------

    female_df = df[
        (df["gender1"] == "females")
        & (df["gender2"] == "females")
    ].copy()

    male_df = df[
        (df["gender1"] == "males")
        & (df["gender2"] == "males")
    ].copy()

    print(f"\nTotal pairs: {len(df)}")
    print(f"Female-Female pairs: {len(female_df)}")
    print(f"Male-Male pairs: {len(male_df)}")

    if len(female_df) == 0 or len(male_df) == 0:
        raise RuntimeError(
            "One of the same-gender groups contains no pairs."
        )

    # --------------------------------------------------------
    # Threshold sweep
    # --------------------------------------------------------

    rows = []

    for threshold in THRESHOLDS:

        female = calculate_metrics(
            female_df,
            threshold
        )

        male = calculate_metrics(
            male_df,
            threshold
        )

        rows.append({
            "threshold": threshold,

            "female_fmr": female["fmr"],
            "male_fmr": male["fmr"],

            "female_fnmr": female["fnmr"],
            "male_fnmr": male["fnmr"],

            "female_tar": female["tar"],
            "male_tar": male["tar"],

            "female_accuracy": female["accuracy"],
            "male_accuracy": male["accuracy"],

            "fmr_disparity": abs(
                female["fmr"] - male["fmr"]
            ),

            "fnmr_disparity": abs(
                female["fnmr"] - male["fnmr"]
            ),

            "tar_disparity": abs(
                female["tar"] - male["tar"]
            ),

            "accuracy_disparity": abs(
                female["accuracy"] - male["accuracy"]
            ),
        })

    results = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    csv_path = (
        OUTPUT_DIR /
        "threshold_fairness.csv"
    )

    results.to_csv(
        csv_path,
        index=False
    )

    # --------------------------------------------------------
    # Find best fairness thresholds
    # --------------------------------------------------------

    results["mean_disparity"] = (
        results["fmr_disparity"]
        + results["fnmr_disparity"]
        + results["tar_disparity"]
        + results["accuracy_disparity"]
    ) / 4.0

    best_fairness = results.loc[
        results["mean_disparity"].idxmin()
    ]

    # --------------------------------------------------------
    # Plot 1 — FMR disparity
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        results["threshold"],
        results["fmr_disparity"],
        label="FMR disparity"
    )

    plt.xlabel("Similarity Threshold")
    plt.ylabel("Absolute disparity")
    plt.title(
        "Female-Male FMR Disparity Across Thresholds"
    )

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "fmr_disparity.png",
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # Plot 2 — FNMR disparity
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        results["threshold"],
        results["fnmr_disparity"],
        label="FNMR disparity"
    )

    plt.xlabel("Similarity Threshold")
    plt.ylabel("Absolute disparity")
    plt.title(
        "Female-Male FNMR Disparity Across Thresholds"
    )

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "fnmr_disparity.png",
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # Plot 3 — TAR disparity
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        results["threshold"],
        results["tar_disparity"],
        label="TAR disparity"
    )

    plt.xlabel("Similarity Threshold")
    plt.ylabel("Absolute disparity")
    plt.title(
        "Female-Male TAR Disparity Across Thresholds"
    )

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "tar_disparity.png",
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # Plot 4 — Overall fairness disparity
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        results["threshold"],
        results["mean_disparity"],
        label="Mean demographic disparity"
    )

    plt.axvline(
        best_fairness["threshold"],
        linestyle="--",
        label=(
            f"Best fairness threshold = "
            f"{best_fairness['threshold']:.2f}"
        )
    )

    plt.xlabel("Similarity Threshold")
    plt.ylabel("Mean disparity")
    plt.title(
        "Overall Demographic Disparity Across Thresholds"
    )

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "overall_fairness_disparity.png",
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "dataset": "BFW",
        "model": "ArcFace",
        "evaluation": "same-gender threshold fairness stress test",
        "total_pairs": int(len(df)),
        "female_female_pairs": int(len(female_df)),
        "male_male_pairs": int(len(male_df)),
        "threshold_min": float(THRESHOLDS.min()),
        "threshold_max": float(THRESHOLDS.max()),
        "threshold_step": 0.01,
        "best_fairness_threshold": float(
            best_fairness["threshold"]
        ),
        "minimum_mean_disparity": float(
            best_fairness["mean_disparity"]
        ),
        "best_fairness_fmr_disparity": float(
            best_fairness["fmr_disparity"]
        ),
        "best_fairness_fnmr_disparity": float(
            best_fairness["fnmr_disparity"]
        ),
        "best_fairness_tar_disparity": float(
            best_fairness["tar_disparity"]
        ),
        "best_fairness_accuracy_disparity": float(
            best_fairness["accuracy_disparity"]
        ),
    }

    with open(
        OUTPUT_DIR / "threshold_fairness_summary.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("THRESHOLD FAIRNESS TEST COMPLETE")
    print("=" * 70)

    print(
        f"Best fairness threshold: "
        f"{best_fairness['threshold']:.2f}"
    )

    print(
        f"Minimum mean disparity: "
        f"{best_fairness['mean_disparity']:.6f}"
    )

    print(
        f"FMR disparity: "
        f"{best_fairness['fmr_disparity']:.6f}"
    )

    print(
        f"FNMR disparity: "
        f"{best_fairness['fnmr_disparity']:.6f}"
    )

    print(
        f"TAR disparity: "
        f"{best_fairness['tar_disparity']:.6f}"
    )

    print(
        f"Accuracy disparity: "
        f"{best_fairness['accuracy_disparity']:.6f}"
    )

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    print("=" * 70)


if __name__ == "__main__":
    main()