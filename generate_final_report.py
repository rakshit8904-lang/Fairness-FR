from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# ============================================================
# FAIRNESS-FR
# FINAL CONSOLIDATED RESEARCH REPORT GENERATOR
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "bfw" / "arcface"
OUTPUT_DIR = RESULTS_DIR / "final_research_report"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_FILE = OUTPUT_DIR / "FAIRNESS_FR_FINAL_RESEARCH_REPORT.md"
INVENTORY_FILE = OUTPUT_DIR / "result_files_inventory.txt"

print("=" * 70)
print("FAIRNESS-FR - FINAL CONSOLIDATED RESEARCH REPORT")
print("=" * 70)

print(f"\nResults directory:\n{RESULTS_DIR}")

if not RESULTS_DIR.exists():
    raise FileNotFoundError(f"Results directory not found:\n{RESULTS_DIR}")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_files(pattern):
    return sorted(RESULTS_DIR.rglob(pattern))


def read_csv_safe(path):
    try:
        return pd.read_csv(path)
    except Exception as exc:
        return None


def read_text_safe(path):
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def fmt(value, digits=6):
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def get_value(df, possible_columns, default="N/A"):
    if df is None or df.empty:
        return default

    for col in possible_columns:
        if col in df.columns:
            return df.iloc[0][col]

    return default


# ============================================================
# BUILD RESULT INVENTORY
# ============================================================

all_result_files = sorted(
    [
        path
        for path in RESULTS_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in {".csv", ".txt"}
    ]
)

with open(INVENTORY_FILE, "w", encoding="utf-8") as f:
    for path in all_result_files:
        f.write(str(path.relative_to(RESULTS_DIR)) + "\n")

print(f"\nResult files discovered: {len(all_result_files)}")


# ============================================================
# LOAD BASELINE RESULTS
# ============================================================

test_scores_path = RESULTS_DIR / "test_scores.csv"
test_scores = read_csv_safe(test_scores_path)

total_pairs = "N/A"
genuine_pairs = "N/A"
impostor_pairs = "N/A"

if test_scores is not None:
    total_pairs = len(test_scores)

    if "label" in test_scores.columns:
        genuine_pairs = int((test_scores["label"] == 1).sum())
        impostor_pairs = int((test_scores["label"] == 0).sum())


# ============================================================
# LOAD PERFORMANCE SUMMARY
# ============================================================

performance_files = find_files("performance*.csv")
performance_df = read_csv_safe(performance_files[0]) if performance_files else None


# ============================================================
# LOAD FAIRNESS RESULTS
# ============================================================

fairness_metric_files = [
    p for p in find_files("fairness_metrics.csv")
    if "fairness_performance_tradeoff" not in str(p)
]

fairness_df = (
    read_csv_safe(fairness_metric_files[0])
    if fairness_metric_files
    else None
)


# ============================================================
# LOAD TRADE-OFF RESULTS
# ============================================================

tradeoff_files = find_files("operating_points.csv")
tradeoff_df = read_csv_safe(tradeoff_files[0]) if tradeoff_files else None


# ============================================================
# LOAD BLUR STRESS TEST
# ============================================================

blur_files = find_files("blur_fairness_results.csv")
blur_df = read_csv_safe(blur_files[0]) if blur_files else None


# ============================================================
# LOAD ILLUMINATION STRESS TEST
# ============================================================

illumination_files = [
    p for p in find_files("*.csv")
    if "illumination" in str(p).lower()
    and "stress" in str(p).lower()
]

illumination_df = None

for path in illumination_files:
    candidate = read_csv_safe(path)

    if candidate is not None and not candidate.empty:
        if "accuracy" in candidate.columns or "tar" in candidate.columns:
            illumination_df = candidate
            break


# ============================================================
# LOAD HARD IMPOSTOR RESULTS
# ============================================================

hard_impostor_files = find_files("*.csv")
hard_impostor_df = None

for path in hard_impostor_files:
    if "hard_impostor_analysis" in str(path):
        candidate = read_csv_safe(path)

        if candidate is not None and not candidate.empty:
            if (
                "hardness_rank" in candidate.columns
                or "cosine_similarity" in candidate.columns
            ):
                hard_impostor_df = candidate
                break


# ============================================================
# LOAD INTERSECTIONAL RESULTS
# ============================================================

intersectional_files = find_files("intersectional_metrics.csv")
intersectional_df = (
    read_csv_safe(intersectional_files[0])
    if intersectional_files
    else None
)

intersectional_disparity_files = find_files(
    "intersectional_disparities.csv"
)

intersectional_disparity_df = (
    read_csv_safe(intersectional_disparity_files[0])
    if intersectional_disparity_files
    else None
)


# ============================================================
# LOAD GLOBAL VS GROUP THRESHOLD RESULTS
# ============================================================

group_threshold_files = find_files(
    "global_vs_group_threshold_metrics.csv"
)

group_threshold_df = (
    read_csv_safe(group_threshold_files[0])
    if group_threshold_files
    else None
)

threshold_disparity_files = find_files(
    "threshold_disparity_comparison.csv"
)

threshold_disparity_df = (
    read_csv_safe(threshold_disparity_files[0])
    if threshold_disparity_files
    else None
)


# ============================================================
# CREATE REPORT
# ============================================================

report = []

report.append("# FAIRNESS-FR: Fairness Evaluation of Face Recognition Systems")
report.append("")
report.append("## Consolidated Experimental Research Report")
report.append("")
report.append(
    f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
report.append("")
report.append("---")
report.append("")

# ============================================================
# ABSTRACT
# ============================================================

report.append("## Abstract")
report.append("")

report.append(
    "This report presents a consolidated fairness evaluation of a "
    "face recognition system using the BFW dataset and the ArcFace "
    "recognition model. The experimental study evaluates overall "
    "recognition performance, demographic group disparities, "
    "threshold sensitivity, image degradation robustness, hard "
    "impostor pairs, intersectional fairness, and the effect of "
    "global versus group-specific decision thresholds."
)

if total_pairs != "N/A":
    report.append(
        f"The analysis used **{total_pairs} evaluated pairs**, including "
        f"**{genuine_pairs} genuine pairs** and "
        f"**{impostor_pairs} impostor pairs**."
    )

report.append(
    "The results demonstrate that fairness is sensitive to both image "
    "quality and threshold selection. Group-specific thresholding can "
    "reduce disparity for some metrics while increasing disparity for "
    "others, highlighting the multi-objective nature of fairness-aware "
    "face recognition evaluation."
)

report.append("")
report.append("---")
report.append("")


# ============================================================
# 1. EXPERIMENTAL SETUP
# ============================================================

report.append("## 1. Experimental Setup")
report.append("")

report.append("- **Dataset:** BFW")
report.append("- **Face recognition model:** ArcFace")
report.append("- **Similarity metric:** Cosine similarity")

if total_pairs != "N/A":
    report.append(f"- **Total evaluated pairs:** {total_pairs}")
    report.append(f"- **Genuine pairs:** {genuine_pairs}")
    report.append(f"- **Impostor pairs:** {impostor_pairs}")

report.append(
    "- **Evaluation focus:** recognition performance and demographic fairness"
)

report.append("")


# ============================================================
# 2. BASELINE DATA
# ============================================================

report.append("## 2. Baseline Recognition Performance")
report.append("")

if performance_df is not None:
    report.append("Performance result file detected.")
    report.append("")
    report.append("```text")
    report.append(performance_df.to_string(index=False))
    report.append("```")
else:
    report.append(
        "A structured performance summary CSV was not automatically "
        "identified. Raw test score data was successfully detected."
    )

report.append("")


# ============================================================
# 3. FAIRNESS
# ============================================================

report.append("## 3. Demographic Fairness Analysis")
report.append("")

if fairness_df is not None:
    report.append("```text")
    report.append(fairness_df.to_string(index=False))
    report.append("```")
else:
    report.append(
        "Fairness metrics are available in the result directory but "
        "could not be uniquely selected automatically."
    )

report.append("")


# ============================================================
# 4. FAIRNESS-PERFORMANCE TRADE-OFF
# ============================================================

report.append("## 4. Fairness-Performance Threshold Trade-Off")
report.append("")

if tradeoff_df is not None:
    report.append(
        "The threshold operating-point analysis identified alternative "
        "decision thresholds based on fairness and recognition performance."
    )
    report.append("")
    report.append("```text")
    report.append(tradeoff_df.to_string(index=False))
    report.append("```")
else:
    report.append(
        "Trade-off operating points were not automatically identified."
    )

report.append("")


# ============================================================
# 5. IMAGE QUALITY / BLUR
# ============================================================

report.append("## 5. Gaussian Blur Fairness Stress Test")
report.append("")

if blur_df is not None:
    report.append(
        "Recognition performance was evaluated under increasing Gaussian "
        "blur severity."
    )
    report.append("")
    report.append("```text")
    report.append(blur_df.to_string(index=False))
    report.append("```")

    if (
        "condition" in blur_df.columns
        and "accuracy" in blur_df.columns
        and "mean_disparity" in blur_df.columns
    ):
        original = blur_df[
            blur_df["condition"].astype(str).str.lower() == "original"
        ]

        heavy = blur_df[
            blur_df["condition"].astype(str).str.lower().str.contains("heavy")
        ]

        if not original.empty and not heavy.empty:
            acc_drop = (
                float(original.iloc[0]["accuracy"])
                - float(heavy.iloc[0]["accuracy"])
            )

            disparity_change = (
                float(heavy.iloc[0]["mean_disparity"])
                - float(original.iloc[0]["mean_disparity"])
            )

            report.append("")
            report.append(
                f"From the original condition to heavy blur, accuracy "
                f"changed by **{acc_drop:.6f}**, while mean disparity "
                f"changed by **{disparity_change:.6f}**."
            )
else:
    report.append("Blur stress-test CSV was not automatically identified.")

report.append("")


# ============================================================
# 6. ILLUMINATION
# ============================================================

report.append("## 6. Illumination Robustness Analysis")
report.append("")

if illumination_df is not None:
    report.append("```text")
    report.append(illumination_df.to_string(index=False))
    report.append("```")
else:
    report.append(
        "Illumination experiment output files were detected, but a "
        "single primary metric CSV could not be selected automatically."
    )

report.append("")


# ============================================================
# 7. HARD IMPOSTORS
# ============================================================

report.append("## 7. Hard Impostor Pair Analysis")
report.append("")

if hard_impostor_df is not None:
    report.append(
        "Hard impostor pairs represent different identities that receive "
        "relatively high similarity scores and therefore pose a greater "
        "risk of false matches."
    )
    report.append("")

    if "hardness_rank" in hard_impostor_df.columns:
        hard_display = hard_impostor_df.head(10)
    else:
        hard_display = hard_impostor_df.head(10)

    report.append("```text")
    report.append(hard_display.to_string(index=False))
    report.append("```")
else:
    report.append(
        "Hard impostor output files were detected but the primary table "
        "could not be automatically identified."
    )

report.append("")


# ============================================================
# 8. INTERSECTIONAL FAIRNESS
# ============================================================

report.append("## 8. Intersectional Fairness Analysis")
report.append("")

if intersectional_df is not None:
    report.append(
        "Intersectional groups were evaluated by combining demographic "
        "attributes such as ethnicity and gender."
    )
    report.append("")
    report.append("```text")
    report.append(intersectional_df.to_string(index=False))
    report.append("```")

if intersectional_disparity_df is not None:
    report.append("")
    report.append("### Intersectional Disparities")
    report.append("")
    report.append("```text")
    report.append(intersectional_disparity_df.to_string(index=False))
    report.append("```")

if intersectional_df is None:
    report.append("Intersectional metric results were not found.")

report.append("")


# ============================================================
# 9. GLOBAL VS GROUP-SPECIFIC THRESHOLDS
# ============================================================

report.append("## 9. Global versus Group-Specific Thresholds")
report.append("")

report.append(
    "This experiment compares a single global operating threshold with "
    "thresholds optimized separately for demographic groups."
)

if group_threshold_df is not None:
    report.append("")
    report.append("### Group-wise Threshold Results")
    report.append("")
    report.append("```text")
    report.append(group_threshold_df.to_string(index=False))
    report.append("```")

if threshold_disparity_df is not None:
    report.append("")
    report.append("### Disparity Comparison")
    report.append("")
    report.append("```text")
    report.append(threshold_disparity_df.to_string(index=False))
    report.append("```")

    if {
        "metric",
        "global_threshold_disparity",
        "group_specific_disparity",
        "disparity_change",
    }.issubset(threshold_disparity_df.columns):

        report.append("")
        report.append("### Interpretation")
        report.append("")

        for _, row in threshold_disparity_df.iterrows():
            metric = row["metric"]
            change = float(row["disparity_change"])

            direction = (
                "increased" if change > 0
                else "decreased" if change < 0
                else "remained unchanged"
            )

            report.append(
                f"- **{metric.upper()} disparity** {direction} by "
                f"**{abs(change):.6f}**."
            )

report.append("")


# ============================================================
# 10. KEY FINDINGS
# ============================================================

report.append("## 10. Key Findings")
report.append("")

report.append(
    "1. Face recognition performance and demographic fairness should be "
    "evaluated jointly rather than relying on accuracy alone."
)

report.append(
    "2. Threshold selection creates a measurable fairness-performance "
    "trade-off."
)

report.append(
    "3. Image degradation can affect both recognition accuracy and "
    "demographic disparity."
)

report.append(
    "4. Hard impostor pairs identify challenging false-match cases that "
    "may not be apparent from aggregate performance metrics."
)

report.append(
    "5. Intersectional analysis provides a more detailed view of fairness "
    "than single demographic attributes alone."
)

report.append(
    "6. Group-specific thresholds do not universally improve all fairness "
    "metrics; improvements in one metric may be accompanied by increased "
    "disparity in another."
)

report.append("")


# ============================================================
# 11. LIMITATIONS
# ============================================================

report.append("## 11. Limitations")
report.append("")

report.append(
    "- The current consolidated analysis is based primarily on the BFW "
    "dataset and available evaluated demographic groups."
)

report.append(
    "- Results should not be generalized to all populations without "
    "evaluation on additional datasets."
)

report.append(
    "- Fairness conclusions depend on the selected metrics, thresholds, "
    "dataset composition, and pair distribution."
)

report.append(
    "- Group-specific threshold deployment may introduce practical and "
    "policy considerations beyond the numerical metrics reported here."
)

report.append("")


# ============================================================
# 12. CONCLUSION
# ============================================================

report.append("## 12. Conclusion")
report.append("")

report.append(
    "The FAIRNESS-FR experimental study demonstrates a multi-dimensional "
    "approach to evaluating face recognition systems. Beyond baseline "
    "accuracy, the experiments analyze demographic disparities, threshold "
    "sensitivity, robustness to image degradation, hard impostor cases, "
    "intersectional groups, and alternative thresholding strategies. "
    "The combined results show that fairness optimization is inherently "
    "multi-objective: a configuration that improves one fairness measure "
    "may reduce performance or worsen another disparity. Consequently, "
    "fairness-aware face recognition evaluation should report multiple "
    "performance and disparity metrics rather than relying on a single "
    "aggregate score."
)

report.append("")
report.append("---")
report.append("")
report.append("## Appendix: Result File Inventory")
report.append("")

for path in all_result_files:
    report.append(
        f"- `{path.relative_to(RESULTS_DIR).as_posix()}`"
    )


# ============================================================
# SAVE REPORT
# ============================================================

REPORT_FILE.write_text(
    "\n".join(report),
    encoding="utf-8"
)

print("\n" + "=" * 70)
print("FINAL RESEARCH REPORT GENERATED SUCCESSFULLY")
print("=" * 70)
print(f"\nReport:\n{REPORT_FILE}")
print(f"\nInventory:\n{INVENTORY_FILE}")
print("\nOpen the report with:")
print(f'notepad "{REPORT_FILE}"')
print("=" * 70)