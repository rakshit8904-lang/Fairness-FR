# ⚖️ Fairness-FR
## Demographic Fairness Evaluation of Face Recognition Systems

## 🚀 Live Demo

🔗 [Open FairFaceEval – Fairness Evaluation of Face Recognition](https://fairness-fr-urnisa.streamlit.app)

---

## 📌 Overview

**Fairness-FR** is an end-to-end research and evaluation framework designed to
study the **recognition performance and demographic fairness of face
recognition systems**.

The framework goes beyond evaluating a face recognition model using only
overall accuracy. It combines biometric performance evaluation with
demographic-group analysis to determine whether recognition behaviour remains
consistent across different demographic groups.

The complete framework follows:

```text
Dataset
   ↓
Dataset Preprocessing
   ↓
Genuine / Impostor Pair Generation
   ↓
Face Embedding Extraction
   ↓
Similarity Score Calculation
   ↓
Threshold Analysis
   ↓
Biometric Performance Evaluation
   ↓
Demographic Fairness Evaluation
   ↓
Cross-Model Comparison
   ↓
Model Ranking
   ↓
Visualizations + Reports
   ↓
Interactive Streamlit Dashboard
```

The project currently evaluates multiple face recognition models under a
common experimental pipeline and produces quantitative metrics, fairness
measurements, plots, comparison tables, rankings, and summary reports.

> **Important:** Fairness-FR is an academic/research evaluation framework.
> The reported fairness measurements are specific to the selected dataset,
> demographic groups, pair-generation protocol, thresholds, and evaluation
> methodology. They should not be interpreted as universal evidence of
> fairness across all populations or deployment conditions.

---

# 🎯 Project Aim

The primary aim of Fairness-FR is to develop a reproducible framework that can
evaluate face recognition systems using both:

```text
Recognition Performance
        +
Demographic Fairness
```

Instead of asking only:

> "Which model is the most accurate?"

the framework also asks:

> "Does the model behave consistently across the demographic groups being
> evaluated?"

This enables a more comprehensive comparison of face recognition systems.

---

# 🚀 Key Contributions

The project provides:

- 🔍 End-to-end face recognition evaluation pipeline
- 🧹 Dataset preprocessing
- 👥 Genuine and impostor pair generation
- 🧠 Face embedding extraction
- 📊 Similarity-score calculation
- 🎚️ Threshold analysis
- 📈 Biometric performance evaluation
- ⚖️ Demographic fairness evaluation
- 👤 Group-level performance analysis
- 📉 FAR / FRR / TAR / EER analysis
- 📊 ROC and DET curves
- 🔥 Metric heatmaps
- 📦 Score-distribution analysis
- 🏆 Multi-model comparison
- 🥇 Automatic model ranking
- 📑 CSV, JSON, and text report generation
- 📊 Nine model-comparison visualizations
- 🖥️ Interactive Streamlit dashboard
- 📱 Dashboard designed for browser-based demonstration
- ⚙️ Configuration-driven experiment management

---

# 🏗️ System Architecture

The complete Fairness-FR architecture is:

```text
                         ┌─────────────────────┐
                         │       Dataset       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Dataset Preprocessing│
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Pair Generation     │
                         │ Genuine / Impostor  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Embedding Extraction│
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Similarity Scoring  │
                         └──────────┬──────────┘
                                    │
                         ┌──────────┴──────────┐
                         ▼                     ▼
              ┌──────────────────┐   ┌──────────────────┐
              │ Performance      │   │ Fairness         │
              │ Evaluation       │   │ Evaluation       │
              └────────┬─────────┘   └────────┬─────────┘
                       │                      │
                       └──────────┬───────────┘
                                  ▼
                       ┌──────────────────────┐
                       │ Model Comparison     │
                       └──────────┬───────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │ Model Ranking        │
                       └──────────┬───────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │ Results & Plots      │
                       └──────────┬───────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │ Streamlit Dashboard  │
                       └──────────────────────┘
```

---

# 🧩 Architecture Components

| Layer | Function |
|---|---|
| Dataset Layer | Provides face images and demographic information |
| Preprocessing Layer | Prepares the dataset for evaluation |
| Pairing Layer | Generates genuine and impostor verification pairs |
| Model Layer | Extracts face embeddings |
| Scoring Layer | Calculates similarity scores |
| Performance Layer | Calculates biometric performance metrics |
| Fairness Layer | Evaluates demographic-group behaviour |
| Comparison Layer | Compares multiple evaluated models |
| Visualization Layer | Generates plots and charts |
| GUI Layer | Provides an interactive Streamlit interface |

---

# 🧠 Evaluation Philosophy

Fairness-FR evaluates face recognition systems across two major dimensions.

## 1. Recognition Performance

This measures how accurately the system performs face verification.

The framework evaluates:

- Accuracy
- Precision
- Recall
- F1-score
- FAR
- FRR
- FMR
- FNMR
- TAR
- TNR
- EER

---

## 2. Demographic Fairness

This evaluates whether the recognition behaviour differs across demographic
groups.

The framework analyses:

- Group-level accuracy
- Group-level FAR
- Group-level FRR
- Group-level TAR
- Group-level EER
- Score distributions
- Group-level metric differences
- Demographic disparity

---

# 📊 Evaluation Pipeline

The complete experimental pipeline consists of the following stages:

```text
1. Preprocessing
       ↓
2. Pair Generation
       ↓
3. Embedding Extraction
       ↓
4. Similarity Scoring
       ↓
5. Performance Evaluation
       ↓
6. Fairness Evaluation
       ↓
7. Model Comparison
```

Each stage produces outputs that can be consumed by subsequent stages.

---

# 📁 Dataset Configuration

Datasets are configured using YAML files:

```text
configs/
└── datasets/
    ├── bfw.yaml
    └── rfw.yaml
```

The dataset configuration controls how the corresponding dataset is loaded
and interpreted by the evaluation framework.

The current generated model-comparison experiment is based on the available
BFW evaluation results.

---

# 🤖 Face Recognition Models

The framework supports configuration of multiple face recognition models.

Currently configured models include:

```text
ArcFace
FaceNet512
GhostFaceNet
MobileFaceNet
SFace
```

Model configuration files are stored in:

```text
configs/
└── models/
    ├── arcface.yaml
    ├── facenet.yaml
    ├── ghostfacenet.yaml
    ├── mobilefacenet.yaml
    └── sface.yaml
```

The completed four-model comparison currently contains:

```text
ArcFace
FaceNet512
GhostFaceNet
SFace
```

---

# 👥 Genuine and Impostor Pairs

Face verification requires comparing pairs of face images.

## Genuine Pair

A genuine pair contains two images belonging to the same identity.

```text
Identity A
 ├── Image 1
 └── Image 2
       ↓
  Genuine Pair
```

The expected decision is:

```text
ACCEPT
```

---

## Impostor Pair

An impostor pair contains images belonging to different identities.

```text
Identity A ── Image 1
                  │
                  ▼
            Impostor Pair
                  ▲
                  │
Identity B ── Image 2
```

The expected decision is:

```text
REJECT
```

---

# 🧠 Face Embedding Extraction

Each face image is converted into a numerical representation called a
**face embedding**.

```text
Face Image
     ↓
Face Recognition Model
     ↓
Embedding Vector
```

Embeddings are then compared to calculate similarity scores.

The implementation is located under:

```text
src/fairness_fr/models/
```

---

# 📐 Similarity Scoring

After extracting embeddings, the framework calculates similarity scores for
the generated genuine and impostor pairs.

Conceptually:

```text
Image A
   ↓
Embedding A
   │
   ├──── Similarity Score ────► Verification Decision
   │
   ↓
Embedding B
   ↑
Image B
```

The similarity score is compared against a decision threshold.

---

# 🎚️ Threshold Analysis

The decision threshold controls whether a pair is accepted or rejected.

```text
Similarity Score ≥ Threshold
            ↓
          ACCEPT


Similarity Score < Threshold
            ↓
          REJECT
```

Changing the threshold affects:

- FAR
- FRR
- TAR
- EER
- Verification operating point

Threshold configuration is controlled through:

```text
configs/thresholds.yaml
```

The framework generates:

```text
threshold_analysis.csv
```

and corresponding threshold plots.

---

# 📈 Biometric Performance Metrics

| Metric | Description |
|---|---|
| Accuracy | Overall percentage of correct decisions |
| Precision | Correct positive predictions among all positive predictions |
| Recall | Genuine cases correctly identified |
| F1-score | Harmonic combination of precision and recall |
| FAR | False Acceptance Rate |
| FRR | False Rejection Rate |
| FMR | False Match Rate |
| FNMR | False Non-Match Rate |
| TAR | True Acceptance Rate |
| TNR | True Negative Rate |
| EER | Equal Error Rate |

---

# 🚨 FAR and FRR

## False Acceptance Rate — FAR

FAR measures the proportion of impostor attempts that are incorrectly
accepted.

```text
Lower FAR
    ↓
Fewer impostor acceptances
```

---

## False Rejection Rate — FRR

FRR measures the proportion of genuine attempts that are incorrectly rejected.

```text
Lower FRR
    ↓
Fewer genuine-user rejections
```

---

# ✅ True Acceptance Rate — TAR

TAR represents the percentage of genuine attempts correctly accepted.

```text
Higher TAR
    ↓
Better genuine-user recognition
```

---

# ⚖️ Equal Error Rate — EER

EER is the operating point at which false acceptance and false rejection
errors are approximately equal.

For comparative evaluation:

```text
Lower EER
    ↓
Better verification performance
```

---

# ⚖️ Demographic Fairness Evaluation

The fairness stage examines whether the model behaves differently across the
demographic groups represented in the dataset.

The framework evaluates:

```text
Group FAR
Group FRR
Group TAR
Group EER
Group Score Distribution
Group Metric Differences
Mean Demographic Disparity
```

A lower measured disparity indicates smaller differences according to the
implemented disparity calculation.

However:

> A low disparity value should not be interpreted as proof of universal
> fairness. It represents the measured behaviour under the selected dataset
> and experimental protocol.

---

# 📊 Fairness Visualizations

The framework generates group-level visualizations such as:

```text
group_far_by_group.png
group_frr_by_group.png
group_tar_by_group.png
group_metric_heatmap.png
group_score_boxplot.png
group_score_distribution.png
```

These visualizations help identify demographic differences that may not be
visible from overall performance metrics.

---

# 📈 Performance Visualizations

For evaluated models, the framework generates:

```text
Confusion Matrix
ROC Curve
DET Curve
Score Distribution
Threshold vs FAR
Threshold vs FRR
Threshold vs TAR
```

Examples:

```text
test_confusion_matrix.png
test_det_curve.png
test_roc_curve.png
test_score_distribution.png
test_threshold_vs_far.png
test_threshold_vs_frr.png
test_threshold_vs_tar.png
```

Corresponding training and validation plots are also generated where
applicable.

---

# 📉 ROC Curve

The ROC curve shows the relationship between true-positive and false-positive
behaviour across different decision thresholds.

It provides a visual representation of verification performance as the
threshold changes.

---

# 📉 DET Curve

The DET curve visualizes the relationship between false acceptance and false
rejection behaviour over different operating points.

It is particularly useful for biometric system evaluation.

---

# 📦 Score Distribution

Score-distribution plots show how similarity scores are distributed.

They help examine:

- Genuine/impostor separation
- Distribution overlap
- Threshold sensitivity
- Group-level score differences

---

# 🏆 Multi-Model Comparison

Once multiple models have completed evaluation, Fairness-FR can compare their
results.

Run:

```powershell
python .\run_model_comparison.py
```

The comparison results are stored in:

```text
results/model_comparison/
```

Generated files include:

```text
model_comparison.csv
model_comparison.json
model_rankings.csv
model_summary.txt
```

---

# 📊 Model Comparison Plots

The comparison module generates nine plots:

```text
results/model_comparison/plots/model_comparison/
```

### Generated plots

```text
accuracy_comparison.png
eer_comparison.png
fairness_disparity_comparison.png
far_comparison.png
frr_comparison.png
metric_heatmap.png
overall_ranking.png
radar_chart.png
tar_comparison.png
```

These provide side-by-side comparisons of the evaluated models.

---

# 🧪 Current Experimental Results

The completed four-model comparison produced the following results:

| Model | Accuracy | Precision | Recall | F1 | FAR | FRR | TAR | EER | Mean Disparity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **FaceNet512** | **90.52%** | **90.61%** | **90.41%** | **90.51%** | **9.37%** | **9.59%** | **90.41%** | **9.59%** | 0.01771 |
| **ArcFace** | 88.67% | 88.67% | 88.67% | 88.67% | 11.33% | 11.33% | 88.67% | 11.33% | **0.00623** |
| **GhostFaceNet** | 85.84% | 85.84% | 85.84% | 85.84% | 14.16% | 14.16% | 85.84% | 14.16% | 0.00979 |
| **SFace** | 82.35% | 82.35% | 82.35% | 82.35% | 17.65% | 17.65% | 82.35% | 17.65% | 0.02456 |

---

# 🥇 FaceNet512

FaceNet512 achieved the strongest overall biometric recognition performance
in the current experiment.

```text
Accuracy   = 90.52%
Precision  = 90.61%
Recall     = 90.41%
F1-score   = 90.51%

FAR        = 9.37%
FRR        = 9.59%

TAR        = 90.41%
EER        = 9.59%
```

It achieved:

```text
Highest Accuracy
Highest Precision
Highest Recall
Highest F1-score
Lowest FAR
Lowest EER
Highest TAR
```

Therefore, FaceNet512 ranks first according to the current overall comparison
methodology.

---

# 🥈 ArcFace

ArcFace achieved:

```text
Accuracy       = 88.67%
TAR            = 88.67%
EER            = 11.33%
```

However, ArcFace achieved the lowest measured demographic disparity:

```text
Mean Disparity = 0.00623
```

Therefore:

```text
Strong overall performance
        +
Lowest measured demographic disparity
```

makes ArcFace particularly important from the fairness perspective.

---

# 🥉 GhostFaceNet

GhostFaceNet achieved:

```text
Accuracy       = 85.84%
TAR            = 85.84%
EER            = 14.16%
Mean Disparity = 0.00979
```

It occupies the third position in the current multi-model ranking.

---

# 4️⃣ SFace

SFace achieved:

```text
Accuracy       = 82.35%
TAR            = 82.35%
EER            = 17.65%
Mean Disparity = 0.02456
```

Among the four evaluated models, SFace currently has:

```text
Lowest Accuracy
Highest FAR
Highest FRR
Highest EER
Highest Mean Disparity
```

under the current experimental configuration.

---

# 🏅 Overall Model Ranking

The final ranking produced by the comparison module is:

```text
🥇 1. FaceNet512
🥈 2. ArcFace
🥉 3. GhostFaceNet
   4. SFace
```

Ranking scores:

| Rank | Model | Overall Rank Score |
|---:|---|---:|
| 1 | FaceNet512 | 1.50 |
| 2 | ArcFace | 1.75 |
| 3 | GhostFaceNet | 2.75 |
| 4 | SFace | 4.00 |

The ranking combines multiple evaluation dimensions rather than relying
solely on accuracy.

---

# 🔬 Main Experimental Finding

The most important finding of the current experiment is that:

```text
Best Recognition Performance
             ↓
         FaceNet512


Lowest Measured Demographic Disparity
             ↓
           ArcFace
```

Therefore, the best-performing model and the model with the lowest measured
demographic disparity are **not the same model**.

This demonstrates why evaluating face recognition systems using only overall
accuracy can be insufficient.

A model-selection decision can instead consider:

```text
Recognition Performance
        +
Biometric Error Rates
        +
Demographic Behaviour
```

---

# 📋 Pipeline Execution

## Activate Virtual Environment

From the project root:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## Run Complete Pipeline

```powershell
python .\run_pipeline.py --all
```

---

## Run Individual Stages

### Preprocessing

```powershell
python .\run_pipeline.py --preprocess
```

### Pair Generation

```powershell
python .\run_pipeline.py --pairs
```

### Embedding Extraction

```powershell
python .\run_pipeline.py --embeddings
```

### Similarity Scores

```powershell
python .\run_pipeline.py --scores
```

### Performance Evaluation

```powershell
python .\run_pipeline.py --performance
```

### Fairness Evaluation

```powershell
python .\run_pipeline.py --fairness
```

### Model Comparison

After at least two models have complete evaluation results:

```powershell
python .\run_model_comparison.py
```

---

# 🖥️ Interactive Streamlit Dashboard

Fairness-FR includes an interactive Streamlit dashboard for presenting the
generated results.

From the project root:

```powershell
streamlit run .\app.py
```

The application normally starts at:

```text
http://localhost:8501
```

The dashboard can display:

- Dataset information
- Model information
- Performance metrics
- Fairness metrics
- Model comparison
- Model rankings
- ROC curves
- DET curves
- Score distributions
- Threshold analysis
- Fairness plots
- Comparison charts

---

# 📱 Mobile Demonstration

The Streamlit dashboard is browser-based and can be accessed from a phone.

For a phone connected to the **same Wi-Fi/network** as the computer running
the application:

```text
http://<COMPUTER-IP>:8501
```

For example:

```text
http://192.168.28.16:8501
```

For access from a completely different network, the application should be
deployed to a public hosting platform.

A public Streamlit deployment can provide a URL similar to:

```text
https://fairness-fr.streamlit.app
```

The public URL can then be opened from:

```text
Computer
Phone
Tablet
Another Wi-Fi network
Mobile data
```

---

# 📁 Project Structure

```text
Fairness-FR/
│
├── README.md
├── app.py
├── run_model_comparison.py
│
├── configs/
│   ├── datasets/
│   │   ├── bfw.yaml
│   │   └── rfw.yaml
│   │
│   ├── models/
│   │   ├── arcface.yaml
│   │   ├── facenet.yaml
│   │   ├── ghostfacenet.yaml
│   │   ├── mobilefacenet.yaml
│   │   └── sface.yaml
│   │
│   ├── experiment.yaml
│   ├── pairing.yaml
│   └── thresholds.yaml
│
├── results/
│   ├── bfw/
│   │   ├── arcface/
│   │   ├── facenet512/
│   │   ├── ghostfacenet/
│   │   └── sface/
│   │
│   └── model_comparison/
│       ├── model_comparison.csv
│       ├── model_comparison.json
│       ├── model_rankings.csv
│       ├── model_summary.txt
│       └── plots/
│
└── src/
    └── fairness_fr/
        │
        ├── config/
        │   ├── config.py
        │   ├── constants.py
        │   └── settings.py
        │
        ├── data/
        │   ├── generate_pairs.py
        │   └── preprocess.py
        │
        ├── evaluation/
        │   ├── calculate_scores.py
        │   ├── evaluate_fairness.py
        │   ├── evaluate_performance.py
        │   └── model_comparator.py
        │
        ├── models/
        │   ├── extract_embeddings.py
        │   └── extract_embeddings_backup.py
        │
        ├── gui/
        │   ├── __init__.py
        │   ├── __main__.py
        │   ├── app.py
        │   ├── components.py
        │   ├── data_loader.py
        │   ├── plots.py
        │   └── styles.py
        │
        └── utils/
            ├── logging.py
            └── utils.py
```

---

# ⚙️ Configuration

Fairness-FR uses YAML configuration files instead of hard-coding experiment
parameters throughout the source code.

```text
configs/
│
├── datasets/
│   ├── bfw.yaml
│   └── rfw.yaml
│
├── models/
│   ├── arcface.yaml
│   ├── facenet.yaml
│   ├── ghostfacenet.yaml
│   ├── mobilefacenet.yaml
│   └── sface.yaml
│
├── experiment.yaml
├── pairing.yaml
└── thresholds.yaml
```

This makes it easier to modify:

- Dataset settings
- Model settings
- Pair-generation strategy
- Threshold configuration
- Experiment configuration

without changing the core evaluation implementation.

---

# 📦 Generated Results

A typical model result directory contains outputs such as:

```text
results/bfw/<model>/
│
├── performance/
├── fairness/
├── plots/
│   ├── fairness/
│   ├── test_confusion_matrix.png
│   ├── test_det_curve.png
│   ├── test_roc_curve.png
│   ├── test_score_distribution.png
│   ├── test_threshold_vs_far.png
│   ├── test_threshold_vs_frr.png
│   └── test_threshold_vs_tar.png
│
├── roc_points.csv
├── scoring_log.csv
├── test_scores.csv
├── train_scores.csv
├── validation_scores.csv
└── threshold_analysis.csv
```

---

# 📊 Model Comparison Outputs

The model comparison directory contains:

```text
results/model_comparison/
│
├── model_comparison.csv
├── model_comparison.json
├── model_rankings.csv
├── model_summary.txt
│
└── plots/
    └── model_comparison/
        ├── accuracy_comparison.png
        ├── eer_comparison.png
        ├── fairness_disparity_comparison.png
        ├── far_comparison.png
        ├── frr_comparison.png
        ├── metric_heatmap.png
        ├── overall_ranking.png
        ├── radar_chart.png
        └── tar_comparison.png
```

---

# 🧰 Software Stack

| Layer | Technology |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Plotly |
| Configuration | YAML |
| GUI | Streamlit |
| Face Recognition | Configured recognition models |
| Experiment Management | YAML configuration |
| Version Control | Git |
| Repository | GitHub |

---

# 🔄 Reproducibility Workflow

A complete experiment can be reproduced using:

```text
1. Configure dataset
        ↓
2. Configure models
        ↓
3. Configure pair generation
        ↓
4. Configure thresholds
        ↓
5. Preprocess dataset
        ↓
6. Generate pairs
        ↓
7. Extract embeddings
        ↓
8. Calculate similarity scores
        ↓
9. Evaluate performance
        ↓
10. Evaluate fairness
        ↓
11. Compare models
        ↓
12. Generate plots and reports
        ↓
13. Open Streamlit dashboard
```

---

# 🧪 Current Pipeline Verification

The pipeline has been successfully executed with:

```text
Completed stages:
    preprocess
    pairs
    embeddings
    scores
    performance
    fairness

Failed stages:
    0
```

The multi-model comparison successfully loaded:

```text
ArcFace
FaceNet512
SFace
GhostFaceNet
```

and generated:

```text
9 model-comparison plots
model_comparison.csv
model_comparison.json
model_rankings.csv
model_summary.txt
```

---

# ⚠️ Important Interpretation Notes

The current results are experimental results obtained under the selected
evaluation configuration.

The following factors influence the results:

- Dataset composition
- Demographic group representation
- Image quality
- Pair-generation strategy
- Number of genuine pairs
- Number of impostor pairs
- Similarity metric
- Threshold selection
- Evaluation split
- Fairness metric implementation

Therefore:

> The results should be interpreted as measurements of model behaviour under
> the implemented experimental protocol rather than universal statements
> about the models.

---

# 🧯 Challenges and Engineering Issues

During development, several implementation issues were encountered.

### 1. Configuration Path Issues

The pipeline initially failed when a model configuration file was missing:

```text
FileNotFoundError:
configs/models/facenet512.yaml
```

The model configuration structure was subsequently aligned with the available
model YAML files.

---

### 2. GUI Package Naming Issues

The GUI initially contained filenames with unintended spaces such as:

```text
components .py
plots .py
__init__ .py
```

These were corrected to:

```text
components.py
plots.py
__init__.py
```

and:

```text
__main__.py
```

This resolved Python package import problems.

---

### 3. Streamlit Application Path

The dashboard was initially launched using an incorrect path.

The actual application entry point was located at:

```text
app.py
```

and can be launched using:

```powershell
streamlit run .\app.py
```

---

### 4. Model Comparison

The normal pipeline can skip the comparison stage if fewer than two models
are configured.

The independent comparison module was therefore used:

```powershell
python .\run_model_comparison.py
```

This successfully loaded four evaluated models.

---

### 5. Mobile Demonstration

Local Streamlit access through:

```text
localhost:8501
```

works only on the machine running Streamlit.

For other devices, the computer's network IP or a public deployment URL must
be used.

---

# 🔬 Scientific Interpretation

The current experiment demonstrates an important distinction between
**recognition performance** and **demographic fairness**.

FaceNet512 produced:

```text
Highest Accuracy
Lowest EER
Highest TAR
```

while ArcFace produced:

```text
Lowest Mean Demographic Disparity
```

Therefore:

```text
Highest Recognition Performance
              ≠
Lowest Demographic Disparity
```

This supports the motivation behind Fairness-FR: model evaluation should
consider multiple dimensions rather than relying on a single aggregate
performance metric.

---

# 🔮 Future Scope

Future development can extend the framework in several directions:

- Evaluate additional face recognition models
- Add additional demographic fairness metrics
- Support more datasets
- Perform cross-dataset evaluation
- Add confidence intervals
- Add statistical significance testing
- Add bootstrap-based uncertainty analysis
- Compare threshold-specific fairness
- Investigate group-specific FAR and FRR trade-offs
- Add calibration analysis
- Add automated experiment reports
- Add additional dashboard controls
- Improve mobile dashboard presentation
- Deploy the dashboard publicly
- Add experiment-history tracking
- Support larger-scale evaluation
- Investigate fairness-performance trade-offs

---

# 📌 Final Conclusion

Fairness-FR provides an end-to-end framework for evaluating face recognition
systems using both **biometric performance** and **demographic fairness**.

The current four-model experiment demonstrates:

```text
FaceNet512
→ Best overall recognition performance

ArcFace
→ Lowest measured demographic disparity

GhostFaceNet
→ Intermediate performance

SFace
→ Lowest overall recognition performance
```

The key conclusion is:

> **A face recognition model should not be evaluated solely on overall
> accuracy. Recognition performance, biometric error rates, and demographic
> behaviour should be considered together when assessing model suitability.**

The framework provides the complete pipeline required to move from dataset
preparation to model comparison and interactive visualization.

---

# 👨‍💻 Author

**Urnisa Rakshit**

B.Tech — Artificial Intelligence & Data Science

---

# ⭐ Project Highlights

```text
⚖️ Demographic Fairness
🤖 Multiple Face Recognition Models
📊 Biometric Performance Evaluation
📈 ROC / DET Analysis
🎚️ Threshold Analysis
👥 Group-Level Evaluation
🏆 Automatic Model Ranking
📉 Fairness Disparity Analysis
📑 CSV / JSON / TXT Reports
📊 9 Model Comparison Plots
🖥️ Interactive Streamlit Dashboard
📱 Browser-Based Demonstration
⚙️ Configuration-Driven Pipeline
```

---

# 📄 License

This repository is intended for academic and research purposes.

Please review and add an appropriate open-source license before redistributing
the project or its components.

---

# 🙌 Acknowledgement

This project was developed as an academic/research-oriented implementation
for studying the interaction between biometric recognition performance and
demographic fairness in face recognition systems.
