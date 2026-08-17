results/bfw/sface/plots/fairness/group_far_by_group.png
results/bfw/sface/plots/fairness/group_frr_by_group.png
results/bfw/sface/plots/fairness/group_metric_heatmap.png
results/bfw/sface/plots/fairness/group_score_boxplot.png
results/bfw/sface/plots/fairness/group_score_distribution.png
results/bfw/sface/plots/fairness/group_tar_by_group.png
results/bfw/sface/plots/test_confusion_matrix.png
results/bfw/sface/plots/test_det_curve.png
results/bfw/sface/plots/test_roc_curve.png
results/bfw/sface/plots/test_score_distribution.png
results/bfw/sface/plots/test_threshold_vs_far.png
results/bfw/sface/plots/test_threshold_vs_frr.png
results/bfw/sface/plots/test_threshold_vs_tar.png
results/bfw/sface/plots/train_confusion_matrix.png
results/bfw/sface/plots/train_det_curve.png
results/bfw/sface/plots/train_roc_curve.png
results/bfw/sface/plots/train_score_distribution.png
results/bfw/sface/plots/train_threshold_vs_far.png
results/bfw/sface/plots/train_threshold_vs_frr.png
results/bfw/sface/plots/train_threshold_vs_tar.png
results/bfw/sface/plots/validation_confusion_matrix.png
results/bfw/sface/plots/validation_det_curve.png
results/bfw/sface/plots/validation_roc_curve.png
results/bfw/sface/plots/validation_score_distribution.png
results/bfw/sface/plots/validation_threshold_vs_far.png
results/bfw/sface/plots/validation_threshold_vs_frr.png
results/bfw/sface/plots/validation_threshold_vs_tar.png
results/bfw/sface/roc_points.csv
results/bfw/sface/scoring_log.csv
results/bfw/sface/test_scores.csv
results/bfw/sface/threshold_analysis.csv
results/bfw/sface/train_scores.csv
results/bfw/sface/validation_scores.csv
results/model_comparison/model_comparison.csv
results/model_comparison/model_comparison.json
results/model_comparison/model_rankings.csv
results/model_comparison/model_summary.txt
results/model_comparison/plots/model_comparison/accuracy_comparison.png
results/model_comparison/plots/model_comparison/eer_comparison.png
results/model_comparison/plots/model_comparison/fairness_disparity_comparison.png
results/model_comparison/plots/model_comparison/far_comparison.png
results/model_comparison/plots/model_comparison/frr_comparison.png
results/model_comparison/plots/model_comparison/metric_heatmap.png
results/model_comparison/plots/model_comparison/overall_ranking.png
results/model_comparison/plots/model_comparison/radar_chart.png
results/model_comparison/plots/model_comparison/tar_comparison.png
src/fairness_fr/__init__.py
src/fairness_fr/config/__init__.py
src/fairness_fr/config/config.py
src/fairness_fr/config/constants.py
src/fairness_fr/config/settings.py
src/fairness_fr/data/__init__.py
src/fairness_fr/data/generate_pairs.py
src/fairness_fr/data/preprocess.py
src/fairness_fr/evaluation/__init__.py
src/fairness_fr/evaluation/calculate_scores.py
src/fairness_fr/evaluation/evaluate_fairness.py
src/fairness_fr/evaluation/evaluate_performance.py
src/fairness_fr/evaluation/model_comparator.py
src/fairness_fr/gui/__init__.py
src/fairness_fr/gui/__main__ .py
src/fairness_fr/gui/app.py
src/fairness_fr/gui/components.py
src/fairness_fr/gui/data_loader.py
src/fairness_fr/gui/plots.py
src/fairness_fr/gui/styles.py
src/fairness_fr/improvement_week7.py
src/fairness_fr/models/__init__.py
src/fairness_fr/models/extract_embeddings.py
src/fairness_fr/models/extract_embeddings_backup.py
src/fairness_fr/utils/__init__.py
src/fairness_fr/utils/logging.py
src/fairness_fr/utils/utils.py
(.venv) PS C:\Users\Rakshit\Downloads\drdo\deployment> (git ls-tree -r --name-only HEAD | Measure-Object -Line).Lines
270
(.venv) PS C:\Users\Rakshit\Downloads\drdo\deployment> git ls-tree -r --name-only HEAD | Select-String "gui|app.py|results/model_comparison|configs"

app.py
configs/datasets/bfw.yaml
configs/datasets/rfw.yaml
configs/experiment.yaml
configs/models/arcface.yaml
configs/models/facenet.yaml
configs/models/ghostfacenet.yaml
configs/models/mobilefacenet.yaml
configs/models/sface.yaml
configs/pairing.yaml
configs/thresholds.yaml
results/model_comparison/model_comparison.csv
results/model_comparison/model_comparison.json
results/model_comparison/model_rankings.csv
results/model_comparison/model_summary.txt
results/model_comparison/plots/model_comparison/accuracy_compari
son.png
results/model_comparison/plots/model_comparison/eer_comparison.p
ng
results/model_comparison/plots/model_comparison/fairness_dispari
ty_comparison.png
results/model_comparison/plots/model_comparison/far_comparison.p
ng
results/model_comparison/plots/model_comparison/frr_comparison.p
ng
results/model_comparison/plots/model_comparison/metric_heatmap.p
ng
results/model_comparison/plots/model_comparison/overall_ranking.
png
results/model_comparison/plots/model_comparison/radar_chart.png
results/model_comparison/plots/model_comparison/tar_comparison.p
ng
src/fairness_fr/gui/__init__.py
src/fairness_fr/gui/__main__ .py
src/fairness_fr/gui/app.py
src/fairness_fr/gui/components.py
src/fairness_fr/gui/data_loader.py
src/fairness_fr/gui/plots.py
src/fairness_fr/gui/styles.py


(.venv) PS C:\Users\Rakshit\Downloads\drdo\deployment> Rename-Item ".\src\fairness_fr\gui\__main__ .py" "__main__.py"
(.venv) PS C:\Users\Rakshit\Downloads\drdo\deployment> git add .
warning: in the working copy of 'src/fairness_fr/gui/__main__.py', LF will be replaced by CRLF the next time Git touches it
(.venv) PS C:\Users\Rakshit\Downloads\drdo\deployment> git commit -m "Fix GUI filenames"
[main f4fd1dc] Fix GUI filenames
 1 file changed, 0 insertions(+), 0 deletions(-)
 rename src/fairness_fr/gui/{__main__ .py => __main__.py} (100%)
(.venv) PS C:\Users\Rakshit\Downloads\drdo\deployment> git push
Enumerating objects: 9, done.
Counting objects: 100% (9/9), done.
Delta compression using up to 16 threads
Compressing objects: 100% (4/4), done.
Writing objects: 100% (5/5), 399 bytes | 399.00 KiB/s, done.
Total 5 (delta 3), reused 0 (delta 0), pack-reused 0 (from 0)
remote: Resolving deltas: 100% (3/3), completed with 3 local objects.
To https://github.com/rakshit8904-lang/Fairness-FR.git
   84dd866..f4fd1dc  main -> main
(.venv) PS C:\Users\Rakshit\Downloads\drdo\deployment> 
