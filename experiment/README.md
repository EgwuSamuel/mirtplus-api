# MIRT — Experiment & Reproducibility Package

Code, benchmark data references, result tables, and figures for the manuscript
**"MIRT: A Similarity-Modulated, Difficulty-Aware Resampling Technique for
Multiclass Imbalanced Classification"** (preliminary evaluation; the scripts of the
current, corrected benchmark are described in the top-level README).

MIRT++ (`mirtplus.py`) is an `imbalanced-learn`-style resampler:

```python
from mirtplus import MIRTPlusResampler
X_res, y_res = MIRTPlusResampler().fit_resample(X, y)
```

## Datasets

All benchmark datasets are **public** and downloaded on demand from
[OpenML](https://www.openml.org) via `sklearn.datasets.fetch_openml`; nothing is
redistributed here. The loaders (`load_*`) and the full registry live in
[`kbs_run_final.py`](kbs_run_final.py) (`DATASETS` dict). OpenML identifiers used:

| Dataset | OpenML | Dataset | OpenML |
|---|---|---|---|
| new-thyroid | `data_id=40` | segment | `name='segment', v1` |
| hayes-roth | `data_id=9960` | satimage | `name='satimage', v1` |
| balance-scale | `data_id=11` | vowel | `name='vowel', v2` |
| car | `data_id=21` | abalone | `name='abalone', v1` |
| cmc | `data_id=23` | ann-thyroid | `data_id=40474` |
| ecoli | `data_id=39` | winequality-white | `name='wine-quality-white', v1` |
| glass | `data_id=41` | hepatitis | `data_id=55` |
| wine | `data_id=187` | splice | `name='splice', v1` |
| dermatology | `data_id=35` | vehicle | `name='vehicle', v1` |
| yeast | `data_id=181` | cleveland | `data_id=179` (5-class) |
| winequality-red | `data_id=40691` | flare / pendigits | see `load_flare` / `load_pendigits` |

Three datasets not conveniently on OpenML are provided directly under
[`test_datasets/`](test_datasets/): `page-blocks.csv`, `shuttle.csv`,
`steel-plates-fault.csv`. Rare classes (fewer than 12 samples) are dropped by the
`_filter` helper so every class survives stratified cross-validation.

## How to reproduce

Requires Python 3.9+, `numpy`, `pandas`, `scikit-learn`, `imbalanced-learn`,
`smote-variants`, `scipy`, `matplotlib`.

```bash
# 1. Run the full benchmark (11 resamplers x datasets x seeds).
#    Writes raw per-fold results to results/kbs_results_raw.csv (+ timings).
python kbs_run_final.py

# 2. Ablation study (MIRT++ component removals) -> table_T12_ablation.csv
python run_ablation.py

# 3. Similarity-vs-confusability validation -> table_T6_confusability.csv
python compute_confusability.py

# 4. Hard-regime (IR>=8 and k>=5) tables -> table_T10 / table_T11
python make_hard_regime_tables.py

# 5. Regenerate all manuscript figures (fig_F*.png)
python regen_figures.py
```

`kbs_run_final.py` checkpoints to `results/kbs_results_raw.csv`; delete that file
to force a clean re-run. Set `KBS_OUTPUT_DIR` to redirect outputs (e.g. on Colab).

The 11 methods compared: None (baseline), SMOTE, Borderline-SMOTE, ADASYN,
SMOTE-ENN, SMOTE-Tomek, KMeans-SMOTE, Safe-Level-SMOTE, MWMOTE, ProWSyn, MIRT++.

## Result tables (as cited in the manuscript)

| File | Content |
|---|---|
| `table_T1_dataset_characteristics.csv` | n, features, classes, imbalance ratio |
| `table_T2_mean_performance.csv` | mean performance per method |
| `table_T3_per_dataset_gmean.csv` | per-dataset G-mean |
| `table_T4_average_ranks.csv` | average ranks |
| `table_T5_win_tie_loss_gmean.csv` | win/tie/loss vs MIRT++ |
| `table_T6_confusability.csv` | ESDA similarity vs empirical confusability |
| `table_T7_wilcoxon_gmean.csv` | Wilcoxon signed-rank tests |
| `table_T8_timing.csv` | wall-clock timing |
| `table_T9_medical_roi.csv` | medical case-study deltas |
| `table_T10_hard_regime_summary.csv` | hard-regime summary (main result) |
| `table_T11_hard_regime_perdataset.csv` | hard-regime per dataset |
| `table_T12_ablation.csv` | component ablation |

`ablation_raw.csv` holds the raw ablation runs; `kbs_analysis.ipynb` is the
end-to-end analysis notebook.

## Honest-results note

Across all datasets, MIRT++ is **not** significantly better than plain SMOTE
overall (Wilcoxon p = 0.148) and is substantially more expensive. Its advantage
is concentrated in the **hard regime** (imbalance ratio >= 8 and >= 5 classes),
where it ranks first among the 11 methods (tables T10/T11). See the manuscript
for the full discussion.
