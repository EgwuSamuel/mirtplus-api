# MIRT — Multiclass Informative Resampling Technique

Code, data references and complete results for the article
**"MIRT: A Similarity-Modulated, Difficulty-Aware Resampling Technique for Multiclass Imbalanced
Classification"** (D.A. Dako, S.O. Egwu; submitted to the
*Journal of Intelligent Information Systems*).

```python
from mirtplus import MIRTPlusResampler          # MIRT, imbalanced-learn style
X_res, y_res = MIRTPlusResampler().fit_resample(X, y)
```

## Reproducing the paper

Python 3.13, `numpy`, `scipy`, `pandas`, `scikit-learn 1.9`, `imbalanced-learn 0.14`, `lightgbm 4.7`,
`smote-variants 1.0.1`, `baycomp`, `knnor`, `matplotlib`.

| Step | Command | Output |
|---|---|---|
| 1. Benchmark (32 datasets × 18 methods × 5 classifiers, 5×2 CV) | `python run_extended_benchmark.py --split eval --only <18 methods> --jobs 3` | `results_ext/` (one CSV per fold) |
| 2. Ablation of MIRT | `python run_ablation_corrected.py` | `results_ablation/` |
| 3. ESDA vs. confusion | `python esda_validity_corrected.py` | `results_ext/tables/T_esda_confusability.csv` |
| 4. Statistics | `python analyse_extended.py --proposed MIRT-v1` | `results_ext/tables/` |
| 5. Every table, figure and number of the paper | `cd manuscript && python make_assets.py` | `tab_*.tex`, `Fig*.pdf`, `numbers.tex` |

Datasets are downloaded from OpenML **by numeric identifier** (`benchmark_datasets.py`); names on OpenML are
ambiguous (e.g. id 40 is *sonar*, not new-thyroid). Characteristics in Table 1 are computed from the loaded
data. In cardiotocography (id 1466) the attributes V26–V35 are a one-hot copy of the class label and are removed.

## Earlier evaluation (kept for transparency)
`kbs_run_final.py` and the `table_T*.csv` files are the scripts and outputs of a preliminary evaluation. Its
dataset registry contained three wrong OpenML identifiers (id 40 = sonar used as new-thyroid, id 1565 = heart-h
as cleveland, id 40474 = thyroid-allbp as ann-thyroid), Table 1 was typed by hand, and its significance tests
used folds rather than datasets as the unit. `VERIFICATION_REPORT.md` documents a re-run of that pipeline.
The paper uses only the corrected benchmark above.

## Third-party code
- `third_party/multi_imbalance/`: subset of *multi-imbalance* (MIT licence, © Horna, Pluciński, Klimczak, Grycza).
- MC-CCR: the authors' code (github.com/michalkoziarski/MC-CCR) has **no licence file**, so it is **not
  redistributed**. Download `algorithms.py` from that repository into `third_party/mc_ccr_algorithms.py`.
- MC-RBO (vectorised) and SWIM-Maha are re-implemented in `baselines.py`; `validate_mcrbo.py` compares MC-RBO
  with the authors' code (github.com/michalkoziarski/MultiClassRBO; save its `algorithms.py` as
  `third_party/mc_rbo_algorithms.py` to run the check).
- KNNOR: the `knnor` package (MIT).

