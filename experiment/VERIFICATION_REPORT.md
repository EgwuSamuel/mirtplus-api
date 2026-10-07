# Verification of the result tables in the repository

Re-run of the unchanged `kbs_run_final.py` in `C:\Users\USER\Desktop\MIRT Final\repro_original`; tables recomputed exactly
as in `kbs_analysis.ipynb` / `make_hard_regime_tables.py` and compared with the CSVs in the repo.

| Table | cells | within tolerance | max abs. difference |
|---|---|---|---|
| T2 mean G-mean/F1/AUC (tol 0.0005) | 33 | 23 (70%) | 0.0032 |
| T3 per-dataset G-mean (tol 0.0005) | 253 | 185 (73%) | 0.0327 |
| T4 average ranks (tol 0.005) | 33 | 2 (6%) | 0.1983 |
| T5 win/tie/loss counts (tol 0) | 30 | 2 (7%) | 7.0000 |
| T7 Wilcoxon p (tol 0.0005) | 10 | 6 (60%) | 0.0329 |
| T9 medical G-mean (tol 0.0005) | 10 | 9 (90%) | 0.0006 |
| T10 hard-regime G-mean/rank (tol 0.005) | 22 | 12 (55%) | 0.2334 |
| T11 hard-regime per-dataset (tol 0.0005) | 77 | 61 (79%) | 0.0059 |

## Datasets in the re-run

abalone, ann-thyroid, balance-scale, car, cleveland, cmc, dermatology, ecoli, flare, glass, hayes-roth, hepatitis, new-thyroid, pendigits, satimage, segment, splice, vehicle, vowel, wine, winequality-red, winequality-white, yeast

23 datasets, 7260 rows.
## Interpretation

**The published tables T2–T11 reproduce in substance but not to the last digit.** The re-run used today's
libraries (scikit-learn 1.9.1, imbalanced-learn 0.14.2, NumPy 2.5); the original run used earlier versions,
which changes k-means, random-forest and SVM internals slightly. Per-dataset G-means differ by at most 0.033
(cmc; all other datasets ≤ 0.013), and the headline conclusions of the preliminary evaluation are reproduced:

| Quantity (preliminary protocol, fold-level units) | Published | Re-run |
|---|---|---|
| Best average rank on G-mean (T4) | MIRT 5.350 | MIRT 5.311 |
| Runner-up | SMOTE 5.359 | SMOTE 5.328 |
| Wilcoxon MIRT > SMOTE (T7) | p = 0.148 | p = 0.145 |
| Wilcoxon MIRT > Borderline-SMOTE | p = 0.004 | p = 0.001 |
| Wilcoxon MIRT > SMOTE-Tomek | p = 0.013 | p = 0.028 |
| Wilcoxon MIRT > ProWSyn | p = 0.050 | p = 0.083 |

Rank-based quantities (T4, T5, T10) differ more in the third decimal because fold-level ranks among 11
near-tied methods flip with tiny score changes; the win/tie/loss counts in T5 use a 0.005 band on
individual runs and move by up to 7.

**Issues of the preliminary evaluation that are independent of reproducibility** (all corrected in the
benchmark used for the JIIS manuscript):
1. Three datasets are not what their names say: OpenML id 40 is *sonar* (2 classes, used as "new-thyroid"),
   1565 is *heart-h* (used as "cleveland"), 40474 is *thyroid-allbp* (used as "ann-thyroid").
2. Table 1 (T1) is typed by hand in `kbs_analysis.ipynb` and does not match the loaded data.
3. `results/kbs_gmean_by_dataset.csv` comes from a different run than Table 3 (120 of 253 cells differ;
   it contains page-blocks, which Table 3 does not).
4. Friedman, Nemenyi and Wilcoxon tests use (dataset × fold × classifier) rows as independent samples.
5. `geometric_mean_score(average='macro')` is the one-vs-rest G-mean, not the geometric mean of class
   recalls described in the manuscript.
6. Four datasets are balanced (IR ≤ 1.1), and hepatitis is binary.
