"""
Validate ESDA: correlate the MIRT++ class-similarity matrix mu with a
classifier's empirical (cross-validated) confusability, per dataset.
Produces table_T6_confusability.csv. Numbers feed Method Section 3.3.
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from mirtplus import compute_enhanced_similarity, similarity_vs_confusability
from kbs_run_final import DATASETS

rows = []
for name, loader in DATASETS.items():
    try:
        X, y = loader()
        X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        mu, classes, _ = compute_enhanced_similarity(X, y)
        r_p, r_s = similarity_vs_confusability(X, y, mu, classes)
        k = len(classes)
        rows.append({'dataset': name, 'k': k, 'pearson_r': r_p, 'spearman_r': r_s,
                     'meaningful': k >= 3})   # k=2 gives a degenerate +/-1 correlation
        print(f'  {name:<20s} k={k:<2d} pearson={r_p:+.3f} spearman={r_s:+.3f}'
              f'{"" if k>=3 else "  (k=2, excluded)"}', flush=True)
        pd.DataFrame(rows).to_csv('table_T6_confusability.csv', index=False)  # incremental
    except Exception as e:
        print(f'  {name:<20s} FAIL {str(e)[:70]}', flush=True)

df = pd.DataFrame(rows)
df.to_csv('table_T6_confusability.csv', index=False)
val = df[(df['meaningful']) & df['pearson_r'].notna()]
print('\n=== SUMMARY (ESDA similarity vs empirical confusability, k>=3 only) ===')
print(f'multiclass datasets with a defined correlation: {len(val)}')
print(f'mean Pearson r  = {val.pearson_r.mean():+.3f}  (median {val.pearson_r.median():+.3f})')
print(f'mean Spearman r = {val.spearman_r.mean():+.3f}  (median {val.spearman_r.median():+.3f})')
print(f'positive Pearson: {(val.pearson_r>0).sum()}/{len(val)}   '
      f'Pearson>=0.3: {(val.pearson_r>=0.3).sum()}/{len(val)}')
print('saved table_T6_confusability.csv')
