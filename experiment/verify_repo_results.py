"""
Verify the result tables published in the repo (table_T2 ... table_T11) against an
independent re-run of the ORIGINAL, unchanged pipeline (kbs_run_final.py).

The tables are recomputed from the re-run's kbs_results_raw.csv / kbs_timings.csv
exactly as in kbs_analysis.ipynb and make_hard_regime_tables.py (same groupings,
same per-run units, same tie band), then compared cell by cell with the CSVs in the
repo. Writes VERIFICATION_REPORT.md next to this script.

  python verify_repo_results.py <folder with kbs_results_raw.csv>
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', '..', 'repro_original')
M = ['None', 'SMOTE', 'Borderline-SMOTE', 'ADASYN', 'SMOTE-ENN', 'SMOTE-Tomek',
     'KMeans-SMOTE', 'Safe-Level-SMOTE', 'MWMOTE', 'ProWSyn', 'MIRT++']
HARD = ['cleveland', 'ecoli', 'flare', 'glass', 'winequality-red', 'winequality-white', 'yeast']
MEDICAL = ['ann-thyroid', 'cleveland', 'dermatology', 'hepatitis', 'new-thyroid']

raw = pd.read_csv(os.path.join(RUN, 'kbs_results_raw.csv'))
raw['resampler'] = raw['resampler'].fillna('None')
tim_path = os.path.join(RUN, 'kbs_timings.csv')
tim = pd.read_csv(tim_path) if os.path.exists(tim_path) else None
if tim is not None:
    tim['method'] = tim['method'].fillna('None')


def piv(df, metric):
    return df.pivot_table(index=['dataset', 'fold', 'clf'], columns='resampler', values=metric)[M].dropna()


def wilcoxon_tab(p):
    out = {}
    for m in M:
        if m != 'MIRT++':
            out[m] = stats.wilcoxon(p['MIRT++'], p[m], alternative='greater').pvalue
    return pd.Series(out)


rec = {}
rec['T2'] = raw.groupby('resampler')[['gmean', 'f1', 'auc']].mean().reindex(M)
rec['T3'] = raw.groupby(['dataset', 'resampler'])['gmean'].mean().unstack()[M]
rec['T4'] = pd.DataFrame({k: piv(raw, k).rank(axis=1, ascending=False).mean() for k in ['gmean', 'f1', 'auc']})
pg = piv(raw, 'gmean')
d = pg.sub(pg['MIRT++'], axis=0).mul(-1)                    # MIRT - other
rec['T5'] = pd.DataFrame({'Win': (d > 0.005).sum(), 'Tie': (d.abs() <= 0.005).sum(),
                          'Loss': (d < -0.005).sum()}).drop('MIRT++')
rec['T7'] = wilcoxon_tab(pg)
h = raw[raw.dataset.isin(HARD)]
ph = piv(h, 'gmean')
rec['T10'] = pd.DataFrame({'gmean': ph.mean(), 'rank': ph.rank(axis=1, ascending=False).mean(),
                           'p': wilcoxon_tab(ph)})
rec['T11'] = h.groupby(['dataset', 'resampler'])['gmean'].mean().unstack()[M]
gm = rec['T3']
rec['T9'] = pd.DataFrame({'MIRT': gm.loc[gm.index.intersection(MEDICAL), 'MIRT++'],
                          'SMOTE': gm.loc[gm.index.intersection(MEDICAL), 'SMOTE']})

pub = {
    'T2': pd.read_csv(os.path.join(HERE, 'table_T2_mean_performance.csv'), index_col=0),
    'T3': pd.read_csv(os.path.join(HERE, 'table_T3_per_dataset_gmean.csv'), index_col=0),
    'T4': pd.read_csv(os.path.join(HERE, 'table_T4_average_ranks.csv'), index_col=0),
    'T5': pd.read_csv(os.path.join(HERE, 'table_T5_win_tie_loss_gmean.csv'), index_col=0),
    'T7': pd.read_csv(os.path.join(HERE, 'table_T7_wilcoxon_gmean.csv'), index_col=0),
    'T9': pd.read_csv(os.path.join(HERE, 'table_T9_medical_roi.csv'), index_col=0),
    'T10': pd.read_csv(os.path.join(HERE, 'table_T10_hard_regime_summary.csv'), index_col=0),
    'T11': pd.read_csv(os.path.join(HERE, 'table_T11_hard_regime_perdataset.csv'), index_col=0),
}
for k in pub:
    pub[k].index = [('None' if (isinstance(i, float) and np.isnan(i)) or i in ('', 'NONE') else i)
                    for i in pub[k].index]


def compare(a, b, tol):
    """a, b: aligned DataFrames (recomputed, published) -> (n cells, n within tol, max |diff|)."""
    a, b = a.astype(float), b.astype(float)
    diff = (a - b).abs()
    return int(diff.size), int((diff <= tol).sum().sum()), float(np.nanmax(diff.values))


pairs = {
    'T2 mean G-mean/F1/AUC': (rec['T2'].loc[pub['T2'].index],
                              pub['T2'][['Mean G-Mean', 'Mean F1', 'Mean AUC']].set_axis(['gmean', 'f1', 'auc'], axis=1), 5e-4),
    'T3 per-dataset G-mean': (rec['T3'].loc[pub['T3'].index, pub['T3'].columns.str.replace('NONE', 'None')]
                              .set_axis(pub['T3'].columns, axis=1), pub['T3'], 5e-4),
    'T4 average ranks': (rec['T4'].loc[pub['T4'].index],
                         pub['T4'].set_axis(['gmean', 'f1', 'auc'], axis=1), 5e-3),
    'T5 win/tie/loss counts': (rec['T5'].loc[pub['T5'].index], pub['T5'], 0),
    'T7 Wilcoxon p': (rec['T7'].loc[pub['T7'].index].to_frame('p'),
                      pub['T7'][['p (MIRT++ > x)']].set_axis(['p'], axis=1), 5e-4),
    'T9 medical G-mean': (rec['T9'].loc[pub['T9'].index],
                          pub['T9'][['MIRT++ G-Mean', 'SMOTE G-Mean']].set_axis(['MIRT', 'SMOTE'], axis=1), 5e-4),
    'T10 hard-regime G-mean/rank': (rec['T10'].loc[pub['T10'].index, ['gmean', 'rank']],
                                    pub['T10'][['Mean G-Mean', 'Avg Rank (G-Mean)']].set_axis(['gmean', 'rank'], axis=1), 5e-3),
    'T11 hard-regime per-dataset': (rec['T11'].loc[pub['T11'].index], pub['T11'], 5e-4),
}

lines = ['# Verification of the result tables in the repository', '',
         f'Re-run of the unchanged `kbs_run_final.py` in `{os.path.abspath(RUN)}`; tables recomputed exactly',
         'as in `kbs_analysis.ipynb` / `make_hard_regime_tables.py` and compared with the CSVs in the repo.', '',
         '| Table | cells | within tolerance | max abs. difference |', '|---|---|---|---|']
for name, (a, b, tol) in pairs.items():
    try:
        n, ok, mx = compare(a, b, tol)
        lines.append(f'| {name} (tol {tol}) | {n} | {ok} ({ok / n:.0%}) | {mx:.4f} |')
    except Exception as e:
        lines.append(f'| {name} | - | comparison failed: {str(e)[:60]} | - |')
lines += ['', '## Datasets in the re-run', '', ', '.join(sorted(raw.dataset.unique())),
          '', f'{raw.dataset.nunique()} datasets, {len(raw)} rows.']
report = '\n'.join(lines)
open(os.path.join(HERE, 'VERIFICATION_REPORT.md'), 'w', encoding='utf-8').write(report)
print(report)
