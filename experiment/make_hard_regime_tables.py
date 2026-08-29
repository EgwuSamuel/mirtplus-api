"""
Build drop-in tables for the 'hard regime' (high imbalance AND many classes:
IR >= 8 and k >= 5). Emits:
  table_T10_hard_regime_summary.csv   -- per-method mean G-Mean, mean F1, ranks, Wilcoxon vs MIRT++
  table_T11_hard_regime_perdataset.csv -- per-dataset G-Mean for the 7 hard datasets
Every number traces to kbs_results_raw.csv.
"""
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

df = pd.read_csv('kbs_results_raw.csv')
df['resampler'] = df['resampler'].fillna('None')

METHODS = ['None','SMOTE','Borderline-SMOTE','ADASYN','SMOTE-ENN','SMOTE-Tomek',
           'KMeans-SMOTE','Safe-Level-SMOTE','MWMOTE','ProWSyn','MIRT++']
CHARS = {  # (IR, k) from table_T1
 'segment':(1.0,7),'vowel':(1.0,11),'vehicle':(1.1,4),'pendigits':(1.1,10),
 'wine':(1.5,3),'splice':(1.8,3),'cmc':(1.9,3),'hayes-roth':(2.1,3),
 'satimage':(2.4,6),'hepatitis':(3.8,2),'new-thyroid':(5.0,3),'dermatology':(5.6,6),
 'balance-scale':(5.9,3),'glass':(8.4,6),'abalone':(9.7,3),'cleveland':(12.5,5),
 'car':(18.6,4),'yeast':(23.1,8),'flare':(23.3,6),'ecoli':(39.1,5),
 'ann-thyroid':(45.8,3),'winequality-red':(68.1,6),'winequality-white':(109.9,6)}

HARD = sorted([d for d,(ir,k) in CHARS.items() if ir >= 8.0 and k >= 5])
d = df[df['dataset'].isin(HARD)]

def stars(p):
    return '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.'

# ---- T10: summary table ----
rows = []
for metric in ['gmean','f1']:
    piv = d.pivot_table(index=['dataset','fold','clf'], columns='resampler',
                        values=metric)[METHODS].dropna()
    means = piv.mean()
    ranks = piv.rank(axis=1, ascending=False).mean()
    if metric == 'gmean':
        rank_g, mean_g, piv_g = ranks, means, piv
    else:
        rank_f, mean_f = ranks, means

summary = []
for m in METHODS:
    if m == 'MIRT++':
        p_txt, sig = '-', '-'
    else:
        w, p = wilcoxon(piv_g['MIRT++'], piv_g[m], alternative='greater')
        p_txt, sig = f'{p:.4f}', stars(p)
    summary.append({
        'Method': m,
        'Mean G-Mean': round(mean_g[m], 4),
        'Mean F1': round(mean_f[m], 4),
        'Avg Rank (G-Mean)': round(rank_g[m], 3),
        'Avg Rank (F1)': round(rank_f[m], 3),
        'Wilcoxon p (MIRT++ > x)': p_txt,
        'Sig': sig,
    })
T10 = pd.DataFrame(summary).sort_values('Avg Rank (G-Mean)').reset_index(drop=True)
T10.to_csv('table_T10_hard_regime_summary.csv', index=False)

# ---- T11: per-dataset G-Mean on the 7 hard datasets ----
T11 = (d.groupby(['dataset','resampler'])['gmean'].mean()
         .unstack()[METHODS].round(3).loc[HARD])
T11.to_csv('table_T11_hard_regime_perdataset.csv')

print(f'Hard regime = {HARD}')
print(f'{len(HARD)} datasets, {len(piv_g)} paired observations\n')
print('=== T10 summary ===')
print(T10.to_string(index=False))
print('\n=== T11 per-dataset G-Mean ===')
print(T11.to_string())
print('\nSaved: table_T10_hard_regime_summary.csv, table_T11_hard_regime_perdataset.csv')
