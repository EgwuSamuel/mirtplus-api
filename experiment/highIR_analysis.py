"""
High-IR / high-class-count significance sub-analysis for MIRT++.
Tests whether MIRT++ significantly beats the strongest baselines
on the HARD regime (high imbalance ratio, many classes).
"""
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare

df = pd.read_csv('kbs_results_raw.csv')
df['resampler'] = df['resampler'].fillna('None')   # blank == the 'None' (no-resample) method
print('rows:', len(df), '| datasets:', df['dataset'].nunique(),
      '| methods:', df['resampler'].nunique())

METHODS = ['None','SMOTE','Borderline-SMOTE','ADASYN','SMOTE-ENN','SMOTE-Tomek',
           'KMeans-SMOTE','Safe-Level-SMOTE','MWMOTE','ProWSyn','MIRT++']

# Dataset characteristics (from table_T1) — IR and class count
CHARS = {
 'segment':(1.0,7),'vowel':(1.0,11),'vehicle':(1.1,4),'pendigits':(1.1,10),
 'wine':(1.5,3),'splice':(1.8,3),'cmc':(1.9,3),'hayes-roth':(2.1,3),
 'satimage':(2.4,6),'hepatitis':(3.8,2),'new-thyroid':(5.0,3),'dermatology':(5.6,6),
 'balance-scale':(5.9,3),'glass':(8.4,6),'abalone':(9.7,3),'cleveland':(12.5,5),
 'car':(18.6,4),'yeast':(23.1,8),'flare':(23.3,6),'ecoli':(39.1,5),
 'ann-thyroid':(45.8,3),'winequality-red':(68.1,6),'winequality-white':(109.9,6),
}

def subset_test(name, keep_datasets, metric='gmean'):
    d = df[df['dataset'].isin(keep_datasets)]
    piv = d.pivot_table(index=['dataset','fold','clf'], columns='resampler',
                        values=metric)[METHODS].dropna()
    print(f'\n{"="*74}\n{name}  ({len(keep_datasets)} datasets, {len(piv)} paired obs, metric={metric})')
    print('datasets:', sorted(keep_datasets))
    # mean + rank
    means = piv.mean().sort_values(ascending=False)
    ranks = piv.rank(axis=1, ascending=False).mean().sort_values()
    print(f'\n  MIRT++ mean {metric}: {piv["MIRT++"].mean():.4f}  '
          f'(overall best: {means.index[0]}={means.iloc[0]:.4f})')
    print(f'  MIRT++ avg rank: {ranks["MIRT++"]:.3f}  (best rank: {ranks.index[0]}={ranks.iloc[0]:.3f})')
    # Friedman
    stat,p = friedmanchisquare(*[piv[m].values for m in METHODS])
    print(f'  Friedman chi2={stat:.2f}, p={p:.2e} ({"SIG" if p<0.05 else "n.s."})')
    # Wilcoxon MIRT++ > each
    print(f'\n  Wilcoxon (MIRT++ > baseline, one-sided):')
    for m in METHODS:
        if m=='MIRT++': continue
        try:
            w,pp = wilcoxon(piv['MIRT++'], piv[m], alternative='greater')
            sig = '***' if pp<0.001 else '**' if pp<0.01 else '*' if pp<0.05 else 'n.s.'
            delta = piv['MIRT++'].mean()-piv[m].mean()
            print(f'    vs {m:<18s} d={delta:+.4f}  W={w:>8.0f}  p={pp:.4f} {sig}')
        except Exception as e:
            print(f'    vs {m:<18s} error {e}')

# Regime definitions
high_ir   = [d for d,(ir,k) in CHARS.items() if ir >= 9.0]     # extreme imbalance
high_k    = [d for d,(ir,k) in CHARS.items() if k >= 6]        # many classes
hard_both = [d for d,(ir,k) in CHARS.items() if ir >= 8.0 and k >= 5]  # hard both

subset_test('ALL 23 DATASETS (baseline)', list(CHARS.keys()))
subset_test('HIGH IMBALANCE (IR >= 9)', high_ir)
subset_test('HIGH CLASS COUNT (k >= 6)', high_k)
subset_test('HARD REGIME (IR>=8 AND k>=5)', hard_both)
