"""
MIRT++ ablation over all 23 datasets, 3 classifiers, 5-fold CV.
Each variant disables one component. Saves raw results + summary + stats.
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from imblearn.metrics import geometric_mean_score
from scipy.stats import friedmanchisquare, wilcoxon
from mirtplus_ablation import MIRTPlusAblate
from kbs_run_final import DATASETS

VARIANTS = {
    'Full':          dict(),
    '-FeatWeights':  dict(feat_weights=False),
    '-Subgroups':    dict(subgroups=False),
    '-Mahalanobis':  dict(mahalanobis=False),
    '-NoiseGate':    dict(noise_gate=False),
    '-ConfSteer':    dict(conf_steer=False),
}

def safe_resample(Xtr, ytr, kw, seed):
    kmin = pd.Series(ytr).value_counts().min()
    kn = min(5, max(1, kmin - 1))
    try:
        return MIRTPlusAblate(k_neighbors=kn, seed=seed, **kw).fit_resample(Xtr, ytr)
    except Exception:
        return Xtr, ytr

records = []
loaded = {}
for name, loader in DATASETS.items():
    try:
        X, y = loader()
        X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        loaded[name] = (X, y)
    except Exception as e:
        print(f'  FAIL load {name}: {str(e)[:60]}', flush=True)

for dname, (X, y) in loaded.items():
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)
    for fold, (tr, te) in enumerate(rskf.split(X, y)):
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        for vname, kw in VARIANTS.items():
            Xr, yr = safe_resample(Xtr, ytr, kw, 1000 + fold)
            for cname, clf in {
                'KNN': KNeighborsClassifier(5),
                'CART': DecisionTreeClassifier(random_state=fold),
                'SVM': SVC(C=1, random_state=fold),
            }.items():
                clf.fit(Xr, yr); yp = clf.predict(Xte)
                records.append({'dataset': dname, 'fold': fold, 'variant': vname, 'clf': cname,
                                'gmean': geometric_mean_score(yte, yp, average='macro'),
                                'f1': f1_score(yte, yp, average='macro', zero_division=0)})
    pd.DataFrame(records).to_csv('ablation_raw.csv', index=False)
    print(f'  done {dname}', flush=True)

res = pd.DataFrame(records)
res.to_csv('ablation_raw.csv', index=False)
order = list(VARIANTS.keys())

# summary: mean gmean/f1 + average rank (per dataset,fold,clf)
piv = res.pivot_table(index=['dataset','fold','clf'], columns='variant', values='gmean')[order].dropna()
means_g = res.groupby('variant')['gmean'].mean().reindex(order)
means_f = res.groupby('variant')['f1'].mean().reindex(order)
ranks_g = piv.rank(axis=1, ascending=False).mean()
stat, p = friedmanchisquare(*[piv[v].values for v in order])

rows = []
for v in order:
    if v == 'Full':
        p_txt, sig, delta = '-', '-', 0.0
    else:
        w, pv = wilcoxon(piv['Full'], piv[v], alternative='greater')
        delta = piv['Full'].mean() - piv[v].mean()
        sig = '***' if pv<0.001 else '**' if pv<0.01 else '*' if pv<0.05 else 'n.s.'
        p_txt = f'{pv:.4f}'
    rows.append({'Variant': v, 'Mean G-Mean': round(means_g[v],4), 'Mean F1': round(means_f[v],4),
                 'Avg Rank (G-Mean)': round(ranks_g[v],3),
                 'Drop vs Full (G-Mean)': round(delta,4),
                 'Wilcoxon p (Full > x)': p_txt, 'Sig': sig})
T12 = pd.DataFrame(rows)
T12.to_csv('table_T12_ablation.csv', index=False)

print(f'\nFriedman chi2={stat:.2f}, p={p:.2e}')
print(T12.to_string(index=False))
print('\nsaved ablation_raw.csv, table_T12_ablation.csv')
