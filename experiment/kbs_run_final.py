"""
KBS Experiment — Final Run (no shuttle, incremental saves)
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
_DIR = os.path.dirname(os.path.abspath(__file__))
# KBS_OUTPUT_DIR env var lets Colab (or any remote env) redirect results to Drive
_OUTPUT_DIR = os.environ.get('KBS_OUTPUT_DIR', os.path.join(_DIR, 'results'))
os.makedirs(_OUTPUT_DIR, exist_ok=True)
import warnings; warnings.filterwarnings('ignore')
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
from scipy.stats import friedmanchisquare, wilcoxon
import scikit_posthocs as sp
from mirtplus import MIRTPlusResampler

# ═══════════════════════════════════════════════════════════════════════════════
#  CUSTOM RESAMPLERS
# ═══════════════════════════════════════════════════════════════════════════════

class KMeansSMOTE:
    def __init__(self, k_neighbors=5, random_state=42):
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(self, X, y):
        rng = np.random.default_rng(self.random_state)
        classes, counts = np.unique(y, return_counts=True)
        max_count = counts.max()
        X_res, y_res = [X.copy()], [y.copy()]
        for cls, cnt in zip(classes, counts):
            if cnt >= max_count:
                continue
            n_gen = max_count - cnt
            X_cls = X[y == cls]
            n_cls = len(X_cls)
            n_clust = min(max(2, n_cls // 10), 8)
            n_clust = min(n_clust, n_cls)
            if n_clust < 2:
                kn = min(self.k_neighbors, n_cls - 1)
                if kn < 1: continue
                nn = NearestNeighbors(n_neighbors=kn).fit(X_cls)
                _, indices = nn.kneighbors(X_cls)
                synth = []
                for _ in range(n_gen):
                    i = rng.integers(n_cls)
                    j = indices[i, rng.integers(kn)]
                    lam = rng.random()
                    synth.append(X_cls[i] + lam * (X_cls[j] - X_cls[i]))
                X_res.append(np.array(synth)); y_res.append(np.full(n_gen, cls))
                continue
            km = KMeans(n_clusters=n_clust, random_state=self.random_state, n_init=3)
            labels = km.fit_predict(X_cls)
            cluster_sizes = np.bincount(labels, minlength=n_clust)
            weights = 1.0 / (cluster_sizes + 1e-8)
            weights = weights / weights.sum()
            samples_per_cluster = (weights * n_gen).astype(int)
            samples_per_cluster[0] += n_gen - samples_per_cluster.sum()
            synth = []
            for ci in range(n_clust):
                X_ci = X_cls[labels == ci]
                n_ci = len(X_ci)
                n_s = samples_per_cluster[ci]
                if n_ci < 2 or n_s == 0: continue
                kn = min(self.k_neighbors, n_ci - 1)
                nn = NearestNeighbors(n_neighbors=kn).fit(X_ci)
                _, indices = nn.kneighbors(X_ci)
                for _ in range(n_s):
                    i = rng.integers(n_ci)
                    j = indices[i, rng.integers(kn)]
                    lam = rng.random()
                    synth.append(X_ci[i] + lam * (X_ci[j] - X_ci[i]))
            if synth:
                X_res.append(np.array(synth)); y_res.append(np.full(len(synth), cls))
        return np.vstack(X_res), np.concatenate(y_res)


class SafeLevelSMOTE:
    def __init__(self, k_neighbors=5, random_state=42):
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(self, X, y):
        rng = np.random.default_rng(self.random_state)
        classes, counts = np.unique(y, return_counts=True)
        max_count = counts.max()
        X_res, y_res = [X.copy()], [y.copy()]
        nn_all = NearestNeighbors(n_neighbors=self.k_neighbors + 1).fit(X)
        for cls, cnt in zip(classes, counts):
            if cnt >= max_count: continue
            n_gen = max_count - cnt
            X_cls = X[y == cls]
            n_cls = len(X_cls)
            kn = min(self.k_neighbors, n_cls - 1)
            if kn < 1: continue
            nn_cls = NearestNeighbors(n_neighbors=kn).fit(X_cls)
            _, indices_cls = nn_cls.kneighbors(X_cls)
            _, indices_all = nn_all.kneighbors(X_cls)
            safe_levels = np.zeros(n_cls)
            for i in range(n_cls):
                nbrs = indices_all[i, 1:]
                safe_levels[i] = np.sum(y[nbrs] == cls)
            synth = []
            for _ in range(n_gen):
                p = rng.integers(n_cls)
                nn_idx = indices_cls[p, rng.integers(kn)]
                sl_p, sl_n = safe_levels[p], safe_levels[nn_idx]
                if sl_p == 0 and sl_n == 0: continue
                elif sl_p == 0: gap = 0.0
                elif sl_n == 0: gap = 1.0
                else: gap = rng.random() * min(1.0, sl_p / sl_n)
                synth.append(X_cls[p] + gap * (X_cls[nn_idx] - X_cls[p]))
            if synth:
                X_res.append(np.array(synth)); y_res.append(np.full(len(synth), cls))
        return np.vstack(X_res), np.concatenate(y_res)


class MWMOTE:
    def __init__(self, k1=5, k2=3, k3=5, random_state=42):
        self.k1, self.k2, self.k3 = k1, k2, k3
        self.random_state = random_state

    def fit_resample(self, X, y):
        rng = np.random.default_rng(self.random_state)
        classes, counts = np.unique(y, return_counts=True)
        max_count = counts.max()
        X_res, y_res = [X.copy()], [y.copy()]
        for cls, cnt in zip(classes, counts):
            if cnt >= max_count: continue
            n_gen = max_count - cnt
            X_min, X_maj = X[y == cls], X[y != cls]
            n_min = len(X_min)
            if n_min < 2: continue
            k1 = min(self.k1, n_min - 1)
            k2 = min(self.k2, len(X_maj) - 1)
            k3 = min(self.k3, n_min - 1)
            nn_all = NearestNeighbors(n_neighbors=k1 + 1).fit(X)
            _, idx_all = nn_all.kneighbors(X_min)
            borderline_mask = np.zeros(n_min, dtype=bool)
            for i in range(n_min):
                if np.any(y[idx_all[i, 1:]] != cls):
                    borderline_mask[i] = True
            if not borderline_mask.any(): borderline_mask[:] = True
            X_border = X_min[borderline_mask]
            nn_maj = NearestNeighbors(n_neighbors=min(k2, len(X_maj))).fit(X_maj)
            dists_maj, _ = nn_maj.kneighbors(X_border)
            weights = 1.0 / (dists_maj.mean(axis=1) + 1e-8)
            weights = weights / weights.sum()
            nn_min = NearestNeighbors(n_neighbors=k3).fit(X_min)
            _, idx_min = nn_min.kneighbors(X_border)
            synth = []
            for _ in range(n_gen):
                bi = rng.choice(len(X_border), p=weights)
                ni = idx_min[bi, rng.integers(k3)]
                lam = rng.random()
                synth.append(X_border[bi] + lam * (X_min[ni] - X_border[bi]))
            X_res.append(np.array(synth)); y_res.append(np.full(n_gen, cls))
        return np.vstack(X_res), np.concatenate(y_res)


class ProWSyn:
    def __init__(self, k_neighbors=5, random_state=42, L=5):
        self.k_neighbors, self.L = k_neighbors, L
        self.random_state = random_state

    def fit_resample(self, X, y):
        rng = np.random.default_rng(self.random_state)
        classes, counts = np.unique(y, return_counts=True)
        max_count = counts.max()
        X_res, y_res = [X.copy()], [y.copy()]
        for cls, cnt in zip(classes, counts):
            if cnt >= max_count: continue
            n_gen = max_count - cnt
            X_min, X_maj = X[y == cls], X[y != cls]
            n_min = len(X_min)
            if n_min < 2: continue
            nn_maj = NearestNeighbors(n_neighbors=min(3, len(X_maj))).fit(X_maj)
            dists, _ = nn_maj.kneighbors(X_min)
            proximity = dists.min(axis=1)
            p_min, p_max = proximity.min(), proximity.max()
            if p_max - p_min < 1e-10:
                level_weights = np.ones(n_min)
            else:
                normalized = (proximity - p_min) / (p_max - p_min)
                levels = np.clip((normalized * self.L).astype(int), 0, self.L - 1)
                level_weights = (levels + 1).astype(float)
            weights = level_weights / level_weights.sum()
            kn = min(self.k_neighbors, n_min - 1)
            nn_min = NearestNeighbors(n_neighbors=kn).fit(X_min)
            _, indices = nn_min.kneighbors(X_min)
            synth = []
            for _ in range(n_gen):
                i = rng.choice(n_min, p=weights)
                j = indices[i, rng.integers(kn)]
                lam = rng.random()
                synth.append(X_min[i] + lam * (X_min[j] - X_min[i]))
            X_res.append(np.array(synth)); y_res.append(np.full(n_gen, cls))
        return np.vstack(X_res), np.concatenate(y_res)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASETS (no shuttle)
# ═══════════════════════════════════════════════════════════════════════════════

U = r'dataset/dataset/used'
D2 = r'dataset/dataset'

def _filter(X, y, min_count=12):
    vc = pd.Series(y).value_counts()
    keep = vc[vc >= min_count].index
    m = np.isin(y, keep)
    return X[m], y[m]

def load_nt():
    f = ['T3Resin','T4','T3','TSH','MaxTSH']
    d = pd.read_csv(f'{U}/new-thyroid.data', header=None, names=['Class']+f)
    return d[f].values.astype(float), d['Class'].values.astype(int)

def load_hr():
    f = ['f1','f2','f3','f4']
    d = pd.read_csv(f'{U}/hayes-roth.data', header=None, names=['id']+f+['Class'])
    return d[f].values.astype(float), d['Class'].values.astype(int)

def load_bs():
    f = ['LW','LD','RW','RD']
    d = pd.read_csv(f'{U}/balance-scale.data', header=None, names=['Class']+f)
    d['Class'] = d['Class'].map({'B':1,'L':2,'R':3})
    return d[f].values.astype(float), d['Class'].values.astype(int)

def load_ca():
    f = ['buying','maint','doors','persons','lug_boot','safety']
    d = pd.read_csv(f'{U}/car.data', header=None, names=f+['Class'])
    ords = {'buying':['low','med','high','vhigh'],'maint':['low','med','high','vhigh'],
            'doors':['2','3','4','5more'],'persons':['2','4','more'],
            'lug_boot':['small','med','big'],'safety':['low','med','high']}
    for c,o in ords.items(): d[c] = d[c].map({v:i for i,v in enumerate(o)})
    d['Class'] = LabelEncoder().fit_transform(d['Class'])
    return d[f].values.astype(float), d['Class'].values.astype(int)

def load_cmc():
    f = ['age','wedu','hedu','child','wrel','wwork','hocc','sol','media']
    d = pd.read_csv(f'{U}/cmc.data', sep='\t', header=None, names=['id']+f+['Class'])
    return d[f].values.astype(float), d['Class'].values.astype(int)

def load_ecoli():
    d = pd.read_csv(f'{D2}/new_to_use/ecoli.data', header=None, sep=r'\s+')
    X = d.iloc[:,1:-1].values.astype(float)
    y = LabelEncoder().fit_transform(d.iloc[:,-1])
    return _filter(X, y)

def load_glass():
    d = pd.read_csv('Implementation/Implementation/glass.data', header=None)
    X = d.iloc[:,1:-1].values.astype(float)
    y = d.iloc[:,-1].values.astype(int)
    return _filter(X, y)

def load_wine():
    d = pd.read_csv(f'{D2}/wine.data', header=None)
    return d.iloc[:,1:].values.astype(float), d.iloc[:,0].values.astype(int)

def load_dermatology():
    d = pd.read_csv(f'{D2}/dermatology.data', header=None, na_values='?')
    d = d.fillna(d.median(numeric_only=True))
    return _filter(d.iloc[:,:-1].values.astype(float), d.iloc[:,-1].values.astype(int))

def load_yeast():
    d = pd.read_csv(f'{D2}/yeast.data', header=None, sep=r'\s+')
    X = d.iloc[:,1:-1].values.astype(float)
    y = LabelEncoder().fit_transform(d.iloc[:,-1])
    return _filter(X, y)

def load_wqred():
    d = pd.read_csv(f'{D2}/winequality-red.csv', sep=';')
    return _filter(d.iloc[:,:-1].values.astype(float), d.iloc[:,-1].values.astype(int))

def load_cleveland():
    d = pd.read_csv(f'{D2}/processed.cleveland.data', header=None, na_values='?')
    d = d.dropna()
    return d.iloc[:,:-1].values.astype(float), d.iloc[:,-1].values.astype(int)

def load_vehicle():
    d = fetch_openml(name='vehicle', version=1, as_frame=True, parser='auto')
    return d.data.values.astype(float), LabelEncoder().fit_transform(d.target)

def load_segment():
    d = fetch_openml(name='segment', version=1, as_frame=True, parser='auto')
    return d.data.values.astype(float), LabelEncoder().fit_transform(d.target)

def load_satimage():
    d = fetch_openml(name='satimage', version=1, as_frame=True, parser='auto')
    return d.data.values.astype(float), LabelEncoder().fit_transform(d.target)

def load_pageblocks():
    d = fetch_openml(name='page-blocks', version=1, as_frame=True, parser='auto')
    return _filter(d.data.values.astype(float), LabelEncoder().fit_transform(d.target))

def load_vowel():
    d = fetch_openml(name='vowel', version=2, as_frame=True, parser='auto')
    X = d.data.select_dtypes(include=[np.number]).values.astype(float)
    return X, LabelEncoder().fit_transform(d.target)

def load_abalone():
    d = fetch_openml(name='abalone', version=1, as_frame=True, parser='auto')
    df = d.data.copy()
    target = d.target.values.astype(int)
    if 'Sex' in df.columns: df['Sex'] = df['Sex'].map({'M':0, 'F':1, 'I':2})
    X = df.values.astype(float)
    y = np.where(target <= 8, 0, np.where(target <= 11, 1, 2))
    return X, y

def load_thyroid_ann():
    d = fetch_openml(data_id=40474, as_frame=True, parser='auto')
    return _filter(d.data.values.astype(float), LabelEncoder().fit_transform(d.target))

def load_wqwhite():
    d = pd.read_csv(f'{D2}/winequality-white.csv', sep=';')
    return _filter(d.iloc[:,:-1].values.astype(float), d.iloc[:,-1].values.astype(int))

def load_hepatitis():
    d = pd.read_csv(f'{D2}/hepatitis.data', header=None, na_values='?')
    d = d.dropna()
    return d.iloc[:,1:].values.astype(float), d.iloc[:,0].values.astype(int)

def load_splice():
    d = fetch_openml(name='splice', version=1, as_frame=True, parser='auto')
    return pd.get_dummies(d.data).values.astype(float), LabelEncoder().fit_transform(d.target)

def load_flare():
    d = fetch_openml(name='flare', version=1, as_frame=True, parser='auto')
    return _filter(pd.get_dummies(d.data).values.astype(float), LabelEncoder().fit_transform(d.target))

def load_pendigits():
    d = fetch_openml(name='pendigits', version=1, as_frame=True, parser='auto')
    return d.data.values.astype(float), LabelEncoder().fit_transform(d.target)


# NO SHUTTLE — too large, was killing the run
DATASETS = {
    'new-thyroid': load_nt, 'hayes-roth': load_hr, 'balance-scale': load_bs,
    'car': load_ca, 'cmc': load_cmc, 'ecoli': load_ecoli, 'glass': load_glass,
    'wine': load_wine, 'dermatology': load_dermatology, 'yeast': load_yeast,
    'winequality-red': load_wqred, 'cleveland': load_cleveland,
    'vehicle': load_vehicle, 'segment': load_segment, 'satimage': load_satimage,
    'vowel': load_vowel, 'abalone': load_abalone,
    'ann-thyroid': load_thyroid_ann, 'winequality-white': load_wqwhite,
    'hepatitis': load_hepatitis, 'splice': load_splice, 'flare': load_flare,
    'pendigits': load_pendigits,
}

METHODS = [
    'None', 'SMOTE', 'Borderline-SMOTE', 'ADASYN',
    'SMOTE-ENN', 'SMOTE-Tomek', 'KMeans-SMOTE',
    'Safe-Level-SMOTE', 'MWMOTE', 'ProWSyn', 'MIRT++'
]


def make_sampler(name, seed, k_neighbors=5):
    if name == 'None': return None
    elif name == 'SMOTE': return SMOTE(k_neighbors=k_neighbors, random_state=seed)
    elif name == 'Borderline-SMOTE': return BorderlineSMOTE(k_neighbors=k_neighbors, random_state=seed)
    elif name == 'ADASYN': return ADASYN(n_neighbors=k_neighbors, random_state=seed)
    elif name == 'SMOTE-ENN': return SMOTEENN(smote=SMOTE(k_neighbors=k_neighbors, random_state=seed), random_state=seed)
    elif name == 'SMOTE-Tomek': return SMOTETomek(smote=SMOTE(k_neighbors=k_neighbors, random_state=seed), random_state=seed)
    elif name == 'KMeans-SMOTE': return KMeansSMOTE(k_neighbors=k_neighbors, random_state=seed)
    elif name == 'Safe-Level-SMOTE': return SafeLevelSMOTE(k_neighbors=k_neighbors, random_state=seed)
    elif name == 'MWMOTE': return MWMOTE(k1=k_neighbors, random_state=seed)
    elif name == 'ProWSyn': return ProWSyn(k_neighbors=k_neighbors, random_state=seed)
    elif name == 'MIRT++': return MIRTPlusResampler(seed=seed)


def safe_resample(X, y, name, seed):
    kmin = pd.Series(y).value_counts().min()
    kn = min(5, max(1, kmin - 1))
    sampler = make_sampler(name, seed, k_neighbors=kn)
    if sampler is None: return X, y
    try:
        return sampler.fit_resample(X, y)
    except Exception:
        return X, y


def auc_macro(clf, Xte, yte, classes):
    try:
        if not hasattr(clf, 'predict_proba'): return np.nan
        proba = clf.predict_proba(Xte)
        yb = label_binarize(yte, classes=classes)
        if yb.shape[1] == 1: yb = np.hstack([1 - yb, yb])
        return roc_auc_score(yb, proba, multi_class='ovr', average='macro')
    except Exception:
        return np.nan


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    OUTPUT = os.path.join(_OUTPUT_DIR, 'kbs_results_raw.csv')

    # Check if we have partial results to resume from
    completed_datasets = set()
    existing_records = []
    if os.path.exists(OUTPUT):
        existing = pd.read_csv(OUTPUT)
        completed_datasets = set(existing['dataset'].unique())
        existing_records = existing.to_dict('records')
        print(f'Resuming: {len(completed_datasets)} datasets already done: {sorted(completed_datasets)}')

    print(f'\nKBS Experiment: {len(DATASETS)} datasets x {len(METHODS)} methods x 3 classifiers')
    print('='*80)

    # Load datasets
    loaded = {}
    for name, loader in DATASETS.items():
        if name in completed_datasets:
            print(f'  SKIP {name} (already done)', flush=True)
            continue
        try:
            X, y = loader()
            loaded[name] = (X, y)
            classes = np.unique(y)
            counts = pd.Series(y).value_counts()
            ir = counts.max() / counts.min()
            print(f'  OK {name}: n={len(y)}, f={X.shape[1]}, c={len(classes)}, IR={ir:.1f}', flush=True)
        except Exception as e:
            print(f'  FAIL {name}: {str(e)[:80]}', flush=True)

    print(f'\nRunning {len(loaded)} remaining datasets...', flush=True)
    print('='*80)

    records = list(existing_records)
    timings = []

    for dname, (X, y) in loaded.items():
        classes = sorted(np.unique(y).tolist())
        # Fewer repeats for large datasets
        n_repeats = 1 if len(y) > 5000 else 2
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=42)
        t_start = time.time()

        for fold, (tr, te) in enumerate(rskf.split(X, y)):
            Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
            for rname in METHODS:
                t0 = time.time()
                Xr, yr = safe_resample(Xtr, ytr, rname, 1000 + fold)
                t_resample = time.time() - t0

                for cname, clf in {
                    'KNN': KNeighborsClassifier(5),
                    'CART': DecisionTreeClassifier(random_state=fold),
                    'SVM': SVC(C=1, probability=True, random_state=fold),
                }.items():
                    clf.fit(Xr, yr)
                    yp = clf.predict(Xte)
                    records.append({
                        'dataset': dname, 'fold': fold,
                        'resampler': rname, 'clf': cname,
                        'f1': f1_score(yte, yp, average='macro', zero_division=0),
                        'gmean': geometric_mean_score(yte, yp, average='macro'),
                        'auc': auc_macro(clf, Xte, yte, classes),
                    })
                timings.append({'dataset': dname, 'method': rname, 'fold': fold, 'time_s': t_resample})

        elapsed = time.time() - t_start
        print(f'  done: {dname} ({elapsed:.1f}s)', flush=True)

        # INCREMENTAL SAVE after each dataset
        pd.DataFrame(records).to_csv(OUTPUT, index=False)

    results = pd.DataFrame(records)
    results.to_csv(OUTPUT, index=False)
    pd.DataFrame(timings).to_csv(os.path.join(_OUTPUT_DIR, 'kbs_timings.csv'), index=False)

    print(f'\n{"="*80}')
    print(f'ALL DONE — {len(results)} total evaluations saved to {OUTPUT}')
    print(f'{"="*80}')

    # ═══════════════════════════════════════════════════════════════════════════
    #  FULL COMPARISON ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════

    print('\n\n')
    print('╔══════════════════════════════════════════════════════════════════════════════╗')
    print('║                    FULL COMPARISON RESULTS                                  ║')
    print('╚══════════════════════════════════════════════════════════════════════════════╝')

    n_datasets = results['dataset'].nunique()
    print(f'\n{n_datasets} datasets × {len(METHODS)} methods × 3 classifiers')

    # 1. Mean performance
    print('\n\n=== MEAN PERFORMANCE (all datasets × classifiers × folds) ===\n')
    agg = results.groupby('resampler')[['f1','gmean','auc']].mean().reindex(METHODS).round(4)
    print(agg.sort_values('gmean', ascending=False).to_string())

    # 2. Average Ranks
    print('\n\n=== AVERAGE RANKS (lower = better) ===\n')
    for metric, label in [('f1','Macro F1'), ('gmean','G-Mean'), ('auc', 'AUC')]:
        piv = results.pivot_table(index=['dataset','fold','clf'],
                                  columns='resampler', values=metric)[METHODS].dropna()
        ranks = piv.rank(axis=1, ascending=False).mean()
        stat, p = friedmanchisquare(*[piv[m].values for m in METHODS])
        print(f'--- {label} ---')
        print(f'Friedman chi2 = {stat:.2f},  p = {p:.2e}  ({"SIGNIFICANT" if p < 0.05 else "n.s."})')
        print(ranks.sort_values().round(3).to_string())
        print()

    # 3. G-Mean by dataset
    print('\n=== G-MEAN BY DATASET ===\n')
    pivot_g = results.groupby(['dataset','resampler'])['gmean'].mean().unstack()[METHODS].round(3)
    print(pivot_g.to_string())
    pivot_g.to_csv('kbs_gmean_by_dataset.csv')

    # 4. F1 by dataset
    print('\n\n=== MACRO-F1 BY DATASET ===\n')
    pivot_f = results.groupby(['dataset','resampler'])['f1'].mean().unstack()[METHODS].round(3)
    print(pivot_f.to_string())
    pivot_f.to_csv('kbs_f1_by_dataset.csv')

    # 5. Win/Tie/Loss vs MIRT++
    print('\n\n=== WIN/TIE/LOSS vs MIRT++ (G-Mean, per dataset) ===\n')
    ds_mean = results.groupby(['dataset','resampler'])['gmean'].mean().unstack()
    wtl_rows = []
    for method in METHODS:
        if method == 'MIRT++': continue
        wins = (ds_mean['MIRT++'] > ds_mean[method] + 0.002).sum()
        losses = (ds_mean[method] > ds_mean['MIRT++'] + 0.002).sum()
        ties = len(ds_mean) - wins - losses
        wtl_rows.append({'vs': method, 'MIRT++ Wins': wins, 'Tie': ties, 'MIRT++ Losses': losses})
    print(pd.DataFrame(wtl_rows).set_index('vs').to_string())

    # 6. Wilcoxon signed-rank
    print('\n\n=== WILCOXON SIGNED-RANK (MIRT++ vs each, G-Mean) ===\n')
    piv_ds = results.pivot_table(index=['dataset','fold','clf'],
                                  columns='resampler', values='gmean')[METHODS].dropna()
    for method in METHODS:
        if method == 'MIRT++': continue
        try:
            stat, p = wilcoxon(piv_ds['MIRT++'], piv_ds[method], alternative='greater')
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
            print(f'  MIRT++ vs {method:<20s}: W={stat:>8.0f}, p={p:.6f} {sig}')
        except Exception as e:
            print(f'  MIRT++ vs {method:<20s}: error — {e}')

    # 7. Nemenyi post-hoc
    print('\n\n=== NEMENYI POST-HOC (G-Mean) ===\n')
    piv_nem = results.pivot_table(index=['dataset','fold','clf'],
                                   columns='resampler', values='gmean')[METHODS].dropna()
    nem = sp.posthoc_nemenyi_friedman(piv_nem.values)
    nem.index = METHODS; nem.columns = METHODS
    print(nem.round(4).to_string())

    # 8. MIRT++ head-to-head vs SMOTE
    print('\n\n=== MIRT++ vs SMOTE (head-to-head per dataset) ===\n')
    for dname in sorted(ds_mean.index):
        g_mirt = ds_mean.loc[dname, 'MIRT++']
        g_smote = ds_mean.loc[dname, 'SMOTE']
        diff = g_mirt - g_smote
        marker = '>>>' if diff > 0.005 else '>' if diff > 0 else '=' if abs(diff) < 0.002 else '<'
        print(f'  {dname:<20s}: MIRT++={g_mirt:.4f}  SMOTE={g_smote:.4f}  diff={diff:+.4f} {marker}')

    # 9. Medical case study
    print('\n\n' + '='*80)
    print('MEDICAL DOMAIN CASE STUDY')
    print('='*80)
    medical = ['cleveland', 'ann-thyroid', 'dermatology', 'hepatitis', 'new-thyroid']
    med_results = results[results.dataset.isin(medical)]
    if not med_results.empty:
        print('\n=== G-Mean on Medical Datasets ===\n')
        med_pivot = med_results.groupby(['dataset','resampler'])['gmean'].mean().unstack()[METHODS].round(3)
        print(med_pivot.to_string())
        med_pivot.to_csv('kbs_medical_gmean.csv')

        print('\n=== F1 on Medical Datasets ===\n')
        med_f1 = med_results.groupby(['dataset','resampler'])['f1'].mean().unstack()[METHODS].round(3)
        print(med_f1.to_string())
        med_f1.to_csv('kbs_medical_f1.csv')

        print('\n=== Average Rank on Medical Datasets (G-Mean) ===\n')
        med_piv = med_results.pivot_table(index=['dataset','fold','clf'],
                                           columns='resampler', values='gmean')[METHODS].dropna()
        print(med_piv.rank(axis=1, ascending=False).mean().sort_values().round(3).to_string())

    # 10. Scalability
    print('\n\n' + '='*80)
    print('SCALABILITY (avg resampling time in seconds)')
    print('='*80)
    if timings:
        tdf = pd.DataFrame(timings)
        avg_t = tdf.groupby(['dataset','method'])['time_s'].mean().unstack()
        cols = [m for m in METHODS[1:] if m in avg_t.columns]
        print(avg_t[cols].round(3).to_string())
        avg_t.to_csv('kbs_scalability.csv')

    print('\n\nDONE. All results saved.')
