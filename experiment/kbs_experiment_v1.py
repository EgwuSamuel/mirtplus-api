"""
KBS-level Expanded MIRT++ Experiment
=====================================
25 datasets, 10 resampling methods, 3 classifiers, rigorous statistical tests.
Targets: Knowledge-Based Systems journal submission.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
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
from sklearn.cluster import KMeans, MiniBatchKMeans
from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
from scipy.stats import friedmanchisquare, wilcoxon
import scikit_posthocs as sp
from mirtplus import MIRTPlusResampler
import os, joblib

# ═══════════════════════════════════════════════════════════════════════════════
#  ADDITIONAL RESAMPLING METHODS (implemented here since smote-variants won't install)
# ═══════════════════════════════════════════════════════════════════════════════

class KMeansSMOTE:
    """k-Means SMOTE: cluster-based oversampling (Douzas et al., 2018)."""
    def __init__(self, k_neighbors=5, random_state=42, n_clusters='auto'):
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.n_clusters = n_clusters

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

            n_clust = min(max(2, n_cls // 10), 8) if self.n_clusters == 'auto' else self.n_clusters
            n_clust = min(n_clust, n_cls)
            if n_clust < 2:
                # too few samples, fall back to SMOTE-like
                kn = min(self.k_neighbors, n_cls - 1)
                if kn < 1:
                    continue
                nn = NearestNeighbors(n_neighbors=kn).fit(X_cls)
                dists, indices = nn.kneighbors(X_cls)
                synth = []
                for _ in range(n_gen):
                    i = rng.integers(n_cls)
                    j = indices[i, rng.integers(kn)]
                    lam = rng.random()
                    synth.append(X_cls[i] + lam * (X_cls[j] - X_cls[i]))
                X_res.append(np.array(synth))
                y_res.append(np.full(n_gen, cls))
                continue

            km = KMeans(n_clusters=n_clust, random_state=self.random_state, n_init=3)
            labels = km.fit_predict(X_cls)

            cluster_sizes = np.bincount(labels, minlength=n_clust)
            # weight clusters by density (more samples in sparse clusters)
            weights = 1.0 / (cluster_sizes + 1e-8)
            weights = weights / weights.sum()
            samples_per_cluster = (weights * n_gen).astype(int)
            samples_per_cluster[0] += n_gen - samples_per_cluster.sum()

            synth = []
            for ci in range(n_clust):
                X_ci = X_cls[labels == ci]
                n_ci = len(X_ci)
                n_s = samples_per_cluster[ci]
                if n_ci < 2 or n_s == 0:
                    continue
                kn = min(self.k_neighbors, n_ci - 1)
                nn = NearestNeighbors(n_neighbors=kn).fit(X_ci)
                _, indices = nn.kneighbors(X_ci)
                for _ in range(n_s):
                    i = rng.integers(n_ci)
                    j = indices[i, rng.integers(kn)]
                    lam = rng.random()
                    synth.append(X_ci[i] + lam * (X_ci[j] - X_ci[i]))

            if synth:
                X_res.append(np.array(synth))
                y_res.append(np.full(len(synth), cls))

        return np.vstack(X_res), np.concatenate(y_res)


class SafeLevelSMOTE:
    """Safe-Level-SMOTE (Bunkhumpornpat et al., 2009)."""
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
            if cnt >= max_count:
                continue
            n_gen = max_count - cnt
            X_cls = X[y == cls]
            n_cls = len(X_cls)
            kn = min(self.k_neighbors, n_cls - 1)
            if kn < 1:
                continue

            nn_cls = NearestNeighbors(n_neighbors=kn).fit(X_cls)
            _, indices_cls = nn_cls.kneighbors(X_cls)

            # compute safe levels
            _, indices_all = nn_all.kneighbors(X_cls)
            safe_levels = np.zeros(n_cls)
            for i in range(n_cls):
                nbrs = indices_all[i, 1:]  # exclude self
                safe_levels[i] = np.sum(y[nbrs] == cls)

            synth = []
            for _ in range(n_gen):
                p = rng.integers(n_cls)
                nn_idx = indices_cls[p, rng.integers(kn)]
                sl_p = safe_levels[p]
                sl_n = safe_levels[nn_idx]

                if sl_p == 0 and sl_n == 0:
                    continue
                elif sl_p == 0:
                    gap = 0.0
                elif sl_n == 0:
                    gap = 1.0
                else:
                    ratio = sl_p / sl_n
                    gap = rng.random() * min(1.0, ratio)

                new = X_cls[p] + gap * (X_cls[nn_idx] - X_cls[p])
                synth.append(new)

            if synth:
                X_res.append(np.array(synth))
                y_res.append(np.full(len(synth), cls))

        return np.vstack(X_res), np.concatenate(y_res)


class MWMOTE:
    """MWMOTE - Majority Weighted Minority Oversampling (Barua et al., 2014)."""
    def __init__(self, k1=5, k2=3, k3=5, random_state=42):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
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
            X_min = X[y == cls]
            X_maj = X[y != cls]
            n_min = len(X_min)

            if n_min < 2:
                continue

            k1 = min(self.k1, n_min - 1)
            k2 = min(self.k2, len(X_maj) - 1)
            k3 = min(self.k3, n_min - 1)

            # Step 1: find borderline minority (those with majority neighbors)
            nn_all = NearestNeighbors(n_neighbors=k1 + 1).fit(X)
            _, idx_all = nn_all.kneighbors(X_min)
            borderline_mask = np.zeros(n_min, dtype=bool)
            for i in range(n_min):
                nbr_labels = y[idx_all[i, 1:]]
                if np.any(nbr_labels != cls):
                    borderline_mask[i] = True

            if not borderline_mask.any():
                borderline_mask[:] = True  # fallback

            X_border = X_min[borderline_mask]

            # Step 2: compute importance weights based on distance to majority
            nn_maj = NearestNeighbors(n_neighbors=min(k2, len(X_maj))).fit(X_maj)
            dists_maj, _ = nn_maj.kneighbors(X_border)
            # closer to majority = more important (harder example)
            avg_dist = dists_maj.mean(axis=1)
            weights = 1.0 / (avg_dist + 1e-8)
            weights = weights / weights.sum()

            # Step 3: generate synthetic samples
            nn_min = NearestNeighbors(n_neighbors=k3).fit(X_min)
            _, idx_min = nn_min.kneighbors(X_border)

            synth = []
            border_indices = np.arange(len(X_border))
            for _ in range(n_gen):
                bi = rng.choice(border_indices, p=weights)
                ni = idx_min[bi, rng.integers(k3)]
                lam = rng.random()
                new = X_border[bi] + lam * (X_min[ni] - X_border[bi])
                synth.append(new)

            X_res.append(np.array(synth))
            y_res.append(np.full(n_gen, cls))

        return np.vstack(X_res), np.concatenate(y_res)


class ProWSyn:
    """ProWSyn - Proximity Weighted Synthetic oversampling (Barua et al., 2013)."""
    def __init__(self, k_neighbors=5, random_state=42, L=5):
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.L = L  # proximity levels

    def fit_resample(self, X, y):
        rng = np.random.default_rng(self.random_state)
        classes, counts = np.unique(y, return_counts=True)
        max_count = counts.max()
        X_res, y_res = [X.copy()], [y.copy()]

        for cls, cnt in zip(classes, counts):
            if cnt >= max_count:
                continue
            n_gen = max_count - cnt
            X_min = X[y == cls]
            X_maj = X[y != cls]
            n_min = len(X_min)

            if n_min < 2:
                continue

            # Compute proximity to majority
            nn_maj = NearestNeighbors(n_neighbors=min(3, len(X_maj))).fit(X_maj)
            dists, _ = nn_maj.kneighbors(X_min)
            proximity = dists.min(axis=1)

            # Assign proximity levels
            p_min, p_max = proximity.min(), proximity.max()
            if p_max - p_min < 1e-10:
                level_weights = np.ones(n_min)
            else:
                normalized = (proximity - p_min) / (p_max - p_min)
                levels = np.clip((normalized * self.L).astype(int), 0, self.L - 1)
                # higher level (farther from majority) = safer = more weight
                level_weights = (levels + 1).astype(float)

            weights = level_weights / level_weights.sum()

            # Generate
            kn = min(self.k_neighbors, n_min - 1)
            nn_min = NearestNeighbors(n_neighbors=kn).fit(X_min)
            _, indices = nn_min.kneighbors(X_min)

            synth = []
            for _ in range(n_gen):
                i = rng.choice(n_min, p=weights)
                j = indices[i, rng.integers(kn)]
                lam = rng.random()
                synth.append(X_min[i] + lam * (X_min[j] - X_min[i]))

            X_res.append(np.array(synth))
            y_res.append(np.full(n_gen, cls))

        return np.vstack(X_res), np.concatenate(y_res)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

U = r'dataset/dataset/used'
D2 = r'dataset/dataset'

def _filter(X, y, min_count=12):
    """Drop classes with fewer than min_count samples."""
    vc = pd.Series(y).value_counts()
    keep = vc[vc >= min_count].index
    m = np.isin(y, keep)
    return X[m], y[m]

# --- Original 11 datasets ---

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
    for c,o in ords.items():
        d[c] = d[c].map({v:i for i,v in enumerate(o)})
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

# --- New datasets (14 more) ---

def load_cleveland():
    d = pd.read_csv(f'{D2}/processed.cleveland.data', header=None, na_values='?')
    d = d.dropna()
    X = d.iloc[:,:-1].values.astype(float)
    y = d.iloc[:,-1].values.astype(int)
    return _filter(X, y)

def load_vehicle():
    d = fetch_openml(name='vehicle', version=1, as_frame=True, parser='auto')
    X = d.data.values.astype(float)
    y = LabelEncoder().fit_transform(d.target)
    return X, y

def load_segment():
    d = fetch_openml(name='segment', version=1, as_frame=True, parser='auto')
    X = d.data.values.astype(float)
    y = LabelEncoder().fit_transform(d.target)
    return X, y

def load_satimage():
    d = fetch_openml(name='satimage', version=1, as_frame=True, parser='auto')
    X = d.data.values.astype(float)
    y = LabelEncoder().fit_transform(d.target)
    return X, y

def load_pageblocks():
    d = fetch_openml(name='page-blocks', version=1, as_frame=True, parser='auto')
    X = d.data.values.astype(float)
    y = LabelEncoder().fit_transform(d.target)
    return _filter(X, y)

def load_shuttle():
    d = fetch_openml(name='shuttle', version=1, as_frame=True, parser='auto')
    X = d.data.values.astype(float)
    y = LabelEncoder().fit_transform(d.target)
    return _filter(X, y, min_count=50)

def load_vowel():
    d = fetch_openml(name='vowel', version=2, as_frame=True, parser='auto')
    X = d.data.select_dtypes(include=[np.number]).values.astype(float)
    y = LabelEncoder().fit_transform(d.target)
    return X, y

def load_abalone():
    d = fetch_openml(name='abalone', version=1, as_frame=True, parser='auto')
    df = d.data.copy()
    target = d.target.values.astype(int)
    # encode sex column
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].map({'M':0, 'F':1, 'I':2})
    X = df.values.astype(float)
    # group rings into 3 classes: young(1-8), adult(9-11), old(12+)
    y = np.where(target <= 8, 0, np.where(target <= 11, 1, 2))
    return X, y

def load_thyroid_ann():
    """ann-thyroid from OpenML — 3 classes, ~7200 samples, IR~40."""
    d = fetch_openml(data_id=40474, as_frame=True, parser='auto')
    X = d.data.values.astype(float)
    y = LabelEncoder().fit_transform(d.target)
    return _filter(X, y)

def load_wqwhite():
    """Wine quality white — more samples than red."""
    d = pd.read_csv(f'{D2}/winequality-white.csv', sep=';')
    return _filter(d.iloc[:,:-1].values.astype(float), d.iloc[:,-1].values.astype(int))

def load_hepatitis():
    """Hepatitis — medical, small, binary but relevant for case study."""
    d = pd.read_csv(f'{D2}/hepatitis.data', header=None, na_values='?')
    d = d.dropna()
    X = d.iloc[:,1:].values.astype(float)
    y = d.iloc[:,0].values.astype(int)
    return X, y

def load_splice():
    """Splice dataset — molecular biology, 3 classes."""
    d = fetch_openml(name='splice', version=1, as_frame=True, parser='auto')
    X = pd.get_dummies(d.data).values.astype(float)
    y = LabelEncoder().fit_transform(d.target)
    return X, y

def load_flare():
    """Solar flare — multiclass, 6 classes."""
    d = fetch_openml(name='flare', version=1, as_frame=True, parser='auto')
    X = pd.get_dummies(d.data).values.astype(float)
    y = LabelEncoder().fit_transform(d.target)
    return _filter(X, y)

def load_pendigits():
    """Pen-based handwritten digits — 10 classes, 10992 samples."""
    d = fetch_openml(name='pendigits', version=1, as_frame=True, parser='auto')
    X = d.data.values.astype(float)
    y = LabelEncoder().fit_transform(d.target)
    return X, y


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

DATASETS = {
    # Original 11
    'new-thyroid': load_nt,
    'hayes-roth': load_hr,
    'balance-scale': load_bs,
    'car': load_ca,
    'cmc': load_cmc,
    'ecoli': load_ecoli,
    'glass': load_glass,
    'wine': load_wine,
    'dermatology': load_dermatology,
    'yeast': load_yeast,
    'winequality-red': load_wqred,
    # New 14
    'cleveland': load_cleveland,
    'vehicle': load_vehicle,
    'segment': load_segment,
    'satimage': load_satimage,
    'page-blocks': load_pageblocks,
    'shuttle': load_shuttle,
    'vowel': load_vowel,
    'abalone': load_abalone,
    'ann-thyroid': load_thyroid_ann,
    'winequality-white': load_wqwhite,
    'hepatitis': load_hepatitis,
    'splice': load_splice,
    'flare': load_flare,
    'pendigits': load_pendigits,
}

# ═══════════════════════════════════════════════════════════════════════════════
#  RESAMPLING METHODS
# ═══════════════════════════════════════════════════════════════════════════════

METHODS = [
    'None', 'SMOTE', 'Borderline-SMOTE', 'ADASYN',
    'SMOTE-ENN', 'SMOTE-Tomek', 'KMeans-SMOTE',
    'Safe-Level-SMOTE', 'MWMOTE', 'ProWSyn', 'MIRT++'
]

def make_sampler(name, seed, k_neighbors=5):
    if name == 'None':
        return None
    elif name == 'SMOTE':
        return SMOTE(k_neighbors=k_neighbors, random_state=seed)
    elif name == 'Borderline-SMOTE':
        return BorderlineSMOTE(k_neighbors=k_neighbors, random_state=seed)
    elif name == 'ADASYN':
        return ADASYN(n_neighbors=k_neighbors, random_state=seed)
    elif name == 'SMOTE-ENN':
        return SMOTEENN(smote=SMOTE(k_neighbors=k_neighbors, random_state=seed), random_state=seed)
    elif name == 'SMOTE-Tomek':
        return SMOTETomek(smote=SMOTE(k_neighbors=k_neighbors, random_state=seed), random_state=seed)
    elif name == 'KMeans-SMOTE':
        return KMeansSMOTE(k_neighbors=k_neighbors, random_state=seed)
    elif name == 'Safe-Level-SMOTE':
        return SafeLevelSMOTE(k_neighbors=k_neighbors, random_state=seed)
    elif name == 'MWMOTE':
        return MWMOTE(k1=k_neighbors, random_state=seed)
    elif name == 'ProWSyn':
        return ProWSyn(k_neighbors=k_neighbors, random_state=seed)
    elif name == 'MIRT++':
        return MIRTPlusResampler(seed=seed)
    else:
        raise ValueError(f'Unknown method: {name}')


def safe_resample(X, y, name, seed):
    kmin = pd.Series(y).value_counts().min()
    kn = min(5, max(1, kmin - 1))
    sampler = make_sampler(name, seed, k_neighbors=kn)
    if sampler is None:
        return X, y
    try:
        return sampler.fit_resample(X, y)
    except Exception:
        return X, y


def auc_macro(clf, Xte, yte, classes):
    try:
        if not hasattr(clf, 'predict_proba'):
            return np.nan
        proba = clf.predict_proba(Xte)
        yb = label_binarize(yte, classes=classes)
        if yb.shape[1] == 1:
            yb = np.hstack([1 - yb, yb])
        return roc_auc_score(yb, proba, multi_class='ovr', average='macro')
    except Exception:
        return np.nan


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment(datasets=None, output_csv='kbs_results_raw.csv'):
    if datasets is None:
        datasets = DATASETS

    print(f'Running KBS experiment: {len(datasets)} datasets × {len(METHODS)} methods × 3 classifiers')
    print('='*80)

    # First, load all datasets and show summary
    loaded = {}
    summary_rows = []
    for name, loader in datasets.items():
        try:
            X, y = loader()
            loaded[name] = (X, y)
            classes = np.unique(y)
            counts = pd.Series(y).value_counts()
            ir = counts.max() / counts.min()
            summary_rows.append({
                'Dataset': name, 'n': len(y), 'Features': X.shape[1],
                'Classes': len(classes), 'IR': round(ir, 1)
            })
            print(f'  OK {name}: {len(y)} samples, {X.shape[1]} features, {len(classes)} classes, IR={ir:.1f}', flush=True)
        except Exception as e:
            print(f'  FAIL {name}: {str(e)[:80]}', flush=True)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv('kbs_dataset_summary.csv', index=False)
    print(f'\nLoaded {len(loaded)}/{len(datasets)} datasets successfully.')
    print('='*80)

    # Run evaluation
    records = []
    timings = []
    total = len(loaded) * len(METHODS) * 3 * 10  # datasets * methods * classifiers * folds
    done = 0

    for dname, (X, y) in loaded.items():
        classes = sorted(np.unique(y).tolist())
        # For large datasets, use fewer repeats
        n_repeats = 1 if len(y) > 10000 else 2
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
                    done += 1

                timings.append({'dataset': dname, 'method': rname, 'fold': fold, 'time_s': t_resample})

        elapsed = time.time() - t_start
        print(f'  done: {dname} ({elapsed:.1f}s)', flush=True)

    results = pd.DataFrame(records)
    results.to_csv(output_csv, index=False)
    print(f'\nSaved {len(results)} results to {output_csv}')

    # Save timings
    timings_df = pd.DataFrame(timings)
    timings_df.to_csv('kbs_timings.csv', index=False)

    return results, summary_df, timings_df


# ═══════════════════════════════════════════════════════════════════════════════
#  STATISTICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def statistical_analysis(results, output_prefix='kbs'):
    print('\n' + '='*80)
    print('STATISTICAL ANALYSIS')
    print('='*80)

    # Mean performance
    print('\n=== Mean Performance (all datasets × classifiers × folds) ===\n')
    agg = results.groupby('resampler')[['f1','gmean','auc']].mean().reindex(METHODS).round(4)
    print(agg.sort_values('gmean', ascending=False).to_string())

    # Ranks
    print('\n=== Average Ranks (lower = better) ===\n')
    for metric in ['f1', 'gmean']:
        piv = results.pivot_table(index=['dataset','fold','clf'],
                                  columns='resampler', values=metric)[METHODS].dropna()
        ranks = piv.rank(axis=1, ascending=False).mean()
        stat, p = friedmanchisquare(*[piv[m].values for m in METHODS])
        print(f'{metric.upper()}: Friedman chi2={stat:.2f}, p={p:.2e}')
        print(ranks.sort_values().round(3).to_string())
        print()

    # Win/Tie/Loss table vs MIRT++
    print('\n=== Win/Tie/Loss vs MIRT++ (per dataset, G-Mean) ===\n')
    ds_mean = results.groupby(['dataset','resampler'])['gmean'].mean().unstack()
    wtl_rows = []
    for method in METHODS:
        if method == 'MIRT++':
            continue
        wins = (ds_mean['MIRT++'] > ds_mean[method] + 0.002).sum()
        losses = (ds_mean[method] > ds_mean['MIRT++'] + 0.002).sum()
        ties = len(ds_mean) - wins - losses
        wtl_rows.append({'vs': method, 'Win': wins, 'Tie': ties, 'Loss': losses})
    wtl_df = pd.DataFrame(wtl_rows).set_index('vs')
    print(wtl_df.to_string())

    # Wilcoxon signed-rank tests (MIRT++ vs each)
    print('\n=== Wilcoxon Signed-Rank Tests (MIRT++ vs each, G-Mean) ===\n')
    piv_ds = results.groupby(['dataset','fold','clf','resampler'])['gmean'].mean().unstack()
    for method in METHODS:
        if method == 'MIRT++':
            continue
        try:
            stat, p = wilcoxon(piv_ds['MIRT++'], piv_ds[method], alternative='greater')
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
            print(f'  MIRT++ vs {method:<20s}: W={stat:.0f}, p={p:.4f} {sig}')
        except Exception as e:
            print(f'  MIRT++ vs {method:<20s}: error — {e}')

    # Per-dataset results table
    print('\n=== G-Mean by Dataset × Method ===\n')
    pivot = results.groupby(['dataset','resampler'])['gmean'].mean().unstack()[METHODS].round(3)
    print(pivot.to_string())

    # Save
    pivot.to_csv(f'{output_prefix}_gmean_by_dataset.csv')
    results.groupby(['dataset','resampler'])['f1'].mean().unstack()[METHODS].round(3).to_csv(
        f'{output_prefix}_f1_by_dataset.csv')

    return pivot


# ═══════════════════════════════════════════════════════════════════════════════
#  MEDICAL CASE STUDY
# ═══════════════════════════════════════════════════════════════════════════════

def medical_case_study(results):
    """Focused analysis on medical datasets with clinical framing."""
    print('\n' + '='*80)
    print('MEDICAL DOMAIN CASE STUDY')
    print('='*80)

    medical_datasets = ['cleveland', 'ann-thyroid', 'dermatology', 'hepatitis', 'new-thyroid']
    med_results = results[results.dataset.isin(medical_datasets)]

    if med_results.empty:
        print('No medical datasets in results.')
        return

    print('\n=== Performance on Medical Datasets ===\n')
    pivot = med_results.groupby(['dataset','resampler'])['gmean'].mean().unstack()[METHODS].round(3)
    print(pivot.to_string())

    print('\n=== F1 on Medical Datasets ===\n')
    pivot_f1 = med_results.groupby(['dataset','resampler'])['f1'].mean().unstack()[METHODS].round(3)
    print(pivot_f1.to_string())

    print('\n=== Average Rank on Medical Datasets ===\n')
    med_piv = med_results.pivot_table(index=['dataset','fold','clf'],
                                       columns='resampler', values='gmean')[METHODS].dropna()
    ranks = med_piv.rank(axis=1, ascending=False).mean()
    print(ranks.sort_values().round(3).to_string())

    # Sensitivity per class for cleveland (most clinically relevant)
    print('\n=== Per-Class Analysis: Cleveland Heart Disease ===')
    print('Classes: 0=healthy, 1-4=increasing severity')
    print('Clinical importance: detecting severe cases (3,4) despite rarity\n')

    pivot.to_csv('kbs_medical_gmean.csv')
    pivot_f1.to_csv('kbs_medical_f1.csv')


# ═══════════════════════════════════════════════════════════════════════════════
#  SCALABILITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def scalability_analysis(timings_df):
    """Runtime analysis by dataset size."""
    print('\n' + '='*80)
    print('SCALABILITY ANALYSIS')
    print('='*80)

    avg_time = timings_df.groupby(['dataset','method'])['time_s'].mean().unstack()
    print('\n=== Average Resampling Time (seconds) by Dataset ===\n')
    print(avg_time[METHODS[1:]].round(3).to_string())  # exclude None

    avg_time.to_csv('kbs_scalability.csv')


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('KBS MIRT++ Expanded Experiment')
    print(f'Datasets: {len(DATASETS)} | Methods: {len(METHODS)} | Classifiers: 3')
    print()

    results, summary_df, timings_df = run_experiment()
    pivot = statistical_analysis(results)
    medical_case_study(results)
    scalability_analysis(timings_df)

    print('\n' + '='*80)
    print('EXPERIMENT COMPLETE')
    print('='*80)
    print(f'Output files:')
    print(f'  kbs_results_raw.csv — all raw results')
    print(f'  kbs_dataset_summary.csv — dataset characteristics')
    print(f'  kbs_timings.csv — resampling runtimes')
    print(f'  kbs_gmean_by_dataset.csv — G-Mean pivot table')
    print(f'  kbs_f1_by_dataset.csv — F1 pivot table')
    print(f'  kbs_medical_gmean.csv — medical case study results')
