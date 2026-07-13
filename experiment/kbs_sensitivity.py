"""
Parameter Sensitivity Analysis for MIRT++
==========================================
Varies beta, rare_ratio, and m_max to show MIRT++ is robust.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from imblearn.metrics import geometric_mean_score
from mirtplus import MIRTPlusResampler
import matplotlib.pyplot as plt
import seaborn as sns

# Use a representative subset of datasets for sensitivity
from kbs_experiment import (
    load_nt, load_ca, load_ecoli, load_glass, load_yeast,
    load_dermatology, load_bs, load_cmc, load_wqred, load_wine, load_hr
)

SENS_DATASETS = {
    'new-thyroid': load_nt, 'car': load_ca, 'ecoli': load_ecoli,
    'glass': load_glass, 'yeast': load_yeast, 'dermatology': load_dermatology,
    'balance-scale': load_bs, 'cmc': load_cmc, 'winequality-red': load_wqred,
    'wine': load_wine, 'hayes-roth': load_hr,
}


def eval_config(datasets, **kwargs):
    """Evaluate MIRT++ with given parameters, return mean G-Mean."""
    gm_scores = []
    for name, loader in datasets.items():
        X, y = loader()
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)
        for fold, (tr, te) in enumerate(rskf.split(X, y)):
            try:
                resampler = MIRTPlusResampler(seed=1000+fold, **kwargs)
                Xr, yr = resampler.fit_resample(X[tr], y[tr])
            except Exception:
                Xr, yr = X[tr], y[tr]
            for clf in [KNeighborsClassifier(5),
                        DecisionTreeClassifier(random_state=fold),
                        SVC(C=1, random_state=fold)]:
                clf.fit(Xr, yr)
                gm = geometric_mean_score(y[te], clf.predict(X[te]), average='macro')
                gm_scores.append(gm)
    return np.mean(gm_scores)


def sensitivity_beta():
    """Vary beta (similarity modulation strength)."""
    print('Sensitivity: beta (similarity modulation strength)')
    betas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0]
    results = []
    for b in betas:
        gm = eval_config(SENS_DATASETS, beta=b)
        results.append({'beta': b, 'G-Mean': round(gm, 4)})
        print(f'  beta={b:.2f} → G-Mean={gm:.4f}')
    return pd.DataFrame(results)


def sensitivity_rare_ratio():
    """Vary rare_ratio (threshold for full SMOTE reach on tiny classes)."""
    print('Sensitivity: rare_ratio')
    ratios = [2, 3, 4, 5, 6, 8, 10, 15, 20]
    results = []
    for r in ratios:
        gm = eval_config(SENS_DATASETS, rare_ratio=r)
        results.append({'rare_ratio': r, 'G-Mean': round(gm, 4)})
        print(f'  rare_ratio={r} → G-Mean={gm:.4f}')
    return pd.DataFrame(results)


def sensitivity_m_max():
    """Vary m_max (max subgroups per class in ESDA)."""
    print('Sensitivity: m_max (max subgroups)')
    m_values = [1, 2, 3, 4, 5, 7, 10]
    results = []
    for m in m_values:
        gm = eval_config(SENS_DATASETS, m_max=m)
        results.append({'m_max': m, 'G-Mean': round(gm, 4)})
        print(f'  m_max={m} → G-Mean={gm:.4f}')
    return pd.DataFrame(results)


def sensitivity_strategy():
    """Compare resampling strategies."""
    print('Sensitivity: strategy')
    strategies = ['max', 'mean']
    results = []
    for s in strategies:
        gm = eval_config(SENS_DATASETS, strategy=s)
        results.append({'strategy': s, 'G-Mean': round(gm, 4)})
        print(f'  strategy={s} → G-Mean={gm:.4f}')
    return pd.DataFrame(results)


def plot_sensitivity(df_beta, df_rare, df_mmax):
    """Create publication-quality sensitivity plots."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(df_beta['beta'], df_beta['G-Mean'], 'o-', color='#2c3e50', linewidth=2, markersize=6)
    axes[0].axvline(x=0.15, color='red', linestyle='--', alpha=0.7, label='Default (0.15)')
    axes[0].set_xlabel('β (similarity modulation)', fontsize=11)
    axes[0].set_ylabel('Mean G-Mean', fontsize=11)
    axes[0].set_title('(a) Sensitivity to β', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df_rare['rare_ratio'], df_rare['G-Mean'], 's-', color='#2c3e50', linewidth=2, markersize=6)
    axes[1].axvline(x=6, color='red', linestyle='--', alpha=0.7, label='Default (6)')
    axes[1].set_xlabel('rare_ratio threshold', fontsize=11)
    axes[1].set_ylabel('Mean G-Mean', fontsize=11)
    axes[1].set_title('(b) Sensitivity to rare_ratio', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(df_mmax['m_max'], df_mmax['G-Mean'], 'D-', color='#2c3e50', linewidth=2, markersize=6)
    axes[2].axvline(x=3, color='red', linestyle='--', alpha=0.7, label='Default (3)')
    axes[2].set_xlabel('m_max (subgroups per class)', fontsize=11)
    axes[2].set_ylabel('Mean G-Mean', fontsize=11)
    axes[2].set_title('(c) Sensitivity to m_max', fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('kbs_sensitivity_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('\nSaved: kbs_sensitivity_plots.png')


if __name__ == '__main__':
    print('='*60)
    print('MIRT++ Parameter Sensitivity Analysis')
    print('='*60 + '\n')

    df_beta = sensitivity_beta()
    print()
    df_rare = sensitivity_rare_ratio()
    print()
    df_mmax = sensitivity_m_max()
    print()
    df_strat = sensitivity_strategy()

    # Save results
    all_sens = pd.concat([
        df_beta.assign(param='beta').rename(columns={'beta':'value'}),
        df_rare.assign(param='rare_ratio').rename(columns={'rare_ratio':'value'}),
        df_mmax.assign(param='m_max').rename(columns={'m_max':'value'}),
    ])
    all_sens.to_csv('kbs_sensitivity_results.csv', index=False)

    plot_sensitivity(df_beta, df_rare, df_mmax)

    print('\n' + '='*60)
    print('DONE — Results saved to kbs_sensitivity_results.csv')
    print('='*60)
