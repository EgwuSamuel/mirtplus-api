"""Online Resource 1 (ESM_1): per-dataset results that do not fit the 25-page limit.
Writes ESM_1.tex (compiled to ESM_1.pdf) and the underlying CSV files."""
import os
import sys
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = next(p for p in (os.path.join(HERE, '..', 'experiment'),                   # inside the repo
                       os.path.join(HERE, '..', 'mirtplus-api', 'experiment'))    # local working copy
           if os.path.isdir(p))
sys.path.insert(0, EXP)

V2 = ['MIRT-v2', 'MIRT-v2-Lite', 'MIRT-v2[uniform-mu]', 'MIRT-v2[crisp-clean]', 'MIRT-v2[no-clean]']
df = pd.read_csv(os.path.join(EXP, 'results_ext', 'ALL_RESULTS.csv'))
df = df[~df.method.isin(V2)].copy()
df['method'] = df['method'].replace({'MIRT-v1': 'MIRT', 'NoResample': 'None'})
ab = {'Borderline-SMOTE': 'B-SMOTE', 'Safe-Level-SMOTE': 'SL-SMOTE', 'KMeans-SMOTE': 'KM-SMOTE',
      'Static-SMOTE': 'St-SMOTE', 'SMOTE-Tomek': 'S-Tomek', 'SMOTE-ENN': 'S-ENN'}

TABLE = r"""\begin{center}\scriptsize\setlength{\tabcolsep}{2pt}
\textbf{Table S%d.} Mean %s per dataset over 5$\times$2 cross-validation and five classifiers
(columns ordered by average rank; best value per dataset in bold)\\[4pt]
\begin{adjustbox}{max width=\linewidth, max totalheight=0.80\textheight}
\begin{tabular}{@{}l%s@{}}\toprule
%s\midrule
%s
\bottomrule\end{tabular}\end{adjustbox}\end{center}\clearpage"""

blocks = []
for metric, label in [('gmean', 'G-mean'), ('bacc', 'balanced accuracy'), ('f1', 'macro-F1')]:
    P = df.groupby(['dataset', 'method'])[metric].mean().unstack()
    order = P.rank(axis=1, ascending=False).mean().sort_values().index.tolist()
    P = P[order]
    P.round(4).to_csv(os.path.join(HERE, f'ESM_1_{metric}_per_dataset.csv'))
    head = ' & '.join(['Dataset'] + [r'\rotatebox{90}{%s}' % ab.get(m, m) for m in order]) + r'\\'
    rows = []
    for d, r in P.iterrows():
        best = r.max()
        cells = [(r'\textbf{%.3f}' % v) if v == best else '%.3f' % v for v in r.values]
        rows.append(d + ' & ' + ' & '.join(cells) + r'\\')
    blocks.append(TABLE % (len(blocks) + 1, label, 'r' * len(order), head, '\n'.join(rows)))

E = pd.read_csv(os.path.join(EXP, 'results_ext', 'tables', 'T_esda_confusability.csv'))
E.round(3).to_csv(os.path.join(HERE, 'ESM_1_esda_confusability.csv'), index=False)

HEADER = r"""\documentclass[10pt,a4paper,landscape]{article}
\usepackage[margin=1.5cm]{geometry}
\usepackage{booktabs,graphicx,adjustbox}
\pagestyle{empty}
\begin{document}
\noindent\textbf{Online Resource 1} -- \emph{MIRT: A Similarity-Modulated, Difficulty-Aware Resampling
Technique for Multiclass Imbalanced Classification}.
D.A. Dako, S.O. Egwu (corresponding author, egwuonucheojosamuel@gmail.com), Department of Software
Engineering, Veritas University Abuja, Nigeria. \emph{Journal of Intelligent Information Systems}.
"""
open(os.path.join(HERE, 'ESM_1.tex'), 'w', encoding='utf-8').write(HEADER + '\n'.join(blocks) + '\n\\end{document}\n')
print('ESM_1.tex and CSVs written')
