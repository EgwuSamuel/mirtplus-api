"""Fig. 1 - MIRT architecture (same content as experiment/fig_architecture.py, re-laid-out
for a 174 mm Springer page width without overlapping boxes)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({'font.family': 'Arial', 'font.size': 7.5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'mathtext.fontset': 'dejavusans'})
INK, BLUE, FILL = '#0b0b0b', '#2a78d6', '#eef4fc'
fig, ax = plt.subplots(figsize=(6.85, 3.9))
ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis('off')


def box(x, y, w, h, title, lines=(), dashed=False, fill='white'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.25,rounding_size=1.2',
                                lw=0.9, ec=INK, fc=fill, ls='--' if dashed else '-'))
    ax.text(x + w / 2, y + h - 1.6, title, ha='center', va='top', fontweight='bold', fontsize=7.6)
    for i, t in enumerate(lines):
        ax.text(x + 1.2, y + h - 5.6 - i * 3.0, t, ha='left', va='top', fontsize=6.6, color=INK)


def arrow(p, q, dashed=False, color=INK):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=8, lw=0.9, color=color,
                                 ls='--' if dashed else '-', shrinkA=0, shrinkB=0))


# row 1 (y 33-58): data -> ESDA -> mu -> validation
box(0.5, 33, 12.5, 25, 'Training fold', ['multiclass,', 'imbalanced', r'$(X, y)$'])
box(16, 33, 40.5, 25, 'Stage 1: ESDA (Sect. 3.2)',
    ['1  standardise features', r'2  random-forest feature weights $w_f$',
     '3  per-class subgroups (k-means + silhouette)', '4  weighted Mahalanobis subgroup distances',
     r'5  $s=\exp(-\alpha d)$, median heuristic for $\alpha$',
     '6  size-weighted aggregation, shrinkage, min-max'])
box(58.5, 41, 16, 17, r'Similarity $\mu$', [r'$\mu \in [0,1]^{K\times K}$', r'$\mu_{ii}=1$'], fill=FILL)
box(78, 41, 21.5, 17, 'Validation (Sect. 3.3)',
    [r'$\mu$ vs. cross-validated', 'confusion matrix', '(Pearson, Spearman)'], dashed=True)
# row 2 (y 2-24): typing -> synthesis -> balanced -> classifier
box(0.5, 2, 21, 22, 'Stage 2: typing',
    ['k-NN in weighted space', 'Safe / Borderline /', 'Rare / Outlier', r'noise $\nu_c$ = share of R+O'])
box(25.5, 2, 42.5, 22, 'Stage 3: synthesis (Sect. 3.5)',
    [r'$s^{*}_c=\mathrm{max}_{j\neq c}\;\mu_{cj}$  (most confusable class)',
     r'$\kappa_c=1$ if rare, else $\mathrm{max}(\kappa_{\min},\,1-\beta s^{*}_c\nu_c)$',
     r'$x_{new}=x_s+U(0,\kappa_c)\,(x_n-x_s)$',
     r'$x_n$: same-class neighbour; grow to $n_{maj}$'])
box(71.5, 7, 12, 14, 'Balanced', ['training', 'set'])
box(86, 7, 13.5, 14, 'Classifier', ['k-NN, CART,', 'SVM, RF,', 'LightGBM'])

arrow((13.4, 45.5), (15.6, 45.5))
arrow((56.9, 49.5), (58.1, 49.5))
arrow((74.9, 49.5), (77.6, 49.5), dashed=True)
arrow((26, 32.6), (11, 24.4))                          # weighted space -> typing
arrow((66.5, 40.6), (58, 24.4), color=BLUE)            # mu -> synthesis
ax.text(63.6, 31.5, r'$\mu$', color=BLUE, fontsize=8)
arrow((21.9, 13), (25.1, 13))
arrow((68.4, 14), (71.1, 14))
arrow((83.9, 14), (85.6, 14))
for e in ('eps', 'pdf', 'png'):
    fig.savefig(f'Fig1.{e}', bbox_inches='tight', facecolor='white', **({'dpi': 600} if e == 'png' else {}))
print('Fig1 written')
