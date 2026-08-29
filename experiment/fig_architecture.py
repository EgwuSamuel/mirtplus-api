"""
Black-and-white architecture diagram of MIRT++ for the KBS paper.
Pure B/W: white fills, black strokes, black text. Renders PNG (600 dpi) + PDF.
Structure traced to mirtplus.py (ESDA -> difficulty typing -> synthesis).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({'font.family': 'serif', 'font.size': 9})

fig, ax = plt.subplots(figsize=(7.2, 10.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')

def box(x, y, w, h, lines, title=None, style='round', lw=1.3, ls='-', fs=9, align='left'):
    p = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle=('round,pad=0.6,rounding_size=2' if style=='round'
                                 else 'square,pad=0.6'),
                       fc='white', ec='black', lw=lw, linestyle=ls, zorder=2)
    ax.add_patch(p)
    ty = y + h/2 - 3.2
    if title:
        ax.text(x, ty, title, ha='center', va='top', fontweight='bold', fontsize=fs+0.5)
        ty -= 4.6
    if lines:
        if align == 'left' and len(lines) > 1:
            ax.text(x - w/2 + 5, ty, '\n'.join(lines), ha='left', va='top',
                    fontsize=fs, linespacing=1.5)
        else:
            yc = y if not title else ty
            ax.text(x, yc, '\n'.join(lines), ha='center', va=('center' if not title else 'top'),
                    fontsize=fs, linespacing=1.5)
    return (x, y - h/2, y + h/2)  # center-x, bottom, top

def arrow(x, y0, y1, ls='-', lw=1.3):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle='-|>',
                 mutation_scale=13, lw=lw, color='black', linestyle=ls, zorder=1))

cx = 42
# Input
b_in  = box(cx, 96.5, 46, 5.5, ['Imbalanced multiclass data  (X, y)'], style='square')
# Stage 1 ESDA
b_s1  = box(cx, 80.5, 62, 24,
      ['1.  Standardize features',
       '2.  Random-forest feature weighting  w',
       '3.  Per-class subgroup clustering (KMeans + silhouette)',
       '4.  Weighted-Mahalanobis subgroup distances  d',
       '5.  Similarity  s = exp(-alpha d),  median-heuristic alpha',
       '6.  Size-weighted aggregation, regularization, normalization'],
      title='Stage 1  -  ESDA (Enhanced Similarity Degree Algorithm)')
# mu output
b_mu  = box(cx, 63.5, 50, 6.5,
      ['Class-similarity matrix   mu in [0,1]^(KxK)'], style='square', lw=1.6)
# Validation side branch (right)
b_val = box(84, 63.5, 26, 13,
      ['Correlate mu with', 'empirical confusability', '(Pearson, Spearman)'],
      title='Validation', ls='--', fs=8.5)
# Stage 2
b_s2  = box(cx, 51, 62, 10,
      ['Types:   Safe  /  Borderline  /  Rare  /  Outlier'],
      title='Stage 2  -  Difficulty typing (k-NN)')
# Stage 3
b_s3  = box(cx, 33, 66, 22,
      ['a.  Seed selection per minority class',
       'b.  Most-confusable class from mu :  s* = max_j mu_cj',
       'c.  Noise-gated step cap  kappa  (rare class -> full reach)',
       'd.  Interpolate:  x_new = x_s + U(0, kappa) (x_n - x_s)'],
      title='Stage 3  -  Similarity-modulated, difficulty-aware synthesis')
# Output balanced
b_out = box(cx, 17.5, 48, 6, ['Balanced data  (X_res, y_res)'], style='square')
# Classifier
b_clf = box(cx, 7.5, 44, 6, ['Classifier   (KNN / CART / SVM)'], style='square')

# vertical arrows down the spine
arrow(cx, b_in[1],  b_s1[2])
arrow(cx, b_s1[1],  b_mu[2])
arrow(cx, b_mu[1],  b_s2[2])
arrow(cx, b_s2[1],  b_s3[2])
arrow(cx, b_s3[1],  b_out[2])
arrow(cx, b_out[1], b_clf[2])

# mu feeds Stage 3 (curved feedback on the left) and Validation (dashed, right)
ax.add_patch(FancyArrowPatch((cx-25, 63.5), (cx-33, 63.5), arrowstyle='-',
             lw=1.1, color='black'))
ax.add_patch(FancyArrowPatch((cx-33, 63.5), (cx-33, 33), arrowstyle='-',
             lw=1.1, color='black'))
ax.add_patch(FancyArrowPatch((cx-33, 33), (cx-33, 33), arrowstyle='-|>',
             mutation_scale=13, lw=1.1, color='black'))
ax.add_patch(FancyArrowPatch((cx-33, 33), (b_s3[0]-33, 33), arrowstyle='-|>',
             mutation_scale=13, lw=1.1, color='black'))
ax.text(cx-34.5, 48, 'mu', rotation=90, ha='center', va='center', fontsize=8.5, style='italic')
# dashed mu -> validation
ax.add_patch(FancyArrowPatch((cx+25, 63.5), (71, 63.5), arrowstyle='-|>',
             mutation_scale=12, lw=1.1, color='black', linestyle='--'))

plt.tight_layout(pad=0.4)
fig.savefig('fig_F0_architecture.png', dpi=600, bbox_inches='tight', facecolor='white')
fig.savefig('fig_F0_architecture.pdf', bbox_inches='tight', facecolor='white')
print('saved fig_F0_architecture.png and .pdf')
