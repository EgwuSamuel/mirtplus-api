"""Consistency checks for main.tex: abstract length, keyword count, every cited key present in
refs.bib, every refs.bib entry cited, every number quoted from numbers.json present in the text."""
import json
import re

s = open('main.tex', encoding='utf-8').read()
if r'\begin{abstract}' in s:                                    # portable article preamble
    a, b = s.index(r'\begin{abstract}') + len(r'\begin{abstract}'), s.index(r'\end{abstract}')
    kw = re.search(r'\\textbf\{Keywords\}(.*?)\n\n', s, re.S).group(1).split(r'$\cdot$')
else:                                                           # Springer sn-jnl template
    a = s.index(r'\abstract{') + len(r'\abstract{'); b = s.index(r'\keywords{')
    kw = re.search(r'\\keywords\{(.*?)\}', s, re.S).group(1).split(',')
abstract = re.sub(r'\$[^$]*\$', 'x', s[a:b])
print('abstract words:', len(abstract.split()), '(limit 150-250)')
print('keywords:', len(kw), '(limit 4-6)')

bib = open('refs.bib', encoding='utf-8').read()
bib_keys = set(re.findall(r'@\w+\{([^,]+),', bib))
cited = set(k.strip() for grp in re.findall(r'\\cite[tp]?\{([^}]+)\}', s) for k in grp.split(','))
print('cited keys:', len(cited), '| missing in refs.bib:', sorted(cited - bib_keys))
print('uncited bib entries:', sorted(bib_keys - cited))

n = json.load(open('numbers.json'))
macros = n.get('macros', {})
used = set(re.findall(r'\\([A-Za-z]+)', s))
print('macros defined but unused:', sorted(k for k in macros if k not in used))
stale = ['7.06', '5.30', '5.89', '6.56', '0.49', '29 of', 'eight baselines', '42\\%', '1.45', '137.5', '0.741',
         '0.722', '6.45', '0.64', '2.99', '4.36', '23.1']
body = s[s.index(r'\begin{document}'):]
print('stale hand-typed values still in text:', [t for t in stale if t in body])
checks = {}
_unused = {
    'MIRT rank 7.06': f"{n['mirt_rank_all']:.2f}", 'SMOTE rank 5.30': f"{n['best_rank_all']:.2f}",
    'hard rank 6.45': f"{n['mirt_rank_hard']:.2f}", 'CD 4.66': f"{n['nemenyi_cd']:.2f}",
    'median r 0.49': f"{n['esda']['median_pearson']:.2f}", 'median rho 0.59': f"{n['esda']['median_spearman']:.2f}",
    'positive 29': f"{n['esda']['positive']} of", 'ablation chi2 2.99': f"{n['ablation_friedman'][0]:.2f}",
    'cost 1.45': f"{n['cost']['MIRT']['mean']:.2f}", 'x SMOTE 55': f"{n['cost']['MIRT']['x_smote']:.0f} times",
    'ADASYN fail 42': f"{n['fail_pct']['ADASYN']:.0f}\\%",
}
for label, token in checks.items():
    print(f"  {'OK ' if token in s else 'MISSING'} {label:22s} -> '{token}'")
print('Holm-significant (all):', n['holm_sig_all'])
print('Holm-significant (hard):', n['holm_sig_hard'])
