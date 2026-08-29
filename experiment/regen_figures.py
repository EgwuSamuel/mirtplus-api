# -*- coding: utf-8 -*-
"""Regenerate all fig_F*.png from the notebook's plotting cells, with the
top-of-figure titles removed (suptitles and 'Figure N:' set_title calls).
Functional panel labels (Classifier: X, dataset names, metric labels) are kept."""
import json, os
import matplotlib
matplotlib.use('Agg')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
nb = json.load(open('kbs_analysis.ipynb', encoding='utf-8'))
CELLS = [2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 18, 19, 21, 22, 24, 25, 26, 27, 28]

def remove_call(src, method):
    """Remove every `<recv>.<method>( ... )` call (balanced parens). Returns new src."""
    out = src
    while True:
        j = out.find('.' + method + '(')
        if j < 0:
            break
        # walk back to include the receiver token (ax, fig, plt, ...)
        s = j
        while s > 0 and (out[s-1].isalnum() or out[s-1] in '_].)'):
            s -= 1
        k = j + len('.' + method + '(')
        depth = 1
        while k < len(out) and depth > 0:
            if out[k] == '(':
                depth += 1
            elif out[k] == ')':
                depth -= 1
            k += 1
        out = out[:s] + 'None' + out[k:]
    return out

def remove_figure_set_titles(src):
    """Remove set_title(...) only when its argument is a figure title
    (text contains 'Figure ' or the bare variable `title`)."""
    out = src
    idx = 0
    while True:
        j = out.find('.set_title(', idx)
        if j < 0:
            break
        s = j
        while s > 0 and (out[s-1].isalnum() or out[s-1] in '_].)'):
            s -= 1
        k = j + len('.set_title(')
        depth = 1
        while k < len(out) and depth > 0:
            if out[k] == '(':
                depth += 1
            elif out[k] == ')':
                depth -= 1
            k += 1
        inner = out[j+len('.set_title('):k-1]
        is_fig = ('Figure ' in inner) or inner.strip().startswith('title')
        if is_fig:
            out = out[:s] + 'None' + out[k:]
            idx = s + 4
        else:
            idx = k
    return out

code = "import matplotlib; matplotlib.use('Agg')\n"
for i in CELLS:
    src = ''.join(nb['cells'][i]['source'])
    src = remove_call(src, 'suptitle')
    src = remove_call(src, 'set_title')   # remove ALL titles (top and per-panel)
    code += '\n# ==== cell %d ====\n' % i + src + '\n'

ns = {'__name__': '__main__'}
exec(compile(code, 'notebook_cells', 'exec'), ns)
print('Regenerated figures with titles removed.')
