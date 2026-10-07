"""Make an editable Word copy of the manuscript (MIRT_JIIS_manuscript.docx) from main.tex.

The LaTeX source stays the master file for submission (JIIS accepts LaTeX only). This script:
expands the generated number macros (numbers.tex), inlines the generated tables, swaps the PDF figures
for their 600-dpi PNG versions, turns the algorithm box into a numbered list, and converts with pandoc
(native Word equations, numbered references from refs.bib in Springer basic style).
"""
import os
import re
import pypandoc

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'main_portable.tex'), encoding='utf-8').read()  # article-class twin of main.tex

# 1. number macros -> literal values
macros = dict(re.findall(r'\\newcommand\{\\(\w+)\}\{(.*)\}', open(os.path.join(HERE, 'numbers.tex'), encoding='utf-8').read()))
src = src.replace(r'\input{numbers.tex}', '')
for name in sorted(macros, key=len, reverse=True):
    src = re.sub(r'\\' + name + r'(\{\})?(?![A-Za-z])', lambda m, v=macros[name]: v.replace('\\', '\\\\') if False else v, src)

# 2. inline generated tables (expanding macros in them too)
def inline(m):
    t = open(os.path.join(HERE, m.group(1)), encoding='utf-8').read()
    t = re.sub(r'\\(scriptsize|footnotesize|small)\b', '', t)
    t = re.sub(r'\\setlength\{\\tabcolsep\}\{[^}]*\}', '', t)
    t = re.sub(r'\\cmidrule\([^)]*\)\{[^}]*\}', '', t)
    t = t.replace(r'\cmidrule{', r'%\cmidrule{')
    return t
src = re.sub(r'\\input\{(tab_\w+\.tex)\}', inline, src)

# 3. figures: PDF -> PNG (Word cannot embed PDF graphics)
src = re.sub(r'(\\includegraphics\[[^\]]*\]\{Fig\d)\.pdf\}', r'\1.png}', src)

# 4. algorithm box -> numbered list
ALG = r"""\noindent\textbf{Algorithm 1} MIRT resampling. \emph{Input:} training data $(X,y)$; parameters
$m_{\max},\lambda,\beta,\kappa_{\min},\rho,k$. \emph{Output:} the balanced training set.
\begin{enumerate}
\item $\tilde X\gets$ standardise$(X)$; $w\gets$ random-forest importances; $\hat X\gets\tilde X\odot\sqrt{w}$ (Eq.~1).
\item $\mu\gets$ ESDA$(\hat X,y)$ (Eqs.~2--4).
\item Type every example by $k$-NN in $\hat X$ (Eq.~5).
\item For each class $c$ with $n_c<n_{\mathrm{maj}}$: $\nu_c\gets$ share of rare/outlier examples of $c$;
$s^{*}_c\gets\max_{j\neq c}\mu_{cj}$; $\kappa_c\gets 1$ if $n_{\mathrm{maj}}/n_c\geq\rho$, else
$\max(\kappa_{\min},1-\beta s^{*}_c\nu_c)$ (Eq.~6).
\item Repeat $n_{\mathrm{maj}}-n_c$ times: draw a seed $x_s\in D_c$ and a same-class neighbour $x_n$ (in
$\hat X$); add $x_{\mathrm{new}}=x_s+u(x_n-x_s)$ with $u\sim U(0,\kappa_c\eta_s)$ (Eq.~7).
\item Return the original and synthetic examples, shuffled.
\end{enumerate}"""
src = re.sub(r'\\begin\{algorithm\}.*?\\end\{algorithm\}', lambda m: ALG, src, flags=re.S)

# 5. equation numbers: pandoc does not number display equations, so write them in explicitly
eq_labels = re.findall(r'\\label\{(eq:[^}]+)\}', src)
counter = iter(range(1, len(eq_labels) + 1))


def number_equation(m):
    body = re.sub(r'\\label\{eq:[^}]+\}', '', m.group(1))
    return r'\begin{equation}' + body.rstrip() + r'\qquad\text{(%d)}' % next(counter) + '\n' + r'\end{equation}'


src = re.sub(r'\\begin\{equation\}(.*?)\\end\{equation\}', number_equation, src, flags=re.S)
for i, lab in enumerate(eq_labels, 1):
    src = src.replace(r'\ref{%s}' % lab, str(i))

# 6. small compatibility fixes for pandoc's LaTeX reader
src = src.replace(r'\bm{', r'\boldsymbol{')
src = re.sub(r'\\DeclareMathOperator\*?\{[^}]*\}\{[^}]*\}', '', src)
src = src.replace(r'\citep{', r'\cite{')
# author block: plain paragraphs after the title
src = re.sub(r'\\author\{.*?\n\\date\{\}', r'\\date{}', src, flags=re.S)
AUTH = r"""\begin{center}
Dickson Apaleokhai Dako\textsuperscript{1} and Samuel Onuche-Ojo Egwu\textsuperscript{1,*}

\textsuperscript{1}Department of Software Engineering, Veritas University Abuja, Abuja, Nigeria

\textsuperscript{*}Corresponding author: egwuonucheojosamuel@gmail.com
\end{center}
"""
src = src.replace(r'\maketitle', r'\maketitle' + '\n' + AUTH)

tmp = os.path.join(HERE, 'main_word.tex')
open(tmp, 'w', encoding='utf-8').write(src)
out = os.path.join(HERE, 'MIRT_JIIS_manuscript.docx')
pypandoc.convert_file(tmp, 'docx', outputfile=out, extra_args=[
    '--citeproc', '--bibliography=refs.bib', '--csl=springer-basic-numeric.csl',
    '--resource-path=' + HERE, '--number-sections', '--metadata=reference-section-title:References',
], cworkdir=HERE)
print('written', out)
