"""Convert main.tex from the portable article preamble to the Springer Nature template (sn-jnl,
sn-basic reference style, as required by JIIS). Only the front matter and back matter change."""
import re

s = open('main_portable.tex', encoding='utf-8').read()

PREAMBLE = r"""%% MIRT manuscript for the Journal of Intelligent Information Systems (Springer Nature template,
%% sn-jnl.cls Version 3.1 December 2024, reference style sn-basic as required by JIIS).
\documentclass[pdflatex,sn-basic]{sn-jnl}
\usepackage{graphicx}%
\usepackage{multirow}%
\usepackage{amsmath,amssymb,amsfonts}%
\usepackage{bm}%
\usepackage{booktabs}%
\usepackage{algorithm}%
\usepackage{algorithmicx}%
\usepackage{algpseudocode}%
\newcommand{\MIRT}{\textsc{mirt}}
\input{numbers.tex}   % every number quoted in the text, generated from the results by make_assets.py
\raggedbottom

\begin{document}

\title[MIRT: similarity-modulated multiclass resampling]{MIRT: A Similarity-Modulated, Difficulty-Aware
Resampling Technique for Multiclass Imbalanced Classification}

\author[1]{\fnm{Dickson Apaleokhai} \sur{Dako}}
\author*[1]{\fnm{Samuel Onuche-Ojo} \sur{Egwu}}\email{egwuonucheojosamuel@gmail.com}
\affil*[1]{\orgdiv{Department of Software Engineering}, \orgname{Veritas University Abuja},
\orgaddress{\city{Abuja}, \country{Nigeria}}}

"""

# front matter: everything up to the end of the old keywords line is replaced
abs_body = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', s, re.S).group(1).strip()
kw = re.search(r'\\noindent\\textbf\{Keywords\}(.*?)\n\n', s, re.S).group(1)
kw = ', '.join(k.strip() for k in kw.replace('\n', ' ').split(r'$\cdot$'))
front_end = s.index('\n\n', s.index(r'\noindent\textbf{Keywords}')) + 2
body = s[front_end:]
front = PREAMBLE + '\\abstract{' + abs_body + '}\n\n\\keywords{' + kw + '}\n\n\\maketitle\n\n'

# back matter: Springer \backmatter + Statements and Declarations, bibliography via class option
a = body.index(r'\section*{Statements and Declarations}')
b = body.index(r'\bibliographystyle{unsrtnat}')
decl = body[a:b]
decl = decl.replace(r'\section*{Statements and Declarations}', '')
decl = re.sub(r'\\paragraph\{([^}]*)\}', r'\\bmhead{\1}', decl)
back = (r"""\backmatter

\bmhead{Supplementary information}
Online Resource 1 contains the per-dataset G-mean, balanced accuracy and macro-F1 of all 18 methods.

\section*{Statements and Declarations}
""" + decl.strip() + '\n\n' + r'\bibliography{refs}' + '\n\n' + r'\end{document}' + '\n')
body = body[:a] + back
# the class provides Fig./Table labels and hyperref; nothing else in the body needs to change
out = front + body
open('main.tex', 'w', encoding='utf-8').write(out)
print('main.tex now uses sn-jnl; keywords:', kw)
