"""Build refs.bib from the DOI registry (BibTeX via DOI content negotiation = publisher metadata).
Entries without a DOI (JMLR, NeurIPS, Scand. J. Stat.) were verified on the publishers' pages
and are written by hand below. Run: python build_bib.py  -> refs.bib + bib_check.txt
"""
import re
import time
import urllib.request

DOIS = {
    # surveys / foundations
    'he2009learning': '10.1109/TKDE.2008.239', 'krawczyk2016challenges': '10.1007/s13748-016-0094-0',
    'fernandez2018smote': '10.1613/jair.1.11192', 'chen2024survey': '10.1007/s10462-024-10759-6',
    'wang2012multiclass': '10.1109/TSMCB.2012.2187280', 'fernandez2013multiple': '10.1016/j.knosys.2013.01.018',
    'lango2022difficult': '10.1016/j.eswa.2022.116962', 'saez2016types': '10.1016/j.patcog.2016.03.012',
    'elreedy2023theory': '10.1007/s10994-022-06296-4', 'kovacs2019asoc': '10.1016/j.asoc.2019.105662',
    'araf2024cost': '10.1007/s10462-023-10652-8', 'salmi2024medical': '10.1007/s10462-024-10884-2',
    # JIIS
    'napierala2016types': '10.1007/s10844-015-0368-1', 'lango2018rbb': '10.1007/s10844-017-0446-7',
    'napierala2012bracid': '10.1007/s10844-011-0193-0', 'zefrehi2023mamipot': '10.1007/s10844-022-00763-z',
    'khleel2023sdp': '10.1007/s10844-023-00793-1',
    # SMOTE family and recent oversamplers
    'chawla2002smote': '10.1613/jair.953', 'han2005borderline': '10.1007/11538059_91',
    'he2008adasyn': '10.1109/IJCNN.2008.4633969', 'batista2004study': '10.1145/1007730.1007735',
    'douzas2018kmeans': '10.1016/j.ins.2018.06.056', 'bunkhumpornpat2009safe': '10.1007/978-3-642-01307-2_43',
    'barua2014mwmote': '10.1109/TKDE.2012.232', 'barua2013prowsyn': '10.1007/978-3-642-37456-2_27',
    'islam2022knnor': '10.1016/j.asoc.2021.108288', 'kachan2025simplicial': '10.1145/3690624.3709268',
    'zhu2023orem': '10.1109/TKDE.2022.3171706', 'sowah2022hcbst': '10.1145/3488280',
    # multiclass-dedicated and related
    'abdi2016mdo': '10.1109/TKDE.2015.2458858', 'janicka2019soup': '10.2478/amcs-2019-0057',
    'koziarski2020mcccr': '10.1016/j.knosys.2020.106223', 'krawczyk2020mcrbo': '10.1109/TNNLS.2019.2913673',
    'sharma2018swim': '10.1109/ICDM.2018.00060', 'bellinger2020swim': '10.1007/s10115-019-01380-z',
    'naglik2024gmm': '10.1007/s10994-023-06416-8', 'khorshidi2025somm': '10.1007/s10115-025-02394-6',
    'zhu2017smom': '10.1016/j.patcog.2017.07.024', 'li2024tkde': '10.1109/TKDE.2024.3384961',
    'vluymans2018frovoco': '10.1007/s10115-017-1126-1', 'liu2020spe': '10.1109/ICDE48307.2020.00078',
    'siers2021cost': '10.1145/3415156', 'ren2023fsvm': '10.1145/3579050',
    # tools and methods
    'breiman2001rf': '10.1023/A:1010933404324', 'rousseeuw1987silhouette': '10.1016/0377-0427(87)90125-7',
    'maesschalck2000mahalanobis': '10.1016/S0169-7439(99)00047-7', 'vanschoren2013openml': '10.1145/2641190.2641198',
    'kovacs2019smotevariants': '10.1016/j.neucom.2019.06.100', 'grycza2021multiimbalance': '10.1007/978-3-030-67670-4_36',
    'fernandeznavarro2011static': '10.1016/j.patcog.2011.02.019',
}
MANUAL = r"""
@article{demsar2006, author={Dem{\v{s}}ar, Janez}, title={Statistical comparisons of classifiers over multiple data sets},
  journal={Journal of Machine Learning Research}, volume={7}, pages={1--30}, year={2006}}
@article{garcia2008, author={Garc{\'i}a, Salvador and Herrera, Francisco}, title={An extension on ``statistical comparisons of classifiers over multiple data sets'' for all pairwise comparisons},
  journal={Journal of Machine Learning Research}, volume={9}, pages={2677--2694}, year={2008}}
@article{benavoli2017, author={Benavoli, Alessio and Corani, Giorgio and Dem{\v{s}}ar, Janez and Zaffalon, Marco}, title={Time for a change: a tutorial for comparing multiple classifiers through {B}ayesian analysis},
  journal={Journal of Machine Learning Research}, volume={18}, number={77}, pages={1--36}, year={2017}}
@article{pedregosa2011, author={Pedregosa, Fabian and Varoquaux, Ga{\"e}l and Gramfort, Alexandre and others}, title={Scikit-learn: machine learning in {P}ython},
  journal={Journal of Machine Learning Research}, volume={12}, pages={2825--2830}, year={2011}}
@article{lemaitre2017, author={Lema{\^i}tre, Guillaume and Nogueira, Fernando and Aridas, Christos K.}, title={Imbalanced-learn: a {P}ython toolbox to tackle the curse of imbalanced datasets in machine learning},
  journal={Journal of Machine Learning Research}, volume={18}, number={17}, pages={1--5}, year={2017}}
@article{holm1979, author={Holm, Sture}, title={A simple sequentially rejective multiple test procedure},
  journal={Scandinavian Journal of Statistics}, volume={6}, number={2}, pages={65--70}, year={1979}}
@inproceedings{ke2017lightgbm, author={Ke, Guolin and Meng, Qi and Finley, Thomas and Wang, Taifeng and Chen, Wei and Ma, Weidong and Ye, Qiwei and Liu, Tie-Yan}, title={{LightGBM}: a highly efficient gradient boosting decision tree},
  booktitle={Advances in Neural Information Processing Systems 30}, pages={3146--3154}, year={2017}}
@misc{garreau2017median, author={Garreau, Damien and Jitkrittum, Wittawat and Kanagawa, Motonobu}, title={Large sample analysis of the median heuristic},
  howpublished={arXiv preprint arXiv:1707.07269}, year={2017}, doi={10.48550/arXiv.1707.07269}}
"""


def fetch(doi):
    req = urllib.request.Request('https://doi.org/' + doi, headers={
        'Accept': 'application/x-bibtex; charset=utf-8', 'User-Agent': 'mirt-refcheck (mailto:egwuonucheojosamuel@gmail.com)'})
    with urllib.request.urlopen(req, timeout=40) as r:
        return r.read().decode('utf-8')


import json                                     # noqa: E402


def crossref(doi):
    req = urllib.request.Request('https://api.crossref.org/works/' + doi + '?mailto=egwuonucheojosamuel@gmail.com',
                                 headers={'User-Agent': 'mirt-refcheck'})
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.load(r)['message']


def tidy(b, meta):
    b = re.sub(r',\s*month\s*=\s*\w+', '', b)                       # bare month macros
    b = re.sub(r',\s*(url|ISSN|ISBN)\s*=\s*\{[^}]*\}', '', b)        # DOI already links the record
    b = re.sub(r'title=\{(.+?)\},', lambda m: 'title={{%s}},' % m.group(1), b, count=1)  # keep capitals
    b = re.sub(r',\s*pages=\{1[–-]1\}', '', b)                         # early-access placeholder
    pp = (meta.get('published-print') or meta.get('published') or {}).get('date-parts', [[None]])[0][0]
    if pp:
        b = re.sub(r'year=\{\d{4}\}', 'year={%d}' % pp, b)
    ct = meta.get('container-title') or []
    if meta.get('type') == 'book-chapter' and len(ct) > 1:
        b = re.sub(r'^@\w+\{', '@incollection{', b, count=1)
        extra = ('' if 'booktitle=' in b else ', booktitle={%s}' % ct[1]) + ', series={%s}' % ct[0]
        b = b.rstrip().rstrip('}').rstrip() + extra + '}'
    return b


def retry(f, *a, n=4):
    for i in range(n):
        try:
            return f(*a)
        except Exception:
            if i == n - 1:
                raise
            time.sleep(2 * (i + 1))


out, report = [], []
for key, doi in DOIS.items():
    try:
        b = retry(fetch, doi).strip()
        b = tidy(b, retry(crossref, doi))
        b = re.sub(r'^@(\w+)\{[^,]*,', lambda m: '@%s{%s,' % (m.group(1), key), b, count=1)
        t = re.search(r'title\s*=\s*\{(.+?)\},', b)
        out.append(b)
        report.append(f'OK   {key:28s} {doi:38s} {t.group(1)[:80] if t else "?"}')
    except Exception as e:
        report.append(f'FAIL {key:28s} {doi:38s} {e}')
    time.sleep(0.4)
open('refs.bib', 'w', encoding='utf-8').write('\n\n'.join(out) + '\n' + MANUAL)
open('bib_check.txt', 'w', encoding='utf-8').write('\n'.join(report))
print('\n'.join(report))
print(f'\n{sum(r.startswith("OK") for r in report)}/{len(DOIS)} DOI entries + manual entries written to refs.bib')
