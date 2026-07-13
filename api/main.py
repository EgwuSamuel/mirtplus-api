"""
MIRT++ REST API — FastAPI service exposing the enhanced multiclass
imbalanced-data resampling technique and a trained classifier.

Run locally:
    cd "ML Project"
    uvicorn api.main:app --reload --port 8000
Then open http://localhost:8000/ui for the web UI, or /docs for Swagger.

Auth: set env var MIRTPLUS_API_KEY to require an `X-API-Key` header on data
endpoints. If unset, auth is disabled (development mode).

Endpoints
    GET  /                  service metadata
    GET  /health            liveness probe
    GET  /ui                minimal web UI
    GET  /similarity-matrix learned ESDA inter-class similarity
    POST /predict           classify feature rows (JSON)
    POST /predict/csv       batch classify an uploaded CSV
    POST /explain           predict + difficulty + most-confusable class
    POST /resample          balance an uploaded dataset (JSON)
    POST /resample/csv      balance an uploaded CSV, download the result
"""
import os
import io
from typing import List

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mirtplus import MIRTPlusResampler   # shared implementation

# ─────────────────────────────────────────────────────────────────────────────
ARTIFACT_PATH = os.path.join(os.path.dirname(__file__), 'mirtplus_model.joblib')

app = FastAPI(
    title='MIRT++ API',
    description='Enhanced Multiclass Informative Resampling Technique — '
                'validated class-similarity + difficulty-aware synthetic resampling.',
    version='1.1.0',
)

_artifact = None


def get_artifact():
    global _artifact
    if _artifact is None:
        if not os.path.exists(ARTIFACT_PATH):
            raise HTTPException(503, 'Model artifact not found. Run the '
                                     '"Train & persist" cell in the notebook.')
        _artifact = joblib.load(ARTIFACT_PATH)
    return _artifact


# ── Schemas ──────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    instances: List[List[float]] = Field(
        ..., description='Rows of feature values, ordered as feature_names.',
        examples=[[[110, 10.4, 1.6, 1.6, 2.7]]])


class ResampleRequest(BaseModel):
    features: List[List[float]]
    labels: List
    strategy: str = 'max'
    clean_frac: float = 0.0
    seed: int = 42


# ── Meta routes ──────────────────────────────────────────────────────────────
@app.get('/')
def root():
    art = get_artifact()
    return {'service': 'MIRT++ API', 'version': '1.1.0',
            'dataset': art.get('dataset'), 'classes': art['classes'],
            'feature_names': art['feature_names'], 'model': type(art['model']).__name__,
            'auth_required': False, 'ui': '/ui', 'docs': '/docs'}


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.get('/similarity-matrix')
def similarity_matrix():
    art = get_artifact()
    return {'classes': art['classes'], 'similarity': art['similarity']}


# ── Prediction ───────────────────────────────────────────────────────────────
def _predict_matrix(X):
    art = get_artifact()
    if X.ndim != 2 or X.shape[1] != len(art['feature_names']):
        raise HTTPException(422, f'Each instance needs {len(art["feature_names"])} '
                                 f'features: {art["feature_names"]}')
    model = art['model']
    preds = model.predict(X)
    proba = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
    return art, model, preds, proba


@app.post('/predict')
def predict(req: PredictRequest):
    _, model, preds, proba = _predict_matrix(np.asarray(req.instances, dtype=float))
    out = {'predictions': [int(p) if np.issubdtype(type(p), np.integer) else p for p in preds]}
    if proba is not None:
        out['probabilities'] = [{str(c): float(p) for c, p in zip(model.classes_, row)}
                                for row in proba]
    return out


@app.post('/predict/csv')
async def predict_csv(file: UploadFile = File(...)):
    art = get_artifact()
    df = pd.read_csv(io.BytesIO(await file.read()))
    feats = art['feature_names']
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise HTTPException(422, f'CSV missing feature columns: {missing}')
    X = df[feats].values.astype(float)
    _, _, preds, _ = _predict_matrix(X)
    df['prediction'] = preds
    buf = io.StringIO(); df.to_csv(buf, index=False); buf.seek(0)
    return StreamingResponse(iter([buf.getvalue()]), media_type='text/csv',
                             headers={'Content-Disposition': 'attachment; filename=predictions.csv'})


@app.post('/explain')
def explain(req: PredictRequest):
    art, model, preds, proba = _predict_matrix(np.asarray(req.instances, dtype=float))
    sim = art['similarity']
    results = []
    for i, p in enumerate(preds):
        row = {'predicted_class': int(p) if np.issubdtype(type(p), np.integer) else p}
        if proba is not None:
            row['confidence'] = round(float(max(proba[i])), 4)
        others = {k: v for k, v in sim.get(str(p), {}).items() if k != str(p)}
        if others:
            mc = max(others, key=others.get)
            row['most_confusable_with'] = mc
            row['similarity'] = round(float(others[mc]), 4)
        results.append(row)
    return {'explanations': results}


# ── Resampling ───────────────────────────────────────────────────────────────
@app.post('/resample')
def resample(req: ResampleRequest):
    X = np.asarray(req.features, dtype=float)
    y = np.asarray(req.labels)
    if len(X) != len(y):
        raise HTTPException(422, 'features and labels must have equal length.')
    if len(np.unique(y)) < 2:
        raise HTTPException(422, 'Need at least two classes.')
    rs = MIRTPlusResampler(strategy=req.strategy, clean_frac=req.clean_frac, seed=req.seed)
    Xr, yr = rs.fit_resample(X, y)
    vc = lambda a: {str(k): int(v) for k, v in pd.Series(a).value_counts().sort_index().items()}
    return {'distribution_before': vc(y), 'distribution_after': vc(yr),
            'n_synthetic': int(len(yr) - len(y)),
            'similarity_matrix': {str(k): {str(k2): round(float(v2), 4) for k2, v2 in v.items()}
                                  for k, v in rs.mu.items()},
            'features': Xr.tolist(), 'labels': yr.tolist()}


@app.post('/resample/csv')
async def resample_csv(file: UploadFile = File(...),
                       label_column: str = 'Class', strategy: str = 'max'):
    df = pd.read_csv(io.BytesIO(await file.read()))
    if label_column not in df.columns:
        raise HTTPException(422, f'label_column "{label_column}" not in {list(df.columns)}')
    y = df[label_column].values
    X = df.drop(columns=[label_column]).values.astype(float)
    feat_cols = [c for c in df.columns if c != label_column]
    Xr, yr = MIRTPlusResampler(strategy=strategy, seed=42).fit_resample(X, y)
    out = pd.DataFrame(Xr, columns=feat_cols); out[label_column] = yr
    buf = io.StringIO(); out.to_csv(buf, index=False); buf.seek(0)
    return StreamingResponse(iter([buf.getvalue()]), media_type='text/csv',
                             headers={'Content-Disposition': 'attachment; filename=balanced.csv'})


# ── Minimal web UI ───────────────────────────────────────────────────────────
@app.get('/ui', response_class=HTMLResponse)
def ui():
    art = get_artifact()
    feats = art['feature_names']
    fields = ''.join(
        f'<label>{f}<input name="{f}" type="number" step="any" value="0"></label>'
        for f in feats)
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>MIRT++ API</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
 *{{box-sizing:border-box}}
 body{{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:860px;margin:0 auto;padding:1.5rem 1rem 4rem;color:#1a1a2e;background:#f8f8fc}}
 h1{{color:#3a0ca3;margin:0 0 .25rem}} h2{{margin:.2rem 0 1rem;color:#3a0ca3;font-size:1.1rem}}
 .card{{background:#fff;border:1px solid #e0e0f0;border-radius:14px;padding:1.4rem 1.6rem;margin:1rem 0;box-shadow:0 2px 8px rgba(58,12,163,.06)}}
 .hero{{background:linear-gradient(135deg,#3a0ca3 0%,#7209b7 100%);color:#fff;border-radius:14px;padding:1.8rem 1.8rem 1.6rem;margin-bottom:1.2rem}}
 .hero h1{{color:#fff;font-size:1.8rem}} .hero p{{margin:.5rem 0 0;opacity:.9;font-size:.97rem;line-height:1.6}}
 .badge-row{{display:flex;flex-wrap:wrap;gap:.5rem;margin-top:1rem}}
 .badge{{display:inline-block;background:rgba(255,255,255,.18);color:#fff;border-radius:20px;padding:.2rem .85rem;font-size:.8rem;border:1px solid rgba(255,255,255,.3)}}
 .guide-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:.8rem;margin:.8rem 0 0}}
 .guide-item{{background:#f3f0ff;border-radius:10px;padding:.9rem 1rem}}
 .guide-item .gi-icon{{font-size:1.4rem;margin-bottom:.3rem}}
 .guide-item .gi-title{{font-weight:600;font-size:.88rem;color:#3a0ca3;margin-bottom:.25rem}}
 .guide-item .gi-desc{{font-size:.8rem;color:#555;line-height:1.5}}
 label{{display:inline-block;margin:.3rem .6rem .3rem 0;font-size:.9rem}}
 input[type=text],input[type=number],select{{padding:.38rem .6rem;border:1px solid #ccc;border-radius:6px;margin-left:.3rem;font-size:.9rem}}
 input[type=file]{{margin:.4rem 0}}
 button{{background:#3a0ca3;color:#fff;border:0;padding:.55rem 1.2rem;border-radius:8px;cursor:pointer;font-size:.92rem;margin-right:.4rem}}
 button:hover{{background:#560bad}} button.sec{{background:#f0ebff;color:#3a0ca3;border:1px solid #c4b5f7}}
 button.sec:hover{{background:#e0d4ff}}
 pre{{background:#0f0f1e;color:#c3f0ca;padding:1rem;border-radius:8px;overflow:auto;font-size:.82rem;margin-top:.8rem}}
 .key-section{{background:#fff8e1;border:1px solid #ffe082;border-radius:10px;padding:1rem 1.2rem;margin:.8rem 0}}
 .key-section h3{{margin:0 0 .5rem;font-size:.95rem;color:#7a5c00}}
 .key-section code{{background:#fff3cd;padding:.1rem .4rem;border-radius:4px;font-size:.85rem;color:#5d4037}}
 .key-section ol{{margin:.5rem 0 0 1rem;padding:0;font-size:.88rem;color:#444;line-height:1.9}}
 .step-label{{display:inline-block;background:#3a0ca3;color:#fff;border-radius:50%;width:20px;height:20px;text-align:center;font-size:.75rem;line-height:20px;margin-right:.4rem;flex-shrink:0}}
 .divider{{border:0;border-top:1px solid #e8e4f4;margin:1.2rem 0}}
 .status-row{{display:flex;gap:1.5rem;flex-wrap:wrap;font-size:.85rem;margin-top:.6rem}}
 .status-item span{{color:#888}}
 .status-val{{font-weight:600;color:#3a0ca3}}
 a{{color:#3a0ca3}}
</style></head><body>

<div class="hero">
  <h1>MIRT++ API</h1>
  <p>Enhanced Multiclass Informative Resampling Technique — a research API that fixes <strong>class imbalance</strong>
  in multiclass datasets using similarity-aware, difficulty-modulated synthetic oversampling.
  Upload any imbalanced CSV and download a balanced version instantly.</p>
  <div class="badge-row">
    <span class="badge">23 datasets evaluated</span>
    <span class="badge">11 methods compared</span>
    <span class="badge">Friedman p &lt; 1e-36</span>
    <span class="badge">KBS submission</span>
  </div>
</div>

<div class="card">
  <h2>What is this?</h2>
  <div class="guide-grid">
    <div class="guide-item">
      <div class="gi-icon">⚖️</div>
      <div class="gi-title">Resample CSV</div>
      <div class="gi-desc">Upload any imbalanced dataset as a CSV. MIRT++ balances all classes and returns the new CSV for download. Works on any domain.</div>
    </div>
    <div class="guide-item">
      <div class="gi-icon">🔍</div>
      <div class="gi-title">Predict &amp; Explain</div>
      <div class="gi-desc">Enter feature values for the demo thyroid dataset. Get a class prediction, confidence score, and which class it's most likely to confuse.</div>
    </div>
    <div class="guide-item">
      <div class="gi-icon">🧬</div>
      <div class="gi-title">ESDA Similarity</div>
      <div class="gi-desc">View the learned inter-class similarity matrix computed by the Enhanced Similarity Degree Algorithm — the core novelty of this work.</div>
    </div>
    <div class="guide-item">
      <div class="gi-icon">📡</div>
      <div class="gi-title">REST API</div>
      <div class="gi-desc">All features are available as JSON endpoints. See <a href="/docs">/docs</a> for the full Swagger reference and request/response schemas.</div>
    </div>
  </div>
  <hr class="divider">
  <div class="status-row">
    <div class="status-item"><span>Demo model: </span><span class="status-val">{art.get('dataset','—')}</span></div>
    <div class="status-item"><span>Classifier: </span><span class="status-val">{type(art['model']).__name__}</span></div>
    <div class="status-item"><span>Classes: </span><span class="status-val">{art['classes']}</span></div>
    <div class="status-item"><span>Auth: </span><span class="status-val">Open</span></div>
    <div class="status-item"><span>Docs: </span><a class="status-val" href="/docs">/docs</a></div>
  </div>
</div>

<div class="key-section" style="background:#eaf3de;border-color:#c0dd97">
  <h3 style="color:#3b6d11">✅ Open access — no API key required</h3>
  <p style="font-size:.88rem;color:#444;margin:.3rem 0 0">
    All endpoints are freely accessible. Use the UI below or call the REST endpoints directly from your code.
    Full reference at <a href="/docs">/docs</a>.
  </p>
</div>

<div class="card">
  <h2>⚖️ Resample a CSV &nbsp;<span style="font-size:.8rem;font-weight:400;color:#888">— works on any dataset</span></h2>
  <p style="font-size:.9rem;color:#555;margin:.2rem 0 1rem">Upload your imbalanced CSV file. MIRT++ will balance all classes using similarity-aware synthetic oversampling and return the balanced file.</p>
  <label>Label column <input id="lbl" value="Class"></label>
  <label>Strategy
    <select id="strat">
      <option value="max">max — oversample all to the majority count</option>
      <option value="mean">mean — oversample minorities to the average count</option>
    </select>
  </label><br><br>
  <input id="csv" type="file" accept=".csv">
  <br><br>
  <button onclick="resampleCsv()">⬇ Balance &amp; download</button>
  <span id="resample-status" style="font-size:.85rem;color:#888;margin-left:.5rem"></span>
</div>

<div class="card">
  <h2>🔍 Predict &amp; Explain &nbsp;<span style="font-size:.8rem;font-weight:400;color:#888">— demo model: {art.get('dataset','')}</span></h2>
  <p style="font-size:.9rem;color:#555;margin:.2rem 0 1rem">Enter feature values for the <strong>{art.get('dataset','')}</strong> dataset and get a class prediction with confidence score and the most confusable class (ESDA-powered).</p>
  <div>{fields}</div>
  <button onclick="predict()">Predict</button>
  <button class="sec" onclick="explain()">Explain</button>
  <pre id="out">Results will appear here…</pre>
</div>

<div class="card">
  <h2>🧬 ESDA Similarity matrix</h2>
  <p style="font-size:.9rem;color:#555;margin:.2rem 0 1rem">The learned inter-class similarity computed by the Enhanced Similarity Degree Algorithm (ESDA) — feature-weighted, Mahalanobis-based, validated against classifier confusability.</p>
  <button class="sec" onclick="simMatrix()">Load similarity matrix</button>
  <pre id="sim">Click the button to load…</pre>
</div>

<script>
const feats={feats!r};
function vec(){{return feats.map(f=>parseFloat(document.querySelector(`[name="${{f}}"]`).value)||0);}}
const hdr={{'Content-Type':'application/json'}};
async function post(url,body){{const r=await fetch(url,{{method:'POST',headers:hdr,body:JSON.stringify(body)}});return r.json();}}
async function predict(){{document.getElementById('out').textContent=JSON.stringify(await post('/predict',{{instances:[vec()]}}),null,2);}}
async function explain(){{document.getElementById('out').textContent=JSON.stringify(await post('/explain',{{instances:[vec()]}}),null,2);}}
async function simMatrix(){{const r=await fetch('/similarity-matrix');document.getElementById('sim').textContent=JSON.stringify(await r.json(),null,2);}}
async function resampleCsv(){{
 const f=document.getElementById('csv').files[0];
 if(!f){{alert('Please choose a CSV file first.');return;}}
 const status=document.getElementById('resample-status');
 status.textContent='Processing…';
 const fd=new FormData();fd.append('file',f);
 const url=`/resample/csv?label_column=${{encodeURIComponent(document.getElementById('lbl').value)}}&strategy=${{document.getElementById('strat').value}}`;
 try{{
   const r=await fetch(url,{{method:'POST',body:fd}});
   if(!r.ok){{const t=await r.text();status.textContent='';alert('Error: '+t);return;}}
   const blob=await r.blob();
   const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='balanced.csv';a.click();
   status.textContent='Done! File downloaded.';
   setTimeout(()=>status.textContent='',4000);
 }}catch(e){{status.textContent='';alert('Request failed: '+e);}}
}}
</script></body></html>"""
