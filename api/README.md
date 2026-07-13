# MIRT++ API

A REST service wrapping **MIRT++** — the enhanced Multiclass Informative Resampling Technique
(validated ESDA class-similarity + difficulty-aware synthetic resampling).

## 1. Create the model artifact

Run the **"Train & persist the production model"** cell (Section 9) in
`MIRT_Plus_Enhanced.ipynb`. It writes `api/mirtplus_model.joblib`.

## 2. Run locally

```bash
cd "ML Project"
pip install -r api/requirements.txt
uvicorn api.main:app --reload --port 8000
```

- **Web UI:** http://localhost:8000/ui
- **Swagger docs:** http://localhost:8000/docs

## 3. Run with Docker

```bash
cd "ML Project"
docker build -t mirtplus-api -f api/Dockerfile .
docker run -p 8000:8000 -e MIRTPLUS_API_KEY=secret123 mirtplus-api
```

## Authentication

Set the environment variable `MIRTPLUS_API_KEY` to require an `X-API-Key` header on
all data endpoints (`/predict`, `/predict/csv`, `/explain`, `/resample`, `/resample/csv`).
If the variable is unset, auth is disabled (development mode). `GET /`, `/health`,
`/similarity-matrix`, and `/ui` are always open.

```bash
export MIRTPLUS_API_KEY=secret123        # Windows: set MIRTPLUS_API_KEY=secret123
uvicorn api.main:app --port 8000
curl -H "X-API-Key: secret123" -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" -d '{"instances": [[110,10.4,1.6,1.6,2.7]]}'
```

## Endpoints

| Method | Path | Auth | Purpose |
|---|---|:--:|---|
| GET  | `/`                   |  | Service metadata |
| GET  | `/health`             |  | Liveness probe |
| GET  | `/ui`                 |  | Minimal web UI (predict / explain / resample CSV / similarity) |
| GET  | `/similarity-matrix`  |  | Learned ESDA inter-class similarity |
| POST | `/predict`            | ✔ | Classify feature rows (JSON) |
| POST | `/predict/csv`        | ✔ | Batch classify an uploaded CSV → CSV with a `prediction` column |
| POST | `/explain`            | ✔ | Predict + confidence + most-confusable class |
| POST | `/resample`           | ✔ | Balance a dataset (JSON in/out) |
| POST | `/resample/csv`       | ✔ | Upload a CSV, download the balanced CSV |

## Examples

**Predict** (new-thyroid: `[T3Resin, T4, T3, TSH, MaxTSH]`)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [[110, 10.4, 1.6, 1.6, 2.7]]}'
```

**Batch predict a CSV** (columns must match the model's feature names)

```bash
curl -X POST http://localhost:8000/predict/csv -F "file=@samples.csv" -o predictions.csv
```

**Resample** an imbalanced dataset

```bash
curl -X POST http://localhost:8000/resample \
  -H "Content-Type: application/json" \
  -d '{"features": [[1,2],[1,3],[2,2],[8,9],[9,9]], "labels": [0,0,0,1,1]}'
```

**Balance a CSV** and download the result

```bash
curl -X POST "http://localhost:8000/resample/csv?label_column=Class&strategy=max" \
  -F "file=@imbalanced.csv" -o balanced.csv
```

## Tests

```bash
python api/test_api.py   # offline TestClient smoke test of every endpoint
```

## Notes
- The API imports `mirtplus.py` from the project root, so the served resampler is
  identical to the one evaluated in the notebook.
- To serve a different task, retrain and re-persist the artifact with the desired
  dataset/model in Section 9 of the notebook.
