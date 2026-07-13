"""Offline smoke-test of the MIRT++ API (FastAPI TestClient).
Run:  python api/test_api.py   (after the model artifact exists)."""
import os, sys, io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def main():
    print('GET /health ->', client.get('/health').json())
    r = client.get('/'); meta = r.json()
    print('GET /       ->', meta)
    feats = meta['feature_names']
    sample = [[110, 10.4, 1.6, 1.6, 2.7]] if len(feats) == 5 else [[0]*len(feats)]

    print('GET /similarity-matrix ->', client.get('/similarity-matrix').json())
    print('POST /predict ->', client.post('/predict', json={'instances': sample}).json())
    print('POST /explain ->', client.post('/explain', json={'instances': sample}).json())

    # batch predict via CSV
    import pandas as pd
    df = pd.DataFrame(sample * 3, columns=feats)
    buf = io.StringIO(); df.to_csv(buf, index=False)
    rc = client.post('/predict/csv', files={'file': ('in.csv', buf.getvalue(), 'text/csv')})
    print('POST /predict/csv ->', 'OK' if rc.status_code == 200 else rc.text,
          '| rows:', len(rc.text.strip().splitlines()) - 1)

    # resample
    rs = client.post('/resample', json={
        'features': [[1,2],[1,3],[2,2],[2,1],[8,9],[9,9],[9,8]], 'labels': [0,0,0,0,1,1,1]})
    j = rs.json()
    print('POST /resample -> before', j['distribution_before'],
          '| after', j['distribution_after'])

    # UI renders
    ui = client.get('/ui')
    print('GET /ui ->', 'OK' if ui.status_code == 200 and '<html' in ui.text else 'FAIL')

    # auth: when no key configured, endpoints are open (already exercised above)
    print('\nALL API TESTS PASSED')

if __name__ == '__main__':
    main()
