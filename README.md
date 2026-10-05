# Secure ML Model Deployment Pipeline

A DevSecMLOps project that trains, validates, secures, tests, and deploys an Iris classification model.

## Current components

- Reproducible model training with a validation quality gate.
- Decision-tree model export with a content-derived version.
- FastAPI prediction service.
- API-key authentication.
- Strict input validation.
- Automated pytest tests.
- Exported-model parity testing.
- Dependency and static security scanning planned.
- Docker deployment planned.
- Input-drift monitoring planned.

## Local commands

Activate the environment:

```bash
source .venv/bin/activate
```

Train the model:

```bash
python -m src.train
```

Run tests:

```bash
python -m pytest -q
```

Start the local API:

```bash
source .env
export API_KEY
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

The API is intentionally bound to `127.0.0.1` for local development.
