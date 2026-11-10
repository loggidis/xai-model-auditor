# XAI Model Auditor

A REST API for auditing object detection models with Explainable AI. Upload a YOLO model and a dataset, run an audit with several XAI methods, compare the heatmaps, rate the results, and export the problematic images for retraining.

## Features

- **Auth**: JWT access tokens, rotating refresh tokens, email verification, rate limiting.
- **Model registry**: Ultralytics YOLO detection models (`.pt`). Architecture, classes, input size and candidate layers are detected automatically.
- **Datasets**: zip upload with optional YOLO labels, safe extraction (no zip slip, size limits).
- **XAI methods**: Grad-CAM++, Integrated Gradients, SHAP (Partition) and LIME, each as a plugin of a common interface, with per-method parameters, masks and timeouts.
- **Audit pipeline**: background jobs (Celery + Redis), batches, per-image error isolation, live progress (SSE), cancel, retry of failed images, notifications.
- **Agreement and auto tags**: pairwise IoU and Pearson between heatmaps, aggregated into `agreement_score`; tags `consistent`, `needs_review`, `uncertain`, `inconclusive`, `no_detection` with configurable thresholds.
- **Evaluations**: a trust score (0-100) and a category per image (`correct_right_reason`, `correct_wrong_reason`, `wrong_prediction`).
- **Reports**: per audit and global; flagged images; export of flagged images (images, YOLO labels, metadata) as a background job.
- **LLM summary** (Groq): asynchronous, available as JSON or PDF; a `dry-run` mode builds a local summary without calling the LLM.
- **Maintenance**: cleanup job for orphaned files and expired tokens.

## Tech stack

Python 3.12, FastAPI, SQLAlchemy 2.0, Alembic, PostgreSQL 16, Redis 7, Celery, PyTorch, Ultralytics, pytorch-grad-cam, Captum, LIME, SHAP, ReportLab, Docker Compose.


## Getting started

**Requirements:** Python 3.12, Docker Desktop, Git. The commands are for Windows PowerShell.

```powershell
# 1. Virtual environment and dependencies
python -m venv venv
.\venv\Scripts\Activate.ps1
cd backend
pip install -r requirements.txt
```

```powershell
# 2. Configuration: two .env files (see "Configuration")
#    <root>\.env          POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB
#    backend\.env         copy backend\.env.example and fill in the values
```

```powershell
# 3. Database and Redis (from the project root)
docker compose up -d
```

```powershell
# 4. Migrations (from backend\)
python -m alembic upgrade head
```

```powershell
# 5. API (terminal 1, from backend\)
python -m uvicorn app.main:app --reload

# 6. Audit worker (terminal 2)
python -m celery -A app.worker.celery_app worker --pool=threads --concurrency=1 --loglevel=info -Q celery -n audit@%h

# 7. Light worker: exports, summaries, cleanup (terminal 3)
python -m celery -A app.worker.celery_app worker --pool=threads --concurrency=1 --loglevel=info -Q exports -n light@%h
```

Steps 6 and 7 can also be started with `python -m scripts.run_workers` (one audit worker per GPU, or one on CPU, plus the light worker; add `--beat` for the daily cleanup). Workers do not reload automatically: restart them after changing code or `.env`.

Open the interactive docs at <http://localhost:8000/docs>. Authorize with the email as the username.

## Scripts

Run them from `backend/`.

```powershell
python -m scripts.run_xai --model best.pt --image img.jpg --method gradcam_pp   # one heatmap PNG, no API
python -m scripts.benchmark_xai --model best.pt --images .\images --write       # runtime per method
python -m scripts.cleanup --dry-run                                             # orphaned files, expired tokens
python -m scripts.send_test_email --to you@example.com                          # check the SMTP settings
```

## Tests

```powershell
cd backend
python -m pytest -q
```

The tests need PostgreSQL and Redis (`docker compose up -d`) and use the real database. They use a small YOLO model with random weights, and they never spend LLM tokens or send real emails. 


