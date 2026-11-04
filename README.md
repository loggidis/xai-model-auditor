# XAI Model Auditor

Web εφαρμογή για audit μοντέλων computer vision με μεθόδους Explainable AI (Grad-CAM++, Integrated Gradients, LIME, SHAP).

**Κατάσταση:** Sprint 2 (auth, model registry, datasets, file storage)

## Τεχνολογίες

Python 3.12, FastAPI, SQLAlchemy 2.0, Alembic, PostgreSQL 16 (Docker), PyJWT, bcrypt

## Εγκατάσταση

```powershell
# 1. Dependencies
python -m venv venv
venv\Scripts\activate
cd backend
pip install -r requirements.txt
```

**2. Δημιούργησε δύο αρχεία `.env`** (ίδιος user/κωδικός/βάση και στα δύο):

`.env` (ρίζα, για το Docker):

```
POSTGRES_USER=xai_user
POSTGRES_PASSWORD=<κωδικός>
POSTGRES_DB=xai_auditor
```

`backend/.env` (για το FastAPI):

```
DATABASE_URL=postgresql+psycopg://xai_user:<κωδικός>@localhost:5432/xai_auditor
SECRET_KEY=<python -c "import secrets; print(secrets.token_hex(32))">
```

**3. Βάση και migrations**

```powershell
docker compose up -d db      # από τη ρίζα
cd backend
alembic upgrade head
```

**4. Εκκίνηση**

```powershell
uvicorn app.main:app --reload
```

Swagger UI: http://localhost:8000/docs

## Endpoints

Όλα, εκτός από `register`, `login` και `health`, απαιτούν token (`Authorization: Bearer <JWT>`).

| Method | Path | Περιγραφή |
|---|---|---|
| POST | `/api/auth/register` | Εγγραφή |
| POST | `/api/auth/login` | Login, επιστρέφει JWT (email στο πεδίο `username`) |
| GET | `/api/auth/me` | Τρέχων χρήστης |
| POST | `/api/models` | Upload μοντέλου (`.pt`/`.pth`) με `name`, `task`, προαιρετικά `description`, `num_classes`, `input_size` |
| GET | `/api/models` | Λίστα μοντέλων του χρήστη |
| GET | `/api/models/{id}` | Λεπτομέρειες μοντέλου |
| DELETE | `/api/models/{id}` | Διαγραφή (βάση και αρχείο) |
| POST | `/api/datasets` | Upload zip εικόνων (+ προαιρετικά YOLO labels) με `name` |
| GET | `/api/datasets` | Λίστα datasets του χρήστη |
| GET | `/api/datasets/{id}` | Λεπτομέρειες dataset |
| GET | `/api/datasets/{id}/images` | Λίστα εικόνων (`limit`, `offset`) |
| DELETE | `/api/datasets/{id}` | Διαγραφή (βάση και αρχεία) |
| GET | `/api/health` | Liveness check |

## Storage

Τα αρχεία αποθηκεύονται στο `storage/` (δεν γίνεται commit). Στη βάση κρατιούνται σχετικά paths.

```
storage/
├── models/<user_id>/<model_id>/model.pt
└── datasets/<user_id>/<dataset_id>/
    ├── images/
    └── labels/          (YOLO, προαιρετικά)
```

Το zip δέχεται επίπεδη δομή ή `images/` + `labels/`. Τα labels αντιστοιχίζονται με τις εικόνες μέσω ονόματος (`img001.jpg` ↔ `img001.txt`). Τα όρια μεγέθους ορίζονται στο `app/core/config.py`.

## Migrations και tests

Όλες οι εντολές τρέχουν από το `backend/`.

```powershell
alembic revision --autogenerate -m "περιγραφή"
alembic upgrade head
python -m pytest
```

Κάθε νέο model πρέπει να γίνεται import στο `app/models/__init__.py`. Τα tests τρέχουν προς την πραγματική βάση.