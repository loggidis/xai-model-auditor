# XAI Model Auditor

Web εφαρμογή για audit μοντέλων computer vision με μεθόδους Explainable AI (Grad-CAM++, Integrated Gradients, LIME, SHAP).

**Κατάσταση:** Sprint 1 (FastAPI, PostgreSQL, Alembic, JWT auth)

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

| Method | Path | Περιγραφή |
|---|---|---|
| POST | `/api/auth/register` | Εγγραφή |
| POST | `/api/auth/login` | Login, επιστρέφει JWT (email στο πεδίο `username`) |
| GET | `/api/auth/me` | Τρέχων χρήστης (απαιτεί token) |
| GET | `/api/health` | Liveness check |

## Migrations και tests

Όλες οι εντολές τρέχουν από το `backend/`.

```powershell
alembic revision --autogenerate -m "περιγραφή"
alembic upgrade head
pytest
```

Κάθε νέο model πρέπει να γίνεται import στο `app/models/__init__.py`.