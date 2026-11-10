"""Manual cleanup.

python -m scripts.cleanup --dry-run     # shows what would be deleted
python -m scripts.cleanup               # deletes
"""
import argparse
import json
from dataclasses import asdict

from app.db.session import SessionLocal
from app.services.cleanup_service import run_cleanup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    with SessionLocal() as db:
        report = run_cleanup(db, dry_run=a.dry_run)
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()