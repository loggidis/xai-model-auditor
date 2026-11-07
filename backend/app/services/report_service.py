import uuid
from collections import Counter
from dataclasses import dataclass, field

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.audit import Audit, AuditResult
from app.models.dataset import Dataset, DatasetImage
from app.models.evaluation import Evaluation
from app.models.ml_model import MLModel
from app.models.user import User
from app.schemas.report import (AuditReport, AuditReportRow, ClassStats, FlaggedItem, GlobalReport,
                                PieSlice)
from app.services.progress import progress_counts
from app.xai.tagging import TAGS

CATEGORIES = ("correct_right_reason", "correct_wrong_reason", "wrong_prediction")
STATUSES = ("pending", "running", "completed", "failed", "cancelled")


# ---------- helpers ----------

def _pct(n: int, total: int) -> float:
    return round(100 * n / total, 1) if total else 0.0


def _pie(counts: dict[str, int], order: tuple[str, ...] | list[str], total: int) -> list[PieSlice]:
    keys = list(order) + sorted(k for k in counts if k not in order)
    return [PieSlice(key=k, count=counts.get(k, 0), percent=_pct(counts.get(k, 0), total)) for k in keys]


def _avg(total: float, n: int, digits: int) -> float | None:
    return round(total / n, digits) if n else None


@dataclass
class _ClassAcc:
    detections: int = 0
    conf_sum: float = 0.0
    images: int = 0
    evaluated: int = 0
    trust_sum: float = 0.0


class _Acc:
    """Accumulates report statistics over a stream of processed (done) results."""

    def __init__(self) -> None:
        self.done = 0
        self.evaluated = 0
        self.tags: Counter = Counter()
        self.cats: Counter = Counter()
        self.trust_sum = 0.0
        self.conf_sum, self.conf_n = 0.0, 0
        self.agr_sum, self.agr_n = 0.0, 0
        self.classes: dict[str, _ClassAcc] = {}

    def add(self, auto_tag: str | None, agreement: float | None, detections: list | None,
            ev: tuple[str, int] | None) -> None:
        self.done += 1
        self.tags[auto_tag or "untagged"] += 1
        if agreement is not None:
            self.agr_sum += agreement
            self.agr_n += 1

        seen: set[str] = set()
        primary, best = None, -1.0               # primary class = class of the most confident detection
        for d in detections or []:
            name = str(d.get("class_name") or d.get("class_id"))
            conf = float(d.get("conf", 0.0))
            c = self.classes.setdefault(name, _ClassAcc())
            c.detections += 1
            c.conf_sum += conf
            self.conf_sum += conf
            self.conf_n += 1
            if name not in seen:
                seen.add(name)
                c.images += 1
            if conf > best:
                best, primary = conf, name

        if ev is not None:
            category, trust = ev
            self.evaluated += 1
            self.cats[category] += 1
            self.trust_sum += trust
            if primary is not None:
                c = self.classes[primary]
                c.evaluated += 1
                c.trust_sum += trust

    # ---- outputs ----
    def tag_pie(self) -> list[PieSlice]:
        return _pie(dict(self.tags), list(TAGS), self.done)

    def category_pie(self) -> list[PieSlice]:
        counts = {c: self.cats.get(c, 0) for c in CATEGORIES}
        counts["unevaluated"] = self.done - self.evaluated
        return _pie(counts, [*CATEGORIES, "unevaluated"], self.done)

    def class_stats(self) -> list[ClassStats]:
        out = [
            ClassStats(class_name=name, detections=c.detections, images=c.images,
                       avg_confidence=_avg(c.conf_sum, c.detections, 4),
                       evaluated_images=c.evaluated, avg_trust=_avg(c.trust_sum, c.evaluated, 1))
            for name, c in self.classes.items()
        ]
        return sorted(out, key=lambda s: (-s.detections, s.class_name))

    @property
    def avg_confidence(self) -> float | None:
        return _avg(self.conf_sum, self.conf_n, 4)

    @property
    def avg_trust(self) -> float | None:
        return _avg(self.trust_sum, self.evaluated, 1)

    @property
    def avg_agreement(self) -> float | None:
        return _avg(self.agr_sum, self.agr_n, 4)


def _result_rows(db: Session, audit_ids: list[uuid.UUID]):
    return db.execute(
        select(AuditResult.audit_id, AuditResult.image_id, AuditResult.auto_tag,
               AuditResult.agreement_score, AuditResult.detections)
        .where(AuditResult.audit_id.in_(audit_ids), AuditResult.status == "done")
    )


def _evaluations(db: Session, audit_ids: list[uuid.UUID]) -> dict[tuple[uuid.UUID, uuid.UUID], tuple[str, int]]:
    rows = db.execute(
        select(Evaluation.audit_id, Evaluation.image_id, Evaluation.category, Evaluation.trust_score)
        .where(Evaluation.audit_id.in_(audit_ids))
    )
    return {(a, i): (c, t) for a, i, c, t in rows}


# ---------- report of one audit ----------

def build_audit_report(db: Session, audit: Audit) -> AuditReport:
    counts = progress_counts(db, audit.id, audit.total_images)
    evs = _evaluations(db, [audit.id])
    acc = _Acc()
    for aid, image_id, tag, agreement, detections in _result_rows(db, [audit.id]):
        acc.add(tag, agreement, detections, evs.get((aid, image_id)))
    return AuditReport(
        audit_id=audit.id, status=audit.status, total_images=audit.total_images,
        processed_images=acc.done, failed_images=counts["failed"],
        remaining_images=counts["pending"] + counts["processing"], evaluated_images=acc.evaluated,
        by_auto_tag=acc.tag_pie(), by_category=acc.category_pie(),
        avg_confidence=acc.avg_confidence, avg_trust=acc.avg_trust, avg_agreement=acc.avg_agreement,
        classes=acc.class_stats(),
    )


# ---------- report of all audits of a user ----------

def build_global_report(db: Session, user: User, limit: int, offset: int) -> GlobalReport:
    rows = db.execute(
        select(Audit, MLModel.name.label("model_name"), Dataset.name.label("dataset_name"))
        .join(MLModel, MLModel.id == Audit.model_id)
        .join(Dataset, Dataset.id == Audit.dataset_id)
        .where(Audit.user_id == user.id)
        .order_by(Audit.created_at.desc())
    ).all()
    ids = [r[0].id for r in rows]

    glob = _Acc()
    per: dict[uuid.UUID, _Acc] = {i: _Acc() for i in ids}
    failed: dict[uuid.UUID, int] = {}
    if ids:
        evs = _evaluations(db, ids)
        for aid, image_id, tag, agreement, detections in _result_rows(db, ids):
            ev = evs.get((aid, image_id))
            per[aid].add(tag, agreement, detections, ev)
            glob.add(tag, agreement, detections, ev)
        failed = {aid: n for aid, n in db.execute(
            select(AuditResult.audit_id, func.count())
            .where(AuditResult.audit_id.in_(ids), AuditResult.status == "failed")
            .group_by(AuditResult.audit_id))}

    page = []
    for audit, model_name, dataset_name in rows[offset:offset + limit]:
        a = per[audit.id]
        page.append(AuditReportRow(
            audit_id=audit.id, created_at=audit.created_at, status=audit.status,
            model_id=audit.model_id, model_name=model_name,
            dataset_id=audit.dataset_id, dataset_name=dataset_name,
            total_images=audit.total_images, processed_images=a.done,
            failed_images=failed.get(audit.id, 0), evaluated_images=a.evaluated,
            needs_review_images=a.tags.get("needs_review", 0),
            problematic_evaluations=a.cats.get("correct_wrong_reason", 0) + a.cats.get("wrong_prediction", 0),
            avg_confidence=a.avg_confidence, avg_trust=a.avg_trust, avg_agreement=a.avg_agreement,
        ))

    status_counts = Counter(r[0].status for r in rows)
    return GlobalReport(
        audits_total=len(rows), audits_by_status=_pie(dict(status_counts), STATUSES, len(rows)),
        images_processed=glob.done, evaluated_images=glob.evaluated,
        by_auto_tag=glob.tag_pie(), by_category=glob.category_pie(),
        avg_confidence=glob.avg_confidence, avg_trust=glob.avg_trust, avg_agreement=glob.avg_agreement,
        classes=glob.class_stats(), audits=page, limit=limit, offset=offset,
    )


# ---------- flagged images ----------

@dataclass
class FlaggedRow:
    image_id: uuid.UUID
    filename: str
    image_index: int
    has_label: bool
    reason: str
    sort_score: float | None
    score_source: str
    category: str | None
    trust_score: int | None
    notes: str | None
    auto_tag: str | None
    agreement_score: float | None
    detections: list = field(default_factory=list)

    @property
    def detection_count(self) -> int:
        return len(self.detections)

    @property
    def mean_confidence(self) -> float | None:
        confs = [float(d.get("conf", 0.0)) for d in self.detections]
        return round(sum(confs) / len(confs), 4) if confs else None


def flagged_items(db: Session, audit: Audit, trust_below: int = 50,
                  include_unevaluated: bool = True) -> list[FlaggedRow]:
    """Problematic images, sorted by ascending trust (agreement * 100 as fallback when not evaluated)."""
    evs = {
        image_id: (category, trust, notes)
        for image_id, category, trust, notes in db.execute(
            select(Evaluation.image_id, Evaluation.category, Evaluation.trust_score, Evaluation.notes)
            .where(Evaluation.audit_id == audit.id, Evaluation.user_id == audit.user_id))
    }
    rows = db.execute(
        select(AuditResult.image_id, AuditResult.image_index, AuditResult.auto_tag,
               AuditResult.agreement_score, AuditResult.detections,
               DatasetImage.filename, DatasetImage.has_label)
        .join(DatasetImage, DatasetImage.id == AuditResult.image_id)
        .where(AuditResult.audit_id == audit.id, AuditResult.status == "done")
    ).all()

    out: list[FlaggedRow] = []
    for image_id, index, tag, agreement, detections, filename, has_label in rows:
        ev = evs.get(image_id)
        category = trust = notes = None
        if ev is not None:
            category, trust, notes = ev
            if category == "wrong_prediction":
                reason = "user:wrong_prediction"
            elif category == "correct_wrong_reason":
                reason = "user:correct_wrong_reason"
            elif trust < trust_below:
                reason = "user:low_trust"
            else:
                continue
            sort_score, source = float(trust), "evaluation"
        else:
            if not include_unevaluated or tag != "needs_review":
                continue
            reason = "auto:needs_review"
            if agreement is not None:
                sort_score, source = round(agreement * 100, 1), "agreement"
            else:
                sort_score, source = None, "none"
        out.append(FlaggedRow(
            image_id=image_id, filename=filename, image_index=index, has_label=has_label,
            reason=reason, sort_score=sort_score, score_source=source, category=category,
            trust_score=trust, notes=notes, auto_tag=tag, agreement_score=agreement,
            detections=detections or [],
        ))
    out.sort(key=lambda r: (r.sort_score is None, r.sort_score if r.sort_score is not None else 0.0,
                            r.image_index))
    return out


def to_flagged_item(r: FlaggedRow) -> FlaggedItem:
    return FlaggedItem(
        image_id=r.image_id, filename=r.filename, image_index=r.image_index, reason=r.reason,
        score_source=r.score_source, sort_score=r.sort_score, category=r.category,
        trust_score=r.trust_score, notes=r.notes, auto_tag=r.auto_tag,
        agreement_score=r.agreement_score, mean_confidence=r.mean_confidence,
        detection_count=r.detection_count,
    )