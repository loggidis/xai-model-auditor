import logging

from celery import Celery
from celery.schedules import crontab
from celery.signals import worker_ready

from app.core.config import settings

log = logging.getLogger(__name__)

celery_app = Celery(
    "xai_auditor", broker=settings.REDIS_URL,
    include=["app.worker.tasks", "app.worker.export_tasks",
             "app.worker.summary_tasks", "app.worker.maintenance_tasks"],
)
celery_app.conf.update(
    task_acks_late=True,                 # if a worker dies, the task is delivered again
    worker_prefetch_multiplier=1,        # one heavy task at a time
    worker_concurrency=1,                # one audit at a time per worker (GPU/RAM memory)
    task_ignore_result=True,             # state lives in Postgres, not in Redis
    broker_connection_timeout=2,
    broker_connection_retry_on_startup=True,
    task_publish_retry_policy={"max_retries": 1, "interval_start": 0,
                               "interval_step": 0.2, "interval_max": 0.5},
    broker_transport_options={
        "visibility_timeout": 24 * 3600,  # audits that run for hours
        "socket_connect_timeout": 2,
        "socket_timeout": 5,
    },
    # Light jobs (no model, no GPU) run on their own worker so they never wait behind a long audit.
    task_routes={
        "exports.run": {"queue": "exports"},
        "summaries.run": {"queue": "exports"},
        "maintenance.cleanup": {"queue": "exports"},
    },
    beat_schedule={
        "daily-cleanup": {"task": "maintenance.cleanup", "schedule": crontab(hour=3, minute=0)},
    },
    timezone="UTC",
)


@worker_ready.connect
def _log_worker_setup(sender=None, **kwargs):
    conc = getattr(getattr(sender, "controller", None), "concurrency", None)
    log.info("XAI worker ready: device=%s concurrency=%s", settings.XAI_DEVICE, conc)
    if conc and conc > 1 and settings.XAI_DEVICE != "cpu":
        log.warning("concurrency=%s on a GPU worker: audits are serialised by a lock, "
                    "run one worker per GPU instead of more threads.", conc)