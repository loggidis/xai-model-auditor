from celery import Celery
from app.core.config import settings

celery_app = Celery("xai_auditor", broker=settings.REDIS_URL, include=["app.worker.tasks"])
celery_app.conf.update(
    task_acks_late=True,                 
    worker_prefetch_multiplier=1,        
    task_ignore_result=True,             
    broker_transport_options={"visibility_timeout": 24 * 3600},   
    timezone="UTC",
)