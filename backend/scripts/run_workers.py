"""Starts one Celery worker per GPU (or one on CPU), the light-jobs worker, and optionally the scheduler.

python -m scripts.run_workers                       # all GPUs, otherwise 1 CPU worker
python -m scripts.run_workers --devices cuda:0      # specific devices
python -m scripts.run_workers --beat                # also start the scheduler (daily cleanup)
"""
import argparse
import os
import subprocess
import sys

CELERY = [sys.executable, "-m", "celery", "-A", "app.worker.celery_app"]
WORKER = [*CELERY, "worker", "--pool=threads", "--concurrency=1", "--loglevel=info"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default=None, help="comma separated list: cuda:0,cuda:1 or cpu")
    ap.add_argument("--beat", action="store_true", help="start the Celery beat scheduler")
    a = ap.parse_args()

    if a.devices:
        devices = [d.strip() for d in a.devices.split(",") if d.strip()]
    else:
        import torch
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]

    procs = []
    for i, dev in enumerate(devices):
        env = {**os.environ, "XAI_DEVICE": dev}          # read by Settings
        node = f"w{i}-{dev.replace(':', '')}@%h"
        print(f"Starting audit worker {node} on {dev}")
        procs.append(subprocess.Popen([*WORKER, "-Q", "celery", "-n", node], env=env))

    print("Starting light-jobs worker (exports, summaries, cleanup)")
    procs.append(subprocess.Popen([*WORKER, "-Q", "exports", "-n", "light@%h"]))
    if a.beat:
        print("Starting scheduler")
        procs.append(subprocess.Popen([*CELERY, "beat", "--loglevel=info"]))
    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()