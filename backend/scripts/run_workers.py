"""Starts one Celery worker per GPU (or one on CPU).

python -m scripts.run_workers                       # all GPUs, otherwise 1 CPU worker
python -m scripts.run_workers --devices cuda:0      # specific devices
python -m scripts.run_workers --devices cpu,cpu     # e.g. 2 CPU workers (watch the RAM)
"""
import argparse
import os
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default=None, help="comma-separated list: cuda:0,cuda:1 or cpu")
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
        cmd = [sys.executable, "-m", "celery", "-A", "app.worker.celery_app", "worker",
               "--pool=threads", "--concurrency=1", "--loglevel=info", "-n", node]
        print(f"Starting worker {node} on {dev}")
        procs.append(subprocess.Popen(cmd, env=env))
    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
    