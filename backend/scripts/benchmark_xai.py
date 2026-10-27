"""Παράδειγμα:
python -m scripts.benchmark_xai --model best.pt --images C:\\path\\to\\images --n 20 --write
"""
import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from app.xai.base import save_runtime_estimate
from app.xai.heatmap_utils import read_image_rgb
from app.xai.inference import detect
from app.xai.loader import load_model
from app.xai.registry import get_method

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TARGET_BATCH_SECONDS = 10   # θέλουμε ενημέρωση progress κάθε ~10s


def timed(device, fn):
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return out, time.perf_counter() - t


def stats(xs):
    return statistics.median(xs), statistics.mean(xs), float(np.percentile(xs, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--write", action="store_true", help="αποθήκευση στο runtime_estimates.json")
    a = ap.parse_args()

    model = load_model(a.model)
    method = get_method("gradcam_pp")
    files = sorted(p for p in a.images.iterdir() if p.suffix.lower() in EXT)[: a.n + a.warmup]
    if len(files) <= a.warmup:
        raise SystemExit("Χρειάζονται περισσότερες εικόνες από το --warmup")
    images = [read_image_rgb(p) for p in files]
    print(f"Device: {model.device} | images: {len(files) - a.warmup} (+{a.warmup} warmup)\n")

    default_median = None
    print(f"{'layer':<12}{'detect med':>12}{'xai med':>10}{'xai mean':>10}{'xai p95':>10}")
    for layer in model.info.recommended_layers:
        det_t, xai_t = [], []
        for i, img in enumerate(images):
            _, td = timed(model.device, lambda: detect(model, img, a.conf))
            _, tx = timed(model.device, lambda: method.run(model, img, layer.name, "full_image",
                                                           conf_threshold=a.conf))
            if i >= a.warmup:
                det_t.append(td)
                xai_t.append(tx)
        med, mean, p95 = stats(xai_t)
        tag = " (default)" if layer.default else ""
        print(f"{layer.name + tag:<12}{statistics.median(det_t):>11.3f}s{med:>9.3f}s{mean:>9.3f}s{p95:>9.3f}s")
        if layer.default:
            default_median, default_det = med, statistics.median(det_t)

    per_image = default_median + default_det
    print(f"\nΑνά εικόνα (default layer, detect + xai, ανά μέθοδο): ~{per_image:.2f}s")
    print(f"Για progress update ανά ~{TARGET_BATCH_SECONDS}s και 1 μέθοδο: batch_size ≈ "
          f"{max(1, round(TARGET_BATCH_SECONDS / per_image))}")
    if a.write:
        save_runtime_estimate(method.name, default_median, model.device.type)
        print("Γράφτηκε στο app/xai/runtime_estimates.json")


if __name__ == "__main__":
    main()