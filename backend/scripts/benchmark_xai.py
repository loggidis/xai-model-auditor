"""Example:
python -m scripts.benchmark_xai --model best.pt --images C:\\path\\to\\images --n 10 --slow-n 3 --write
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
from app.xai.registry import list_methods

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TARGET_BATCH_SECONDS = 10   # we want a progress update every ~10s


def timed(device, fn):
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return out, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--n", type=int, default=10, help="images for the fast methods")
    ap.add_argument("--slow-n", type=int, default=3, help="images for the slow methods")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--methods", default=None, help="e.g. gradcam_pp,lime_image (default: all)")
    ap.add_argument("--write", action="store_true", help="save to runtime_estimates.json")
    a = ap.parse_args()

    model = load_model(a.model)
    wanted = set(a.methods.split(",")) if a.methods else None
    methods = [m for m in list_methods(model.info.architecture_type) if not wanted or m.name in wanted]

    files = sorted(p for p in a.images.iterdir() if p.suffix.lower() in EXT)
    need = a.warmup + max(a.n, a.slow_n)
    if len(files) < need:
        raise SystemExit(f"At least {need} images are needed (found {len(files)})")
    images = [read_image_rgb(p) for p in files[:need]]
    print(f"Device: {model.device} | methods: {[m.name for m in methods]}\n")

    det = [timed(model.device, lambda: detect(model, img, a.conf))[1] for img in images[a.warmup:a.warmup + a.n]]
    det_med = statistics.median(det)
    print(f"detect: median {det_med:.3f}s\n")

    summary: dict[str, float] = {}
    print(f"{'method / layer':<34}{'n':>3}{'median':>10}{'mean':>10}{'p95':>10}")
    for m in methods:
        n = a.slow_n if m.slow else a.n
        layers = model.info.recommended_layers if m.uses_target_layer else [None]
        props = m.params_schema.get("properties", {})
        params = {"timeout_s": 1800} if "timeout_s" in props else {}     # no timeout in the benchmark
        for layer in layers:
            times = []
            try:
                for i, img in enumerate(images[:a.warmup + n]):
                    _, t = timed(model.device, lambda: m.run(
                        model, img, layer.name if layer else None, "full_image",
                        conf_threshold=a.conf, **params))
                    if i >= a.warmup:
                        times.append(t)
            except Exception as e:
                print(f"{m.name:<34} FAILED: {type(e).__name__}: {e}")
                continue
            label = m.name + (f" [{layer.name}{'*' if layer.default else ''}]" if layer else "")
            med = statistics.median(times)
            print(f"{label:<34}{len(times):>3}{med:>9.2f}s{statistics.mean(times):>9.2f}s"
                  f"{float(np.percentile(times, 95)):>9.2f}s")
            if layer is None or layer.default:
                summary[m.name] = med

    if not summary:
        raise SystemExit("No method completed.")
    fast = sum(v for k, v in summary.items() if not next(x for x in methods if x.name == k).slow)
    total = sum(summary.values())
    print(f"\nPer image (default layer): detect {det_med:.2f}s + XAI "
          f"[fast {fast:.2f}s | all {total:.2f}s]")
    for label, per_image in (("fast methods", det_med + fast), ("all methods", det_med + total)):
        print(f"batch_size for progress every ~{TARGET_BATCH_SECONDS}s ({label}): "
              f"{max(1, round(TARGET_BATCH_SECONDS / per_image))}")
    if a.write:
        for name, med in summary.items():
            save_runtime_estimate(name, med, model.device.type)
        print("Written to app/xai/runtime_estimates.json")


if __name__ == "__main__":
    main()