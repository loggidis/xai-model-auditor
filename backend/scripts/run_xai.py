"""Παράδειγμα:
python -m scripts.run_xai --model C:\\path\\best.pt --image C:\\path\\img.jpg --conf 0.25
"""
import argparse
import time
from pathlib import Path

from app.xai.heatmap_utils import draw_detections, overlay, read_image_rgb, write_png
from app.xai.inference import detect
from app.xai.loader import load_model
from app.xai.registry import get_method

import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Model + image -> XAI heatmap PNG")
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("xai_out"))
    ap.add_argument("--method", default="gradcam_pp")
    ap.add_argument("--layer", default=None, help="π.χ. model.18 (default: το προτεινόμενο)")
    ap.add_argument("--mask", default="full_image")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--detection-index", type=int, default=None)
    a = ap.parse_args()

    t = time.perf_counter()
    model = load_model(a.model)
    info = model.info
    print(f"Loaded in {time.perf_counter() - t:.1f}s on {model.device}: {info.architecture_type}, "
          f"{info.num_classes} classes {info.class_names}, input {info.input_size}")
    for l in info.recommended_layers:
        print(f"  layer {l.name}{' (default)' if l.default else ''}: {l.description}")

    img = read_image_rgb(a.image)
    dets = detect(model, img, a.conf)
    print(f"\nDetections (conf >= {a.conf}): {len(dets)}")
    for d in dets:
        print("  ", d.to_dict())

    res = get_method(a.method).run(model, img, a.layer, a.mask, conf_threshold=a.conf,
                                   detection_index=a.detection_index)
    if not res.targets:
        print("\nΠροσοχή: δεν υπάρχουν detections >= conf, το heatmap είναι κενό. Μείωσε το --conf.")
    print(f"\n{res.method} layer={res.layer} mask={res.mask} targets={len(res.targets)} "
          f"box_focus={res.box_focus} runtime={res.runtime_s:.2f}s")

    tag = f"{a.image.stem}_{res.method}_{res.layer.replace('.', '_')}_{res.mask}"
    over = draw_detections(overlay(img, res.heatmap), res.targets)
    write_png(a.out / f"{tag}.png", over)
    write_png(a.out / f"{tag}.raw.png", (np.clip(res.heatmap, 0, 1) * 255).astype(np.uint8))
    print(f"Saved: {a.out / (tag + '.png')}")


if __name__ == "__main__":
    main()