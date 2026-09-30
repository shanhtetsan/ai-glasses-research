#!/usr/bin/env python3
"""Batch model testing tool: run a folder of images through local model
weights and record every detection (and every miss) to a CSV, for research
comparison across models/images/confidence thresholds.

Usage:
    python3 scripts/test_models.py --images test_images/ \
        --models yolo-seg,trafficlight,yoloe --out results.csv

    # Lower confidence, also save annotated images:
    python3 scripts/test_models.py --images test_images/ --models yoloe \
        --out results.csv --conf 0.15 --save-annotated annotated/

Model paths are relative to the current working directory (run from the
repository root), matching scripts/download_models.sh and
scripts/precompute_yoloe_embeddings.py's own convention.

For "yoloe", classes are set from the precomputed whitelist embeddings
(scripts/precompute_yoloe_embeddings.py's OBSTACLE_WHITELIST +
model/yoloe-whitelist-embeddings.pt) — get_text_pe() is never called here,
matching yolo_service/app.py's _load_obstacle_model().
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2

# Sibling script in the same directory; Python puts scripts/ on sys.path[0]
# when this file is run directly (`python3 scripts/test_models.py`).
from precompute_yoloe_embeddings import OBSTACLE_WHITELIST

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# name -> checkpoint path, relative to the current working directory.
MODEL_PATHS = {
    "detect": "yolov8n.pt",
    "yolo-seg": "model/yolo-seg.pt",
    "trafficlight": "model/trafficlight.pt",
    "shoppingbest5": "model/shoppingbest5.pt",
    "yoloe": "model/yoloe-11l-seg.pt",
}
YOLOE_EMBEDDINGS_PATH = "model/yoloe-whitelist-embeddings.pt"


def load_model(name: str):
    """Load one named model. Task (detect/segment) is auto-detected by
    Ultralytics from each checkpoint's own metadata — verified against every
    checkpoint in model/ rather than assumed, so it isn't hardcoded here."""
    path = MODEL_PATHS[name]
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    if name == "yoloe":
        import torch
        from ultralytics import YOLOE

        embeddings_path = Path(YOLOE_EMBEDDINGS_PATH)
        if not embeddings_path.is_file():
            raise FileNotFoundError(
                f"{embeddings_path} missing — run "
                "scripts/precompute_yoloe_embeddings.py first"
            )
        model = YOLOE(path)
        embeddings = torch.load(embeddings_path, map_location="cpu")
        model.set_classes(OBSTACLE_WHITELIST, embeddings)
        return model

    from ultralytics import YOLO
    return YOLO(path)


def find_images(images_dir: Path) -> list:
    return sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def run_one(model, image, conf: float):
    """Returns (detections, inference_ms) where detections is a list of
    (label, confidence) tuples, and the raw Ultralytics Results object."""
    started = time.monotonic()
    result = model.predict(
        source=image, conf=conf, device="cpu", save=False, verbose=False,
    )[0]
    inference_ms = (time.monotonic() - started) * 1000.0
    detections = []
    if result.boxes is not None and len(result.boxes) > 0:
        for cls_id, score in zip(
            result.boxes.cls.tolist(), result.boxes.conf.tolist()
        ):
            detections.append((result.names[int(cls_id)], score))
    return detections, inference_ms, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", required=True, help="Folder of images to test")
    parser.add_argument(
        "--models", required=True,
        help=f"Comma-separated model names: {', '.join(sorted(MODEL_PATHS))}",
    )
    parser.add_argument("--out", required=True, help="CSV output path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default 0.25)")
    parser.add_argument(
        "--save-annotated", default=None, metavar="DIR",
        help="Write annotated images to DIR/<model>/<image> (default: off)",
    )
    args = parser.parse_args()

    images_dir = Path(args.images)
    if not images_dir.is_dir():
        parser.error(f"--images {images_dir} is not a directory")

    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    unknown = [name for name in model_names if name not in MODEL_PATHS]
    if unknown:
        parser.error(
            f"unknown model(s) {unknown}; choose from {sorted(MODEL_PATHS)}"
        )

    images = find_images(images_dir)
    if not images:
        parser.error(f"no images found in {images_dir} (looked for {sorted(IMAGE_EXTENSIONS)})")

    models = {}
    for name in model_names:
        print(f"[LOAD] {name} <- {MODEL_PATHS[name]}", flush=True)
        try:
            models[name] = load_model(name)
        except Exception as exc:
            print(f"[LOAD] {name} FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)

    if not models:
        print("[FATAL] no models loaded successfully", file=sys.stderr)
        sys.exit(1)

    save_dir = Path(args.save_annotated) if args.save_annotated else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "model", "class", "confidence", "inference_ms", "width", "height"])

        for image_path in images:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"[SKIP] {image_path.name}: unreadable image", file=sys.stderr)
                continue
            height, width = image.shape[:2]

            for name, model in models.items():
                try:
                    detections, inference_ms, result = run_one(model, image, args.conf)
                except Exception as exc:
                    print(f"[ERROR] {image_path.name}/{name}: {type(exc).__name__}: {exc}", file=sys.stderr)
                    continue

                if detections:
                    for label, score in detections:
                        writer.writerow([
                            image_path.name, name, label, f"{score:.4f}",
                            f"{inference_ms:.1f}", width, height,
                        ])
                else:
                    writer.writerow([
                        image_path.name, name, "none", "",
                        f"{inference_ms:.1f}", width, height,
                    ])

                print(
                    f"[OK] {image_path.name}/{name}: {len(detections)} detection(s), "
                    f"{inference_ms:.1f}ms",
                    flush=True,
                )

                if save_dir is not None:
                    model_dir = save_dir / name
                    model_dir.mkdir(parents=True, exist_ok=True)
                    annotated = result.plot()
                    cv2.imwrite(str(model_dir / image_path.name), annotated)

    print(f"[DONE] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
