#!/usr/bin/env python3
"""Precompute CLIP text embeddings for the YOLOE obstacle whitelist.

The perception service's /v1/obstacles endpoint calls model.set_classes()
once at startup with a static whitelist. Rather than have the service import
CLIP and download the ViT-B/32 checkpoint from OpenAI's CDN at container
startup (unpinned dependency, ~338MB network fetch, breaks the service's
offline/pinned-build contract), this script computes the embeddings once,
here, wherever CLIP already resolves (e.g. after running
obstacle_detector_client.py locally), and saves the resulting tensor for the
service to load directly with torch.load() + set_classes(whitelist, tensor).

Re-run this only if OBSTACLE_WHITELIST (kept in sync with
obstacle_detector_client.py:52-57 and yolo_service/app.py) ever changes.

Usage:
    python3 scripts/precompute_yoloe_embeddings.py \
        [--model model/yoloe-11l-seg.pt] \
        [--out model/yoloe-whitelist-embeddings.pt]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLOE

# Must stay identical to obstacle_detector_client.py:52-57 and
# yolo_service/app.py's OBSTACLE_WHITELIST.
OBSTACLE_WHITELIST = [
    'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'animal', 'scooter', 'stroller', 'dog',
    'pole', 'post', 'column', 'pillar', 'stanchion', 'bollard', 'utility pole',
    'telegraph pole', 'light pole', 'street pole', 'signpost', 'support post',
    'vertical post', 'bench', 'chair', 'potted plant', 'hydrant', 'cone', 'stone', 'box'
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="model/yoloe-11l-seg.pt")
    parser.add_argument("--out", default="model/yoloe-whitelist-embeddings.pt")
    args = parser.parse_args()

    model = YOLOE(args.model)
    embeddings = model.get_text_pe(OBSTACLE_WHITELIST)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings.detach().cpu(), out_path)
    print(f"[OK] saved {tuple(embeddings.shape)} embeddings for {len(OBSTACLE_WHITELIST)} "
          f"classes to {out_path}")


if __name__ == "__main__":
    main()
