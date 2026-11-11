"""Measure page-orientation accuracy on real handwritten pages (eval/labels.json = clockwise degrees needed).

podman run --rm --env-file .env dms-worker python eval_orientation.py
Baseline measured on these 12 pages: Tesseract OSD 0/12, Tesseract OCR-confidence voting 7/12.
"""
import json
import time
from pathlib import Path

import cv2

import scan

labels = json.loads(Path("eval/labels.json").read_text())
ok = 0
for name, truth in labels.items():
    got = scan.text_rotation(cv2.imread(f"eval/{name}"))
    ok += got == truth
    print(f"{name}: expected {truth:3d}  got {got:3d}  {'OK' if got == truth else 'WRONG'}")
    time.sleep(4)  # stay under the free tier's requests-per-minute limit
print(f"\naccuracy: {ok}/{len(labels)} = {ok / len(labels):.0%}  (model: {scan.GEMINI_MODEL})")
