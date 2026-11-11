"""Phone photo -> clean "scanned" page (like CamScanner 'Magic color').

1. find the sheet (largest 4-corner shape) and flatten its perspective
2. turn it upright (Gemini Vision picks the readable one of 4 rotations;
   Tesseract OSD got 0/12 real handwritten pages right, see eval_orientation.py)
3. whiten the paper (divide by the estimated lighting), keep pen colours
"""
import base64
import json
import os
import time
from pathlib import Path

import cv2
import httpx
import numpy as np
from PIL import Image, ImageOps


def load_upright_exif(path: Path) -> np.ndarray:
    # Telegram "photos" have no EXIF, but photos sent as files do
    with Image.open(path) as im:
        rgb = ImageOps.exif_transpose(im).convert("RGB")
    return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)


def _order(pts: np.ndarray) -> np.ndarray:
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], dtype=np.float32)  # tl tr br bl


def find_sheet(img: np.ndarray) -> np.ndarray | None:
    """Corners of the paper in the photo, or None when it fills the frame / isn't clear."""
    h, w = img.shape[:2]
    k = 800 / max(h, w)
    small = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.dilate(cv2.Canny(gray, 50, 150), np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = small.shape[0] * small.shape[1]
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        # a sheet: 4 corners, convex, between 20% and 98% of the photo (98%+ = the border of the photo itself)
        if len(approx) == 4 and cv2.isContourConvex(approx) and 0.2 * area < cv2.contourArea(approx) < 0.98 * area:
            return _order(approx.reshape(4, 2).astype(np.float32) / k)
    return None


def flatten(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = corners
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(corners, dst), (w, h), flags=cv2.INTER_CUBIC)


GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")
ASK = ("These are the same page of handwritten course notes (French, maths) shown in 4 rotations, "
       "labelled A, B, C, D. Which one is upright: text lines horizontal, letters and formulas readable "
       "left to right, not upside down? Answer with the label only.")


def _jpeg_b64(img: np.ndarray) -> str:
    return base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()).decode()


def text_rotation(img: np.ndarray) -> int:
    """Clockwise degrees (0/90/180/270) that make the page upright; 0 when Gemini is unavailable."""
    if not GEMINI_KEY:
        return 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = 640 / max(gray.shape)  # enough to see writing direction, cheap in tokens
    gray = cv2.resize(gray, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    parts = [{"text": ASK}]
    for label, deg in zip("ABCD", (0, 90, 180, 270)):  # comparing candidates beats asking for an angle
        cand = gray if deg == 0 else cv2.rotate(gray, ROT[deg])
        parts += [{"text": f"Image {label}:"}, {"inline_data": {"mime_type": "image/jpeg", "data": _jpeg_b64(cand)}}]
    body = {"contents": [{"parts": parts}], "generationConfig": {
        "temperature": 0, "responseMimeType": "application/json",
        "responseSchema": {"type": "OBJECT", "properties": {"upright": {"type": "STRING", "enum": list("ABCD")}},
                           "required": ["upright"]}}}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    for attempt in range(3):
        try:
            r = httpx.post(url, headers={"x-goog-api-key": GEMINI_KEY}, json=body, timeout=60)
        except httpx.HTTPError:  # network down / timeout: keep the page as photographed
            return 0
        if r.status_code in (429, 500, 503):  # free tier rate limit or busy: wait and retry
            time.sleep(5 * (attempt + 1))
            continue
        if r.is_error:
            return 0  # never fail a batch over orientation: keep the page as photographed
        label = json.loads(r.json()["candidates"][0]["content"]["parts"][0]["text"])["upright"]
        return {"A": 0, "B": 90, "C": 180, "D": 270}[label]
    return 0


ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def whiten(img: np.ndarray) -> np.ndarray:
    """Even out shadows and turn the paper white, keeping ink colours."""
    # paper colour per pixel = the image with the ink removed (dilate) and smoothed
    bg = cv2.medianBlur(cv2.dilate(img, np.ones((9, 9), np.uint8)), 31)
    out = cv2.divide(img, bg, scale=255)
    # push near-white to white, darken ink a little (light curve)
    lut = np.clip((np.arange(256) - 40) * 255 / (235 - 40), 0, 255).astype(np.uint8)
    return cv2.LUT(out, lut)


def scan(src: Path, dest: Path) -> dict:
    """Write a clean JPEG of the page to dest; returns what was done."""
    img = load_upright_exif(src)
    corners = find_sheet(img)
    if corners is not None:
        img = flatten(img, corners)
    rot = text_rotation(img)
    if rot:
        img = cv2.rotate(img, ROT[rot])
    img = whiten(img)
    cv2.imwrite(str(dest), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return {"cropped": corners is not None, "rotated": rot}
