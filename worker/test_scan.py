"""Scan step on a synthetic phone photo: sheet on a dark desk, in perspective, turned sideways, with a shadow."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import httpx
import pytest
import scan


@pytest.fixture(autouse=True)
def no_real_gemini(monkeypatch):
    monkeypatch.setattr(scan, "GEMINI_KEY", "")  # each test opts in with its own fake


def phone_photo(path: Path, sideways: bool) -> None:
    sheet = Image.new("RGB", (1240, 1754), (235, 232, 225))
    d = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("DejaVuSans.ttf", 44)
    for k in range(18):
        d.text((90, 90 + k * 90), f"Cours de modélisation, ligne {k} : x = A u + b", fill=(20, 20, 20), font=font)
    d.text((90, 1700), "Remarque importante", fill=(200, 30, 30), font=font)  # red pen must survive
    if sideways:
        sheet = sheet.rotate(90, expand=True)
    s = np.asarray(sheet)[:, :, ::-1].copy()
    h, w = s.shape[:2]
    desk = np.full((2400, 2400, 3), 60, np.uint8)
    dst = np.float32([[550, 200], [1750, 300], [1850, 2200], [450, 2100]])  # A4-ish sheet in perspective
    if sideways:  # phone held the other way: the sheet lies across the photo
        dst = np.float32([[200, 550], [2200, 450], [2100, 1850], [300, 1750]])
    m = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), dst)
    warped = cv2.warpPerspective(s, m, (2400, 2400))
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), m, (2400, 2400))
    photo = np.where(mask[..., None] > 0, warped, desk)
    shade = np.tile(np.linspace(0.7, 1.0, 2400), (2400, 1))[..., None]  # lamp on one side
    cv2.imwrite(str(path), (photo * shade).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 90])


class FakeGemini:
    """Stands in for the Gemini API: answers with the label we expect, records what was sent."""

    def __init__(self, answers):
        self.answers, self.calls = list(answers), []

    def __call__(self, url, headers, json, timeout):
        self.calls.append(json)
        body = {"candidates": [{"content": {"parts": [{"text": '{"upright": "%s"}' % self.answers.pop(0)}]}}]}
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))


def test_scan_crops_rotates_and_whitens(tmp_path, monkeypatch):
    monkeypatch.setattr(scan, "GEMINI_KEY", "test")
    fake = FakeGemini(["A", "B"])  # upright photo -> A (0 deg); sideways photo -> B (90 deg clockwise)
    monkeypatch.setattr(scan.httpx, "post", fake)
    for sideways in (False, True):
        src, out = tmp_path / f"in{sideways}.jpg", tmp_path / f"out{sideways}.jpg"
        phone_photo(src, sideways)
        done = scan.scan(src, out)
        img = cv2.imread(str(out))
        h, w = img.shape[:2]
        assert done["cropped"]
        assert h > w, "page must end up portrait, text upright"
        assert done["rotated"] == (90 if sideways else 0)
        paper = np.median(img[h // 2 - 30:h // 2 + 30, w - 120:w - 60])  # right margin, blank paper
        assert paper > 235, "paper should be white after scanning"
        b, g, r = (int(v) for v in img[int(h * 0.96):int(h * 0.985), int(w * 0.08):int(w * 0.5)].reshape(-1, 3).min(0))
        assert r > b + 60, "red pen should stay red"


def test_photo_without_visible_edges_is_kept_whole(tmp_path):
    src, out = tmp_path / "full.jpg", tmp_path / "full_out.jpg"
    page = Image.new("RGB", (1200, 1700), "white")
    ImageDraw.Draw(page).text((80, 80), "Plein cadre", fill="black", font=ImageFont.truetype("DejaVuSans.ttf", 50))
    page.save(src)
    done = scan.scan(src, out)
    assert not done["cropped"] and cv2.imread(str(out)).shape[:2] == (1700, 1200)


def test_gemini_gets_4_rotations_and_failures_keep_the_page(tmp_path, monkeypatch):
    img = np.full((800, 600, 3), 255, np.uint8)
    monkeypatch.setattr(scan, "GEMINI_KEY", "test")
    fake = FakeGemini(["D"])
    monkeypatch.setattr(scan.httpx, "post", fake)
    assert scan.text_rotation(img) == 270
    parts = fake.calls[0]["contents"][0]["parts"]
    assert sum("inline_data" in p for p in parts) == 4
    monkeypatch.setattr(scan.httpx, "post", lambda *a, **k: httpx.Response(400, request=httpx.Request("POST", "x")))
    assert scan.text_rotation(img) == 0
    monkeypatch.setattr(scan, "GEMINI_KEY", "")
    assert scan.text_rotation(img) == 0
