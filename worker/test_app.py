"""Runs the whole /process pipeline on generated files, with Telegram and Notion faked.
python -m pytest -q test_app.py   (inside the container: podman exec dms-worker python -m pytest -q)"""
import shutil
from pathlib import Path

import pikepdf
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

import app as worker
import scan as scan_module


@pytest.fixture(autouse=True)
def no_real_gemini(monkeypatch):
    # tests must not depend on the GEMINI_API_KEY in .env (it made the rotation count change)
    monkeypatch.setattr(scan_module, "GEMINI_KEY", "")


def make_photo(path: Path, text: str, exif_orientation: int = 1) -> None:
    im = Image.new("RGB", (1240, 1754), "white")
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 48)
    except OSError:
        font = ImageFont.load_default(size=48)
    for k in range(12):
        d.text((80, 100 + k * 120), f"{text} ligne {k} : modélisation", fill="black", font=font)
    exif = Image.Exif()
    exif[0x0112] = exif_orientation
    im.save(path, "JPEG", quality=85, exif=exif)


def test_photos_and_pdf_then_merge_with_old(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    make_photo(src / "p1.jpg", "Page un")
    make_photo(src / "p2.jpg", "Page deux", exif_orientation=6)  # phone held sideways
    pdf_in = src / "doc.pdf"
    with pikepdf.new() as pdf:  # 2-page PDF made from photos (no text layer -> gets OCR'd)
        pdf.pages.extend(pikepdf.open(_photo_pdf(src, "Annexe")).pages)
        pdf.save(pdf_in)
    old = src / "old.pdf"
    shutil.copy(_photo_pdf(src, "Ancien"), old)

    by_id = {"A": src / "p1.jpg", "B": src / "p2.jpg", "C": pdf_in}
    uploads = []
    monkeypatch.setattr(worker, "tg_download", lambda c, fid, dest, *a: shutil.copy(by_id[fid], dest))
    monkeypatch.setattr(worker, "stream_to", lambda c, url, dest: shutil.copy(old, dest))
    monkeypatch.setattr(worker, "notion_upload", lambda c, p, name: uploads.append((name, pages_of(p))) or f"up{len(uploads)}")

    r = TestClient(worker.app).post("/process", json={"file_ids": ["A", "B", "C"], "slug": "Modelisation_TD",
                                                      "date": "2026-11-05", "old_url": "https://x/old.pdf"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["new_pages"] == 3 and j["rotated"] == 0  # no Gemini key in tests: pages stay as photographed
    assert j["merged"] and j["had_old"] and j["parts"] == 1
    assert uploads == [("Modelisation_TD_2026-11-05.pdf", 3), ("Modelisation_TD.pdf", 4)]  # old 1 page + 3 new
    assert j["batch_upload_ids"] == ["up1"] and j["merged_upload_ids"] == ["up2"]
    assert j["batch_mb"] < 5


def test_empty_batch_is_rejected():
    r = TestClient(worker.app).post("/process", json={"file_ids": [], "slug": "x", "date": "2026-11-05"})
    assert r.status_code == 400


def _photo_pdf(d: Path, text: str) -> Path:
    jpg, out = d / f"{text}.jpg", d / f"{text}.pdf"
    make_photo(jpg, text)
    out.write_bytes(worker.img2pdf.convert(str(jpg)))
    return out


def pages_of(p: Path) -> int:
    with pikepdf.open(p) as pdf:
        return len(pdf.pages)


def test_shrink_keeps_text_layer(tmp_path, monkeypatch):
    raw, ocred = tmp_path / "raw.pdf", tmp_path / "ocr.pdf"
    jpg = tmp_path / "noisy.jpg"
    make_photo(jpg, "Gros")
    im = Image.open(jpg)
    noise = Image.effect_noise(im.size, 40).convert("RGB")  # real phone photos are noisy, so they compress badly
    Image.blend(im, noise, 0.25).save(jpg, quality=95)
    raw.write_bytes(worker.img2pdf.convert(str(jpg)))
    worker.ocr(raw, ocred)
    monkeypatch.setattr(worker, "LIMIT", 10_000)  # force all Ghostscript passes
    small = worker.shrink(ocred)
    assert small.stat().st_size < ocred.stat().st_size
    with pikepdf.open(small) as pdf:  # OCR text survives the recompression
        assert "/Font" in str(pdf.pages[0].Resources)


def test_too_big_is_split_into_parts_in_order(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    for k in range(4):
        make_photo(src / f"p{k}.jpg", f"Page {k}")
    uploads = []
    monkeypatch.setattr(worker, "tg_download", lambda c, fid, dest, *a: shutil.copy(src / f"{fid}.jpg", dest))
    monkeypatch.setattr(worker, "stream_to", lambda c, url, dest: pytest.fail("no merge when split"))
    monkeypatch.setattr(worker, "notion_upload", lambda c, p, name: uploads.append((name, pages_of(p), p.stat().st_size)) or f"up{len(uploads)}")
    probe = tmp_path / "probe.pdf"
    shutil.copy(src / "p0.jpg", tmp_path / "p0_copy")
    worker.build_batch([tmp_path / "p0_copy"], probe)  # one scanned page, as the pipeline makes it
    one_page = probe.stat().st_size
    monkeypatch.setattr(worker, "shrink", lambda p: p)  # like digital slides: compression can't help
    monkeypatch.setattr(worker, "LIMIT", int(one_page * 2.5))  # about 2 pages fit per part

    r = TestClient(worker.app).post("/process", json={"file_ids": ["p0", "p1", "p2", "p3"], "slug": "S",
                                                      "date": "2026-11-05", "old_url": "https://x/old.pdf"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["parts"] >= 2 and not j["merged"] and j["new_pages"] == 4
    batch = uploads[: j["parts"]]
    assert [n for n, _, _ in batch] == [f"S_2026-11-05_part{i + 1}.pdf" for i in range(j["parts"])]
    assert sum(pg for _, pg, _ in batch) == 4 and all(sz < worker.LIMIT for _, _, sz in batch)
    assert j["merged_filenames"] == [n for n, _, _ in batch]


def test_one_bad_file_does_not_block_the_others(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    make_photo(src / "ok.jpg", "Bonne page")

    def fake_dl(c, fid, dest, chat_id=None, msg_id=None):
        if fid == "bad":
            raise ValueError("Telegram can't give the file: Bad Request: wrong file_id")
        shutil.copy(src / "ok.jpg", dest)

    monkeypatch.setattr(worker, "tg_download", fake_dl)
    monkeypatch.setattr(worker, "ocr", lambda s, o: shutil.copy(s, o))  # no ocrmypdf needed here
    monkeypatch.setattr(worker, "notion_upload", lambda c, p, name: "up")
    files = [{"file_id": "good", "msg_id": 10, "chat_id": 1}, {"file_id": "bad", "msg_id": 11, "chat_id": 1}]
    r = TestClient(worker.app).post("/process", json={"files": files, "slug": "S", "date": "2026-11-08"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["done_msg_ids"] == [10] and [f["msg_id"] for f in j["failed"]] == [11] and j["new_pages"] == 1

    files = [{"file_id": "bad", "msg_id": 11, "chat_id": 1}]
    r = TestClient(worker.app).post("/process", json={"files": files, "slug": "S", "date": "2026-11-08"})
    assert r.status_code == 400 and "wrong file_id" in r.text


def test_file_over_20mb_goes_through_mtproto(tmp_path, monkeypatch):
    class FakeTelegram:
        def get(self, url, params):
            return httpx_response({"ok": False, "error_code": 400, "description": "Bad Request: file is too big"})

    calls = []
    monkeypatch.setattr(worker.telegram_dl, "download_big",
                        lambda fid, dest, chat_id, msg_id: calls.append((fid, chat_id, msg_id)) or dest.write_bytes(b"x"))
    dest = tmp_path / "big"
    worker.tg_download(FakeTelegram(), "BIG", dest, 42, 7)
    assert calls == [("BIG", 42, 7)] and dest.read_bytes() == b"x"


def test_big_file_without_api_keys_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.setattr(worker.telegram_dl, "API_ID", "")
    with pytest.raises(RuntimeError, match="TG_API_ID"):
        worker.telegram_dl.download_big("BIG", tmp_path / "x")


def httpx_response(body):
    import httpx
    return httpx.Response(200, json=body)
