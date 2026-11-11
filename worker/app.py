"""dms-worker: all the heavy file work of the DMS, off n8n's JavaScript heap.

n8n sends JSON only ({files: [{file_id, msg_id, chat_id}], slug, date, old_url}; the old
{file_ids: [...]} still works). The worker downloads the files from Telegram to disk (over
20 MB: through MTProto, see telegram_dl.py), builds one PDF (photos -> PDF, PDFs kept), OCRs it,
shrinks it under Notion's 5 MB limit, merges it with the module's old PDF when it fits, uploads
to Notion and answers with the file_upload ids plus which messages were done / failed, so n8n
only clears the files that really went through.
"""
import math
import os
import subprocess
import tempfile
from pathlib import Path

import httpx
import img2pdf
import pikepdf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import telegram_dl
from scan import scan

TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
NOTION_TOKEN = os.environ.get("NOTION_TOKEN", "")
LIMIT = int(os.environ.get("MAX_BYTES", 4.8 * 1024 * 1024))  # aim under this
MB = 1024 * 1024

app = FastAPI(title="dms-worker")


class FileRef(BaseModel):
    file_id: str
    msg_id: int | None = None
    chat_id: int | None = None


class Job(BaseModel):
    files: list[FileRef] = []
    file_ids: list[str] = []  # old n8n payload
    slug: str
    date: str
    old_url: str | None = None


def run(*cmd: str) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise HTTPException(500, f"{cmd[0]} failed: {(p.stderr or p.stdout).strip()[-400:]}")


def stream_to(client: httpx.Client, url: str, dest: Path) -> None:
    with client.stream("GET", url, follow_redirects=True) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_bytes(1 << 20):  # 1 MB at a time: big files never sit in RAM
                f.write(chunk)


def tg_download(client: httpx.Client, file_id: str, dest: Path, chat_id: int | None = None,
                msg_id: int | None = None) -> None:
    r = client.get(f"https://api.telegram.org/bot{TG_TOKEN}/getFile", params={"file_id": file_id}).json()
    if not r.get("ok"):
        if "too big" in (r.get("description") or "").lower():  # Bot API stops at 20 MB, MTProto doesn't
            return telegram_dl.download_big(file_id, dest, chat_id, msg_id)
        raise ValueError(f"Telegram can't give the file: {r.get('description')}")
    stream_to(client, f"https://api.telegram.org/file/bot{TG_TOKEN}/{r['result']['file_path']}", dest)


def is_pdf(p: Path) -> bool:
    with p.open("rb") as f:
        return f.read(4) == b"%PDF"


A4_WIDTH = (img2pdf.mm_to_pt(210), None)  # every scanned page gets the same width, height follows the sheet


def build_batch(files: list[Path], out: Path) -> dict:
    """Photos -> scanned pages (cropped, upright, white paper), PDFs as they are, in send order.
    Consecutive photos become one PDF; returns how many photos were cropped / turned."""
    parts, photos, stats = [], [], {"cropped": 0, "rotated": 0}

    def flush():
        if photos:
            part = out.with_name(f"photos{len(parts)}.pdf")
            part.write_bytes(img2pdf.convert([str(p) for p in photos], layout_fun=img2pdf.get_layout_fun(A4_WIDTH)))
            parts.append(part)
            photos.clear()

    for k, f in enumerate(files):
        if is_pdf(f):
            flush()
            parts.append(f)
        else:
            page = f.with_name(f"scan{k}.jpg")
            done = scan(f, page)
            stats["cropped"] += done["cropped"]
            stats["rotated"] += bool(done["rotated"])
            photos.append(page)
    flush()
    if len(parts) == 1:
        parts[0].replace(out)
    else:
        with pikepdf.new() as pdf:
            for p in parts:
                with pikepdf.open(p) as src:
                    pdf.pages.extend(src.pages)
            pdf.save(out)
    return stats


def ocr(src: Path, out: Path) -> None:
    # skip-text: digital PDFs stay as they are. No --rotate-pages: Tesseract's guess turned correct
    # handwritten pages sideways; photos are already turned upright by scan.py (Gemini)
    run("ocrmypdf", "--skip-text", "-l", "fra+eng",
        "--output-type", "pdf", "--optimize", "1", "--jobs", "2", str(src), str(out))


def shrink(src: Path) -> Path:
    best = src
    for dpi in (150, 110, 80):  # image resolution steps; the OCR text layer is kept as text
        if best.stat().st_size < LIMIT:
            break
        out = src.with_name(f"gs_{dpi}.pdf")
        # force JPEG: Ghostscript's auto filter picks lossless Flate for noisy photos and the file grows
        run("gs", "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.6", "-dPDFSETTINGS=/ebook",
            "-dPassThroughJPEGImages=false",
            "-dAutoFilterColorImages=false", "-dColorImageFilter=/DCTEncode",
            "-dAutoFilterGrayImages=false", "-dGrayImageFilter=/DCTEncode",
            "-dDownsampleColorImages=true", "-dDownsampleGrayImages=true",
            f"-dColorImageResolution={dpi}", f"-dGrayImageResolution={dpi}",
            "-dColorImageDownsampleThreshold=1.0", "-dGrayImageDownsampleThreshold=1.0",
            "-dNOPAUSE", "-dBATCH", "-dQUIET", f"-sOutputFile={out}", str(src))
        if out.stat().st_size < best.stat().st_size:
            best = out
    return best


def split(src: Path, limit: int) -> list[Path]:
    """Cut a PDF into consecutive page ranges, each under `limit` bytes (order kept)."""
    with pikepdf.open(src) as pdf:
        n = len(pdf.pages)
        k = max(2, math.ceil(src.stat().st_size / (limit * 0.85)))  # 15% margin: fonts get copied into every part
        # shortcut: equal page counts per part, more parts until all fit; fine for course PDFs (< few hundred pages)
        while k <= n:
            step = math.ceil(n / k)
            parts = []
            for i, start in enumerate(range(0, n, step)):
                out = src.with_name(f"part{k}_{i}.pdf")
                with pikepdf.new() as part:
                    part.pages.extend(pdf.pages[start:start + step])
                    part.save(out, compress_streams=True, object_stream_mode=pikepdf.ObjectStreamMode.generate)
                parts.append(out)
            if all(p.stat().st_size < limit for p in parts):
                return parts
            k += 1
    raise HTTPException(413, "A single page is over 5 MB even after compression (Notion free limit). "
                             "Send that page as a photo instead.")


def pages(p: Path) -> int:
    with pikepdf.open(p) as pdf:
        return len(pdf.pages)


def notion_upload(client: httpx.Client, path: Path, name: str) -> str:
    h = {"Authorization": f"Bearer {NOTION_TOKEN}", "Notion-Version": "2022-06-28"}
    r = client.post("https://api.notion.com/v1/file_uploads", headers=h,
                    json={"filename": name, "content_type": "application/pdf"})
    if r.is_error:
        raise HTTPException(502, f"Notion create upload: {r.text[:300]}")
    fid = r.json()["id"]
    with path.open("rb") as f:
        r = client.post(f"https://api.notion.com/v1/file_uploads/{fid}/send", headers=h,
                        files={"file": (name, f, "application/pdf")})
    if r.is_error:
        raise HTTPException(502, f"Notion send: {r.text[:300]}")
    return fid


@app.get("/health")
def health():
    return {"ok": True, "telegram": bool(TG_TOKEN), "notion": bool(NOTION_TOKEN), "big_files": telegram_dl.enabled()}


@app.post("/process")
def process(job: Job):  # plain def: FastAPI runs it in a thread, so the blocking tools are fine
    refs = job.files or [FileRef(file_id=f) for f in job.file_ids]
    if not refs:
        raise HTTPException(400, "No photo or PDF in this batch")
    with tempfile.TemporaryDirectory() as tmp, httpx.Client(timeout=300) as client:
        d = Path(tmp)
        files, done, failed = [], [], []
        for k, ref in enumerate(refs):
            dest = d / f"in{k}"
            try:  # one bad file must not block the others: n8n keeps only the failed ones buffered
                tg_download(client, ref.file_id, dest, ref.chat_id, ref.msg_id)
                files.append(dest)
                done.append(ref.msg_id)
            except Exception as e:
                failed.append({"msg_id": ref.msg_id, "file_id": ref.file_id, "reason": str(e)[:300]})
        if not files:
            raise HTTPException(400, "No file could be downloaded: " + "; ".join(f["reason"] for f in failed))

        raw, ocred = d / "batch_raw.pdf", d / "batch_ocr.pdf"
        stats = build_batch(files, raw)
        ocr(raw, ocred)
        batch = shrink(ocred)
        size = batch.stat().st_size
        # still too big (digital slides: vectors and fonts don't shrink): cut into parts instead of failing
        parts = [batch] if size < LIMIT else split(batch, LIMIT)

        merged, had_old = None, bool(job.old_url)
        if job.old_url and len(parts) == 1:
            old = d / "old.pdf"
            stream_to(client, job.old_url, old)
            if old.stat().st_size + size < LIMIT:
                merged = d / "merged.pdf"
                with pikepdf.open(old) as a, pikepdf.open(batch) as b:
                    a.pages.extend(b.pages)
                    a.save(merged)

        # a Notion file_upload can be attached once: the log blocks and the property each get their own
        many = len(parts) > 1
        names = [f"{job.slug}_{job.date}" + (f"_part{i + 1}" if many else "") + ".pdf" for i in range(len(parts))]
        batch_ids = [notion_upload(client, p, n) for p, n in zip(parts, names)]
        if merged:
            merged_names, merged_ids = [f"{job.slug}.pdf"], [notion_upload(client, merged, f"{job.slug}.pdf")]
        else:
            merged_names, merged_ids = names, [notion_upload(client, p, n) for p, n in zip(parts, names)]
        return {
            "batch_upload_ids": batch_ids,
            "merged_upload_ids": merged_ids,
            "merged_filenames": merged_names,
            "parts": len(parts),
            "merged": merged is not None,
            "had_old": had_old,
            "new_pages": pages(batch),
            "rotated": stats["rotated"],
            "cropped": stats["cropped"],
            "batch_mb": round(size / MB, 2),
            "done_msg_ids": done,
            "failed": failed,
        }
