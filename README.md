# telegram-notion-dms

I take my course notes on paper. This bot turns the photos I send it on Telegram into clean, searchable PDFs, one per module, filed in Notion.

I send a few photos or PDFs with a short caption like `proba td3`. The bot files it under the module "Probabilités", type TD, with the new pages appended to the module's PDF and a dated entry in its upload log.

24-second demo: [chemseddine-kebaily.netlify.app/en/#p-dms](https://chemseddine-kebaily.netlify.app/en/#p-dms)

```
Telegram ──> n8n ──────────────> dms-worker (FastAPI) ──> Notion
             caption -> module    download, scan, OCR,
             (Gemini)             compress, merge, upload
```

## What it does

**Filing (n8n + Gemini).** The caption is typed fast, so it has typos, abbreviations and mixed French and English. Gemini Flash-Lite reads it, returns the document type (Cours, TD, TP, Examen, Corrigé, Notes perso) and matches it against the modules already in Notion. If the module is new or the match is unsure, the bot asks with two buttons instead of guessing, so a typo never creates a duplicate module.

**Scanning (OpenCV).** Each photo goes through the same steps as a scanner app: find the sheet, flatten the perspective, turn it upright and whiten the paper while keeping pen colours.

Turning handwritten pages upright was the hard part. Tesseract's orientation detection got 0 of my 12 test pages right. The worker now shows Gemini the page in all four rotations and asks which one is readable, and that gets 12/12 (`eval_orientation.py`).

**PDF work.** OCRmyPDF adds a French and English text layer, so the notes are searchable in Notion. Ghostscript brings the file under Notion's 5 MB upload limit. If it still doesn't fit (slides with heavy vector graphics), the PDF is split into parts. Otherwise it is merged with the module's existing PDF.

**Large files.** The Telegram Bot API stops at 20 MB. Bigger files are downloaded over MTProto (Pyrogram), still as the bot.

## Why a separate worker

The first version did everything inside n8n and crashed its JavaScript heap on large batches. Now n8n only passes JSON (file ids, module, date) and the worker streams files to disk and does the heavy lifting. If one file fails, the worker reports which messages went through, and n8n keeps only the failed ones buffered for the next try.

## Run it

You need Podman or Docker with Compose, a Telegram bot ([BotFather](https://t.me/BotFather)), a Notion integration and a free Gemini key ([AI Studio](https://aistudio.google.com)).

1. Copy `worker/.env.example` to `worker/.env` and `n8n/.env.example` to `n8n/.env`, then fill them in.
2. Start everything:
   ```sh
   cd n8n && podman compose up -d --build
   ```
3. In Notion, create a database with these properties: `Module` (title), `Type` (select), `Merged PDF` (files), `Pages` (number), `Batches` (number), `Last update` (date). Share it with your integration.
4. In n8n (http://localhost:5678), create a data table `dms_buffer` with columns `group_id`, `file_id`, `caption`, `chat_id`, `sent_date` (text) and `msg_id` (number).
5. Import `n8n/dms-workflow.json`, attach your Telegram, Notion and Gemini credentials, and replace `YOUR_TELEGRAM_CHAT_ID` (Normalize and Report error nodes) and `YOUR_NOTION_DATABASE_ID` (Find module and Results nodes). Activate it.

Only your chat ID is accepted, so nobody else can write to your Notion.

## Tests

```sh
podman exec -w /app n8n_dms-worker_1 python -m pytest -q     # 10 tests
podman exec -w /app n8n_dms-worker_1 python eval_orientation.py
```

The orientation benchmark needs your own photos in `worker/eval/` (`labels.json` gives the clockwise rotation each page needs). My 12 pages are not in the repo.

## Stack

n8n, PostgreSQL, Python, FastAPI, OpenCV, Gemini API, Tesseract / OCRmyPDF, Ghostscript, pikepdf, Pyrogram, Notion API, Podman Compose.
