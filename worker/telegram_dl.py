"""Download Telegram files over 20 MB (the Bot API getFile limit) through MTProto, still as the bot.

Needs TG_API_ID + TG_API_HASH (my.telegram.org) next to TELEGRAM_BOT_TOKEN. The bot keeps using the
normal Bot API for everything else: no_updates=True means this client never takes the webhook's updates.
"""
import asyncio
import os
import threading
from pathlib import Path

from pyrogram import Client

API_ID = os.environ.get("TG_API_ID", "")
API_HASH = os.environ.get("TG_API_HASH", "")
SESSION_DIR = os.environ.get("TG_SESSION_DIR", "/data/session")  # persistent: keeps the login + peer cache

_loop: asyncio.AbstractEventLoop | None = None
_client: Client | None = None
_lock = threading.Lock()


def enabled() -> bool:
    return bool(API_ID and API_HASH and os.environ.get("TELEGRAM_BOT_TOKEN"))


def _start() -> None:
    """One event loop in a background thread: FastAPI runs /process in worker threads, Pyrogram is async."""
    global _loop, _client
    _loop = asyncio.new_event_loop()
    threading.Thread(target=_loop.run_forever, daemon=True, name="mtproto").start()
    Path(SESSION_DIR).mkdir(parents=True, exist_ok=True)
    _client = Client("dms_worker", api_id=int(API_ID), api_hash=API_HASH,
                     bot_token=os.environ["TELEGRAM_BOT_TOKEN"], workdir=SESSION_DIR, no_updates=True)
    asyncio.run_coroutine_threadsafe(_client.start(), _loop).result(timeout=60)


async def _download(file_id: str, chat_id: int | None, msg_id: int | None, dest: Path) -> None:
    try:
        got = await _client.download_media(file_id, file_name=str(dest))
    except Exception:
        # file_reference inside an old file_id can expire: fetch the message again for a fresh one
        if not (chat_id and msg_id):
            raise
        msg = await _client.get_messages(chat_id, msg_id)
        if not msg or msg.empty or not (msg.document or msg.photo):
            raise ValueError(f"message {msg_id} has no file any more")
        got = await _client.download_media(msg, file_name=str(dest))
    if not got:
        raise ValueError("Telegram returned no file")
    if Path(got) != dest:
        Path(got).replace(dest)


def download_big(file_id: str, dest: Path, chat_id: int | None = None, msg_id: int | None = None) -> None:
    if not enabled():
        raise RuntimeError("file over 20 MB and TG_API_ID / TG_API_HASH are not set in the worker")
    with _lock:
        if _client is None:
            _start()
    asyncio.run_coroutine_threadsafe(_download(file_id, chat_id, msg_id, dest), _loop).result(timeout=1800)
