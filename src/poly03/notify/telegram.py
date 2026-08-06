"""Telegram notifications: forwards `poly03 paper run` console output to a
Telegram chat via the Bot API, so a long-running paper session can be
watched remotely. Best-effort only -- a missing/misconfigured token or a
network hiccup never takes down the paper loop.
"""

from __future__ import annotations

import logging

import requests

from poly03.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from poly03.logutil import append_log

logger = logging.getLogger("poly03.notify.telegram")

_API_URL = "https://api.telegram.org/bot{token}/sendMessage"
_MAX_LEN = 4000  # Telegram's hard cap is 4096 chars/message; leave headroom


def is_configured() -> bool:
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


def send_message(text: str) -> bool:
    """Best-effort send; returns False (and logs) instead of raising."""
    if not is_configured():
        return False

    url = _API_URL.format(token=TELEGRAM_BOT_TOKEN)
    ok = True
    for i in range(0, len(text), _MAX_LEN):
        chunk = text[i : i + _MAX_LEN]
        try:
            resp = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": chunk}, timeout=10)
            if resp.status_code != 200:
                logger.warning("Telegram send failed (%s): %s", resp.status_code, resp.text[:200])
                ok = False
        except requests.RequestException as exc:
            logger.warning("Telegram send failed: %s", exc)
            ok = False
    return ok


class TelegramReporter:
    """Drop-in replacement for `print()` that also mirrors output to
    Telegram. Buffers lines between flush() calls and sends each buffered
    group as a single Telegram message (one message per tick / report,
    rather than one per print line -- avoids flood-limit errors)."""

    def __init__(self, enabled: bool | None = None) -> None:
        self._buffer: list[str] = []
        self.enabled = is_configured() if enabled is None else enabled

    def log(self, msg: object = "") -> None:
        msg = str(msg)
        print(msg)
        append_log(msg)
        if self.enabled:
            self._buffer.append(msg)

    def flush(self) -> None:
        if self.enabled and self._buffer:
            send_message("\n".join(self._buffer))
        self._buffer.clear()
