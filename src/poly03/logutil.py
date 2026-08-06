"""Shared helper for mirroring CLI console output to PAPER_TRADE_LOG_FILE."""

from __future__ import annotations

from poly03.config import PAPER_TRADE_LOG_FILE


def append_log(msg: str) -> None:
    with open(PAPER_TRADE_LOG_FILE, "a") as f:
        f.write(f"{msg}\n")
