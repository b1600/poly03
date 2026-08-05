"""Thin wrapper around py-clob-client for the read paths strategy_v1.md
needs: live order books, best bid/ask, tick size, min order size.

Order-book reads and price queries need no auth at all. L1 (private key)
and L2 (api key/secret/passphrase) creds are wired through so this same
client can grow into order placement later without changing its shape --
but nothing here places an order. Phase 0 is data-only.
"""

from __future__ import annotations

from collections.abc import Iterable

import requests
from py_clob_client.client import ClobClient as _RawClobClient
from py_clob_client.clob_types import ApiCreds

from poly03.config import Credentials, get_credentials, get_endpoints
from poly03.data.models import OrderBook, PriceLevel


class ClobClient:
    def __init__(self, creds: Credentials | None = None, host: str | None = None):
        self.creds = creds or get_credentials()
        self.host = host or get_endpoints().clob_api_url

        api_creds = None
        if self.creds.has_l2:
            api_creds = ApiCreds(
                api_key=self.creds.api_key,
                api_secret=self.creds.api_secret,
                api_passphrase=self.creds.api_passphrase,
            )

        self._client = _RawClobClient(
            host=self.host,
            chain_id=self.creds.chain_id,
            key=self.creds.private_key,
            creds=api_creds,
            funder=self.creds.funder_address,
        )

    def derive_api_creds(self) -> ApiCreds:
        """Derive L2 creds from the L1 private key. Run this once and save
        the result into POLYMARKET_CLOB_API_KEY/SECRET/PASSPHRASE -- don't
        call it on every startup."""
        if not self.creds.has_l1:
            raise RuntimeError("POLYMARKET_PRIVATE_KEY is not set; cannot derive API creds")
        return self._client.create_or_derive_api_creds()

    def get_order_book(self, token_id: str) -> OrderBook:
        raw = self._client.get_order_book(token_id)
        return OrderBook(
            market=raw.market,
            asset_id=raw.asset_id,
            timestamp=raw.timestamp,
            bids=[PriceLevel(price=l.price, size=l.size) for l in raw.bids],
            asks=[PriceLevel(price=l.price, size=l.size) for l in raw.asks],
        )

    def get_order_books(self, token_ids: Iterable[str]) -> dict[str, OrderBook]:
        """One request for many books via the batch endpoint, falling back
        to per-token calls if the underlying client doesn't batch this."""
        ids = list(token_ids)
        try:
            from py_clob_client.clob_types import BookParams

            raws = self._client.get_order_books([BookParams(token_id=t) for t in ids])
            return {
                r.asset_id: OrderBook(
                    market=r.market,
                    asset_id=r.asset_id,
                    timestamp=r.timestamp,
                    bids=[PriceLevel(price=l.price, size=l.size) for l in r.bids],
                    asks=[PriceLevel(price=l.price, size=l.size) for l in r.asks],
                )
                for r in raws
            }
        except Exception:
            return {t: self.get_order_book(t) for t in ids}

    def get_price(self, token_id: str, side: str = "buy") -> float:
        return float(self._client.get_price(token_id, side)["price"])

    def get_tick_size(self, token_id: str) -> float:
        return float(self._client.get_tick_size(token_id))

    def get_last_trade_price(self, token_id: str) -> float | None:
        try:
            return float(self._client.get_last_trade_price(token_id)["price"])
        except Exception:
            return None

    def get_fee_rate_bps(self, token_id: str) -> int | None:
        try:
            return self._client.get_fee_rate_bps(token_id)
        except Exception:
            return None

    def get_price_history(
        self, token_id: str, *, interval: str = "max", fidelity: int = 1440
    ) -> list[tuple[int, float]]:
        """Historical (unix_ts, price) points for a token. Not exposed by
        py-clob-client, so this hits the REST endpoint directly. Thin/old
        markets can return very sparse (even single-point) history -- this
        is exactly the survivorship/data-quality gap §8 Phase 0 warns
        about, not a bug in this call."""
        resp = requests.get(
            f"{self.host}/prices-history",
            params={"market": token_id, "interval": interval, "fidelity": fidelity},
            timeout=20.0,
        )
        resp.raise_for_status()
        history = resp.json().get("history", [])
        return [(int(pt["t"]), float(pt["p"])) for pt in history]
