"""Thin wrapper around py-clob-client for the read paths strategy_v1.md
needs: live order books, best bid/ask, tick size, min order size -- plus,
for strategy_v2.md §4 Phase 1, the write paths: post/cancel a limit order,
list open orders. The write methods all require L2 creds and raise loudly
if they're missing; nothing in this file signs or sends anything unless the
caller explicitly asks it to place or cancel an order.
"""

from __future__ import annotations

from collections.abc import Iterable

import requests
from py_clob_client.client import ClobClient as _RawClobClient
from py_clob_client.clob_types import ApiCreds

from poly03.config import Credentials, get_credentials, get_endpoints
from poly03.data.models import OrderBook, PriceLevel

_VALID_TICK_SIZES = ("0.1", "0.01", "0.001", "0.0001")


def _tick_size_literal(tick: float) -> str:
    """py-clob-client's PartialCreateOrderOptions.tick_size wants one of a
    fixed set of strings, not a float. Match on the closest valid tick
    rather than str(tick), since float formatting isn't guaranteed to hit
    e.g. "0.01" exactly."""
    as_float = [float(t) for t in _VALID_TICK_SIZES]
    closest = min(as_float, key=lambda t: abs(t - tick))
    return _VALID_TICK_SIZES[as_float.index(closest)]


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

    # --- strategy_v2.md §4 Phase 1: order placement ---------------------------

    def _require_l2(self) -> None:
        if not self.creds.has_l2:
            raise RuntimeError(
                "L2 API credentials required to place/cancel orders "
                "(POLYMARKET_CLOB_API_KEY/SECRET/PASSPHRASE). Run "
                "ClobClient.derive_api_creds() once and save the result into .env."
            )

    def post_limit_order(
        self,
        *,
        token_id: str,
        price: float,
        size: float,
        side: str,
        tick_size: float,
        neg_risk: bool = False,
    ) -> dict:
        """Sign and rest one GTC limit order. `side` is
        `py_clob_client.order_builder.constants.BUY` or `.SELL`. Requires L2
        creds -- this is the only place in the codebase that spends real
        capital."""
        self._require_l2()
        from py_clob_client.clob_types import OrderArgs, OrderType, PartialCreateOrderOptions

        order_args = OrderArgs(token_id=token_id, price=price, size=size, side=side)
        options = PartialCreateOrderOptions(tick_size=_tick_size_literal(tick_size), neg_risk=neg_risk)
        signed = self._client.create_order(order_args, options)
        return self._client.post_order(signed, OrderType.GTC)

    def cancel_order(self, order_id: str) -> dict:
        self._require_l2()
        return self._client.cancel(order_id)

    def cancel_orders(self, order_ids: Iterable[str]) -> dict:
        ids = list(order_ids)
        if not ids:
            return {}
        self._require_l2()
        return self._client.cancel_orders(ids)

    def cancel_all(self) -> dict:
        self._require_l2()
        return self._client.cancel_all()

    def get_open_orders(self, *, market: str | None = None, asset_id: str | None = None) -> list[dict]:
        """All open orders, optionally scoped to one market/asset. Paginated
        by opaque cursor, same "LTE=" end sentinel as iter_sampling_markets."""
        self._require_l2()
        from py_clob_client.clob_types import OpenOrderParams

        params = OpenOrderParams(market=market, asset_id=asset_id)
        cursor = "MA=="
        out: list[dict] = []
        while True:
            page = self._client.get_orders(params, cursor)
            batch = page.get("data", []) if isinstance(page, dict) else (page or [])
            out.extend(batch)
            cursor = page.get("next_cursor") if isinstance(page, dict) else "LTE="
            if not cursor or cursor == "LTE=":
                return out

    def get_order(self, order_id: str) -> dict | None:
        self._require_l2()
        try:
            return self._client.get_order(order_id)
        except Exception:
            return None

    def iter_sampling_markets(self, *, max_markets: int | None = None) -> Iterable[dict]:
        """Yield raw CLOB markets from /sampling-markets -- the authoritative
        list of *reward-eligible* markets, and the universe Book M quotes from
        (strategy_v2.md §3.1).

        This is the source of truth for reward parameters. Gamma's
        `clobRewards`/`rewardsMinSize`/`rewardsMaxSpread` fields mirror the
        same data but are denormalised onto the market payload; here they
        arrive as one `rewards` block:

            {"rates": [{"rewards_daily_rate": 3, ...}],
             "min_size": 20, "max_spread": 4.5}

        `max_spread` is in *cents* from the midpoint, `min_size` in shares.
        Paginated by opaque cursor; "LTE=" is the documented end sentinel.
        """
        cursor = ""
        yielded = 0
        while True:
            resp = requests.get(
                f"{self.host}/sampling-markets",
                params={"next_cursor": cursor} if cursor else None,
                timeout=20.0,
            )
            resp.raise_for_status()
            payload = resp.json()
            batch = payload.get("data") or []
            if not batch:
                return
            for raw in batch:
                yield raw
                yielded += 1
                if max_markets is not None and yielded >= max_markets:
                    return
            cursor = payload.get("next_cursor") or ""
            if not cursor or cursor == "LTE=":
                return

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
