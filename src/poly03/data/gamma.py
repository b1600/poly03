"""Client for Polymarket's Gamma API (market/event metadata, no auth required).

Gamma is the source for everything in strategy_v1.md §2.1 except live order
books: resolution text, resolution dates, volume/liquidity, and (via the
/events endpoint) the tags an event carries, which §4.3 cluster tagging
depends on.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import requests

from poly03.config import get_endpoints
from poly03.data.models import Event, Market

_DEFAULT_PAGE_SIZE = 100


class GammaClient:
    def __init__(self, base_url: str | None = None, session: requests.Session | None = None, timeout: float = 20.0):
        self.base_url = (base_url or get_endpoints().gamma_api_url).rstrip("/")
        self.session = session or requests.Session()
        self.timeout = timeout

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        resp = self.session.get(f"{self.base_url}{path}", params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _get_page(self, path: str, params: dict[str, Any]) -> list | None:
        """Paginated GET that treats Gamma's offset ceiling as end-of-data.

        Gamma 422s past roughly offset=2000 on the list endpoints, for every
        sort order. That is a hard server-side cap on how deep any scan can
        page, not a transient error -- so callers get a clean stop rather than
        an exception that would take down a long-running loop mid-tick.
        """
        try:
            return self._get(path, params=params)
        except requests.HTTPError as exc:
            resp = exc.response
            if resp is not None and resp.status_code == 422 and params.get("offset"):
                return None
            raise

    # --- events -------------------------------------------------------------

    def get_event(self, event_id: str) -> Event:
        data = self._get(f"/events/{event_id}")
        return self._event_from_raw(data)

    def iter_events(
        self,
        *,
        closed: bool | None = None,
        active: bool | None = None,
        order: str | None = "volume",
        ascending: bool = False,
        page_size: int = _DEFAULT_PAGE_SIZE,
        max_pages: int | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> Iterator[Event]:
        params: dict[str, Any] = {"limit": page_size, "order": order, "ascending": str(ascending).lower()}
        if closed is not None:
            params["closed"] = str(closed).lower()
        if active is not None:
            params["active"] = str(active).lower()
        if extra_params:
            params.update(extra_params)

        offset = 0
        pages = 0
        while True:
            params["offset"] = offset
            batch = self._get_page("/events", params)
            if not batch:
                return
            for raw in batch:
                yield self._event_from_raw(raw)
            offset += len(batch)
            pages += 1
            if len(batch) < page_size or (max_pages is not None and pages >= max_pages):
                return

    def _event_from_raw(self, raw: dict[str, Any]) -> Event:
        ev = Event(**raw, market_ids=[m["id"] for m in raw.get("markets", [])])
        ev.raw_markets = raw.get("markets", [])
        return ev

    # --- markets --------------------------------------------------------------

    def get_market(self, market_id: str) -> Market:
        data = self._get(f"/markets/{market_id}")
        return self._market_from_raw(data)

    def iter_markets(
        self,
        *,
        closed: bool | None = None,
        active: bool | None = None,
        order: str | None = "volume",
        ascending: bool = False,
        page_size: int = _DEFAULT_PAGE_SIZE,
        max_pages: int | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> Iterator[Market]:
        """Fast path: iterate /markets directly. No event tags attached
        (use iter_markets_with_event_context for that)."""
        params: dict[str, Any] = {"limit": page_size, "order": order, "ascending": str(ascending).lower()}
        if closed is not None:
            params["closed"] = str(closed).lower()
        if active is not None:
            params["active"] = str(active).lower()
        if extra_params:
            params.update(extra_params)

        offset = 0
        pages = 0
        while True:
            params["offset"] = offset
            batch = self._get_page("/markets", params)
            if not batch:
                return
            for raw in batch:
                yield self._market_from_raw(raw)
            offset += len(batch)
            pages += 1
            if len(batch) < page_size or (max_pages is not None and pages >= max_pages):
                return

    def iter_markets_with_event_context(
        self,
        *,
        closed: bool | None = None,
        active: bool | None = None,
        order: str | None = "volume",
        ascending: bool = False,
        page_size: int = _DEFAULT_PAGE_SIZE,
        max_pages: int | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> Iterator[Market]:
        """Iterate /events and flatten their embedded markets, tagging each
        Market with event_id/event_title/tags/open_interest. This is the
        path to use when cluster tagging (§4.3) matters. The /events list
        endpoint already embeds full market payloads, so this costs no
        extra requests over iter_events."""
        for event in self.iter_events(
            closed=closed,
            active=active,
            order=order,
            ascending=ascending,
            page_size=page_size,
            max_pages=max_pages,
            extra_params=extra_params,
        ):
            for raw_market in event.raw_markets:
                market = self._market_from_raw(raw_market)
                market.event_id = event.id
                market.event_title = event.title
                market.tags = event.tags
                market.open_interest = event.open_interest
                yield market

    def _market_from_raw(self, raw: dict[str, Any]) -> Market:
        events = raw.get("events") or []
        market = Market(**raw, raw=raw)
        if events:
            e0 = events[0]
            market.event_id = e0.get("id")
            market.event_title = e0.get("title")
            market.open_interest = e0.get("openInterest")
        return market
