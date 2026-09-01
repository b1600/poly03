from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from poly03.making import rebuild as rb


class _FakeGamma:
    """Serves Gamma markets keyed by condition_id, like /markets?condition_ids=."""

    def __init__(self, markets):
        self.markets = {m.condition_id: m for m in markets}
        self.batches = []

    def _get(self, path, params=None):
        assert path == "/markets"
        ids = params["condition_ids"]
        self.batches.append(list(ids))
        return [{"__market__": cid} for cid in ids if cid in self.markets]

    def _market_from_raw(self, raw):
        return self.markets[raw["__market__"]]


class _FakeClob:
    def __init__(self, *, balance=100.0, open_orders=None, chain_balances=None):
        self.balance = balance
        self.open_orders = open_orders or []
        self.chain_balances = chain_balances or {}

    def get_usdc_balance_allowance(self):
        return self.balance, self.balance

    def get_open_orders(self, **kw):
        return [dict(o) for o in self.open_orders]

    def get_conditional_balance(self, token_id):
        return self.chain_balances[token_id]


def _position(asset="tok-yes", cond="0xabc", size=50.0, avg=0.40, cur=0.45, **kw):
    p = {
        "asset": asset,
        "conditionId": cond,
        "size": size,
        "avgPrice": avg,
        "curPrice": cur,
        "title": "Test market?",
        "redeemable": False,
        "negativeRisk": False,
        "currentValue": size * cur,
        "endDate": "2027-01-01T00:00:00Z",
    }
    p.update(kw)
    return p


@pytest.fixture
def market(market_factory):
    m = market_factory("Test market?", days_to_resolution=90)
    m.id = "12345"
    m.condition_id = "0xabc"
    return m


def _build(monkeypatch, positions, market_list, **kw):
    monkeypatch.setattr(rb, "fetch_exchange_positions", lambda address, **_: positions)
    gamma = _FakeGamma(market_list)
    kw.setdefault("clob", None)
    kw.setdefault("cash_usd", 117.04)
    return rb.build_state(address="0xwallet", bankroll_cap_usd=400.0, gamma=gamma, **kw)


def test_positions_come_from_the_exchange_not_local_belief(monkeypatch, market):
    state, report = _build(monkeypatch, [_position()], [market])

    assert len(state.positions) == 1
    pos = state.positions[0]
    assert pos.net_shares == 50.0
    assert pos.avg_price == 0.40
    assert pos.market_id == "12345"  # Gamma's id, not the condition_id
    assert pos.token_id == "tok-yes"
    assert state.cash_usd == pytest.approx(117.04)
    assert state.equity_usd == pytest.approx(117.04 + 50.0 * 0.45)


def test_resolved_positions_are_excluded_as_redemption_claims(monkeypatch, market):
    positions = [
        _position(asset="live", size=50.0),
        _position(asset="won", redeemable=True, curPrice=1.0, currentValue=20.0),
        _position(asset="lost", redeemable=True, curPrice=0.0, currentValue=0.0),
    ]
    state, report = _build(monkeypatch, positions, [market])

    assert [p.token_id for p in state.positions] == ["live"]
    assert len(report.redeemable) == 2
    assert report.redeemable_value_usd == pytest.approx(20.0)


def test_a_worthless_but_unflagged_position_is_also_excluded(monkeypatch, market):
    """curPrice 0 means there is no book to flatten into -- treating it as
    live inventory just produces a failing sell every tick."""
    state, _ = _build(monkeypatch, [_position(curPrice=0.0)], [market])
    assert state.positions == []


def test_a_position_gamma_cannot_resolve_is_kept_not_dropped(monkeypatch, market):
    """Dropping it is what let deployed capital be computed against a book
    missing $240 of inventory. It is kept, keyed on condition_id, so it can
    never match a quotable market -- held, never quoted."""
    orphan = _position(asset="orphan", cond="0xunknown")
    state, report = _build(monkeypatch, [_position(), orphan], [market])

    assert len(state.positions) == 2
    kept = next(p for p in state.positions if p.token_id == "orphan")
    assert kept.market_id == "0xunknown"
    assert len(report.unresolved) == 1


def test_end_date_is_stamped_so_the_flatten_window_survives_a_quiet_market(monkeypatch, market):
    state, _ = _build(monkeypatch, [_position()], [market])
    assert state.positions[0].end_date_iso == market.end_date.isoformat()


def test_trade_history_is_floored_at_now_so_positions_are_not_double_booked(monkeypatch, market):
    before = datetime.now(timezone.utc).timestamp()
    state, _ = _build(monkeypatch, [_position()], [market])
    assert state.trades_booked_through >= before


def test_open_orders_are_adopted_from_the_exchange(monkeypatch, market):
    clob = _FakeClob(
        open_orders=[
            {"id": "0xorder", "asset_id": "tok-yes", "price": "0.41", "original_size": "50", "side": "BUY"},
            {"id": "0xflat", "asset_id": "tok-yes", "price": "0.46", "original_size": "50", "side": "SELL"},
        ]
    )
    state, report = _build(monkeypatch, [_position()], [market], clob=clob)

    assert len(state.open_orders) == 2
    buy = next(o for o in state.open_orders if o.order_id == "0xorder")
    assert buy.side == "buy"
    assert buy.market_id == "12345"  # resolved through the position's context
    assert buy.price == pytest.approx(0.41)
    # A resting SELL is a flatten order from a previous run -- still ours.
    assert next(o for o in state.open_orders if o.order_id == "0xflat").side == "sell"


def test_cash_is_read_from_the_clob_when_not_given(monkeypatch, market):
    clob = _FakeClob(balance=222.22)
    state, _ = _build(monkeypatch, [_position()], [market], clob=clob, cash_usd=None)
    assert state.cash_usd == pytest.approx(222.22)


def test_cash_must_be_supplied_when_there_is_no_clob(monkeypatch, market):
    monkeypatch.setattr(rb, "fetch_exchange_positions", lambda address, **_: [])
    with pytest.raises(ValueError, match="cash_usd"):
        rb.build_state(address="0xw", bankroll_cap_usd=400.0, gamma=_FakeGamma([market]), clob=None, cash_usd=None)


def test_chain_verification_prefers_the_on_chain_balance(monkeypatch, market):
    """data-api and the chain disagreeing is exactly the drift that sized a
    SELL the exchange then rejected. On-chain wins."""
    clob = _FakeClob(chain_balances={"tok-yes": 31.5})
    state, report = _build(monkeypatch, [_position(size=50.0)], [market], clob=clob, verify_chain=True)

    assert state.positions[0].net_shares == pytest.approx(31.5)
    assert len(report.chain_mismatches) == 1


def test_condition_lookups_are_batched_not_one_request_per_position(monkeypatch, market_factory):
    markets = []
    positions = []
    for i in range(45):
        m = market_factory("Q?", days_to_resolution=90)
        m.id = f"id-{i}"
        m.condition_id = f"0x{i:04x}"
        markets.append(m)
        positions.append(_position(asset=f"tok-{i}", cond=m.condition_id))

    monkeypatch.setattr(rb, "fetch_exchange_positions", lambda address, **_: positions)
    gamma = _FakeGamma(markets)
    rb.build_state(address="0xw", bankroll_cap_usd=400.0, gamma=gamma, clob=None, cash_usd=1.0)

    assert len(gamma.batches) == 3  # 45 ids at 20 per batch
    assert sum(len(b) for b in gamma.batches) == 45
