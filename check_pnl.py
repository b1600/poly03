#!/usr/bin/env python
"""Is Book M actually making money? Answered from the exchange, not the bot.

Why this exists separately from `poly03 make live report`:

- That report measures *markouts* -- spread capture and adverse selection on
  a 5/30-minute horizon per fill. Useful, but it is not P&L, and it says
  nothing about the money lost getting *out* of a position.
- It reads `making_live_state.json`, which is the bot's own belief. That
  belief has been wrong in every way that matters at least twice: phantom
  double-booked fills producing a $58 equity gap and a spurious drawdown
  halt (2026-08-31), and fabricated 1000bps fees inventing 44% of a reported
  loss. A book that reports on itself cannot catch its own bookkeeping bugs.
- A dry run cannot produce any of it at all: no orders rest, so nothing
  fills, no rewards accrue, and equity is frozen at whatever the state file
  was resumed with.

So this reads Polymarket's public data API for the wallet directly. Every
number below is what the exchange says happened. It needs no credentials --
only POLYMARKET_FUNDER_ADDRESS, which is a public address -- and it makes no
writes of any kind.

Usage:
    python check_pnl.py                  # since midnight UTC today
    python check_pnl.py --since 2026-08-31
    python check_pnl.py --hours 6
    python check_pnl.py --since 2026-08-31 --by-market
"""

from __future__ import annotations

import argparse
import collections
import sys
from datetime import datetime, timedelta, timezone

import requests

from poly03.config import _env  # also triggers load_dotenv()

DATA_API = "https://data-api.polymarket.com"
TIMEOUT = 30


def _fetch_all_activity(address: str) -> list[dict]:
    """Every TRADE/REDEEM/REWARD/MAKER_REBATE the wallet has, oldest first."""
    out: list[dict] = []
    seen: set[tuple] = set()
    offset = 0
    while True:
        r = requests.get(
            f"{DATA_API}/activity",
            params={"user": address, "limit": 500, "offset": offset},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        page = r.json()
        for x in page:
            # The endpoint can repeat a row across pages; dedupe on the
            # fields that actually identify one event.
            key = (x["transactionHash"], x["asset"], x["timestamp"], x["size"], x.get("type"), x.get("side"))
            if key not in seen:
                seen.add(key)
                out.append(x)
        if len(page) < 500:
            break
        offset += 500
    out.sort(key=lambda x: x["timestamp"])
    return out


def _fetch_positions(address: str) -> dict[str, dict]:
    """asset -> current position, for marking open inventory."""
    r = requests.get(
        f"{DATA_API}/positions",
        params={"user": address, "sizeThreshold": 0.01, "limit": 500},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return {p["asset"]: p for p in r.json()}


def analyse(activity: list[dict], positions: dict[str, dict], since_ts: float) -> dict:
    """P&L over the window, marked at today's prices.

        pnl = sells - buys + income + (net shares acquired x current price)

    Marking the *change* in inventory rather than its absolute value is what
    makes this correct without needing prices as of the window's start: a
    share sold that was bought before the window shows up as a negative
    delta, correctly charged at what it is worth now.
    """
    buys = sells = income = 0.0
    delta: dict[str, float] = collections.defaultdict(float)
    bought: dict[str, dict] = collections.defaultdict(lambda: {"sh": 0.0, "usd": 0.0})
    sold: dict[str, dict] = collections.defaultdict(lambda: {"sh": 0.0, "usd": 0.0})
    per_market: dict[str, dict] = collections.defaultdict(
        lambda: {"buy": 0.0, "sell": 0.0, "delta": collections.defaultdict(float), "title": ""}
    )
    # Per market, per outcome: what we paid on each side, for the pair check.
    pair_sides: dict[str, dict] = collections.defaultdict(
        lambda: collections.defaultdict(lambda: {"sh": 0.0, "usd": 0.0})
    )
    n_trades = 0

    for x in activity:
        if x["timestamp"] < since_ts:
            continue
        key = x.get("slug") or x["title"]
        m = per_market[key]
        m["title"] = x["title"]
        kind, usd, sz, asset = x["type"], x["usdcSize"], x["size"], x["asset"]

        if kind == "TRADE":
            n_trades += 1
            if x["side"] == "BUY":
                buys += usd
                delta[asset] += sz
                m["buy"] += usd
                m["delta"][asset] += sz
                bought[asset]["sh"] += sz
                bought[asset]["usd"] += usd
                o = pair_sides[key][x["outcome"]]
                o["sh"] += sz
                o["usd"] += usd
            else:
                sells += usd
                delta[asset] -= sz
                m["sell"] += usd
                m["delta"][asset] -= sz
                sold[asset]["sh"] += sz
                sold[asset]["usd"] += usd
        elif kind == "REDEEM":
            sells += usd
            delta[asset] -= sz
            m["sell"] += usd
            m["delta"][asset] -= sz
        elif kind in ("REWARD", "MAKER_REBATE"):
            income += usd

    def mark(asset: str) -> float:
        p = positions.get(asset)
        return p["curPrice"] if p else 0.0

    inventory_value = sum(sh * mark(a) for a, sh in delta.items())

    # Round trips: shares both bought and sold inside the window. This is the
    # number that killed the book -- entering as a maker and exiting as a
    # taker, against a 2-4c gross spread.
    rt_pnl = rt_shares = 0.0
    rt_win = rt_lose = 0
    for asset, b in bought.items():
        s = sold.get(asset)
        if not s or s["sh"] < 1e-6 or b["sh"] < 1e-6:
            continue
        q = min(b["sh"], s["sh"])
        bv, sv = b["usd"] / b["sh"], s["usd"] / s["sh"]
        p = q * (sv - bv)
        rt_pnl += p
        rt_shares += q
        if p < 0:
            rt_lose += 1
        elif p > 0:
            rt_win += 1

    # Pairs: both Yes and No bought in the same market. Holding one of each
    # pays exactly $1.00, so a combined cost over par is a locked-in loss.
    pair_locked = 0.0
    pairs_over_par = 0
    pairs_total = 0
    pair_rows = []
    for key, sides in pair_sides.items():
        if len(sides) < 2:
            continue
        (n1, o1), (n2, o2) = list(sides.items())[:2]
        v1, v2 = o1["usd"] / o1["sh"], o2["usd"] / o2["sh"]
        matched = min(o1["sh"], o2["sh"])
        cost = v1 + v2
        locked = matched * (1.0 - cost)
        pair_locked += locked
        pairs_total += 1
        if cost > 1.0:
            pairs_over_par += 1
        pair_rows.append((locked, cost, matched, per_market[key]["title"]))

    rows = []
    for key, m in per_market.items():
        v = sum(sh * mark(a) for a, sh in m["delta"].items())
        rows.append((m["sell"] - m["buy"] + v, m["buy"], m["sell"], v, m["title"]))
    rows.sort()

    return {
        "buys": buys,
        "sells": sells,
        "income": income,
        "inventory_value": inventory_value,
        "pnl": sells - buys + income + inventory_value,
        "n_trades": n_trades,
        "rt_pnl": rt_pnl,
        "rt_shares": rt_shares,
        "rt_win": rt_win,
        "rt_lose": rt_lose,
        "pair_locked": pair_locked,
        "pairs_over_par": pairs_over_par,
        "pairs_total": pairs_total,
        "pair_rows": sorted(pair_rows),
        "rows": rows,
        "n_markets": len(per_market),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--since", help="UTC date, YYYY-MM-DD (default: midnight UTC today)")
    g.add_argument("--hours", type=float, help="look back this many hours instead")
    ap.add_argument("--by-market", action="store_true", help="per-market P&L breakdown")
    ap.add_argument("--address", help="override POLYMARKET_FUNDER_ADDRESS")
    args = ap.parse_args(argv)

    address = args.address or _env("POLYMARKET_FUNDER_ADDRESS")
    if not address:
        print("no POLYMARKET_FUNDER_ADDRESS in .env and no --address given", file=sys.stderr)
        return 2

    now = datetime.now(timezone.utc)
    if args.hours:
        since = now - timedelta(hours=args.hours)
    elif args.since:
        since = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        since = now.replace(hour=0, minute=0, second=0, microsecond=0)

    activity = _fetch_all_activity(address)
    positions = _fetch_positions(address)
    a = analyse(activity, positions, since.timestamp())

    hours = (now - since).total_seconds() / 3600.0
    print(f"=== Book M realised P&L, per the exchange ===")
    print(f"wallet {address}")
    print(f"window {since:%Y-%m-%d %H:%M} -> {now:%Y-%m-%d %H:%M} UTC  ({hours:.1f}h)\n")

    if a["n_trades"] == 0:
        print("  no trades in this window.")
        print("  If the loop is running dry, that is expected and permanent --")
        print("  a dry run places no orders, so it can never produce P&L.")
        return 0

    print(f"  bought                          -${a['buys']:>10,.2f}")
    print(f"  sold / redeemed                 +${a['sells']:>10,.2f}")
    print(f"  liquidity rewards + rebates     +${a['income']:>10,.2f}")
    print(f"  net inventory, at today's marks +${a['inventory_value']:>10,.2f}")
    print(f"  {'-' * 44}")
    print(f"  P&L                              ${a['pnl']:>+10,.2f}    over {a['n_trades']} trades in {a['n_markets']} markets")

    print("\n--- where it came from ---")
    if a["rt_shares"] > 0:
        cps = a["rt_pnl"] / a["rt_shares"] * 100.0
        n = a["rt_win"] + a["rt_lose"]
        print(f"  round trips (bought and sold inside the window):")
        print(f"    {n} legs, {a['rt_lose']} losing / {a['rt_win']} winning")
        print(f"    ${a['rt_pnl']:+,.2f} on {a['rt_shares']:,.0f} shares = {cps:+.2f} c/share")
        if cps < -1.0:
            print(f"    ^ this is the exit cost. Against a 2-4c gross spread, anything")
            print(f"      past about -1c/share means exits are eating the whole book.")
    else:
        print("  round trips: none yet (nothing bought and sold in the same window)")

    if a["pairs_total"]:
        print(f"\n  Yes/No pairs completed: {a['pairs_total']}, of which {a['pairs_over_par']} cost more than $1.00")
        print(f"    locked-in P&L on matched pairs: ${a['pair_locked']:+,.2f}")
        for locked, cost, matched, title in a["pair_rows"][:5]:
            if cost <= 1.0:
                continue
            print(f"      {cost:.4f} per pair x {matched:>6.1f} sh = ${locked:>7.2f}  {title[:44]}")

    print("\n--- the verdict ---")
    if a["income"] > 0:
        ratio = abs(a["pnl"]) / a["income"]
        verb = "earned" if a["pnl"] > 0 else "lost"
        print(f"  ${a['income']:,.2f} of reward income; ${abs(a['pnl']):,.2f} {verb} overall")
        if a["pnl"] < 0:
            print(f"  -> losing ${ratio:,.1f} for every $1 of reward earned")
    else:
        print("  no reward income in this window at all.")
        print("  Rewards are the entire thesis (strategy_v2.md 3.3). If this stays")
        print("  at zero while fills happen, the book is quoting one-sided --")
        print("  rewards.combine_sides scores a one-sided quote as zero.")

    daily = a["pnl"] / hours * 24.0 if hours > 0 else 0.0
    print(f"\n  run rate: ${daily:+,.2f}/day at this window's pace")

    if args.by_market:
        print("\n--- per market ---")
        print(f"  {'pnl':>8} {'bought':>8} {'sold':>8} {'invNow':>8}  market")
        for pnl, buy, sell, inv, title in a["rows"]:
            if abs(pnl) < 0.01:
                continue
            print(f"  {pnl:>8.2f} {buy:>8.2f} {sell:>8.2f} {inv:>8.2f}  {title[:52]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
