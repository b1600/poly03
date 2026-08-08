# poly03 — Polymarket "Boring Edge" Bot

Implementation of `strategy_v1.md`: harvest the favorite-longshot bias on
Polymarket by buying near-certain outcomes at 85-97c and holding to
resolution, filtered through a conservative "is this actually obvious?"
classifier.

Two phases are implemented so far, per `strategy_v1.md` §8's rollout plan
(backtest → **paper** → micro-live → scale):

- **Phase 0 (backtest/research)** — `scan`, `backtest`, `montecarlo`.
- **Phase 1 (paper trading)** — `paper tick|status|report|reset`. Runs the
  full entry/exit/lifecycle loop against live Gamma/CLOB data with a
  simulated bankroll. **Still $0 real capital and no live orders** — see
  "Known gaps" below for exactly what paper trading can't validate.

Nothing in this repo places a live order. Do not skip to Phase 2
(micro-live) until the §8 gate is met — run `poly03 paper report` to check.

## Setup

```bash
uv sync
cp .env.example .env   # fill in credentials if/when you have them
```

Everything in `scan` and `backtest` works against **public, unauthenticated**
Gamma/CLOB endpoints — `.env` is only needed once you wire in authenticated
calls or (much later) order placement. Never commit `.env`.

## Book M (`strategy_v2.md`) — reward-subsidized making

`strategy_v1.md`'s Book A never traded: its edge gate was a tautology (`q` was
defined as `price + margin`, so nothing could fail it) and two exclusion
filters were miscalibrated to the venue. `strategy_v2.md` documents both, and
proposes **Book M**, which earns from liquidity rewards, the maker rebate, and
spread capture — all paid for *placing orders* rather than for being right, so
no `q` is needed anywhere.

```bash
# §3.1: what's quotable right now, and why everything else was dropped
uv run poly03 make scan

# §4 Phase 0: accumulate the reward-share observation series (no orders placed)
uv run poly03 make run --bankroll 10000

# the deliverable: our estimated share of the reward pool, with its caveats
uv run poly03 make report
```

Phase 0 places no orders and **simulates no fills** — deliberately. Fill rate,
the realized maker fee, and adverse selection cannot be paper-traded, so they
are Phase 1 (micro-live, ~$500) measurements. The reward figures come from a
*reconstruction* of Polymarket's published scoring formula; every report says
so, and the gate needs ≥200 ticks over ≥7 days before it reports READY.

Book A still runs (`poly03 paper`) but will no longer enter without a real edge
estimator wired in — see `BOOK_A_REQUIRE_EDGE_ESTIMATE` in `config.py`.

## Layout

```
src/poly03/
  config.py           # every numeric threshold in strategy_v1.md, in one place
  data/                # Gamma API client, CLOB API client, pydantic models
  filters/             # §2.2 hard exclusion filters
  classifier/          # §3 tiered "is this obvious?" classifier
    rules.py           #   mechanical rules/keyword engine (the ceiling)
    llm_veto.py         #   LLM veto/tier-capper interface (no-op by default)
    pipeline.py         #   composes the two + logs every classification
  cluster/             # §4.3 entity/theme/geography/date/source tagging + cap tracking
  scoring/              # §1.1/§2.3 annualized ROC and edge_score
  sizing/               # §4.1 fixed-fractional sizing with a 1/10-Kelly cap
  backtest/
    engine.py           # §8 Phase 0: classifier calibration vs actual resolutions
    montecarlo.py       # §4.2 equity-path simulation, with correlated-loss stress test
  making/               # strategy_v2.md §3: Book M -- reward-subsidized making
    rewards.py          #   reward config (authoritative) + scoring reconstruction
    universe.py         #   §3.1 which markets are worth quoting
    quoting.py          #   §3.2 two-sided quote construction + inventory skew
    engine.py           #   §4 Phase 0 tick: quote -> score -> estimate share
    measurement.py      #   §4 reward-share estimate + Phase 0 gate
  paper/                # §8 Phase 1: paper trading
    state.py            #   PaperState/PaperPosition + JSON persistence, JSONL decision log
    engine.py            #   run_tick(): scan -> manage open positions -> kill switches -> enter
    measurement.py        #   §7: calibration by tier, Brier, drawdown, P&L attribution, Phase-1 gate
  cli/main.py           # `poly03 scan|backtest|montecarlo|paper|make`
```

## CLI

```bash
# scan the live open-market universe for Book A candidates
uv run poly03 scan --max-markets 300

# Phase 0 backtest: classifier calibration against historical resolutions
# (slow -- one CLOB prices-history call per outcome per market)
uv run poly03 backtest --max-markets 100

# §4.2 Monte Carlo equity simulation, doc's own worked example
uv run poly03 montecarlo --n-positions 100 --price 0.91 --true-prob 0.95

# same, with correlated-loss stress (§4.3's actual point)
uv run poly03 montecarlo --n-positions 100 --price 0.91 --true-prob 0.95 \
  --cluster-size 20 --cluster-stress-prob 0.03 --cluster-stress-loss-fraction 0.8

# §8 Phase 1: paper trading. Persists to paper_state.json + appends to
# paper_decisions.jsonl (both gitignored -- override with --state-file /
# --log-file or the PAPER_STATE_FILE / PAPER_DECISION_LOG_FILE env vars).

# One instruction: start with a $100 paper bankroll and run forever,
# ticking every 30 min (§2.1's scan cadence). Ctrl+C stops it and prints
# the §7 report. Leave it running in tmux/screen/systemd for the doc's
# 30-60 day paper window.
uv run poly03 paper run --bankroll 100

# equivalent one-shot pieces, if you'd rather drive it yourself (e.g. from cron):
uv run poly03 paper reset --bankroll 100   # start (or restart) a paper run
uv run poly03 paper tick                    # one scan/manage/enter cycle
uv run poly03 paper status                  # bankroll, open/closed counts, halt state
uv run poly03 paper report                  # §7 measurement + §8 Phase 1 gate readiness
```

Note on `--bankroll 100`: sizing is a fraction of bankroll (§4.1 `base_fraction`
is 0.5%), and Polymarket enforces a real minimum order size per market
(`order_min_size`, typically $5). At $100 nearly every computed stake will
round down below that floor, so don't be surprised if `paper run` ticks for
a long time with `entered=0` -- that's the sizing engine correctly refusing
to place orders it couldn't place for real, not a bug. It's a fine way to
watch the scan/classify/exit-trigger machinery run live; it will not
produce a statistically meaningful §7 calibration sample (§8's gate wants
≥50 resolutions) at this size.

## Known gaps

Still true from Phase 0:

- **No real q (true probability) model.** `scan` still uses the placeholder
  (`price + MIN_MARGIN_PP`) and now prints a warning saying so. As of
  `strategy_v2.md` §1.1 this is treated as disqualifying rather than
  cosmetic: because the placeholder makes the margin gate a tautology, the
  paper engine refuses to enter at all unless a real `edge_estimator` is
  passed in (`BOOK_A_REQUIRE_EDGE_ESTIMATE`). Producing that estimator is
  the actual research problem; nothing here solves it. Book M exists
  precisely so that progress doesn't depend on solving it.
- **No LLM veto wired up.** `classifier/llm_veto.py` defines the interface
  (§3.2: structured output, explicit confidence, cited clause, can only
  lower a tier) but ships a no-op. The rules engine in `rules.py` is
  therefore the tier *ceiling* for the whole pipeline right now, in both
  `scan` and `paper tick`.
- **Backtest entry reconstruction is a proxy, not ground truth.** It uses
  CLOB `/prices-history`, which is sparse-to-empty for old, thin markets —
  a real survivorship gap, not just a caveat. Treat backtest sample size
  as a lower bound.

New in Phase 1 (paper trading), stated plainly rather than left implicit:

- **Fills are assumed, not simulated.** Book A is maker-only (§5.1); paper
  trading has no real matching engine to derive a fill probability from,
  so an entry that survives the filters/sizing/cap checks is assumed to
  fill in full at the current best bid. This is optimistic by
  construction, and it is explicitly *supposed* to be a Phase 2 concern —
  §8 says paper trading "cannot measure adverse selection or fill
  quality." `paper report` prints a note to this effect so a 100% paper
  fill rate is never mistaken for a real one.
- **No news/catalyst feed for the §3.3 falsification watch.** The proxy
  used instead: re-run the classifier against the market's current
  text/state every tick (catches rule amendments and anything the rules
  engine reads differently now) and treat any adverse price move past the
  §6.2 threshold (8¢ default) as signal, never as noise. This is cheaper
  and more conservative than the doc's ideal, but it will miss a
  falsifying catalyst that hasn't yet moved the book.
- **API/oracle-anomaly kill switch (§4.4's last bullet) isn't implemented**
  — stale books, failed settlements, and unexpected status transitions
  aren't detected as a class; only the four measurable kill switches
  (drawdown, loss-rate multiple, losses-in-window, Tier 1 miss) run.
- **Book B/C aren't in the paper engine.** Only Book A (tail harvest) runs
  through `paper tick`; the underround-arb and cheap-field-basket books in
  §5.2 are unimplemented in both phases.
- **No live order placement.** `data/clob.py` has the credential plumbing
  for it, but nothing calls it — Phase 2 (micro-live) is the first phase
  that would need to.

## Tests

```bash
uv run pytest
```
