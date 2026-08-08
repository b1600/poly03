# strategy_v2 — proposal: from "boring edge" to reward-subsidized making

Status: **proposal**, not implemented. Supersedes the Book A thesis in
`strategy_v1.md`. Every number below was measured against live Gamma data on
2026-08-07/08 (~9,900 markets across the top 600 events by volume); the probe
scripts are throwaway but the queries are reproducible.

---

## 1. Why v1 cannot work as written

Three independent defects. Any one of them alone is fatal; the third is the one
that matters.

### 1.1 There is no alpha model — the edge gate is a tautology

`paper/engine.py:167` and `:378` set the "estimated true probability" to

```python
q_placeholder = min(0.999, maker_price + MIN_MARGIN_PP)
```

So `margin_pp = q - maker_price = MIN_MARGIN_PP` by construction, and
`passes_min_margin` (`scoring/edge_score.py:71`) is `margin >= MIN_MARGIN_PP` —
always true. The §2.3 margin-of-error gate, the thing that is supposed to decide
whether a market is mispriced, **cannot ever reject a market for lack of edge.**
The same fabricated `q` is fed to `kelly_fraction()` in
`sizing/position_sizing.py:58`, so position size is Kelly-derived from an
assumption, not an estimate.

v1 is not "buy favorites that are underpriced." It is "buy every favorite that
survives the text filters, and assume 4c of edge." The 4c is the entire P&L
thesis and nothing in the system measures it.

A side effect worth fixing regardless: the `min(0.999, ...)` clamp means every
market priced above 0.959 fails the gate for a purely arithmetic reason.

| price | q | margin | passes |
|---|---|---|---|
| 0.959 | 0.999 | 0.0400 | yes |
| 0.960 | 0.999 | 0.0390 | **no** |
| 0.970 | 0.999 | 0.0290 | **no** |

The effective price band is 0.85–0.959, not the configured 0.85–0.97.

### 1.2 The universe collapses to zero, and two filters are miscalibrated

Of 1,756 open markets with two-sided quotes, **85** sit in the 0.85–0.97 band.
Applying `filters/exclusion.py` to those 85: **0 survive.** Reason counts
(markets trip several):

| reason | count / 85 |
|---|---|
| `thin_book` | 83 |
| `ambiguous_resolution_criteria` | 60 |
| `wide_spread` | 28 |
| `resolution_date_beyond_horizon_cap` | 8 |

The screenshot's read was right that this is "filters working as designed," but
two of them are not calibrated to the venue:

- **`thin_book`.** `MIN_OPEN_INTEREST_USD = 10_000` against a median in-band OI
  of **$2,449**, and `MIN_24H_VOLUME_USD = 2_000` which only **6 of 85** in-band
  markets clear. Each gate independently removes ~95% of the band. These are
  reasonable numbers for a venue with deeper books; they are not reasonable here.
- **`ambiguous_resolution_criteria`.** This is a genuine bug, not a tuning
  issue. Across 1,040 markets the only two keywords that fire are:

  | keyword | hits / 1,040 |
  |---|---|
  | `consensus of credible` | 717 |
  | `credible report` | 687 |

  Those phrases are Polymarket's **standard UMA resolution boilerplate** ("a
  consensus of credible reporting"), present on ~70% of all markets regardless
  of how crisp the resolution criteria actually are. The filter is not detecting
  ambiguity; it is detecting Polymarket. It should be scoped to markets where
  the boilerplate is the *only* resolution language, not used as a keyword veto.

### 1.3 The economics were never good enough to be worth the tail risk

Even with 1.1 and 1.2 fixed, the trade is: pay ~0.93 to collect 1.00 over a
**median 87 days** (in-band, measured). That is ~7c gross, ~32%/yr annualized —
and the loss branch is 93c. The strategy is short a ~7% tail whose true
probability it has explicitly declined to estimate. Sizing controls (§4.1/§4.3)
bound how *much* you lose per event; they do nothing about the fact that the
sign of the expectancy is unknown.

The ~50 available names are also mostly the same trade — safe-seat House races,
uncontested nominations — so the cluster caps in §4.3 will bind long before the
book is diversified enough for the law of large numbers to do the work the
thesis needs it to do.

---

## 2. What the venue actually pays for

The probe turned up two structural facts that `strategy_v1.md` does not account
for, and they point the same direction.

### 2.1 Fees are taker-only, and makers are rebated

8,732 of 9,878 markets have `feesEnabled: true`. The `feeSchedule` payload is
consistent across every fee type found:

| feeType | rate | takerOnly | rebateRate |
|---|---|---|---|
| `sports_fees_v2` | 0.05 | true | 0.15 |
| `politics_fees` | 0.05 | true | 0.25 |
| `crypto_fees_v2` | 0.07 | true | 0.20 |
| `tech_fees`, `finance_prices_fees` | 0.04 | true | 0.25 |
| `general_fees`, `economics_fees`, `culture_fees`, `weather_fees` | 0.05 | true | 0.25 |

Taking liquidity at p=0.93 costs roughly `rate × min(p, 1-p)` ≈ 0.05 × 0.07 =
**0.35c/share** — about 5% of v1's entire 7c gross. Providing it costs zero and
earns a rebate.

> Caveat: the legacy `makerBaseFee`/`takerBaseFee` fields both read `1000` even
> where `feeSchedule.takerOnly` is true. `feeSchedule` is almost certainly
> authoritative, but **confirm the realized maker fee on live fills before
> sizing anything on it.** This is the single assumption in this document that
> a paper run cannot verify.

### 2.2 The venue pays a seven-figure annual subsidy for resting quotes

5,859 open markets carry liquidity-reward parameters (`rewardsMinSize`,
`rewardsMaxSpread`); 1,261 have a funded `clobRewards` config totalling
**$22,359/day advertised**. Narrowing to markets that are actually worth
quoting — open, two-sided, 24h volume ≥ $1,000, funded rewards:

- 492 markets with ≥$1k daily volume
- **340** of those pay rewards, **$15,783/day** total
- **124** of those also have a spread ≥ 2 ticks, i.e. room to improve both sides
- rewards on that quotable subset: **$3,458/day ≈ $1.26M/yr**

Typical parameters are `rewardsMaxSpread: 4.5` (cents from mid) and
`rewardsMinSize: 50` ($50 resting). Top of the book by daily rate:

| $/day | bid | ask | spread | 24h vol | minSize |
|---|---|---|---|---|---|
| 300 | 0.430 | 0.450 | 0.020 | $234,943 | 200 |
| 200 | 0.039 | 0.041 | 0.002 | $227,895 | 200 |
| 194 | 0.308 | 0.310 | 0.002 | $12,037 | 50 |
| 164 | 0.250 | 0.280 | 0.030 | $5,786 | 50 |
| 100 | 0.937 | 0.950 | 0.013 | $48,444 | 200 |

**That $1.26M/yr is split among all makers, and our share is unknown.**
Determining it is the first thing the new phase 0 must do (§4). But note the
asymmetry against v1: this income is *mechanical* — it is paid for order
placement, not for being right — and it is measurable in days rather than in the
50+ resolutions and 30–60 days that §8's Phase 1 gate requires.

### 2.3 What I checked and rejected

For completeness, two ideas that the data does not support:

- **Multi-outcome dutch-book arbitrage.** Using the authoritative `negRisk`
  exclusivity flag (not title heuristics — those produce false positives like
  "Will Trump visit \<state\>", 50 *independent* legs whose bids sum to ~20),
  of 188 fully-negRisk events only 8 have `sum(bid) > 1.00`, none exceed 1.02,
  and the best is **0.28% gross over 39 days**. Most have zero displayed depth.
  Real, but not a business.
- **Selling the longshot tail.** The 0.00–0.02 bucket looks huge (5,136
  markets) until you check the spread: median ask on that bucket is **1.000**,
  i.e. there is no offer. It is dead inventory, not an opportunity.

---

## 3. Proposal: Book M — reward-subsidized two-sided making

**Thesis.** Stop trying to predict outcomes. Get paid the three things the venue
pays mechanically — liquidity rewards, maker rebate, spread capture — and treat
v1's classifier as a *risk* filter (which markets are safe to have inventory in)
rather than an *alpha* filter (which markets are mispriced).

This inverts what v1 optimizes. v1 needs `q` to be right. Book M needs only that
we not be catastrophically adversely selected, which is a bounded, controllable,
and — critically — *observable* problem.

### 3.1 Universe

Quote a market only if all hold:

- funded rewards from the CLOB's `/sampling-markets` (authoritative for
  `min_size`, `max_spread`, `rewards_daily_rate` — Gamma's `clobRewards`
  mirrors it but denormalised)
- 24h volume ≥ $1,000 and `acceptingOrders`
- `0.02 < bid`, `ask < 0.98` (avoid the pinned tails, where inventory is
  un-exitable and reward scoring degenerates)
- spread ≥ 2 × `orderPriceMinTickSize` — we must be able to improve both sides
  without crossing
- passes a **repaired** exclusion filter set (§1.2 fixes applied), restricted
  to the resolution-integrity checks — see the correction below
- **not** resolving within 48h, and no unsettled `umaResolutionStatuses`

Measured today that is ~124 markets, comfortably more than a $10k book can
quote. The implementation reproduces this independently: `poly03 make scan`
returns 126–129 quotable markets and a $3,464/day pool against the $3,458/day
measured in §2.2.

**Corrections found during implementation.** Three of this section's
assumptions did not survive contact with the data:

1. **The classifier cannot be reused as a risk gate.** §3.5 below says to keep
   it for "which markets we'll hold". But Tier 4 means *requires a forecast* —
   a statement about predictability, not about resolution integrity. Nearly
   every market worth quoting is Tier 4, because two-way flow is exactly what
   makes a market unpredictable and profitable to quote. Gating on it collapsed
   the universe to 7 markets. The tier is now recorded for reporting only;
   resolution risk is carried entirely by the filter subset and the flatten
   window.
2. **Sort by 24h volume, not cumulative volume.** Cumulative volume is
   dominated by markets that were busy months ago and are dead now — the
   opposite of what this section wants. Ordering by the same quantity we gate
   on also lets the scan stop at the floor instead of paging blindly.
3. **Scan per-market, not per-event.** Paginating by event volume front-loads
   on mega multi-outcome events whose legs are mostly unfunded, dropping the
   overlap with the reward-eligible set from ~24% to ~3%. (Gamma also 422s past
   roughly `offset=2000` on every sort order, a hard ceiling on scan depth.)

### 3.2 Quoting

- Two-sided, resting, inside `rewardsMaxSpread` of mid (that is the reward
  eligibility condition — outside it we earn nothing).
- Size per side = `max(rewardsMinSize, …)` subject to §4.1 caps. `rewardsMinSize`
  is a **hard floor**: below it the order earns no rewards at all, which makes
  small orders strictly worse than no order. This directly replaces the
  `order_min_size` frustration the README documents — the binding minimum is
  now $50–$200/order, so **$10k is a workable bankroll and $100 is not.**
- Skew quotes against inventory: as net position in a market grows, widen the
  side that would add to it and tighten the side that would reduce it.
- Cancel and re-quote on mid movement beyond a threshold; always cancel into
  the 48h-to-resolution window and flatten.

### 3.3 Revenue lines, ranked by confidence

| line | mechanism | confidence |
|---|---|---|
| Liquidity rewards | paid per resting eligible order | high — advertised, funded, observable |
| Maker rebate | `feeSchedule.rebateRate` | medium — needs live confirmation (§2.1) |
| Spread capture | round-trip inside the quoted spread | medium — net of adverse selection |

Note this is a deliberately *unambitious* stack. None of it requires being right
about anything. If it clears costs it compounds; if it doesn't, we find out in
two weeks instead of two months.

### 3.4 Risks and controls

- **Adverse selection** — the real cost. Informed flow picks off stale quotes.
  Controls: quote only markets with no imminent catalyst; cancel aggressively on
  mid moves; hard per-market inventory cap. **This is what phase 0 must measure,
  and it is exactly what §8 of v1 says paper trading cannot measure** — so this
  strategy needs micro-live sooner than v1 did, at correspondingly small size.
- **Resolution risk on residual inventory** — you will be holding something when
  a market resolves. Controls: the repaired exclusion filters, the 48h flatten
  rule, and the existing tier classifier used as a gate on *which* markets we're
  willing to be caught holding.
- **Reward-program change risk** — rates are set by the venue and can go to zero
  without notice. Control: re-read `clobRewards` every scan; treat any market
  whose rate drops as an immediate quote-pull. Do not build fixed cost structure
  against this income.
- **Correlated inventory** — §4.3's cluster tagging carries over unchanged and
  now applies to net inventory rather than to directional positions.

### 3.5 What carries over from v1

Most of the codebase survives; it is the thesis that changes.

| module | disposition |
|---|---|
| `filters/exclusion.py` | keep, repair per §1.2, repurpose as risk gate |
| `classifier/*` | ~~gates which markets we'll hold~~ — **superseded**, see §3.1's correction 1: Tier 4 is about predictability, not risk. Recorded, not gated on |
| `cluster/tagging.py` | keep, applied to net inventory |
| `sizing/position_sizing.py` | replace Kelly path (no `q` to feed it); keep the caps |
| `scoring/edge_score.py` | **retire** — `annualized_roc × confidence × liquidity` has no meaning without a real `q` |
| `paper/*`, `backtest/*` | keep the harness; replace the tick body |

---

## 4. Revised rollout

The v1 gate ("≥50 resolutions, calibration in tolerance") does not apply — Book M
has no probability estimates to calibrate. Replace it:

- **Phase 0 (1 week, $0).** Poll the reward-eligible universe on the scan
  cadence; record book state and what our quotes *would* have been. Estimate our
  reward share by reconstructing the scoring against observed competing depth.
  Deliverable: a defensible number for "share of $3,458/day," the one input this
  whole proposal rests on. **Implemented** as `poly03 make run|report`; the gate
  requires ≥200 ticks over ≥7 days with a stable estimate (IQR/median ≤ 0.75).

  Three guards were needed to stop this estimate from being nonsense, and they
  are worth stating because each caught a real error:

  - The share denominator must **not** apply the per-maker two-sided scoring
    rule to the aggregated public book, and must **not** apply the `min_size`
    floor to pooled price levels. Both shrank the competition and inflated our
    share into four-figure annualized yields.
  - A market with less than one qualifying competitor's worth of resting score
    is **unidentified**: the raw share goes to 100% for any order size, which
    is an absence of evidence, not an opportunity. These are reported
    separately and excluded from the headline number.
  - No single pool is ever assumed to yield more than 50% to us.

  Even after all three, the current estimate implies an implausible annualized
  yield on collateral, and the report says so in those words. It is an upper
  bound from a static snapshot: it assumes the competing book stays as thin as
  it is today, which is the first thing that stops being true if the
  opportunity is real. Treat Phase 0's output as *evidence the universe exists
  and is worth $500 to probe*, not as a return forecast.
- **Phase 1 (2 weeks, micro-live, ~$500).** Real resting orders, minimum viable
  size. This is the earliest point at which realized fills, the actual maker fee
  (§2.1 caveat), and adverse selection become measurable — none of them can be
  paper-traded. Gate to phase 2: rewards + rebate + spread capture exceed
  realized adverse selection over ≥500 fills.
- **Phase 2.** Scale to the $10k book, capped by per-market depth share.

## 5. Recommended immediate fixes to v1, independent of this proposal

Worth doing whether or not Book M is adopted, because they are bugs:

1. `filters/exclusion.py` — scope `ambiguous_resolution_criteria` so it stops
   firing on standard UMA boilerplate (~70% of all markets).
2. `config.py` — recalibrate `MIN_OPEN_INTEREST_USD` / `MIN_24H_VOLUME_USD` to
   the venue's actual distribution (in-band median OI is $2,449).
3. `scoring/edge_score.py` / `paper/engine.py` — either supply a real `q` or
   delete the margin gate. A gate that cannot reject is worse than no gate: it
   reads as risk control in the code and provides none.
4. `paper/engine.py:493` — `report.scanned = max_markets` should be the count
   actually returned by `scan_universe()`. (Already noted in the earlier review.)
5. Investigate the `paper_state.json` / `paper_trade.log` divergence — state
   shows `n_ticks: 2` against ~1,441 logged ticks, so the long-running process
   is writing to a different state file than the one in the repo.
