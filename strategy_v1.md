# Strategy v1 — Polymarket "Boring Edge" Bot

**Status:** design document, pre-implementation
**Date:** 2026-08-05
**Reference:** `concept.txt` (the viral framing), `polymarket_2b9s_analysis.md` (the actual account)

---

## 0. What we are actually copying (and what we're not)

The tweet in `concept.txt` says: buy the obvious side at 70–90¢, wait, collect $1, "x3–x10 on every trade." That is internally inconsistent — buying at 80¢ and collecting $1 is **+25%**, not 3–10x. The 3–10x numbers in the account's own P&L come from a *different* playbook: the 0.8¢ longshot basket on the 2028 GOP nomination market, where one leg (Vance) went 0.8¢ → 45.9¢.

So the account is running two uncorrelated books, and we should build both explicitly rather than blur them:

| Book | Mechanic | Return shape | Role |
|---|---|---|---|
| **A. Tail harvest** | Buy the near-certain side at 85–97¢, hold to resolution | Small, frequent, positively-skewed-until-it-isn't | Bankroll engine |
| **B. Field basket** | Buy *every* outcome in a multi-outcome market when it's cheap, or buy the whole underround field | Rare, large, convex | Lottery ticket funded by A |
| **C. Underround arb** (new) | Buy every outcome when Σ(asks) < $1 in a mutually-exclusive-and-exhaustive market | Deterministic, capacity-limited | Free money when it appears |

Book A pays the bills. Book B is what produces the screenshot. Book C is the only genuinely risk-free one and will be rare.

**The one sentence that matters:** the edge is not prediction, it is the *favorite–longshot bias* — retail systematically overpays for lottery-shaped outcomes, which mechanically underprices the boring side. Our job is to harvest that bias at scale while not blowing up on the 1-in-15 event that actually happens.

---

## 1. Where the money comes from, and where it goes

### 1.1 The arithmetic of Book A

Buy at price `p`, hold to resolution, collect $1 if right, $0 if wrong.

- Gross return per trade: `(1 - p) / p`
- Breakeven win rate: exactly `p`
- Annualized return on capital: `(1/p)^(365/T) - 1`, where `T` = days to resolution

| Entry | Return | Breakeven win rate | Annualized @ 30d | @ 90d | @ 365d |
|---|---|---|---|---|---|
| 0.85 | +17.6% | 85% | 486% | 91% | 17.6% |
| 0.90 | +11.1% | 90% | 254% | 53% | 11.1% |
| 0.95 | +5.3% | 95% | 89% | 23% | 5.3% |
| 0.97 | +3.1% | 97% | 45% | 13% | 3.1% |

**Two consequences the bot must be built around:**

1. **Time to resolution is a first-class variable, not an afterthought.** Buying "Jesus returns before 2027 — No" at 96.2¢ with 17 months left is ~2.7% annualized. That is worse than a T-bill and it locks capital in an unmargined, non-interest-bearing position. The account in the analysis does this anyway; we should not, except as a small tail-diversifier. **Rank every candidate by annualized ROC, never by raw spread to $1.**

2. **A single loss erases ~10 wins at 90¢.** The strategy is short a tail. It looks flawless right up until it isn't. Everything in §4 (risk) exists because of this line.

### 1.2 The hidden costs

Model these explicitly before any live capital:

- **Fee schedule** — verify current maker/taker fees on the CLOB at build time; do not assume zero. A 1% taker fee on an 11% gross return is 9% of the edge gone.
- **Spread** — crossing a 2¢ spread on a 92¢ position costs more than a fifth of the trade's entire upside. **Book A is a maker-only strategy.** See §5.
- **Capital lockup** — no margin, no interest on posted collateral. Opportunity cost is real; it's why §1.1 point 1 exists.
- **Resolution lag** — UMA dispute windows can add days-to-weeks after the real-world event. Budget for it in ROC calculations.
- **Slippage on size** — these books are thin. Our own orders move the price we're measuring.

---

## 2. Universe construction and market scanning

### 2.1 Data sources

- **Gamma API** — market metadata, categories, resolution dates, resolution source text, volume/liquidity.
- **CLOB API** — live order books, best bid/ask, depth, tick size, min order size, token IDs per outcome.
- **WebSocket feeds** — book updates for markets already in the watchlist; do not poll REST at high frequency.

Scan cadence: full universe sweep every 15–30 min. Watchlist books over websocket continuously.

### 2.2 Hard exclusion filters (applied first, no exceptions)

A market is **disqualified** if any of these are true:

1. **Ambiguous resolution criteria.** If the resolution text depends on a subjective judgment ("widely reported as", "generally considered", "notable"), skip it. This is the #1 source of losses that look like a bug but aren't. Flag any market whose rules text contains subjective qualifiers for manual review, never auto-trade it.
2. **Unreliable or single-point resolution source.** A named primary source (official government release, exchange print, sports league) is required. "Consensus of media reports" is not.
3. **Early/partial resolution possible** in a way that breaks the hold-to-expiry assumption.
4. **Thin book** — sub-threshold 24h volume or open interest (default: <$10k OI, <$2k 24h volume).
5. **Wide spread** — best bid/ask spread > 2¢ on Book A candidates.
6. **Resolution date beyond horizon cap** — default 270 days for Book A.
7. **Already-resolving / in dispute** status.
8. **Correlated-cluster full** — see §4.3.

### 2.3 Book A candidate scoring

For each surviving market and each side, compute:

```
edge_score = annualized_ROC × confidence_multiplier × liquidity_factor
```

Where:

- `annualized_ROC` from §1.1, using **the price we'd actually get filled at as a maker** (i.e. the bid we intend to post, not the mid).
- `confidence_multiplier` — a 0–1 penalty from the classifier in §3.
- `liquidity_factor` — penalizes markets where our target size is >10% of visible depth.

Require a minimum absolute margin: **only enter if our estimated true probability exceeds the ask by ≥ 4 percentage points.** Below that, the estimate error dominates the edge.

**Price band for Book A: 0.85–0.97.**
- Below 0.85 we're no longer harvesting a bias, we're taking a view — leave it.
- Above 0.97 the return doesn't clear fees, spread, and capital cost, and the tail loss is catastrophic relative to the win.

---

## 3. The "is this actually obvious?" classifier

This is the core of the system and the part most likely to be wrong. Everything else is plumbing.

### 3.1 Tiered question taxonomy

Assign each market to a tier. The tier sets `confidence_multiplier` and the position cap.

**Tier 1 — Physical / structural impossibility (multiplier 1.0)**
Outcome would require violating a known constraint or a formal process that cannot complete in the time available. Examples: constitutional processes with fixed minimum timelines; events requiring a step that provably hasn't started; "X happens by date D" where the mandatory lead time to X exceeds D − today.
*These are the only ones where a 96¢ entry is defensible.*

**Tier 2 — Status-quo inertia with a hard clock (multiplier 0.7)**
Nothing has changed, changing takes visible institutional machinery, and the window is short. "Regime falls before date D," "treaty signed by D," "official resigns by D" — where D is <90 days out and there is no live process underway.

**Tier 3 — Base-rate favorites (multiplier 0.4)**
Historically rare events with no current catalyst. Genuinely mispriced by longshot bias, but base rates are estimates and regimes change. Small size only.

**Tier 4 — Anything requiring a forecast (multiplier 0.0 → excluded)**
Elections, sports, price levels, earnings, anything where the market has informed participants and no structural anchor. **We have no edge here.** Excluded from Book A entirely.

The classifier should be **conservative by construction**: unclassifiable → Tier 4 → no trade. The failure mode we care about is a Tier-3 market being scored as Tier 1.

### 3.2 Implementation approach

- Rules/keyword/template matching for the mechanical parts (date extraction, horizon math, resolution-source parsing, "has the process started" checks against a small set of structured feeds).
- LLM-assisted classification for the rules text — used as a **veto and a tier-capper, not a promoter**. An LLM saying "this is obvious" never raises a tier; an LLM flagging ambiguity always lowers one. Require structured output with an explicit confidence and a cited clause from the rules text.
- Every auto-classification is logged with its inputs so we can back-test the classifier separately from the strategy.

### 3.3 Catalyst monitoring (the thing that prevents the blowup)

For every open Book A position, maintain a **falsification watch**: what observable would tell us the "obvious" thing is no longer obvious? News/feed monitoring against those keywords, with an automatic downgrade → exit path (§6.2). A position bought at 94¢ that trades to 80¢ on real news is not a dip to buy; it is the market telling us our classifier was wrong.

---

## 4. Position sizing and risk

### 4.1 Sizing rule

**Do not use full Kelly.** At `p = 0.92` with an estimated true probability `q = 0.99`, Kelly says bet 87.5% of bankroll. That number is an artifact of pretending we know `q` to two decimals. We don't.

Use **fixed-fractional with a Kelly cap**:

```
stake = min(
  base_fraction × bankroll × tier_multiplier,
  kelly_fraction(q, p) × 0.10 × bankroll,     # 1/10 Kelly, hard ceiling
  max_position_usd,
  0.10 × visible_book_depth
)
```

Defaults to start (calibrate after 200+ resolved trades):

| Parameter | Default |
|---|---|
| `base_fraction` | 0.5% of bankroll |
| Tier 1 multiplier | 2.0× (→1.0%) |
| Tier 2 multiplier | 1.0× (→0.5%) |
| Tier 3 multiplier | 0.4× (→0.2%) |
| Max single position | 2% of bankroll, hard |
| Max Book A deployed | 60% of bankroll |
| Max Book B deployed | 10% of bankroll |
| Min cash reserve | 20% of bankroll |

At 0.5% average sizing, the strategy needs **breadth**: 100–300 concurrent positions. That matches the reference account's 35,512 trades. Breadth is not a nice-to-have — it is the only thing converting a 90% win rate into a survivable equity curve.

### 4.2 The loss you must budget for

At 100 positions averaging 91¢ entry with a *true* 95% resolution rate, expect ~5 losses per cycle. Model the equity path with a Monte Carlo before going live, including a stress case where losses cluster (they will — see §4.3). If the drawdown in the 5th-percentile path is unacceptable, the sizing is wrong, not the model.

### 4.3 Correlation is the real risk

500 independent-looking "No" bets on Trump-adjacent political markets are **one bet on one political regime**. The account in the analysis is heavily exposed to exactly this. Enforce:

- **Cluster tagging** on every market: entity, theme, geography, resolution source, resolution date bucket.
- **Cluster caps**: max 15% of bankroll per entity cluster, 25% per theme.
- **Date-bucket caps**: max 20% of bankroll resolving in any single 30-day window — otherwise one news cycle marks the whole book at once.
- **Resolution-source cap**: max 20% depending on any single oracle/source, so one bad UMA resolution can't cascade.

### 4.4 Kill switches

Halt all new entries and alert when any trips:

- Bankroll drawdown > 15% from high-water mark
- Book A realized loss rate exceeds implied loss rate by >2× over trailing 50 resolutions
- ≥3 losses in any 7-day window
- Any Tier 1 position resolves against us (this means the classifier is broken, not unlucky) — **full stop, manual review required**
- API/oracle anomalies: stale books, failed settlements, unexpected market status transitions

---

## 5. Execution

### 5.1 Maker-only for Book A

Non-negotiable. At 92¢, crossing a 2¢ spread costs 2¢ against an 8.7¢ total upside — nearly a quarter of the trade.

- Post limit orders at or just inside the bid at our computed target price.
- **Never** chase. If the market moves away, cancel and re-evaluate; there are hundreds of other markets.
- Accept partial fills; scale into positions over hours or days.
- Order lifetime with automatic re-quote on book movement beyond a threshold.
- Respect tick size and min order size per market; round conservatively (toward a better price for us).

### 5.2 Book B / C execution

- **Book C (underround arb):** only fires when `Σ(best asks across all outcomes) < 1 − fees − buffer` **and** the outcome set is provably mutually exclusive *and exhaustive*. The exhaustiveness check is the whole game: if there's no "Other / None of the above" outcome, the field is **not** exhaustive and this is not an arb — it's a directional bet that a listed candidate wins. Default: require an explicit catch-all outcome, or skip. Execute all legs as near-simultaneously as possible; leg risk is the failure mode. Investigate whether the negRisk adapter's merge/convert mechanics apply — they can change the capital math materially. Verify current behavior against live API docs before relying on it.
- **Book B (cheap field basket):** buy every outcome under a price ceiling (default ≤2¢) in long-dated multi-outcome markets. Total basket cost capped at 1% of bankroll; treat the entire basket as expected-zero. This is the Vance trade. It works because long-dated candidate lists get repriced violently as the field narrows — but it needs a long horizon and is uncapped only in the upside direction, so size it as a donation you'd be fine losing.

### 5.3 Operational

- Idempotent order submission with client-generated order IDs; never double-fill on a retry.
- Full local order/position state, reconciled against the API every cycle. Trust reconciliation, not local assumptions.
- Rate-limit compliance with backoff; a ban mid-book is a real risk.
- Signing keys in a secrets manager, never in the repo. Separate read-only keys for the scanner from the trading key.
- Structured logging of every decision (including *rejections* and why) — the rejection log is the training data for classifier v2.

---

## 6. Position lifecycle

### 6.1 Default: hold to resolution

Book A's edge is realized at settlement. No profit-taking at 98¢ unless the annualized ROC on the remaining 2¢ has fallen below the redeployment hurdle (§6.3).

### 6.2 Early exit triggers

Exit immediately, at market if necessary, when:

- The falsification watch (§3.3) fires — the structural assumption is broken.
- Tier downgrade on re-classification.
- Price drops more than X¢ below entry on volume, with no identified cause — treat unexplained adverse moves as information we don't have yet, not as noise. (Default X = 8¢; tune.)
- Resolution rules are amended or a dispute is filed.

### 6.3 Capital recycling

Maintain a **hurdle rate**: the annualized ROC of the best available unfilled candidate. If an open position's remaining annualized ROC (from current ask to $1 over remaining days) falls below the hurdle, close it and redeploy. This is what turns a static hold-to-expiry book into a compounding one, and it's the main structural improvement over the reference account.

---

## 7. Measurement

Track from day one — the strategy is unfalsifiable without these:

- **Calibration by tier**: predicted vs realized resolution rate, bucketed. This is the single most important number. If Tier 1 isn't resolving ≥99%, the classifier is mislabeled.
- **Brier score** on the classifier's implied probabilities.
- Realized vs modeled annualized ROC (gap = fees + slippage + lockup we underestimated).
- Fill rate on maker orders; adverse selection check (are we only getting filled when we're wrong?).
- Drawdown, time-to-recovery, max cluster concentration actually reached.
- Per-tier and per-cluster P&L attribution.
- **Counterfactual log**: what rejected markets would have returned. Tells us if the filters are too tight.

---

## 8. Rollout phases

| Phase | Duration | Capital | Gate to advance |
|---|---|---|---|
| **0. Backtest** | — | $0 | Reconstruct historical books; validate classifier tiers against actual resolutions. Beware survivorship: include delisted/disputed markets. |
| **1. Paper** | 30–60 days | $0 | ≥50 simulated resolutions; realized calibration within tolerance; no Tier 1 misses. |
| **2. Micro-live** | 60 days | $500–1,000 | Real fills, real fees, real slippage measured. Compare to paper — the gap is the real cost model. |
| **3. Scale** | ongoing | ramp 2× per 90 days if metrics hold | Any kill switch trip resets to prior tier. |

Do not skip phase 2. Paper trading cannot measure adverse selection or fill quality, and those are where maker strategies actually die.

---

## 9. Known weaknesses of this strategy

Stated plainly so v2 has a target list:

1. **We are short a tail.** The equity curve looks like a bond fund until it looks like a car crash. Sizing and cluster caps are the only defense; neither is perfect.
2. **The edge may be capacity-limited.** These books are thin. The strategy that works at $10k may not work at $500k, and our own fills degrade the prices we measured.
3. **The classifier is the entire system.** A systematic mislabel is not diversified away by breadth — it's replicated across every position.
4. **Regime dependence.** Favorite–longshot bias is a behavioral regularity, not a law. If Polymarket's user base institutionalizes, the mispricing compresses.
5. **Platform and oracle risk** are unhedgeable: UMA resolving "wrong," rules amended post-hoc, regulatory action, withdrawal freezes. Cap total platform exposure as a fraction of *net worth*, not just of the trading bankroll.
6. **The reference account's returns are one observation.** We're reverse-engineering a strategy from a winner with no visibility into the distribution of people who ran the same playbook and lost. Treat the 35,512-trade sample as suggestive, not as validation.

---

## 10. Open questions before implementation

- Current Polymarket fee schedule (maker vs taker, any rebates)?
- Does the negRisk adapter's merge/convert change Book C's capital efficiency enough to make it a primary book?
- What's the realistic fill rate for passive quotes in the 85–97¢ band on low-volume markets — is there enough flow to build 100+ positions?
- Historical base rate: how often do markets priced ≥95¢ actually resolve against? (Backtest answers this and sets the real Tier 1 ceiling.)
- Tax/reporting treatment for high trade counts — 35k trades a year is a compliance problem as well as a strategy.
