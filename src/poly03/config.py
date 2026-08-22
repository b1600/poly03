"""Environment/credential loading and the strategy's numeric constants.

The numeric defaults here are transcribed directly from strategy_v1.md and are
the single source of truth for the rest of the codebase -- nothing downstream
should hardcode a threshold that appears in the design doc.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str | None = None) -> str | None:
    val = os.environ.get(name)
    return val if val not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    return float(val) if val not in (None, "") else default


@dataclass(frozen=True)
class Credentials:
    """Polymarket auth material. All optional -- read-only market data works
    without any of this. Populate via .env, never commit real values.

    L1 (private_key) signs orders / derives API creds. L2 (api_key/secret/
    passphrase) is what py-clob-client uses for authenticated REST calls.
    """

    private_key: str | None = field(default_factory=lambda: _env("POLYMARKET_PRIVATE_KEY"))
    funder_address: str | None = field(default_factory=lambda: _env("POLYMARKET_FUNDER_ADDRESS"))
    api_key: str | None = field(default_factory=lambda: _env("POLYMARKET_CLOB_API_KEY"))
    api_secret: str | None = field(default_factory=lambda: _env("POLYMARKET_CLOB_API_SECRET"))
    api_passphrase: str | None = field(default_factory=lambda: _env("POLYMARKET_CLOB_API_PASSPHRASE"))
    chain_id: int = field(default_factory=lambda: int(_env("POLYMARKET_CHAIN_ID", "137")))
    # task 20260818_2012 item 6: py_clob_client.ClobClient.__init__ takes a
    # signature_type (0=EOA, 1=email/magic proxy, 2=browser proxy) that this
    # codebase never passed -- if funder is a Polymarket proxy wallet, orders
    # would sign against the wrong account. Defaults to 0 (EOA), matching
    # py-clob-client's own default; set POLYMARKET_SIGNATURE_TYPE if funder
    # is a proxy wallet. `make live preflight` prints the resolved value.
    signature_type: int = field(default_factory=lambda: int(_env("POLYMARKET_SIGNATURE_TYPE", "0")))

    @property
    def has_l1(self) -> bool:
        return bool(self.private_key)

    @property
    def has_l2(self) -> bool:
        return bool(self.api_key and self.api_secret and self.api_passphrase)


@dataclass(frozen=True)
class Endpoints:
    gamma_api_url: str = field(default_factory=lambda: _env("GAMMA_API_URL", "https://gamma-api.polymarket.com"))
    clob_api_url: str = field(default_factory=lambda: _env("CLOB_API_URL", "https://clob.polymarket.com"))


# --- §2.2 hard exclusion filters -------------------------------------------------

# strategy_v2.md §1.2: the original 10k/2k pair was transcribed from
# strategy_v1.md without checking it against the venue. Measured against the
# live book, median open interest for markets in the Book A price band is
# ~$2,449 and only 6 of 85 in-band markets clear $2,000 of 24h volume -- each
# gate independently removed ~95% of the band, which is why the paper run
# logged candidates=0 for 1,441 consecutive ticks. Recalibrated to the venue's
# actual distribution; override via env if you want v1's behaviour back.
MIN_OPEN_INTEREST_USD = _env_float("MIN_OPEN_INTEREST_USD", 2_000.0)
MIN_24H_VOLUME_USD = _env_float("MIN_24H_VOLUME_USD", 250.0)
MAX_SPREAD_BOOK_A = _env_float("MAX_SPREAD_BOOK_A", 0.02)
BOOK_A_HORIZON_CAP_DAYS = int(_env_float("BOOK_A_HORIZON_CAP_DAYS", 270))

# Phrases that genuinely signal a subjective resolution. See
# AMBIGUOUS_BOILERPLATE_PATTERNS below for the ones that had to be removed.
AMBIGUOUS_RESOLUTION_KEYWORDS = (
    "widely reported",
    "generally considered",
    "generally regarded",
    "notable",
    "notably",
    "significant coverage",
    "consensus of media",
    "in the opinion of",
    "at the discretion of",
    "widely believed",
)

# strategy_v2.md §1.2: "consensus of credible reporting" and "credible report"
# were in the list above and fired on 717/1040 and 687/1040 markets
# respectively -- they are Polymarket's *standard UMA resolution boilerplate*,
# present on ~70% of all markets regardless of how crisp the criteria are. The
# filter was detecting Polymarket, not ambiguity. They are stripped from the
# text before keyword matching rather than deleted outright, so a market that
# leans on the boilerplate *and nothing else* is still caught by
# check_unreliable_source().
AMBIGUOUS_BOILERPLATE_PATTERNS = (
    r"a consensus of credible reporting",
    r"consensus of credible reporting",
    r"consensus of credible",
    r"credible reporting",
    r"credible report(s)?",
)

# --- §2.3 / §1.1 scoring ----------------------------------------------------------

BOOK_A_PRICE_BAND = (0.85, 0.97)
MIN_MARGIN_PP = _env_float("MIN_MARGIN_PP", 0.04)  # min (true_prob - ask), percentage points as a fraction
LIQUIDITY_SIZE_FRACTION_CAP = 0.10  # target size must be <=10% of visible depth else penalized

# --- §3.1 tier taxonomy ------------------------------------------------------------

TIER_CONFIDENCE_MULTIPLIER: dict[int, float] = {
    1: 1.0,
    2: 0.7,
    3: 0.4,
    4: 0.0,
}

TIER2_MAX_HORIZON_DAYS = 90

# --- §4.1 sizing --------------------------------------------------------------------

BASE_FRACTION = 0.005  # 0.5% of bankroll
TIER_SIZE_MULTIPLIER: dict[int, float] = {
    1: 2.0,
    2: 1.0,
    3: 0.4,
    4: 0.0,
}
KELLY_FRACTION_CAP = 0.10  # 1/10 Kelly
MAX_SINGLE_POSITION_FRACTION = 0.02
MAX_BOOK_A_DEPLOYED_FRACTION = 0.60
MAX_BOOK_B_DEPLOYED_FRACTION = 0.10
MIN_CASH_RESERVE_FRACTION = 0.20

# --- §4.3 correlation / cluster caps --------------------------------------------

MAX_ENTITY_CLUSTER_FRACTION = 0.15
MAX_THEME_CLUSTER_FRACTION = 0.25
MAX_DATE_BUCKET_FRACTION = 0.20
DATE_BUCKET_WINDOW_DAYS = 30
MAX_RESOLUTION_SOURCE_FRACTION = 0.20

# --- §4.4 kill switches ------------------------------------------------------------

KILL_DRAWDOWN_FRACTION = 0.15
KILL_LOSS_RATE_MULTIPLE = 2.0
KILL_LOSS_RATE_TRAILING_N = 50
KILL_MAX_LOSSES_IN_WINDOW = 3
KILL_LOSS_WINDOW_DAYS = 7

# --- §5.2 Book B / C ----------------------------------------------------------------

BOOK_B_PRICE_CEILING = 0.02
BOOK_B_BASKET_CAP_FRACTION = 0.01
BOOK_C_FEE_BUFFER = _env_float("BOOK_C_FEE_BUFFER", 0.01)

# --- §6.2 early exit ------------------------------------------------------------------

EARLY_EXIT_ADVERSE_MOVE_CENTS = 8.0

# --- §8 Phase 1: paper trading ----------------------------------------------------------

PAPER_STATE_FILE = _env("PAPER_STATE_FILE", "paper_state.json")
PAPER_DECISION_LOG_FILE = _env("PAPER_DECISION_LOG_FILE", "paper_decisions.jsonl")
PAPER_TRADE_LOG_FILE = _env("PAPER_TRADE_LOG_FILE", "paper_trade.log")
PAPER_STARTING_BANKROLL = _env_float("PAPER_STARTING_BANKROLL", 100_000.0)

# doc: "breadth ... 100-300 concurrent positions"
MAX_CONCURRENT_POSITIONS = int(_env_float("MAX_CONCURRENT_POSITIONS", 300))
# throttle: don't dump the whole candidate list into positions in one tick
PAPER_MAX_NEW_POSITIONS_PER_TICK = int(_env_float("PAPER_MAX_NEW_POSITIONS_PER_TICK", 15))
PAPER_TARGET_SCAN_MARKETS = int(_env_float("PAPER_TARGET_SCAN_MARKETS", 300))

# §8 Phase 1 gate: >=50 simulated resolutions, calibration in tolerance, no Tier 1 misses
PAPER_GATE_MIN_RESOLUTIONS = int(_env_float("PAPER_GATE_MIN_RESOLUTIONS", 50))
PAPER_GATE_CALIBRATION_TOLERANCE_PP = _env_float("PAPER_GATE_CALIBRATION_TOLERANCE_PP", 0.05)

# --- strategy_v2.md §1.1: Book A edge-estimate guard --------------------------------

# v1 fabricated `q = price + MIN_MARGIN_PP`, which made the §2.3 margin gate a
# tautology (margin == MIN_MARGIN_PP by construction, so it could never reject
# anything for lack of edge) and fed that same fabricated number to Kelly
# sizing. strategy_v2.md §5.3: "a gate that cannot reject is worse than no
# gate -- it reads as risk control in the code and provides none."
#
# With this True (the default), Book A refuses to enter rather than trading on
# a placeholder. Set BOOK_A_REQUIRE_EDGE_ESTIMATE=false to restore v1's
# behaviour, which is only defensible once a real estimator supplies q.
BOOK_A_REQUIRE_EDGE_ESTIMATE = _env("BOOK_A_REQUIRE_EDGE_ESTIMATE", "true").lower() not in ("false", "0", "no")

# --- strategy_v2.md §3: Book M -- reward-subsidized two-sided making ----------------

# §3.1 universe. Measured 2026-08-07: of ~7,250 open two-sided markets, 492
# clear $1k/24h, 340 of those pay funded CLOB rewards, and 124 of those also
# have room to improve both sides. That last set is the quotable universe.
MAKING_MIN_24H_VOLUME_USD = _env_float("MAKING_MIN_24H_VOLUME_USD", 1_000.0)
MAKING_MIN_PRICE = _env_float("MAKING_MIN_PRICE", 0.02)
MAKING_MAX_PRICE = _env_float("MAKING_MAX_PRICE", 0.98)
MAKING_MIN_SPREAD_TICKS = _env_float("MAKING_MIN_SPREAD_TICKS", 2.0)
MAKING_MIN_REWARD_DAILY_RATE = _env_float("MAKING_MIN_REWARD_DAILY_RATE", 1.0)
# §3.1/§3.2: never be quoting into a resolution. Pull and flatten inside this.
MAKING_FLATTEN_HOURS_BEFORE_RESOLUTION = _env_float("MAKING_FLATTEN_HOURS_BEFORE_RESOLUTION", 48.0)

# §3.2 quoting. rewardsMaxSpread is expressed in *cents* from the midpoint;
# quote strictly inside it so a one-tick mid move doesn't drop us out of reward
# eligibility before the next scan.
MAKING_QUOTE_SAFETY_MARGIN_CENTS = _env_float("MAKING_QUOTE_SAFETY_MARGIN_CENTS", 0.5)
MAKING_MAX_MARKETS_QUOTED = int(_env_float("MAKING_MAX_MARKETS_QUOTED", 40))
MAKING_MAX_DEPLOYED_FRACTION = _env_float("MAKING_MAX_DEPLOYED_FRACTION", 0.60)
MAKING_MAX_INVENTORY_PER_MARKET_FRACTION = _env_float("MAKING_MAX_INVENTORY_PER_MARKET_FRACTION", 0.02)
# Inventory skew: at this fraction of the per-market inventory cap, the adding
# side is fully suppressed and only the reducing side is quoted.
MAKING_INVENTORY_SKEW_SATURATION = _env_float("MAKING_INVENTORY_SKEW_SATURATION", 1.0)
MAKING_REQUOTE_MID_MOVE_CENTS = _env_float("MAKING_REQUOTE_MID_MOVE_CENTS", 1.0)

# §2.2/§4 reward-scoring reconstruction. These constants encode Polymarket's
# published liquidity-rewards scoring so the Phase 0 estimator has something
# auditable to work from. THEY ARE A RECONSTRUCTION, NOT A CONTRACT -- the only
# way to validate them is to compare estimated vs realized payouts once real
# orders rest (Phase 1). making/rewards.py says so at the call site and the
# report prints it; do not size real capital on the output alone.
REWARD_SCORING_EXPONENT = _env_float("REWARD_SCORING_EXPONENT", 2.0)
REWARD_ONE_SIDED_RATIO_CUTOFF = _env_float("REWARD_ONE_SIDED_RATIO_CUTOFF", 3.0)
# Below/above these midpoints Polymarket scores one-sided quoting, since a
# two-sided quote is not meaningful at the tails.
REWARD_ONE_SIDED_PRICE_FLOOR = _env_float("REWARD_ONE_SIDED_PRICE_FLOOR", 0.10)
REWARD_ONE_SIDED_PRICE_CEILING = _env_float("REWARD_ONE_SIDED_PRICE_CEILING", 0.90)

# Ceiling on the share of any single pool we are willing to assume we'd win.
# Without it, a market with no competing depth inside the scoring window
# estimates at 100% -- i.e. "we collect the entire pool for one minimum-size
# quote", which was producing four-figure annualized yields off $18 positions.
# An uncontested pool is missing evidence, not free money: it attracts other
# makers the moment it is worth attracting them.
REWARD_MAX_ASSUMED_SHARE = _env_float("REWARD_MAX_ASSUMED_SHARE", 0.50)

# Gamma 422s past roughly offset=2100 on its list endpoints, for every sort
# order, so this is the deepest any single scan can reach -- not a tuning knob.
GAMMA_MAX_SCAN_MARKETS = int(_env_float("GAMMA_MAX_SCAN_MARKETS", 2100))

MAKING_STATE_FILE = _env("MAKING_STATE_FILE", "making_state.json")
MAKING_DECISION_LOG_FILE = _env("MAKING_DECISION_LOG_FILE", "making_decisions.jsonl")

# --- strategy_v2.md §4 Phase 1: micro-live execution --------------------------------

# Deliberately a separate file from MAKING_STATE_FILE. Phase 0's state is a
# simulated observation series (no positions, no fills -- see
# making/state.py's docstring); Phase 1 holds real inventory funded with real
# capital. Sharing a file would mix simulated ticks with real fills in the
# same series and corrupt both the Phase 0 estimate and the Phase 1
# measurement.
MAKING_LIVE_STATE_FILE = _env("MAKING_LIVE_STATE_FILE", "making_live_state.json")
MAKING_LIVE_DECISION_LOG_FILE = _env("MAKING_LIVE_DECISION_LOG_FILE", "making_live_decisions.jsonl")

# Hard ceiling on deployed capital for Phase 1, independent of whatever
# MAKING_MAX_DEPLOYED_FRACTION would otherwise allow. strategy_v2.md §4:
# Phase 1 runs at ~$500, not whatever fraction of a hypothetical bankroll the
# Phase 0 defaults imply.
MAKING_LIVE_BANKROLL_CAP_USD = _env_float("MAKING_LIVE_BANKROLL_CAP_USD", 500.0)

# Adverse-selection kill switch (§4 rollout plan, item 3). Book M has no `q`,
# so v1's Tier-1-miss / drawdown switches don't apply -- this book's failure
# mode is getting picked off by informed flow, which shows up as fills that
# mark out against us, not as a resolved loss. Trips when the last N scored
# fills (5-minute markout, see making/execution.py compute_markouts) were ALL
# worse than this many cents/share -- a strict "every recent fill was bad",
# not an average, since the average dilutes a real regime change with older
# good fills.
MAKING_LIVE_KILL_MARKOUT_CONSECUTIVE = int(_env_float("MAKING_LIVE_KILL_MARKOUT_CONSECUTIVE", 5))
MAKING_LIVE_KILL_MARKOUT_CENTS_PER_SHARE = _env_float("MAKING_LIVE_KILL_MARKOUT_CENTS_PER_SHARE", 2.0)

# Absolute equity-drawdown halt (task 20260818_2012 item 5), separate from
# the markout-based switch above: trips from fill #1, not after
# MAKING_LIVE_KILL_MARKOUT_CONSECUTIVE scored fills. Mirrors v1's
# KILL_DRAWDOWN_FRACTION pattern, applied to bankroll_cap_usd rather than a
# paper bankroll -- at the $100 default this is exactly -$15.
MAKING_LIVE_KILL_DRAWDOWN_FRACTION = _env_float("MAKING_LIVE_KILL_DRAWDOWN_FRACTION", 0.15)

# task item 1a/1b: Phase 0's MAKING_MAX_INVENTORY_PER_MARKET_FRACTION (0.02)
# and the §4.3 cluster fractions above are sized for a bankroll many times
# $500 -- at $100 they reject every market's minimum order size before it's
# ever quoted (see the task's own arithmetic: a 20-share pair costs $20,
# breaching a 15%-of-$100 entity cap on its own). These live-specific
# overrides are sized instead for the live bankroll actually being risked:
# MAKING_LIVE_MAX_INVENTORY_PER_MARKET_FRACTION=0.25 means a single market
# can be a quarter of a $100 book (~4-5 markets reachable) -- no smaller
# fraction clears the venue's real min_size distribution at this bankroll.
# The cluster fractions are each set above 0.25 so a single min-size quote
# is never blocked by a cluster cap alone (a "floor" of one quote always
# being permitted, per the task), with headroom for 2-3 markets sharing a
# cluster. These are judgment calls, not derived constants -- override via
# env if real fills suggest otherwise.
MAKING_LIVE_MAX_INVENTORY_PER_MARKET_FRACTION = _env_float("MAKING_LIVE_MAX_INVENTORY_PER_MARKET_FRACTION", 0.25)
MAKING_LIVE_MAX_ENTITY_CLUSTER_FRACTION = _env_float("MAKING_LIVE_MAX_ENTITY_CLUSTER_FRACTION", 0.35)
MAKING_LIVE_MAX_THEME_CLUSTER_FRACTION = _env_float("MAKING_LIVE_MAX_THEME_CLUSTER_FRACTION", 0.50)
MAKING_LIVE_MAX_DATE_BUCKET_FRACTION = _env_float("MAKING_LIVE_MAX_DATE_BUCKET_FRACTION", 0.40)
MAKING_LIVE_MAX_RESOLUTION_SOURCE_FRACTION = _env_float("MAKING_LIVE_MAX_RESOLUTION_SOURCE_FRACTION", 0.40)

# task item 5 "min notional": 20 shares at $0.03 is $0.60, under Polymarket's
# ~$1 minimum order notional. A per-leg guard rather than raising
# MAKING_MIN_PRICE globally, since that constant also gates Phase 0's paper
# universe and shouldn't change for a live-only concern.
MAKING_LIVE_MIN_NOTIONAL_USD = _env_float("MAKING_LIVE_MIN_NOTIONAL_USD", 1.0)

# task item 4: bounds how late a markout horizon can be stamped. Outside
# [horizon, horizon + slack] minutes old, the fill is left unscored rather
# than stamping a stale reading -- see execution.py's compute_markouts.
MAKING_LIVE_MARKOUT_WINDOW_SLACK_MINUTES = _env_float("MAKING_LIVE_MARKOUT_WINDOW_SLACK_MINUTES", 2.0)

# --- Telegram notifications (optional) -------------------------------------------------
# Forwards `poly03 paper run` console output to a Telegram chat. Unset by
# default -- notifications are best-effort and never required for the paper
# loop to run.

TELEGRAM_BOT_TOKEN = _env("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = _env("TELEGRAM_CHAT_ID")


def get_credentials() -> Credentials:
    return Credentials()


def get_endpoints() -> Endpoints:
    return Endpoints()
