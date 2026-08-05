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

MIN_OPEN_INTEREST_USD = _env_float("MIN_OPEN_INTEREST_USD", 10_000.0)
MIN_24H_VOLUME_USD = _env_float("MIN_24H_VOLUME_USD", 2_000.0)
MAX_SPREAD_BOOK_A = _env_float("MAX_SPREAD_BOOK_A", 0.02)
BOOK_A_HORIZON_CAP_DAYS = int(_env_float("BOOK_A_HORIZON_CAP_DAYS", 270))

AMBIGUOUS_RESOLUTION_KEYWORDS = (
    "widely reported",
    "generally considered",
    "generally regarded",
    "notable",
    "notably",
    "significant coverage",
    "consensus of media",
    "consensus of credible",
    "in the opinion of",
    "at the discretion of",
    "credible report",
    "widely believed",
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
PAPER_STARTING_BANKROLL = _env_float("PAPER_STARTING_BANKROLL", 100_000.0)

# doc: "breadth ... 100-300 concurrent positions"
MAX_CONCURRENT_POSITIONS = int(_env_float("MAX_CONCURRENT_POSITIONS", 300))
# throttle: don't dump the whole candidate list into positions in one tick
PAPER_MAX_NEW_POSITIONS_PER_TICK = int(_env_float("PAPER_MAX_NEW_POSITIONS_PER_TICK", 15))
PAPER_TARGET_SCAN_MARKETS = int(_env_float("PAPER_TARGET_SCAN_MARKETS", 300))

# §8 Phase 1 gate: >=50 simulated resolutions, calibration in tolerance, no Tier 1 misses
PAPER_GATE_MIN_RESOLUTIONS = int(_env_float("PAPER_GATE_MIN_RESOLUTIONS", 50))
PAPER_GATE_CALIBRATION_TOLERANCE_PP = _env_float("PAPER_GATE_CALIBRATION_TOLERANCE_PP", 0.05)

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
