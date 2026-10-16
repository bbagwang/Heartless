"""Runtime configuration.

Only the Binance API keys (and a Telegram bot token if you want Telegram) are required.
Every other knob below has a default chosen so the bot can run unattended.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- credentials -------------------------------------------------------------------------
    binance_api_key: str = Field(default="", alias="BINANCE_API_KEY")
    binance_api_secret: str = Field(default="", alias="BINANCE_API_SECRET")
    binance_testnet: bool = Field(default=False, alias="BINANCE_TESTNET")
    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_owner_id: int | None = Field(default=None, alias="TELEGRAM_OWNER_ID")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")

    # --- mode / infrastructure -------------------------------------------------------------
    mode: Literal["paper", "live"] = Field(default="paper", alias="HEARTLESS_MODE")
    data_dir: Path = Field(default=Path("data"), alias="HEARTLESS_DATA_DIR")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    timezone: str = Field(default="Asia/Seoul", alias="TIMEZONE")
    # Loopback by default: the dashboard exposes live-money controls over plain HTTP. Set WEB_HOST=0.0.0.0
    # only behind a TLS reverse proxy / VPN (docker-compose does this and pins the host side to 127.0.0.1).
    web_host: str = Field(default="127.0.0.1", alias="WEB_HOST")
    web_port: int = Field(default=8080, alias="WEB_PORT")
    web_enabled: bool = Field(default=True, alias="WEB_ENABLED")

    # --- universe --------------------------------------------------------------------------
    universe_size: int = Field(default=16, alias="UNIVERSE_SIZE")
    universe_min_quote_volume: float = Field(default=80_000_000.0, alias="UNIVERSE_MIN_QUOTE_VOLUME")
    universe_refresh_minutes: int = Field(default=60, alias="UNIVERSE_REFRESH_MINUTES")
    history_days: int = Field(default=21, alias="HISTORY_DAYS")
    # bulk history from the public data.binance.vision archive (no API weight); REST only fills the recent tail
    archive_backfill: bool = Field(default=True, alias="ARCHIVE_BACKFILL")
    # Data subscription, not a trading guarantee (F0, docs/UNIVERSE.md): symbols whose market data the bot always
    # subscribes, BTC as the regime / market-context reference every alpha reads. They trade only when they pass the
    # same eligibility rules as every other symbol (heartless/data/universe.py); was BTCUSDT,ETHUSDT,SOLUSDT.
    always_include: str = Field(default="BTCUSDT", alias="ALWAYS_INCLUDE")

    # --- risk (defaults are deliberately conservative; the optimizer never touches these) -----
    risk_per_trade_pct: float = Field(default=0.5, alias="RISK_PER_TRADE_PCT")  # % of equity lost at stop
    max_risk_per_trade_pct: float = Field(default=1.25, alias="MAX_RISK_PER_TRADE_PCT")
    max_positions: int = Field(default=5, alias="MAX_POSITIONS")
    max_same_direction: int = Field(default=3, alias="MAX_SAME_DIRECTION")
    max_position_leverage: float = Field(default=5.0, alias="MAX_POSITION_LEVERAGE")  # notional / equity
    max_gross_leverage: float = Field(default=12.0, alias="MAX_GROSS_LEVERAGE")
    exchange_leverage: int = Field(default=10, alias="EXCHANGE_LEVERAGE")  # margin leverage set per symbol
    daily_loss_limit_pct: float = Field(default=3.0, alias="DAILY_LOSS_LIMIT_PCT")
    weekly_loss_limit_pct: float = Field(default=7.0, alias="WEEKLY_LOSS_LIMIT_PCT")
    max_drawdown_halt_pct: float = Field(default=15.0, alias="MAX_DRAWDOWN_HALT_PCT")
    paper_initial_balance: float = Field(default=10_000.0, alias="PAPER_INITIAL_BALANCE")
    taker_fee: float = Field(default=0.0005, alias="TAKER_FEE")
    maker_fee: float = Field(default=0.0002, alias="MAKER_FEE")
    backtest_slippage_bps: float = Field(default=1.5, alias="BACKTEST_SLIPPAGE_BPS")  # simulated market-order slippage

    # --- learning ----------------------------------------------------------------------------
    # 120 (E2; was 360 in C2-B, 240 before): a cycle now tunes the champion's enabled alphas, the disabled ones not marked
    # failed and RESEARCH_FAILED_PER_CYCLE failed ones (below). With the shipped defaults (2 failed alphas) the heaviest
    # cycle - the 92-day warm-up pair ts_trend_tsmom + vol_breakout_dbox - took 1,865 s of a core (docs/SELF_IMPROVEMENT.md
    # 3.1): 26% of 2 hours. ResearchManager.due() also waits twice the last cycle's length, so a longer cycle (more
    # enabled alphas) never pushes the research duty cycle past one half
    research_interval_minutes: int = Field(default=120, alias="RESEARCH_INTERVAL_MINUTES")
    # 120 days (was 30): a 30-day window holds too few independent days for 1h alphas (docs/SELF_IMPROVEMENT.md)
    research_lookback_days: int = Field(default=120, alias="RESEARCH_LOOKBACK_DAYS")
    research_folds: int = Field(default=3, alias="RESEARCH_FOLDS")  # walk-forward confirmation windows
    # history kept in the database for research / discovery (live feature views only use HISTORY_DAYS)
    # (candles, funding, metrics and equity rows; 180 days (was 90) keep the 120-day research window plus margin).
    # A floor since E2: raised to the research window + the longest research warm-up + slack when that is longer
    # (Settings._retention_covers_research below; default 214 days)
    data_retention_days: int = Field(default=180, alias="DATA_RETENTION_DAYS")
    discovery_interval_hours: int = Field(default=24, alias="DISCOVERY_INTERVAL_HOURS")
    discovery_min_days: int = Field(default=45, alias="DISCOVERY_MIN_DAYS")
    # meta-labeling: learn per alpha which contexts its signals win in and veto/scale accordingly (off until validated)
    meta_label: bool = Field(default=False, alias="META_LABEL")
    meta_min_samples: int = Field(default=120, alias="META_MIN_SAMPLES")
    research_candidates: int = Field(default=10, alias="RESEARCH_CANDIDATES")
    challengers: int = Field(default=2, alias="CHALLENGERS")
    promotion_min_trades: int = Field(default=25, alias="PROMOTION_MIN_TRADES")
    graduation_min_trades: int = Field(default=60, alias="GRADUATION_MIN_TRADES")
    graduation_min_profit_factor: float = Field(default=1.25, alias="GRADUATION_MIN_PF")
    graduation_max_drawdown_pct: float = Field(default=8.0, alias="GRADUATION_MAX_DD_PCT")
    daily_report_hour: int = Field(default=9, alias="DAILY_REPORT_HOUR")  # local hour (timezone above)

    # --- self-improvement hardening (ROADMAP P4, docs/SELF_IMPROVEMENT.md) ----------------------
    # promotion / retirement are judged on day-clustered statistics: a challenger needs PROMOTION_MIN_TRADES closed
    # trades on at least PROMOTION_MIN_DAYS distinct UTC exit days (the absolute-path bar constants live in
    # heartless/learning/research.py next to the rule they belong to)
    promotion_min_days: int = Field(default=14, alias="PROMOTION_MIN_DAYS")
    # per-alpha kill switch on live (paper or real) closed trades: one-sided lower CUSUM of (R - expected) / sd with
    # slack k crossing -h (k and h in standard deviations of the alpha's R, sd clamped to [1, 2]R), or the R sum of the
    # alpha's last `window` trades below window_r (a backstop for absolute losses). Disabling never needs a challenger.
    kill_switch: bool = Field(default=True, alias="KILL_SWITCH")
    kill_switch_k: float = Field(default=0.25, alias="KILL_SWITCH_K")
    kill_switch_h: float = Field(default=8.0, alias="KILL_SWITCH_H")
    kill_switch_window: int = Field(default=30, alias="KILL_SWITCH_WINDOW")
    kill_switch_window_r: float = Field(default=-20.0, alias="KILL_SWITCH_WINDOW_R")
    # execution-quality monitor: rolling window of fills per engine, and the taker-fill count from which the reports
    # suggest raising BACKTEST_SLIPPAGE_BPS (when realized slippage exceeds twice the model; never changed automatically)
    exec_quality_window: int = Field(default=200, alias="EXEC_QUALITY_WINDOW")
    exec_quality_min_fills: int = Field(default=50, alias="EXEC_QUALITY_MIN_FILLS")

    # --- market context (P2a, heartless/data/context.py) ---------------------------------------
    # hourly cross-sectional features over a fixed universe (alphas read them as ctx.market); symbols without data at
    # an hour are simply left out of that hour's cross-section
    context_universe: str = Field(default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,DOGEUSDT,ADAUSDT,AVAXUSDT,LINKUSDT,LTCUSDT,"
                                          "BCHUSDT,BNBUSDT,SUIUSDT,DOTUSDT,ATOMUSDT,NEARUSDT,FILUSDT,ETCUSDT,TRXUSDT,"
                                          "UNIUSDT,AAVEUSDT", alias="CONTEXT_UNIVERSE")
    # SQLite file holding the market_context table (empty: <data_dir>/context.db, see context_db_path)
    context_db: str = Field(default="", alias="CONTEXT_DB")
    # lab / discover build missing context years on demand (otherwise only stored rows are used)
    context_auto_build: bool = Field(default=True, alias="CONTEXT_AUTO_BUILD")

    # --- C2-B: self-improvement and operations (docs/SELF_IMPROVEMENT.md, docs/REPLAY.md) ------------------------
    # decision grids the daily discovery mines in ONE run, so its multiple-testing N counts the rules of all of them
    discovery_grids: str = Field(default="15m,1h", alias="DISCOVERY_GRIDS")
    # a candidate whose change (params diff) equals that of a challenger judged and retired within this many days is
    # not installed again: every fresh trial of the same change is another look at the same noise
    challenger_retry_days: int = Field(default=90, alias="CHALLENGER_RETRY_DAYS")

    # --- E1: entry sizing and portfolio construction (ROADMAP P3, docs/PORTFOLIO.md) ------------------------------
    # A market entry is decided on a bar close and fills after it (live: a second later; backtest: the next open), plus
    # slippage. It is sized for a fill BACKTEST_SLIPPAGE_BPS + this many bps + one tick beyond the decision touch, so
    # the risk measured at the fill stays within the per-trade budget (the next-open gap is a tick or less on ~99.9% of
    # 2026 1m bars; the rest covers the paper account's size impact of up to ~1.5 bp).
    market_entry_buffer_bps: float = Field(default=1.5, alias="MARKET_ENTRY_BUFFER_BPS")
    # Signal batching + risk budgets (heartless/execution/allocator.py), used by the live engines and the joint
    # backtester alike. Off: every symbol's entry is admitted in the order its bar is processed (FIFO) under the count
    # limits. On: the entry candidates of one bar close are ranked by expected value, then admitted while the open
    # risk stays within MAX_TOTAL_RISK_PCT and the BTC-beta weighted risk of each side within
    # MAX_SAME_DIRECTION_RISK_PCT (MAX_POSITIONS / MAX_SAME_DIRECTION stay as caps). Off by default: the P3 acceptance
    # rule needs an enabled alpha whose joint backtest improves with it (docs/PORTFOLIO.md).
    portfolio_allocator: bool = Field(default=False, alias="PORTFOLIO_ALLOCATOR")
    max_total_risk_pct: float = Field(default=2.5, alias="MAX_TOTAL_RISK_PCT")  # 5 positions x 0.5%
    max_same_direction_risk_pct: float = Field(default=1.5, alias="MAX_SAME_DIRECTION_RISK_PCT")  # 3 x 0.5% at beta 1
    beta_lookback_days: int = Field(default=30, alias="BETA_LOOKBACK_DAYS")  # BTC beta from this many days of 1h returns

    # --- E2: research-cycle cost and the history it needs (docs/SELF_IMPROVEMENT.md 3.1, 9) ------------------------
    # A research cycle tunes the champion's enabled alphas, the disabled alphas the evidence registry does not mark
    # failed, and RESEARCH_FAILED_PER_CYCLE of the failed ones in a fixed rotation (ResearchManager.research_selection;
    # each failed alpha is revisited every ceil(failed / k) cycles). Tuning all 15 parametrised alphas every cycle cost
    # about 3.5 hours of a core, mostly on alphas that already failed out of sample.
    research_failed_per_cycle: int = Field(default=2, alias="RESEARCH_FAILED_PER_CYCLE")

    @model_validator(mode="after")
    def _retention_covers_research(self):
        """The research worker loads the warm-up of the longest-lookback alpha it tunes before its window (E2), so the
        stored history must reach RESEARCH_LOOKBACK_DAYS + that warm-up + RETENTION_SLACK_DAYS. DATA_RETENTION_DAYS is
        raised to research_retention_days() when it is shorter (default 120 + 92 + 2 = 214 > 180 days); the backfill,
        the retention prune and the live context store all follow data_retention_days."""
        need = research_retention_days(self.research_lookback_days)
        if self.data_retention_days < need:
            self.data_retention_days = need
        return self

    # --- F7: speed and optimization (docs/PERFORMANCE.md) ----------------------------------------------------------
    # Live feature frames are appended bar by bar (heartless/data/incremental.py) instead of recomputed every minute:
    # each frame equals, bit for bit, the batch computation over the bars since its anchor, and is re-anchored (checked
    # against the batch, then rebuilt from the 1m tail) every few hours. Off: the per-bar batch rebuild as before.
    incremental_features: bool = Field(default=True, alias="INCREMENTAL_FEATURES")
    # Local order book per traded symbol from the depth streams (heartless/exchange/orderbook.py), read by alphas as
    # ctx.book (spread, imbalance, depth within ORDERBOOK_DEPTH_BPS, staleness). Off until an F-wave alpha uses it;
    # a replay then serves the conservative 1m-bar proxy instead. "partial": depth20 snapshots (no REST); "diff":
    # diff-depth kept in sequence on REST snapshots. Speed: 100, 250 or 500 ms.
    orderbook_stream: bool = Field(default=False, alias="ORDERBOOK_STREAM")
    orderbook_mode: Literal["partial", "diff"] = Field(default="partial", alias="ORDERBOOK_MODE")
    orderbook_speed_ms: int = Field(default=100, alias="ORDERBOOK_SPEED_MS")
    orderbook_depth_bps: float = Field(default=10.0, alias="ORDERBOOK_DEPTH_BPS")
    # |local clock - exchange clock| (REST time syncs, stream event times) above this is logged as a warning
    clock_drift_warn_ms: float = Field(default=1000.0, alias="CLOCK_DRIFT_WARN_MS")

    # --- F0: coin-agnostic universes (docs/UNIVERSE.md, heartless/data/universe.py) -------------------------------
    # live eligibility besides UNIVERSE_MIN_QUOTE_VOLUME: listing age, tick size against the price, quoted spread
    universe_min_age_days: int = Field(default=30, alias="UNIVERSE_MIN_AGE_DAYS")
    universe_max_tick_bps: float = Field(default=10.0, alias="UNIVERSE_MAX_TICK_BPS")
    universe_max_spread_bps: float = Field(default=5.0, alias="UNIVERSE_MAX_SPREAD_BPS")
    # SQLite file of the exchange-wide symbol inventory and the 1h futures / spot datasets (empty: <data_dir>/universe.db)
    universe_db: str = Field(default="", alias="UNIVERSE_DB")
    # market-context cross-section: 0 = every CONTEXT_UNIVERSE symbol with data that hour (the fixed list, default);
    # N > 0 = only the N of them with the largest trailing 24h quote volume at that hour
    context_top_n: int = Field(default=0, alias="CONTEXT_TOP_N")

    # --- F-pre-B: shock layer, execution modes, book guard (docs/SHOCK.md, docs/EXECUTION.md, ROADMAP 7.7.2 / 7.8) --
    # The shock layer (heartless/strategy/shock.py): S0 (BTC 5-minute move >= 3% -> 15 minutes without new entries) and
    # S1-S3 (hourly BTC price shock, volatility jump, correlation spike -> 2 hours without new entries, resting entry
    # limits cancelled, shock_exempt alphas excluded). On by default from F-pre-B on (ROADMAP 7.8: responses 1 and 3
    # only make entries rarer and switching an alpha off easier, so they need no gate); the definitions are constants.
    shock_layer: bool = Field(default=True, alias="SHOCK_LAYER")
    # 1m candle database whose BTCUSDT bars feed S0 in a backtest that does not load BTC itself (a lab job of another
    # symbol); empty: <data_dir>/heartless.db. S1-S3 read UNIVERSE_DB (1h, point in time).
    shock_candle_db: str = Field(default="", alias="SHOCK_CANDLE_DB")
    # a prebuilt shock schedule (JSON, shock.save_schedule) that backtests read when it covers their window; one that
    # is missing or too short is built and written. Empty: each process builds (and caches) its own.
    shock_schedule: str = Field(default="", alias="SHOCK_SCHEDULE")
    # Execution mode per alpha (ROADMAP 7.7.2), an engine setting and not an alpha parameter: "taker" (the default:
    # orders exactly as before), "maker_m1" (a market signal becomes a post-only limit at the decision bar's close,
    # cancelled after one decision bar, at most 60 minutes) or "maker_m2" (a retest limit 0.25 signal ATR behind the
    # close, cancelled after three decision bars, at most 240 minutes). One mode for every alpha, or
    # "alpha:mode,alpha:mode" (alphas not named stay taker). The trial ledger hashes it as `exec`.
    exec_mode: str = Field(default="taker", alias="EXEC_MODE")
    # conservative maker fill model (PaperAccount: backtests and paper alike): a resting maker limit fills only when a
    # 1m bar trades through it by this fraction of the synthetic spread (0.5 nominal, 1.0 stress), never in the bar it
    # was placed in, and not in a bar whose quote volume is below MAKER_MIN_VOLUME_MULT x the order notional
    maker_through_spread: float = Field(default=0.5, alias="MAKER_THROUGH_SPREAD")
    maker_min_volume_mult: float = Field(default=20.0, alias="MAKER_MIN_VOLUME_MULT")
    # Book guard (ROADMAP 7.11): skip an entry while the symbol's spread is >= BOOK_GUARD_SPREAD_MULT x its own
    # prior-24h median and >= BOOK_GUARD_MIN_SPREAD_BPS, or its depth within +-ORDERBOOK_DEPTH_BPS is below
    # BOOK_GUARD_DEPTH_FRAC of its own prior-24h median. The live stream book when there is one, else the conservative
    # 1m-bar proxy (backtests, replays, live without ORDERBOOK_STREAM): one rule everywhere.
    book_guard: bool = Field(default=True, alias="BOOK_GUARD")
    book_guard_spread_mult: float = Field(default=3.0, alias="BOOK_GUARD_SPREAD_MULT")
    book_guard_min_spread_bps: float = Field(default=5.0, alias="BOOK_GUARD_MIN_SPREAD_BPS")
    book_guard_depth_frac: float = Field(default=0.25, alias="BOOK_GUARD_DEPTH_FRAC")

    @field_validator("exec_mode")
    @classmethod
    def _known_exec_mode(cls, v: str) -> str:
        parse_exec_modes(v)  # raises on an unknown mode: a typo must not silently trade taker
        return v

    @field_validator("telegram_owner_id", mode="before")
    @classmethod
    def _blank_owner_id_is_none(cls, v):
        # `.env.example` ships `TELEGRAM_OWNER_ID=` ("leave empty to use the pairing flow"); pydantic-settings
        # hands that through as "" which `int | None` rejects, crashing every CLI command at startup.
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @property
    def live_capable(self) -> bool:
        return bool(self.binance_api_key and self.binance_api_secret)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "heartless.db"

    @property
    def always_include_list(self) -> list[str]:
        return [s.strip().upper() for s in self.always_include.split(",") if s.strip()]

    @property
    def context_universe_list(self) -> list[str]:
        return sorted({s.strip().upper() for s in self.context_universe.split(",") if s.strip()})

    # --- C2-A: market-context store -------------------------------------------------------------------------------
    # The context is derived data with its own metadata (source candle database, universe, built hours) and build
    # lock, so it always lives in a file of its own: CONTEXT_DB, by default <data_dir>/context.db, never the candle
    # database itself (heartless/data/context.py refuses that), which keeps read-only research databases untouched.
    @property
    def context_db_path(self) -> Path:
        return Path(self.context_db) if self.context_db else self.data_dir / "context.db"

    # F0: the universe database is research data of its own (inventory, 1h blocks), never the candle database
    @property
    def universe_db_path(self) -> Path:
        return Path(self.universe_db) if self.universe_db else self.data_dir / "universe.db"

    # F-pre-B
    @property
    def shock_candle_db_path(self) -> Path:
        return Path(self.shock_candle_db) if self.shock_candle_db else self.db_path

    @property
    def exec_modes(self) -> tuple[str, dict[str, str]]:
        """EXEC_MODE parsed: (mode of the alphas not named, {alpha: mode})."""
        return parse_exec_modes(self.exec_mode)


# ROADMAP 7.7.2: the execution modes an engine trades an alpha with (taker = the orders as before F-pre-B)
EXEC_MODES = ("taker", "maker_m1", "maker_m2")


def parse_exec_modes(text: str | None) -> tuple[str, dict[str, str]]:
    """EXEC_MODE -> (default mode, {alpha: mode}). "maker_m1" sets every alpha; "a:maker_m1,b:maker_m2" names alphas
    (the others stay taker, or take a bare mode given alongside). ValueError for an unknown mode."""
    default, per = "taker", {}
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        name, _, mode = part.rpartition(":")
        mode = mode.strip()
        if mode not in EXEC_MODES:
            raise ValueError(f"unknown execution mode {mode!r} in EXEC_MODE (one of {', '.join(EXEC_MODES)})")
        if name.strip():
            per[name.strip()] = mode
        else:
            default = mode
    return default, per


# E2: days of stored history beyond the research window and its warm-up (the research window ends at the newest stored
# candle, the backfill starts from the wall clock)
RETENTION_SLACK_DAYS = 2


def research_retention_days(lookback_days: int) -> int:
    """History a research cycle may read: its window (`lookback_days`) plus the longest warm-up of any alpha it may tune
    (learning/optimizer.longest_research_warmup_ms: vol_breakout_dbox's 90-day lookback + the lab's 2-day margin) plus
    RETENTION_SLACK_DAYS."""
    import math

    from heartless.learning.optimizer import longest_research_warmup_ms  # lazy: it imports the strategy stack

    return int(lookback_days) + math.ceil(longest_research_warmup_ms() / 86_400_000) + RETENTION_SLACK_DAYS


def load_settings(**overrides) -> Settings:
    s = Settings(**overrides)
    s.data_dir.mkdir(parents=True, exist_ok=True)
    from heartless.core.secrets import apply_to_settings, load_secrets

    apply_to_settings(s, load_secrets(s.data_dir))
    os.environ.setdefault("TZ", "UTC")
    return s
