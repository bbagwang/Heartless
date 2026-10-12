"""Runtime configuration.

Only the Binance API keys (and a Telegram bot token if you want Telegram) are required.
Every other knob below has a default chosen so the bot can run unattended.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
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
    always_include: str = Field(default="BTCUSDT,ETHUSDT,SOLUSDT", alias="ALWAYS_INCLUDE")

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
    # 360 (was 240, C2-B): one 120-day cycle takes ~2.2 h of a core (docs/SELF_IMPROVEMENT.md 3.1), 54% of a 4-hour
    # interval; 6 hours keep the research worker busy at most ~36% of the time and leave room for the daily discovery
    research_interval_minutes: int = Field(default=360, alias="RESEARCH_INTERVAL_MINUTES")
    # 120 days (was 30): a 30-day window holds too few independent days for 1h alphas (docs/SELF_IMPROVEMENT.md)
    research_lookback_days: int = Field(default=120, alias="RESEARCH_LOOKBACK_DAYS")
    research_folds: int = Field(default=3, alias="RESEARCH_FOLDS")  # walk-forward confirmation windows
    # history kept in the database for research / discovery (live feature views only use HISTORY_DAYS)
    # (candles, funding, metrics and equity rows; 180 days (was 90) keep the 120-day research window plus margin)
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


def load_settings(**overrides) -> Settings:
    s = Settings(**overrides)
    s.data_dir.mkdir(parents=True, exist_ok=True)
    from heartless.core.secrets import apply_to_settings, load_secrets

    apply_to_settings(s, load_secrets(s.data_dir))
    os.environ.setdefault("TZ", "UTC")
    return s
