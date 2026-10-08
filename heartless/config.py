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
    research_interval_minutes: int = Field(default=240, alias="RESEARCH_INTERVAL_MINUTES")
    research_lookback_days: int = Field(default=30, alias="RESEARCH_LOOKBACK_DAYS")
    research_folds: int = Field(default=3, alias="RESEARCH_FOLDS")  # walk-forward confirmation windows
    # history kept in the database for research / discovery (live feature views only use HISTORY_DAYS)
    data_retention_days: int = Field(default=90, alias="DATA_RETENTION_DAYS")
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


def load_settings(**overrides) -> Settings:
    s = Settings(**overrides)
    s.data_dir.mkdir(parents=True, exist_ok=True)
    from heartless.core.secrets import apply_to_settings, load_secrets

    apply_to_settings(s, load_secrets(s.data_dir))
    os.environ.setdefault("TZ", "UTC")
    return s
