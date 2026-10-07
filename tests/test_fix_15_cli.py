"""Regression tests for the CLI fixes: `backtest` must not nest asyncio.run(), `--download` must not persist the
still-forming 1m kline, and `run` must exit non-zero after a fatal error in app.run()."""
import argparse
import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from heartless import cli
from heartless.config import Settings
from heartless.core.models import Candle
from heartless.core.store import Store
from heartless.util.timeutil import MS_MINUTE, now_ms
from synth import synth_candles, synth_symbols

SYM = "BTCUSDT"


def _settings() -> Settings:
    tmp = Path(tempfile.mkdtemp())
    s = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp), WEB_ENABLED=False, TELEGRAM_BOT_TOKEN="")
    s.data_dir.mkdir(exist_ok=True)
    return s


class FakeRest:
    """Minimal stand-in for BinanceRest: returns a fixed candle list for klines_range (like Binance, the last row
    may be the current, still-forming kline) and no funding history."""

    def __init__(self, rows):
        self.rows = rows
        self.closed = False

    async def klines_range(self, symbol, start, end, interval="1m"):
        return [c for c in self.rows if start <= c.open_time <= end]

    async def funding_rate_history(self, symbol, start=None, end=None, limit=1000):
        return []

    async def close(self):
        self.closed = True


def _rest_rows(n_closed: int, now: int) -> list[Candle]:
    """n_closed completed minutes ending right before `now`, plus the forming kline that contains `now`."""
    cur_open = now // MS_MINUTE * MS_MINUTE
    rows = []
    for i in range(n_closed, 0, -1):
        t = cur_open - i * MS_MINUTE
        rows.append(Candle(t, 100.0, 101.0, 99.0, 100.5, 10.0, 1000.0, 10, 5.0, t + MS_MINUTE - 1))
    # the forming bar: REST marks it closed=True, so only its close_time reveals it is incomplete
    rows.append(Candle(cur_open, 100.5, 100.6, 100.4, 100.55, 0.3, 30.0, 1, 0.1, cur_open + MS_MINUTE - 1, closed=True))
    return rows


# --- findings 1 & 3: --download must not persist the still-forming kline -------------------------------------
def test_ensure_candles_drops_forming_kline():
    s = _settings()
    store = Store(s.db_path)
    now = now_ms()
    rest = FakeRest(_rest_rows(600, now))
    asyncio.run(cli._ensure_candles(rest, store, [SYM], days=1))
    lo, hi, n = store.candle_range(SYM)
    cur_open = now // MS_MINUTE * MS_MINUTE
    assert n == 600
    assert hi == cur_open - MS_MINUTE, "last stored bar must be the previous (completed) minute"
    assert all(c.close_time <= now for c in store.load_candles(SYM))


def test_ensure_candles_with_only_forming_kline_stores_nothing():
    s = _settings()
    store = Store(s.db_path)
    now = now_ms()
    rest = FakeRest(_rest_rows(0, now))
    asyncio.run(cli._ensure_candles(rest, store, [SYM], days=1))
    assert store.candle_range(SYM) == (None, None, 0)


# --- findings 2 & 5: backtest subcommand must run inside the already-running loop ---------------------------
def test_backtest_subcommand_runs_under_asyncio_run(monkeypatch, capsys):
    s = _settings()
    store = Store(s.db_path)
    now = now_ms()
    n = 3000
    start = (now - (n + 5) * MS_MINUTE) // MS_MINUTE * MS_MINUTE
    ca = synth_candles(n, seed=1, start_ms=start)
    store.save_candles(SYM, [ca.candle_at(i) for i in range(ca.n)])
    rest = FakeRest([])

    async def fake_load(settings, store_, download, symbols_arg):
        return rest, synth_symbols([SYM]), [SYM]

    monkeypatch.setattr(cli, "_load_symbols_and_universe", fake_load)
    args = argparse.Namespace(days=3, alpha=None, symbols=None, download=False, json=True)
    asyncio.run(cli._backtest(s, args))  # used to raise "asyncio.run() cannot be called from a running event loop"
    out = json.loads(capsys.readouterr().out)
    assert set(out) == {"stats", "by_alpha", "skipped"}
    assert out["stats"]["n"] >= 0
    assert rest.closed


def test_backtest_subcommand_human_output(monkeypatch, capsys):
    s = _settings()
    store = Store(s.db_path)
    now = now_ms()
    n = 3000
    start = (now - (n + 5) * MS_MINUTE) // MS_MINUTE * MS_MINUTE
    ca = synth_candles(n, seed=2, start_ms=start)
    store.save_candles(SYM, [ca.candle_at(i) for i in range(ca.n)])

    async def fake_load(settings, store_, download, symbols_arg):
        return FakeRest([]), synth_symbols([SYM]), [SYM]

    monkeypatch.setattr(cli, "_load_symbols_and_universe", fake_load)
    args = argparse.Namespace(days=3, alpha=None, symbols=None, download=False, json=False)
    asyncio.run(cli._backtest(s, args))
    out = capsys.readouterr().out
    assert "Backtest 3d on 1 symbols" in out
    assert "trades" in out


# --- finding 4: `heartless run` must exit non-zero after a fatal error in app.run() --------------------------
class _FailingApp:
    stopped = False

    def __init__(self, settings):
        pass

    async def run(self):
        raise RuntimeError("exchangeInfo unreachable")

    async def stop(self):
        _FailingApp.stopped = True


class _CleanApp:
    def __init__(self, settings):
        pass

    async def run(self):
        return None

    async def stop(self):
        return None


def test_run_exits_1_after_fatal_error(monkeypatch):
    import heartless.app as app_mod

    monkeypatch.setattr(app_mod, "Heartless", _FailingApp)
    _FailingApp.stopped = False
    with pytest.raises(SystemExit) as ei:
        cli._run(_settings())
    assert ei.value.code == 1
    assert _FailingApp.stopped, "cleanup must still run before exiting"


def test_run_exits_cleanly_when_app_finishes(monkeypatch):
    import heartless.app as app_mod

    monkeypatch.setattr(app_mod, "Heartless", _CleanApp)
    cli._run(_settings())  # must not raise SystemExit
