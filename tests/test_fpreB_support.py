"""Shared fixtures of the F-pre-B tests (no tests here): a scripted alpha that signals at given 1m close times, and a
backtest runner that trades only it."""
from __future__ import annotations

from heartless.config import Settings
from heartless.core.models import EntryStyle, Side, Signal
from heartless.learning.backtester import Backtester
from heartless.strategy.base import Alpha
from heartless.strategy.ensemble import Ensemble
from heartless.strategy.params import StrategyParams
from synth import synth_symbols

ALPHA = "momentum_burst"  # a registered name (its parameter slot and regime affinity 1.0 everywhere) the script borrows
_ENSEMBLE_INIT = Ensemble.__init__  # the real one: several backtests in one test must not chain their patches


class Scripted(Alpha):
    """Signals at the 1m close times in `times` (its decision timeframe `tf` must close there): stop `stop_atr` ATR
    behind the close (ATR = 0.4% of the price), a time exit after `hold` bars; LIMIT style rests `limit_off` ATR behind
    the close for `ttl` bars of `tf`; no loss cool-down; only on `symbols` when given."""
    name = ALPHA

    def __init__(self, times, side: Side = Side.LONG, style: EntryStyle = EntryStyle.MARKET, limit_off: float = 0.0,
                 ttl: int | None = None, hold: int = 30, stop_atr: float = 1.5, tf: str = "1m",
                 symbols: tuple[str, ...] | None = None):
        self.times = {int(t) for t in times}
        self.symbols = symbols
        self.side, self.style, self.limit_off, self.ttl = side, style, limit_off, ttl
        self.hold, self.stop_atr, self.timeframe = hold, stop_atr, tf
        self.seen: list[int] = []

    def evaluate(self, view, ctx, p):
        t = int(view.cursor_time)
        if t not in self.times or (self.symbols is not None and ctx.symbol not in self.symbols):
            return None
        self.seen.append(t)
        px = float(view.price)
        atr = 0.004 * px
        sd = self.side.sign
        lim = px - sd * self.limit_off * atr if self.style is EntryStyle.LIMIT else None
        return Signal(ALPHA, ctx.symbol, self.side, 0.9, "scripted", px - sd * self.stop_atr * atr, None, None,
                      self.style, lim, self.hold, 0.0, atr, self.timeframe, {"ref_price": px},
                      entry_ttl_bars=self.ttl, time_exit=True, cooldown_bars=0)


def backtest(monkeypatch, tmp_path, alpha: Alpha, candles: dict, shock="auto", delisted=None, **settings):
    """Run `alpha` alone (only_alpha) over `candles` with the lab's research settings (no account loss halts), the book
    guard off unless asked for, `settings` on top, and no databases unless named."""
    def patched(self, params, bandit, alphas=None, only_alpha=None):
        _ENSEMBLE_INIT(self, params, bandit, [alpha], only_alpha)

    monkeypatch.setattr(Ensemble, "__init__", patched)
    from heartless.learning.lab import RESEARCH_SETTINGS

    kw = {**RESEARCH_SETTINGS, "BOOK_GUARD": False, **settings}
    s = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path / "nodata"), **kw)
    bt = Backtester(s, synth_symbols(list(candles)), candles, shock=shock, delisted=delisted)
    params = StrategyParams.default()
    params.enabled[ALPHA] = True
    return bt.run(params, only_alpha=ALPHA), bt
