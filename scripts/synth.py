"""Generate synthetic but market-like 1m candles for tests and offline demos."""
from __future__ import annotations

import numpy as np

from heartless.core.models import Candle, SymbolInfo
from heartless.data.candles import CandleArrays


def synth_candles(n: int = 20000, seed: int = 0, start_ms: int = 1_700_006_400_000, price: float = 100.0,
                  vol: float = 0.0008) -> CandleArrays:
    rng = np.random.default_rng(seed)
    ca = CandleArrays("1m", capacity=n + 16)
    p = price
    regime_len = 0
    drift = 0.0
    v_scale = 1.0
    for i in range(n):
        if regime_len <= 0:  # switch between trend / range / volatile regimes
            regime_len = int(rng.integers(300, 1500))
            mode = rng.choice(["trend_up", "trend_dn", "range", "vol"], p=[0.25, 0.25, 0.35, 0.15])
            drift = {"trend_up": 0.00015, "trend_dn": -0.00015, "range": 0.0, "vol": 0.0}[mode]
            v_scale = {"trend_up": 1.0, "trend_dn": 1.0, "range": 0.7, "vol": 2.2}[mode]
        regime_len -= 1
        o = p
        ret = drift + rng.standard_t(4) * vol * v_scale * 0.6
        if drift == 0.0:  # mean reversion pull in ranges
            ret -= 0.02 * np.log(p / price) if price > 0 else 0
        p = max(o * (1 + ret), 0.01)
        wick = abs(rng.normal(0, vol * v_scale * 0.5))
        h = max(o, p) * (1 + wick)
        l = min(o, p) * (1 - abs(rng.normal(0, vol * v_scale * 0.5)))
        base_vol = 50 * v_scale * (1 + 3 * abs(ret) / (vol + 1e-12))
        vlm = abs(rng.normal(base_vol, base_vol * 0.4)) + 1
        tb = vlm * np.clip(0.5 + ret / (vol * 3) + rng.normal(0, 0.1), 0.05, 0.95)
        t = start_ms + i * 60_000
        ca.append(Candle(t, o, h, l, p, vlm, vlm * p, int(vlm), tb, t + 59_999))
    return ca


def synth_symbols(names: list[str]) -> dict[str, SymbolInfo]:
    return {n: SymbolInfo(n, n[:-4], "USDT", tick_size=0.01, step_size=0.001, min_qty=0.001, min_notional=5.0,
                          price_precision=2, quantity_precision=3) for n in names}
