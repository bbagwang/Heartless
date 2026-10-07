import random

from heartless.config import Settings
from heartless.core.models import Decision, EntryStyle, Regime, Side, Signal, SymbolInfo
from heartless.strategy.params import ALPHA_SPECS, StrategyParams
from heartless.strategy.risk import RiskManager


def _settings():
    return Settings(_env_file=None, RISK_PER_TRADE_PCT=0.5, MAX_POSITION_LEVERAGE=5, EXCHANGE_LEVERAGE=10)


def _decision(stop=98.0, entry=100.0):
    sig = Signal("a", "BTCUSDT", Side.LONG, 0.7, "r", stop, 104.0, 102.0, EntryStyle.MARKET, None, 0, 0.0, 1.0, "5m",
                 {"ref_price": entry})
    return Decision("BTCUSDT", Side.LONG, 0.7, 0.7, ["a"], sig, "r", Regime.RANGE, 1.0, 2.0)


def test_sizing_risks_the_configured_fraction_including_fees():
    rm = RiskManager(_settings())
    info = SymbolInfo("BTCUSDT", "BTC", "USDT", 0.1, 0.001, 0.001, 5, 1, 3)
    res = rm.size(_decision(), equity=10_000, info=info, entry_price=100.0)
    assert res.qty > 0
    # 0.5% of 10k = 50 USDT at risk including round-trip costs
    assert abs(res.risk_amount - 50) < 0.5
    assert res.notional <= 10_000 * 5


def test_sizing_caps_notional_by_leverage():
    rm = RiskManager(_settings())
    info = SymbolInfo("BTCUSDT", "BTC", "USDT", 0.1, 0.001, 0.001, 5, 1, 3)
    res = rm.size(_decision(stop=99.95), equity=1_000, info=info, entry_price=100.0)  # tiny stop -> huge qty
    assert res.notional <= 1_000 * 5 + 1e-6


def test_daily_loss_limit_halts_entries():
    s = _settings()
    rm = RiskManager(s)
    t0 = 1_700_000_000_000
    rm.observe_equity(10_000, t0)
    events = rm.observe_equity(10_000 * (1 - s.daily_loss_limit_pct / 100) - 1, t0 + 60_000)
    assert events and rm.state.halted_until > t0
    ok, why = rm.can_open(_decision(), [], t0 + 120_000, 9_600)
    assert not ok and "halted" in why


def test_params_roundtrip_and_perturbation_stays_in_bounds():
    p = StrategyParams.default()
    d = p.to_dict()
    q = StrategyParams.from_dict(d)
    assert q.alphas == p.alphas
    rng = random.Random(3)
    for alpha, specs in ALPHA_SPECS.items():
        for _ in range(50):
            vals = p.perturbed(alpha, rng, scale=1.0)
            for s in specs:
                v = vals[s.name]
                if s.choices:
                    assert v in s.choices
                else:
                    assert s.lo - 1e-9 <= v <= s.hi + 1e-9
    assert StrategyParams.distance("trend_pullback", p.alphas["trend_pullback"], p.alphas["trend_pullback"]) == 0
