"""Alpha registry, positioning features and lab helpers."""
import math

import numpy as np

from heartless.data.extras import MetricsSeries, rows_from_rest
from heartless.learning.lab import apply_overrides, infer_symbol_info, splits
from heartless.strategy.alphas import ALL_ALPHAS, ALPHA_BY_NAME
from heartless.strategy.base import Alpha
from heartless.strategy.params import ALPHA_SPECS, COMMON, StrategyParams
from heartless.strategy.regime import REGIME_AFFINITY

DAY = 86_400_000


def test_registry_discovers_every_alpha_with_its_own_specs():
    names = [a.name for a in ALL_ALPHAS]
    assert len(names) == len(set(names)) >= 6
    for a in ALL_ALPHAS:
        assert isinstance(a, Alpha) and a.name in ALPHA_BY_NAME
        specs = ALPHA_SPECS[a.name]
        assert [s.name for s in specs[:len(COMMON)]] == [s.name for s in COMMON]
        assert len({s.name for s in specs}) == len(specs), f"duplicate param names in {a.name}"
        for s in specs:
            assert s.clip(s.default) == s.default or s.choices, f"{a.name}.{s.name} default outside its bounds"
        assert set(REGIME_AFFINITY[a.name]) and all(0 <= v <= 1.5 for v in REGIME_AFFINITY[a.name].values())
    p = StrategyParams.default()
    assert set(p.alphas) == set(ALPHA_SPECS) and set(p.enabled) == set(ALPHA_SPECS)


def _series(n=600, start=1_700_000_000_000):
    rows = [{"ts": start + i * 300_000, "oi": 1000 + i, "oi_value": 1e6 + i, "top_ls_accounts": 1.2, "top_ls_positions": 1.5 + i * 1e-3,
             "ls_accounts": 1.1, "taker_ls_vol": 2.0 if i % 2 else 0.5} for i in range(n)]
    return MetricsSeries.from_rows(rows), start


def test_snapshot_is_causal_and_computes_changes():
    ms, start = _series()
    t_row = start + 500 * 300_000
    snap_before = ms.snapshot(t_row + 299_999)  # row 500 is not usable until its bucket closed
    snap_after = ms.snapshot(t_row + 300_000)
    assert snap_before["oi"] == 1499 and snap_after["oi"] == 1500
    assert abs(snap_after["oi_chg_1h"] - (1500 / 1488 - 1)) < 1e-12
    assert abs(snap_after["oi_chg_4h"] - (1500 / 1452 - 1)) < 1e-12
    assert abs(snap_after["taker_ratio_1h"] - 1.0) < 1e-9  # geometric mean of 2.0 and 0.5
    assert ms.snapshot(start) == {}  # nothing usable yet
    stale = ms.snapshot(int(ms.ts[-1]) + 300_000 + 31 * 60_000)
    assert stale.get("stale") is True


def test_rest_rows_line_up_with_archive_buckets():
    end_stamped = 1_700_000_100_000  # multiple of 5m: REST stamps the bucket END
    rows = rows_from_rest([{"sumOpenInterest": "5", "sumOpenInterestValue": "50", "timestamp": end_stamped}], [], [], [],
                          [{"buySellRatio": "1.1", "timestamp": end_stamped}])
    assert rows[0]["ts"] == end_stamped - 300_000 and rows[0]["oi"] == 5.0 and rows[0]["taker_ls_vol"] == 1.1


def test_merge_keeps_newest_copy():
    a, start = _series(10)
    b = MetricsSeries.from_rows([{"ts": start + 9 * 300_000, "oi": 77, "oi_value": 1, "top_ls_accounts": 1, "top_ls_positions": 1,
                                  "ls_accounts": 1, "taker_ls_vol": 1}])
    m = a.merge(b)
    assert len(m) == 10 and m.oi[-1] == 77 and np.all(np.diff(m.ts) > 0)


def test_splits_and_overrides_and_symbol_inference():
    sp = splits(0, 280 * DAY)
    assert sp["train"][0] == 12 * DAY and sp["valid"][0] == sp["train"][1] + 1 and sp["holdout"][1] == 280 * DAY
    assert sp["holdout"][0] - sp["valid"][0] == 62 * DAY
    p = apply_overrides(StrategyParams.default(), ["trend_pullback.adx_min=25", "ensemble.entry_threshold=0.6", "enabled.funding_fade=0"])
    assert p.alphas["trend_pullback"]["adx_min"] == 25 and p.ensemble["entry_threshold"] == 0.6 and p.enabled["funding_fade"] is False
    info = infer_symbol_info("BTCUSDT", np.array([85000.0, 85000.1, 85000.3] * 100))
    assert math.isclose(info.tick_size, 0.1) and info.step_size == 0.0001


def test_no_alpha_module_failed_to_import():
    from heartless.strategy.alphas import FAILED_ALPHAS

    assert FAILED_ALPHAS == {}, FAILED_ALPHAS
