"""Fix 11: tp1_r (partial target) must always lie strictly inside tp_r (final target).

tp_r and tp1_r have overlapping ranges and used to be sampled / perturbed independently, so ~10% of
random candidates had tp1_r >= tp_r. The engine then placed the half-size "runner" TP closer than the
"partial" TP1 and inverted the bracket structure (the near fill was treated as the partial, stop moved
to BE, remaining half targeted the farther level).
"""
import random

from heartless.strategy.params import ALPHA_SPECS, StrategyParams

TARGET_ALPHAS = [a for a, specs in ALPHA_SPECS.items() if {"tp_r", "tp1_r"} <= {s.name for s in specs}]
N = 10_000


def _spec(alpha, name):
    return next(s for s in ALPHA_SPECS[alpha] if s.name == name)


def _assert_in_bounds(alpha, vals):
    for s in ALPHA_SPECS[alpha]:
        v = vals[s.name]
        if s.choices:
            assert v in s.choices
        else:
            assert s.lo - 1e-9 <= v <= s.hi + 1e-9, (alpha, s.name, v)


def test_target_alphas_present():
    # every alpha that uses targets() is covered by this test module
    assert set(TARGET_ALPHAS) == {"trend_pullback", "squeeze_breakout", "momentum_burst", "funding_fade", "sweep_reversal"}


def test_spec_bounds_allow_strict_order_at_tp_r_lower_bound():
    # the clamp tp1_r <- tp_r - step must never be pushed back above tp_r by tp1_r.lo
    for alpha in TARGET_ALPHAS:
        tp, tp1 = _spec(alpha, "tp_r"), _spec(alpha, "tp1_r")
        assert tp1.clip(tp.lo - tp1.step) < tp.lo, alpha


def test_random_alpha_params_never_inverts_targets():
    rng = random.Random(11)
    for alpha in TARGET_ALPHAS:
        for _ in range(N):
            vals = StrategyParams.random_alpha_params(alpha, rng)
            assert vals["tp1_r"] < vals["tp_r"], (alpha, vals)
            _assert_in_bounds(alpha, vals)


def test_perturbed_never_inverts_targets_from_defaults():
    base = StrategyParams.default()
    rng = random.Random(12)
    for alpha in TARGET_ALPHAS:
        for i in range(N):
            scale = 1.0 if i % 2 else 0.3
            vals = base.perturbed(alpha, rng, scale=scale)
            assert vals["tp1_r"] < vals["tp_r"], (alpha, vals)
            _assert_in_bounds(alpha, vals)


def test_perturbed_never_inverts_targets_from_edge_base():
    """Base sitting right at the boundary (tp_r at its minimum, tp1_r as high as allowed) is the
    hardest case: any upward move of tp1_r or downward move of tp_r must still be corrected."""
    base = StrategyParams.default()
    for alpha in TARGET_ALPHAS:
        tp, tp1 = _spec(alpha, "tp_r"), _spec(alpha, "tp1_r")
        base = base.with_alpha(alpha, {"tp_r": tp.lo, "tp1_r": tp1.hi})
        assert base.alphas[alpha]["tp1_r"] < base.alphas[alpha]["tp_r"]
    rng = random.Random(13)
    for alpha in TARGET_ALPHAS:
        for _ in range(N):
            vals = base.perturbed(alpha, rng, scale=1.0, n_params=2)
            assert vals["tp1_r"] < vals["tp_r"], (alpha, vals)
            _assert_in_bounds(alpha, vals)


def test_from_dict_corrects_inverted_hand_edited_params():
    d = StrategyParams.default().to_dict()
    d["alphas"]["trend_pullback"]["tp_r"] = 1.2
    d["alphas"]["trend_pullback"]["tp1_r"] = 1.8
    d["alphas"]["momentum_burst"]["tp_r"] = 0.8
    d["alphas"]["momentum_burst"]["tp1_r"] = 0.8  # equal is also inverted (TP1 would never be the partial)
    p = StrategyParams.from_dict(d)
    tp = p.alphas["trend_pullback"]
    assert tp["tp_r"] == 1.2  # final target is authoritative, partial is pulled inside
    assert abs(tp["tp1_r"] - 1.1) < 1e-9
    mb = p.alphas["momentum_burst"]
    assert mb["tp_r"] == 0.8 and abs(mb["tp1_r"] - 0.7) < 1e-9
    for alpha in TARGET_ALPHAS:
        assert p.alphas[alpha]["tp1_r"] < p.alphas[alpha]["tp_r"]
        _assert_in_bounds(alpha, p.alphas[alpha])


def test_from_dict_partial_override_of_tp_r_only_still_ordered():
    # legacy file only lowers tp_r below the default tp1_r
    d = {"alphas": {"squeeze_breakout": {"tp_r": 1.2}}}  # default tp1_r is 1.2
    p = StrategyParams.from_dict(d)
    sb = p.alphas["squeeze_breakout"]
    assert sb["tp_r"] == 1.2 and sb["tp1_r"] < 1.2


def test_with_alpha_enforces_order_for_external_seeds():
    base = StrategyParams.default()
    c = base.with_alpha("funding_fade", {"tp_r": 1.0, "tp1_r": 1.8})
    assert c.alphas["funding_fade"]["tp_r"] == 1.0
    assert c.alphas["funding_fade"]["tp1_r"] < 1.0
    # the source object is untouched (with_alpha works on a clone)
    assert base.alphas["funding_fade"] == StrategyParams.default().alphas["funding_fade"]


def test_enforcement_is_noop_for_valid_and_targetless_params():
    p = StrategyParams.default()
    q = StrategyParams.from_dict(p.to_dict())
    assert q.alphas == p.alphas  # defaults are already ordered -> byte-identical roundtrip
    rng = random.Random(14)
    for _ in range(200):
        vals = StrategyParams.random_alpha_params("mean_reversion", rng)
        assert "tp_r" not in vals and "tp1_r" not in vals
        _assert_in_bounds("mean_reversion", vals)
