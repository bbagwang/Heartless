"""Regression tests for heartless/data/indicators.py (fix round 6).

Finding: hurst_proxy was biased low (mean ~0.469 on pure Gaussian random walks instead of 0.5) because the
variance ratio used the plain sample variance of overlapping 5-bar sums and then log(VR); the consumers
(regime.py `hurst15 > 0.55`, mean_reversion.py `hurst < 0.45`) therefore fired on ~23% / ~42% of random-walk
bars.  The estimator now uses the Lo-MacKinlay small-sample correction which centres the null at ~0.5.

NOTE: the null *noise* floor at the window currently used by features.py (n=100) is std ~0.107 regardless of
estimator, so the 0.45/0.55 consumer thresholds still sit well inside one sigma.  The tests below pin down the
null distribution so that the follow-up calibration (wider window in features.py and/or ~2-sigma consumer
thresholds 0.35/0.65) cannot silently regress.
"""
import numpy as np

from heartless.data import indicators as I

BARS = 20_000
SEEDS = 20


def _random_walk(seed: int, bars: int = BARS, sigma: float = 0.002) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(0.0, sigma, bars)))


def _ar1_walk(phi: float, seed: int, bars: int = BARS, sigma: float = 0.002) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, sigma, bars)
    r = np.empty(bars)
    r[0] = e[0]
    for i in range(1, bars):
        r[i] = phi * r[i - 1] + e[i]
    return 100.0 * np.exp(np.cumsum(r))


def _null_sample(n: int) -> np.ndarray:
    hs = [I.hurst_proxy(_random_walk(s), n) for s in range(SEEDS)]
    h = np.concatenate(hs)
    return h[~np.isnan(h)]


def test_hurst_proxy_centred_on_random_walk():
    # window used by features.py
    h = _null_sample(100)
    assert abs(h.mean() - 0.5) < 0.02, h.mean()
    assert h.mean() > 0.48  # pre-fix estimator read 0.469
    # the two consumer thresholds should now be hit roughly symmetrically (pre-fix: 42.7% vs 23.1%)
    lo, hi = (h < 0.45).mean(), (h > 0.55).mean()
    assert abs(lo - hi) < 0.10, (lo, hi)
    # longer window converges further towards 0.5
    h200 = _null_sample(200)
    assert abs(h200.mean() - 0.5) < 0.01, h200.mean()


def test_hurst_proxy_null_noise_band():
    """Documents the null noise floor so consumer thresholds can be calibrated against it."""
    h100 = _null_sample(100)
    assert 0.09 < h100.std() < 0.13, h100.std()
    # ~2-sigma band at n=100: each side fires on well under 15% of random-walk bars
    assert (h100 > 0.65).mean() < 0.10, (h100 > 0.65).mean()
    assert (h100 < 0.35).mean() < 0.15, (h100 < 0.35).mean()
    # with the verifier-suggested window (features.py n=200) the 0.35/0.65 band is a real 2-sigma band
    h200 = _null_sample(200)
    assert (h200 > 0.65).mean() < 0.05, (h200 > 0.65).mean()
    assert (h200 < 0.35).mean() < 0.05, (h200 < 0.35).mean()


def test_hurst_proxy_discriminates_persistence():
    """Positively autocorrelated returns must read trending, negatively autocorrelated mean-reverting."""
    trend = np.concatenate([I.hurst_proxy(_ar1_walk(0.3, s), 100) for s in range(3)])
    revert = np.concatenate([I.hurst_proxy(_ar1_walk(-0.3, s), 100) for s in range(3)])
    assert np.nanmean(trend) > 0.60, np.nanmean(trend)
    assert np.nanmean(revert) < 0.40, np.nanmean(revert)
    # majority of bars land on the right side of the consumer thresholds
    assert (trend[~np.isnan(trend)] > 0.55).mean() > 0.80
    assert (revert[~np.isnan(revert)] < 0.45).mean() > 0.80


def test_hurst_proxy_alignment_bounds_and_causality():
    c = _random_walk(7, bars=600)
    n = 100
    h = I.hurst_proxy(c, n)
    assert len(h) == len(c)
    assert np.isnan(h[: n - 1]).all() and not np.isnan(h[n - 1:]).any()
    assert np.nanmin(h) >= 0.0 and np.nanmax(h) <= 1.0
    # causality: changing the last close must not alter any earlier value
    c2 = c.copy()
    c2[-1] *= 1.05
    h2 = I.hurst_proxy(c2, n)
    assert np.allclose(h[:-1], h2[:-1], equal_nan=True)
    assert h[-1] != h2[-1]
    # too short -> all NaN; degenerate windows -> all NaN rather than a division error
    assert np.isnan(I.hurst_proxy(c[: n + 4], n)).all()
    assert np.isnan(I.hurst_proxy(c, 5)).all()


def test_hurst_proxy_constant_price_is_neutral():
    c = np.full(300, 123.45)
    h = I.hurst_proxy(c, 100)
    assert np.allclose(h[99:], 0.5)
