from heartless.strategy.alphas.funding_fade import FundingFade
from heartless.strategy.alphas.mean_reversion import MeanReversion
from heartless.strategy.alphas.momentum_burst import MomentumBurst
from heartless.strategy.alphas.squeeze_breakout import SqueezeBreakout
from heartless.strategy.alphas.sweep_reversal import SweepReversal
from heartless.strategy.alphas.trend_pullback import TrendPullback

ALL_ALPHAS = [TrendPullback(), SqueezeBreakout(), MeanReversion(), MomentumBurst(), FundingFade(), SweepReversal()]
ALPHA_BY_NAME = {a.name: a for a in ALL_ALPHAS}

__all__ = ["ALL_ALPHAS", "ALPHA_BY_NAME", "TrendPullback", "SqueezeBreakout", "MeanReversion", "MomentumBurst",
           "FundingFade", "SweepReversal"]
