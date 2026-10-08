from __future__ import annotations

import math
from decimal import ROUND_DOWN, ROUND_HALF_EVEN, Decimal


def round_step(value: float, step: float, mode: str = "down") -> float:
    """Round value to a multiple of step (Binance stepSize / tickSize)."""
    if step <= 0:
        return value
    d_step = Decimal(str(step))
    d_val = Decimal(str(value))
    ratio = d_val / d_step
    if mode == "down":
        # tolerate binary floating-point residue: 0.0564 - 0.0282 = 0.028199999999999996 must stay 0.0282, not
        # drop a whole step (which would leave a one-step position that no bracket covers)
        ratio += Decimal("1e-9")
    q = ratio.quantize(Decimal("1"), rounding=ROUND_DOWN if mode == "down" else ROUND_HALF_EVEN)
    return float(q * d_step)


def round_up_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    d_step = Decimal(str(step))
    d_val = Decimal(str(value))
    q = (d_val / d_step - Decimal("1e-9")).to_integral_value(rounding="ROUND_CEILING")
    return float(q * d_step)


def decimals_of(step: float) -> int:
    s = f"{step:.12f}".rstrip("0")
    if "." not in s:
        return 0
    return len(s.split(".")[1])


def fmt_price(p: float, tick: float | None = None) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "-"
    if tick:
        return f"{p:.{decimals_of(tick)}f}"
    if p >= 1000:
        return f"{p:,.2f}"
    if p >= 1:
        return f"{p:.4f}"
    return f"{p:.6f}"


def fmt_usd(v: float, sign: bool = True) -> str:
    if v is None:
        return "-"
    s = f"{v:+,.2f}" if sign else f"{v:,.2f}"
    return f"{s} USDT"


def fmt_pct(v: float, digits: int = 2) -> str:
    return f"{v:+.{digits}f}%"


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x
