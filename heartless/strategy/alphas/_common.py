from __future__ import annotations

from heartless.core.models import Side


def protective_stop(side: Side, entry: float, atr: float, sl_atr: float, structure: float | None,
                    min_atr: float = 0.7) -> float:
    """Closer of ATR stop and structure stop, but never tighter than min_atr * ATR."""
    atr_stop = entry - side.sign * sl_atr * atr
    stop = atr_stop
    if structure is not None:
        # structure stop must be on the protective side
        if (side is Side.LONG and structure < entry) or (side is Side.SHORT and structure > entry):
            stop = max(atr_stop, structure) if side is Side.LONG else min(atr_stop, structure)
    if abs(entry - stop) < min_atr * atr:
        stop = entry - side.sign * min_atr * atr
    return stop


def targets(side: Side, entry: float, stop: float, tp_r: float, tp1_r: float) -> tuple[float, float]:
    r = abs(entry - stop)
    return entry + side.sign * tp_r * r, entry + side.sign * tp1_r * r


def clamp_conf(c: float) -> float:
    return max(0.0, min(0.98, c))
