"""Alpha registry.

Every module in this package that defines a subclass of `Alpha` with a unique `name` is registered
automatically, so adding an alpha is a single new file: the class declares `timeframe`, `param_specs`
and `regime_affinity`, and the parameter schema, ensemble and optimizer pick it up. Modules whose name
starts with "_" are helpers and are skipped. Set `enabled_by_default = False` on an experimental alpha to
ship it switched off (the research loop can still evaluate and enable it).
"""
from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil

from heartless.strategy.base import Alpha

# stable ordering for UIs and charts; newly discovered alphas are appended alphabetically
_PREFERRED = ["trend_pullback", "squeeze_breakout", "mean_reversion", "momentum_burst", "funding_fade", "sweep_reversal"]


log = logging.getLogger(__name__)
# module name -> error for alpha modules that failed to import; the rest of the registry still loads so that one
# broken experimental file cannot take the whole bot down (startup and tests surface these loudly)
FAILED_ALPHAS: dict[str, str] = {}


def _discover() -> list[Alpha]:
    found: dict[str, Alpha] = {}
    for mod in pkgutil.iter_modules(__path__):
        if mod.name.startswith("_"):
            continue
        try:
            module = importlib.import_module(f"{__name__}.{mod.name}")
        except Exception as e:  # noqa: BLE001
            FAILED_ALPHAS[mod.name] = f"{type(e).__name__}: {e}"
            log.error("alpha module %s failed to import and is disabled: %s", mod.name, e)
            continue
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, Alpha) and cls is not Alpha and cls.__module__ == module.__name__ and not inspect.isabstract(cls):
                inst = cls()
                if inst.name in found:
                    raise RuntimeError(f"duplicate alpha name {inst.name!r} in {module.__name__}")
                found[inst.name] = inst
    order = [n for n in _PREFERRED if n in found] + sorted(n for n in found if n not in _PREFERRED)
    return [found[n] for n in order]


ALL_ALPHAS: list[Alpha] = _discover()
ALPHA_BY_NAME: dict[str, Alpha] = {a.name: a for a in ALL_ALPHAS}

__all__ = ["ALL_ALPHAS", "ALPHA_BY_NAME", "FAILED_ALPHAS"]
