"""Full-system replay soak harness (docs/REPLAY.md).

Runs the real Heartless orchestrator over stored history on a virtual-time event loop: the market stream and the
public REST endpoints are served from a store, Telegram goes to a capture stub, the network is blocked, and a set of
invariants is checked every virtual minute. Entry point: `heartless replay` / `run_replay()`.
"""
from __future__ import annotations

__all__ = ["ReplayConfig", "VirtualTimeLoop", "run_replay", "run_virtual"]


def __getattr__(name: str):
    # lazy: importing the package (e.g. for the clock alone) must not pull in the orchestrator
    if name in ("VirtualTimeLoop", "run_virtual"):
        from heartless.replay import clock

        return getattr(clock, name)
    if name in ("ReplayConfig", "run_replay"):
        from heartless.replay import runner

        return getattr(runner, name)
    raise AttributeError(name)
