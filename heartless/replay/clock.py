"""Virtual-time asyncio event loop for full-system replays.

The orchestrator's own coroutines (scheduler, kline worker, market stream, Telegram sender/poller) run unchanged; only
the loop's notion of time differs:

* `loop.time()` is virtual. Whenever the loop has nothing ready and would block until its next timer, it jumps the
  virtual clock straight to that deadline instead of sleeping (`--speed max`), or sleeps a scaled-down real interval
  (`speed=N`: N virtual seconds per real second).
* Work handed to an executor (`asyncio.to_thread` for the per-bar feature rebuild, the research/discovery process
  pool) freezes virtual time until it reports back: the loop really waits for it. Results therefore do not depend on
  how fast the machine is, and everything the app does at one instant completes at that instant, like on an
  infinitely fast host.
* `heartless.util.timeutil.now_ms()` is routed to the same clock while `run_virtual()` runs, so timestamps, day
  boundaries, schedules and records all follow replayed time. Production never installs this loop.

Virtual time is kept as integer microseconds since the replay origin, so long runs cannot accumulate float drift and a
jump always lands exactly on (or a fraction of a microsecond past) the timer that caused it.
"""
from __future__ import annotations

import asyncio
import logging
import math
import selectors
import time
from typing import Any, Awaitable, Callable

from heartless.util import timeutil

log = logging.getLogger(__name__)

REAL_WAIT_S = 0.5  # real-time slice used while executor work is in flight or the loop looks idle


class _VirtualSelector(selectors.BaseSelector):
    """Delegates to the platform selector; a blocking select() is answered by the owning VirtualTimeLoop."""

    def __init__(self) -> None:
        self._sel = selectors.DefaultSelector()
        self.loop: VirtualTimeLoop | None = None

    def register(self, fileobj, events, data=None):
        return self._sel.register(fileobj, events, data)

    def unregister(self, fileobj):
        return self._sel.unregister(fileobj)

    def modify(self, fileobj, events, data=None):
        return self._sel.modify(fileobj, events, data)

    def get_key(self, fileobj):
        return self._sel.get_key(fileobj)

    def get_map(self):
        return self._sel.get_map()

    def close(self) -> None:
        self._sel.close()

    def select(self, timeout=None):
        if self.loop is None or (timeout is not None and timeout <= 0):
            return self._sel.select(timeout)
        return self.loop._idle_select(self._sel, timeout)


class VirtualTimeLoop(asyncio.SelectorEventLoop):
    """SelectorEventLoop whose clock is virtual (see the module docstring)."""

    def __init__(self, start_ms: int, speed: float | None = None, idle_grace_s: float = 60.0):
        sel = _VirtualSelector()
        super().__init__(sel)
        sel.loop = self
        self.origin_ms = int(start_ms)
        self._vus = 0  # virtual microseconds since origin_ms
        # timers are compared against time() + resolution: a microsecond is far above the float spacing of the
        # virtual seconds value (~1e-9 s for months), so a jump onto a deadline always makes that timer due
        self._clock_resolution = 1e-6
        self.speed = speed if speed and speed > 0 else None  # None = as fast as possible
        self.idle_grace_s = idle_grace_s
        self.inflight = 0  # executor jobs whose result has not been delivered to the loop yet
        self.executor_jobs = 0
        self.jumps = 0
        self.frozen_wall_s = 0.0  # real time spent waiting for executor work (virtual time frozen)
        self.deadlocked = False
        self._idle_since: float | None = None

    # --- clock -------------------------------------------------------------------------------------
    def time(self) -> float:
        return self._vus / 1e6

    def now_ms(self) -> int:
        return self.origin_ms + self._vus // 1000

    def advance_to_ms(self, ms: int) -> None:
        """Move the clock forward to `ms` (never backwards). Only for setup code running outside the loop."""
        self._vus = max(self._vus, (int(ms) - self.origin_ms) * 1000)

    async def sleep_until_ms(self, ms: int) -> None:
        """Sleep until virtual epoch ms `ms` with microsecond precision (asyncio.sleep from a ms difference may land
        up to 1 ms late because now_ms() floors)."""
        delay = (int(ms) - self.origin_ms) / 1000 - self.time()
        await asyncio.sleep(max(delay, 0.0))

    # --- executor tracking -------------------------------------------------------------------------
    def run_in_executor(self, executor, func, *args):
        fut = super().run_in_executor(executor, func, *args)
        self.inflight += 1
        self.executor_jobs += 1
        fut.add_done_callback(self._job_done)
        return fut

    def _job_done(self, _fut) -> None:
        self.inflight -= 1

    # --- the jump ----------------------------------------------------------------------------------
    def _idle_select(self, sel: selectors.BaseSelector, timeout: float | None):
        """Called instead of a blocking select: nothing is ready and the next timer is `timeout` seconds away."""
        if self.inflight:
            # executor work is running (feature rebuild thread, research process): time stands still until it reports
            t0 = time.perf_counter()
            events = sel.select(REAL_WAIT_S)
            self.frozen_wall_s += time.perf_counter() - t0
            return events
        events = sel.select(0)
        if events:
            return events
        if timeout is None:
            # no timer, no executor job, no ready callback: only outside I/O could wake the loop and a replay has none.
            # Allow a late thread wake-up, then declare the deadlock instead of hanging forever.
            events = sel.select(REAL_WAIT_S)
            if events:
                self._idle_since = None
                return events
            now = time.monotonic()
            if self._idle_since is None:
                self._idle_since = now
            elif now - self._idle_since > self.idle_grace_s:
                self.deadlocked = True
                log.error("virtual-time loop deadlocked: no timer, executor job or I/O for %.0fs", self.idle_grace_s)
                self.stop()
            return []
        self._idle_since = None
        if self.speed:
            time.sleep(timeout / self.speed)
        # ceil (minus float noise) so the clock never lands just short of the deadline
        self._vus += max(0, math.ceil(timeout * 1e6 - 1e-3))
        self.jumps += 1
        return []


def _cancel_all_tasks(loop: asyncio.AbstractEventLoop) -> None:
    tasks = [t for t in asyncio.all_tasks(loop) if not t.done()]
    for t in tasks:
        t.cancel()
    if tasks:
        loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))


def run_virtual(main: Callable[[VirtualTimeLoop], Awaitable[Any]], start_ms: int, *, speed: float | None = None,
                idle_grace_s: float = 60.0,
                exception_handler: Callable[[asyncio.AbstractEventLoop, dict], None] | None = None) -> Any:
    """Run `main(loop)` on a fresh VirtualTimeLoop starting at epoch ms `start_ms`, with now_ms() routed to it.

    The previous clock and event loop are restored afterwards, also on failure, so tests can run replays in-process."""
    loop = VirtualTimeLoop(start_ms, speed=speed, idle_grace_s=idle_grace_s)
    if exception_handler is not None:
        loop.set_exception_handler(exception_handler)
    prev_clock = timeutil.set_clock(loop.now_ms)
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(main(loop))
    finally:
        try:
            _cancel_all_tasks(loop)
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.run_until_complete(loop.shutdown_default_executor())
        finally:
            timeutil.set_clock(prev_clock)
            asyncio.set_event_loop(None)
            loop.close()
