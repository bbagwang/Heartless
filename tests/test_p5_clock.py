"""P5 virtual time: the replay loop jumps straight to the next timer, freezes while executor work runs, routes
timeutil.now_ms() to virtual time only while a replay runs, and reports a deadlock instead of hanging."""
from __future__ import annotations

import asyncio
import time

import pytest

from heartless.replay.clock import run_virtual
from heartless.util import timeutil
from heartless.util.timeutil import MS_DAY, now_ms

START = 1_772_323_200_000  # 2026-03-01 00:00 UTC


def test_timers_jump_and_now_ms_follows_virtual_time():
    async def main(loop):
        seen = []

        async def tick(period: float, n: int, tag: str):
            for _ in range(n):
                await asyncio.sleep(period)
                seen.append((now_ms(), tag))

        t0 = time.perf_counter()
        await asyncio.gather(tick(5, 17_280, "5s"), tick(3600, 24, "1h"))  # one virtual day of a 5 s scheduler
        return seen, time.perf_counter() - t0, loop

    seen, wall, loop = run_virtual(main, START)
    assert seen[0] == (START + 5_000, "5s") and seen[-1][0] == START + MS_DAY
    assert [t for t, tag in seen if tag == "1h"][:2] == [START + 3_600_000, START + 7_200_000]
    assert wall < 5 and loop.jumps >= 17_280
    assert abs(now_ms() - time.time() * 1000) < 5_000  # the wall clock is back


def test_executor_work_freezes_virtual_time():
    async def main(loop):
        def work():
            time.sleep(0.3)  # real time passes ...
            return now_ms()

        async def scheduler():
            for _ in range(100):
                await asyncio.sleep(1)

        sched = asyncio.create_task(scheduler())
        t_in_thread = await asyncio.to_thread(work)
        frozen_at = now_ms()
        await sched
        return t_in_thread, frozen_at, now_ms(), loop.frozen_wall_s

    t_in_thread, frozen_at, end, frozen = run_virtual(main, START)
    assert t_in_thread == frozen_at == START  # ... but no virtual time while the job ran
    assert end == START + 100_000 and frozen >= 0.25


def test_sleep_until_lands_exactly_and_speed_paces_real_time():
    async def main(loop):
        await loop.sleep_until_ms(START + 59_999)
        a = now_ms()
        await loop.sleep_until_ms(START + 60_000 * 1440 + 1_000)
        return a, now_ms()

    assert run_virtual(main, START) == (START + 59_999, START + MS_DAY + 1_000)
    t0 = time.perf_counter()
    run_virtual(lambda loop: asyncio.sleep(2), START, speed=10)  # 2 virtual s at 10x ~ 0.2 real s
    assert 0.15 < time.perf_counter() - t0 < 2


def test_clock_hook_is_restored_and_production_uses_the_wall_clock():
    assert timeutil._clock is None
    prev = timeutil.set_clock(lambda: 42)
    try:
        assert now_ms() == 42
    finally:
        timeutil.set_clock(prev)
    assert abs(now_ms() - time.time() * 1000) < 5_000
    with pytest.raises(ValueError):
        run_virtual(lambda loop: _boom(), START)
    assert timeutil._clock is None


async def _boom():
    raise ValueError("x")


def test_a_deadlock_stops_the_loop_instead_of_hanging():
    async def main(loop):
        await asyncio.Event().wait()  # nothing will ever set it, no timer, no executor job

    t0 = time.perf_counter()
    with pytest.raises(RuntimeError, match="Event loop stopped"):
        run_virtual(main, START, idle_grace_s=0.5)
    assert time.perf_counter() - t0 < 10
