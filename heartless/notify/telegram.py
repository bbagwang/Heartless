"""Owner-only Telegram bot built directly on the Bot API (no heavy framework).

Pairing: unless TELEGRAM_OWNER_ID is set, the first user to send `/start <code>` (in a private chat) with
the pairing code printed in the logs becomes the only owner. Everyone else is ignored silently. The code is a
fresh high-entropy secret on every process start (never persisted), it expires after PAIRING_TTL_SEC, and
pairing is disarmed after PAIRING_MAX_FAILURES wrong codes (restart to re-arm). Setting TELEGRAM_OWNER_ID
skips pairing entirely and is the recommended setup.
"""
from __future__ import annotations

import asyncio
import logging
import secrets
import time

import httpx

from heartless.notify import formatter as F
from heartless.util.timeutil import fmt_ts, now_ms

log = logging.getLogger(__name__)

COMMANDS = [
    ("status", "현재 상태"), ("positions", "열린 포지션"), ("pnl", "손익 (today|week|month|all)"),
    ("alphas", "알파 신뢰도"), ("research", "마지막 리서치 결과"), ("challengers", "챌린저 현황"),
    ("params", "현재 파라미터"), ("trades", "최근 거래"), ("universe", "거래 종목"), ("health", "시스템 상태"),
    ("pause", "신규 진입 중지"), ("resume", "거래 재개"), ("close", "포지션 청산: /close BTCUSDT | all"),
    ("mode", "모드 전환: /mode paper|live"), ("golive", "실거래 전환"), ("kill", "전량 청산 + 정지"),
    ("web", "웹 대시보드 링크"), ("report", "일일 리포트 즉시 전송"), ("optimize", "리서치 사이클 즉시 실행"), ("discover", "알파 자동 발굴 즉시 실행"), ("setkeys", "Binance API 키 등록 (메시지 자동 삭제)"),
    ("help", "도움말"),
]

PAIRING_TTL_SEC = 600          # a pairing code is only honoured this long after start-up
PAIRING_MAX_FAILURES = 5       # wrong `/start <code>` attempts (any sender) before pairing is disarmed
TELEGRAM_API_ATTEMPTS = 4      # one request + up to three re-sends after a 429 flood-control reply


class TelegramBot:
    def __init__(self, app):
        self.app = app
        self.s = app.s
        self.token = self.s.telegram_bot_token
        self.base = f"https://api.telegram.org/bot{self.token}"
        self.client = httpx.AsyncClient(timeout=70)
        self.owner_id: int | None = self.s.telegram_owner_id or app.store.get("telegram.owner")
        self.pairing_code: str | None = None
        self._pairing_born = time.monotonic()
        self._pairing_failures = 0
        if not self.owner_id:
            # Fresh high-entropy code per process start. It is deliberately NOT read from / written to the store:
            # the old persisted 6-digit code survived restarts, never expired and could be brute-forced.
            self.pairing_code = secrets.token_urlsafe(12)
            log.warning("=" * 70)
            log.warning("TELEGRAM PAIRING CODE: %s  -> send '/start %s' to the bot (private chat) to become the owner",
                        self.pairing_code, self.pairing_code)
            log.warning("pairing expires in %d min or after %d wrong codes (restart to re-arm); "
                        "set TELEGRAM_OWNER_ID to skip pairing", PAIRING_TTL_SEC // 60, PAIRING_MAX_FAILURES)
            log.warning("=" * 70)
        self._offset = 0
        self._queue: asyncio.Queue = asyncio.Queue()
        self._pending_confirm: dict[str, tuple[str, float]] = {}
        self._error_bucket: list[float] = []
        self._stop = False
        bus = app.bus
        for topic in ("position_opened", "position_closed", "partial_tp", "entry_cancelled", "risk_event", "error",
                      "research_started", "research_done", "promotion", "graduation_ready", "daily_report", "mode_changed",
                      "control", "started", "universe", "advisor_review", "reconcile", "challenger_retired"):
            bus.subscribe(topic, self._make_handler(topic))

    # --- transport -----------------------------------------------------------------------------
    async def api(self, method: str, **params):
        data: dict = {"ok": False}
        for attempt in range(1, TELEGRAM_API_ATTEMPTS + 1):
            try:
                r = await self.client.post(f"{self.base}/{method}", json=params)
                data = r.json()
            except Exception as e:  # noqa: BLE001
                log.warning("telegram %s error: %s", method, e)
                return {"ok": False}
            if data.get("ok"):
                return data
            if data.get("error_code") != 429:
                log.warning("telegram %s failed: %s", method, data.get("description"))
                return data
            # Flood control: Telegram rejected the request, so re-sending the identical call after retry_after
            # is safe. Previously the failure was returned after the sleep and the message silently dropped.
            wait = float((data.get("parameters") or {}).get("retry_after", 3))
            log.warning("telegram %s rate limited (attempt %d/%d), retrying in %.0fs", method, attempt,
                        TELEGRAM_API_ATTEMPTS, wait)
            await asyncio.sleep(wait)
        return data

    def send(self, text: str, keyboard: list[list[dict]] | None = None, chat_id: int | None = None) -> None:
        self._queue.put_nowait((text, keyboard, chat_id))

    async def _sender(self) -> None:
        while not self._stop:
            text, keyboard, chat_id = await self._queue.get()
            cid = chat_id or self.owner_id
            if not cid:
                continue
            chunks = _chunks(text, 3900)
            for i, chunk in enumerate(chunks):
                params = {"chat_id": cid, "text": chunk, "parse_mode": "HTML", "disable_web_page_preview": True}
                if keyboard and i == len(chunks) - 1:
                    params["reply_markup"] = {"inline_keyboard": keyboard}
                res = await self.api("sendMessage", **params)
                if not res.get("ok") and "parse" in str(res.get("description", "")).lower():
                    params.pop("parse_mode")
                    await self.api("sendMessage", **params)
                await asyncio.sleep(0.35)

    async def run(self) -> None:
        await self.api("setMyCommands", commands=[{"command": c, "description": d} for c, d in COMMANDS])
        sender = asyncio.create_task(self._sender())
        if self.owner_id:
            self.send(self.app.startup_summary())
        try:
            while not self._stop:
                try:
                    data = await self.api("getUpdates", offset=self._offset, timeout=50,
                                          allowed_updates=["message", "callback_query"])
                    for upd in data.get("result", []) or []:
                        self._offset = upd["update_id"] + 1
                        try:
                            await self._handle_update(upd)
                        except Exception:  # noqa: BLE001
                            log.exception("telegram update handling failed")
                    if not data.get("ok"):
                        await asyncio.sleep(3)
                except Exception as e:  # noqa: BLE001
                    log.warning("telegram poll error: %s", e)
                    await asyncio.sleep(3)
        finally:
            sender.cancel()

    # --- auth ----------------------------------------------------------------------------------
    def _is_owner(self, user_id: int) -> bool:
        return self.owner_id is not None and int(user_id) == int(self.owner_id)

    async def _handle_update(self, upd: dict) -> None:
        if "callback_query" in upd:
            cq = upd["callback_query"]
            uid = cq["from"]["id"]
            if not self._is_owner(uid):
                await self.api("answerCallbackQuery", callback_query_id=cq["id"])
                return
            await self._handle_callback(cq)
            return
        msg = upd.get("message")
        if not msg or "text" not in msg:
            return
        uid = msg["from"]["id"]
        chat_id = msg["chat"]["id"]
        text = msg["text"].strip()
        if not self._is_owner(uid):
            parts = text.split()
            if parts and parts[0].startswith("/start") and self.owner_id is None and len(parts) >= 2 and self.pairing_code:
                chat = msg["chat"]
                # Pair only from the sender's private chat: the confirmation carries the startup summary (web
                # token), and a group/supergroup would expose it to every member.
                if chat.get("type", "private") != "private" or int(chat.get("id", uid)) != int(uid):
                    log.warning("telegram pairing attempt ignored outside a private chat (uid=%s chat=%s)", uid, chat.get("id"))
                    return
                if time.monotonic() - self._pairing_born > PAIRING_TTL_SEC:
                    self.pairing_code = None
                    log.error("telegram pairing disabled (code expired after %ds); restart to get a new code or set "
                              "TELEGRAM_OWNER_ID", PAIRING_TTL_SEC)
                    return
                if not secrets.compare_digest(parts[1].encode("utf-8"), self.pairing_code.encode("utf-8")):
                    self._pairing_failures += 1
                    log.warning("telegram pairing attempt rejected from uid=%s (%d/%d)", uid, self._pairing_failures,
                                PAIRING_MAX_FAILURES)
                    if self._pairing_failures >= PAIRING_MAX_FAILURES:
                        self.pairing_code = None
                        log.error("telegram pairing disabled after %d wrong codes; restart to get a new code or set "
                                  "TELEGRAM_OWNER_ID", self._pairing_failures)
                    return
                self.owner_id = int(uid)
                self.app.store.set("telegram.owner", self.owner_id)
                self.app.store.set("telegram.pairing", None)  # drop any code persisted by older versions
                self.pairing_code = None
                log.info("telegram owner paired: %s", uid)
                # No explicit chat_id: like every other reply this resolves to the owner's private chat, so the
                # web token in the startup summary is never posted into the chat the /start arrived from.
                self.send("✅ 페어링 완료. 이제 이 계정만 Heartless 를 제어할 수 있습니다.\n\n" + self.app.startup_summary())
                self.send(self._help())
            return  # silently ignore strangers
        if text.split()[0].lower().split("@")[0] == "/setkeys":
            # the message carries secrets: remove it from the chat history before doing anything else
            mid = msg.get("message_id")
            if mid is not None:
                await self.api("deleteMessage", chat_id=chat_id, message_id=mid)
        await self._handle_command(text, chat_id)

    # --- commands ------------------------------------------------------------------------------
    def _help(self) -> str:
        return "<b>명령어</b>\n" + "\n".join(f"/{c} — {F.esc(d)}" for c, d in COMMANDS)

    async def _handle_command(self, text: str, chat_id: int) -> None:
        parts = text.split()
        cmd = parts[0].lower().split("@")[0]
        args = parts[1:]
        app = self.app
        tz = self.s.timezone
        if cmd in ("/start", "/help"):
            self.send(self._help())
        elif cmd == "/status":
            self.send(F.fmt_status(app.status(), tz))
        elif cmd == "/positions":
            st = app.status()
            self.send(F.fmt_positions(st, self._ticks()), keyboard=self._position_keyboard(st))
        elif cmd == "/pnl":
            period = args[0] if args and args[0] in ("today", "week", "month", "all") else "today"
            self.send(F.fmt_pnl(app.pnl_report(period), tz))
        elif cmd == "/alphas":
            self.send(F.fmt_alphas(app.bandit.snapshot(), app.params))
        elif cmd == "/research":
            self.send(F.fmt_research(app.research.last_summary, tz) + "\n\n" + F.fmt_discovery(app.store.get("discovery.last"), tz))
        elif cmd == "/optimize":
            if app.research.running:
                self.send("리서치가 이미 진행 중입니다")
            else:
                asyncio.create_task(app.research.cycle(force=True))
                self.send("🔬 리서치 사이클을 시작했습니다 (수 분 소요)")
        elif cmd == "/discover":
            if app.research.running:
                self.send("리서치/발굴이 이미 진행 중입니다")
            else:
                asyncio.create_task(app.research.discovery_cycle(force=True))
                self.send("⛏ 알파 발굴을 시작했습니다 (수 분 소요). 통과한 규칙만 챌린저로 투입됩니다")
        elif cmd == "/challengers":
            self.send(self._challengers_text())
        elif cmd == "/params":
            self.send(self._params_text(args[0] if args else None))
        elif cmd == "/trades":
            n = int(args[0]) if args and args[0].isdigit() else 10
            self.send(self._trades_text(n))
        elif cmd == "/universe":
            regs = app.status()["regimes"]
            self.send("<b>유니버스</b>\n" + "\n".join(f"{F.esc(s)} — {regs.get(s, '-')}" for s in app.universe))
        elif cmd == "/health":
            h = app.health
            self.send("<b>시스템</b>\n" + "\n".join(f"{F.esc(k)}: {F.esc(v)}" for k, v in h.items()))
        elif cmd == "/pause":
            await app.pause()
        elif cmd == "/resume":
            await app.resume()
        elif cmd == "/close":
            if not args:
                self.send("사용법: /close BTCUSDT 또는 /close all")
            elif args[0].lower() == "all":
                self._confirm("close_all", "모든 포지션을 시장가로 청산할까요?")
            else:
                n = await app.close_symbol(args[0].upper())
                self.send(f"{args[0].upper()} 청산 요청 {'완료' if n else '실패(포지션 없음)'}")
        elif cmd == "/mode":
            if not args or args[0] not in ("paper", "live"):
                self.send(f"현재 모드: {app.mode}\n사용법: /mode paper | /mode live")
            elif args[0] == "live":
                self._confirm("golive", "🔴 실제 자금으로 거래를 시작합니다. 계속할까요?")
            else:
                self._confirm("gopaper", "라이브 포지션을 모두 청산하고 페이퍼 모드로 전환할까요?")
        elif cmd == "/golive":
            g = app.research.graduation_status()
            warn = "" if g["ok"] else "\n⚠️ 페이퍼 졸업 기준을 아직 충족하지 못했습니다."
            self._confirm("golive", f"🔴 실제 자금으로 거래를 시작합니다.{warn}\n계속할까요?")
        elif cmd == "/kill":
            self._confirm("kill", "🛑 모든 포지션을 청산하고 봇을 일시정지합니다. 계속할까요?")
        elif cmd == "/setkeys":
            if len(args) < 2:
                self.send("사용법: <code>/setkeys API키 시크릿 [testnet]</code>\n"
                          "메시지는 즉시 삭제됩니다. 출금 권한 없는 선물 전용 키를 쓰고, 가능하면 IP 제한을 거세요. "
                          "더 안전한 방법은 웹 대시보드의 설정 화면(HTTPS)입니다.")
            else:
                testnet = (args[2].lower() in ("testnet", "test", "1", "true")) if len(args) > 2 else None
                self.send("🔑 키 확인 중...")
                self.send(F.esc(await app.set_credentials(args[0], args[1], testnet)))
        elif cmd == "/web":
            self.send(f"웹 대시보드 토큰: <code>{app.web_token}</code>\n"
                      f"접속: http://&lt;서버주소&gt;:{self.s.web_port}/?token={app.web_token}")
        elif cmd == "/report":
            await app.daily_report()
        else:
            self.send("알 수 없는 명령입니다. /help")

    def _confirm(self, action: str, question: str) -> None:
        token = secrets.token_hex(4)
        self._pending_confirm[token] = (action, time.time())
        self.send(question, keyboard=[[{"text": "✅ 확인", "callback_data": f"cf:{token}"},
                                       {"text": "취소", "callback_data": "cancel"}]])

    async def _handle_callback(self, cq: dict) -> None:
        data = cq.get("data", "")
        app = self.app
        await self.api("answerCallbackQuery", callback_query_id=cq["id"])
        if data == "cancel":
            self.send("취소했습니다")
            return
        if data.startswith("cf:"):
            token = data[3:]
            item = self._pending_confirm.pop(token, None)
            if not item or time.time() - item[1] > 300:
                self.send("확인 요청이 만료되었습니다. 다시 시도하세요")
                return
            action = item[0]
            if action == "close_all":
                n = await app.close_all()
                self.send(f"{n}개 포지션 청산 요청 완료")
            elif action == "golive":
                self.send(await app.set_mode("live"))
            elif action == "gopaper":
                self.send(await app.set_mode("paper"))
            elif action == "kill":
                n = await app.kill()
                self.send(f"🛑 킬 스위치 실행: {n}개 청산, 신규 진입 정지")
            return
        if data.startswith("close:"):
            _, sym, eng = (data.split(":") + [None])[:3]
            n = await app.close_symbol(sym, eng)
            self.send(f"{sym} 청산 요청 {'완료' if n else '실패(포지션 없음)'}")
        elif data == "golive":
            self._confirm("golive", "🔴 실제 자금으로 거래를 시작합니다. 계속할까요?")

    # --- event handlers ------------------------------------------------------------------------
    def _make_handler(self, topic: str):
        async def handler(payload: dict) -> None:
            if not self.owner_id:
                return
            if not payload.get("notify", True):
                return
            try:
                await self._on_event(topic, payload)
            except Exception:  # noqa: BLE001
                log.exception("telegram event formatting failed for %s", topic)
        return handler

    def _ticks(self) -> dict[str, float]:
        return {s: i.tick_size for s, i in self.app.symbols.items()}

    def _label(self, engine: str) -> str:
        return "🔴 LIVE" if engine == "live" else "🧪 PAPER" if engine == "paper" else f"🧪 {engine}"

    async def _on_event(self, topic: str, p: dict) -> None:
        app = self.app
        tz = self.s.timezone
        eng_name = p.get("engine", "system")
        tick = self._ticks().get(p.get("symbol", ""), None)
        if topic == "position_opened":
            pos = p["position"]
            self.send(F.fmt_position_opened(pos, tick, self._label(eng_name)),
                      keyboard=[[{"text": f"즉시 청산 {pos.symbol}", "callback_data": f"close:{pos.symbol}:{eng_name}"}]])
        elif topic == "position_closed":
            pos, trade = p["position"], p["trade"]
            eng = app.engines.get(eng_name)
            today = eng.stats.realized_today if eng else 0.0
            equity = eng.stats.equity if eng else 0.0
            self.send(F.fmt_position_closed(pos, trade, tick, self._label(eng_name), today, equity))
        elif topic == "partial_tp":
            self.send(F.fmt_partial(p["position"], p["price"], p["qty"], p["pnl"], tick, self._label(eng_name)))
        elif topic == "entry_cancelled":
            pos = p["position"]
            self.send(f"{self._label(eng_name)} · {F.esc(pos.symbol)} 진입 취소: {F.esc(p.get('reason', ''))}")
        elif topic == "risk_event":
            self.send(f"⚠️ <b>리스크</b> {F.esc(p.get('message', ''))}")
        elif topic == "error":
            now = time.time()
            self._error_bucket = [t for t in self._error_bucket if now - t < 600]
            if len(self._error_bucket) < 5:
                self._error_bucket.append(now)
                self.send(f"❗ <b>오류</b> {F.esc(p.get('message', ''))}")
        elif topic == "research_started":
            self.send(f"🔬 {F.esc(p.get('message', ''))}")
        elif topic == "research_done":
            self.send(f"🔬 <b>리서치 완료</b> ({p.get('seconds') or 0:.0f}s)\n{F.esc(p.get('message', ''))}")
        elif topic == "promotion":
            self.send(F.esc(p.get("message", "")))
        elif topic == "graduation_ready":
            self.send(f"🎓 <b>페이퍼 졸업 기준 충족</b>\n14일: {p['n']}건, PF {p['profit_factor']:.2f}, MDD {p['max_dd_pct']:.1f}%, "
                      f"순익 {p['net']:+.1f} USDT, 승률 {p['win_rate']:.0f}%\n실거래로 전환할까요?",
                      keyboard=[[{"text": "🔴 라이브 전환", "callback_data": "golive"}, {"text": "나중에", "callback_data": "cancel"}]])
        elif topic == "daily_report":
            self.send(F.fmt_daily(p["today"], p["week"], p["status"], tz))
        elif topic in ("mode_changed", "control", "universe", "reconcile", "challenger_retired"):
            self.send(F.esc(p.get("message", "")))
        elif topic == "started":
            pass  # startup summary is sent by run()
        elif topic == "advisor_review":
            self.send(f"🧠 <b>AI 리뷰</b>\n{F.esc(p.get('message', ''))}")

    # --- text helpers --------------------------------------------------------------------------
    def _position_keyboard(self, st: dict) -> list[list[dict]] | None:
        prim = st["engines"].get(st["primary"], {}) if st.get("primary") else {}
        rows = prim.get("positions", [])
        if not rows:
            return None
        return [[{"text": f"청산 {r['symbol']}", "callback_data": f"close:{r['symbol']}:{st['primary']}"}] for r in rows[:8]]

    def _challengers_text(self) -> str:
        app = self.app
        if not app.challengers:
            return "활성 챌린저가 없습니다 (다음 리서치 사이클에서 생성)"
        lines = ["<b>챌린저</b>"]
        from heartless.execution.stats import summarize

        champ = summarize(app.store.load_trades(engine="paper", since=min(c.started for c in app.challengers)))
        lines.append(f"챔피언(paper) n={champ['n']} avgR {champ['avg_r']:+.2f} net {champ['net']:+.1f}")
        for c in app.challengers:
            st = summarize(app.store.load_trades(engine=c.name, since=c.started))
            lines.append(f"<code>{F.esc(c.name)}</code> [{F.esc(c.source)}] {F.esc(c.params.version)} · 시작 {fmt_ts(c.started, self.s.timezone, '%m-%d %H:%M')}\n"
                         f"  n={st['n']}/{self.s.promotion_min_trades} avgR {st['avg_r']:+.2f} PF {min(st['profit_factor'], 99):.2f} net {st['net']:+.1f}")
        return "\n".join(lines)

    def _params_text(self, alpha: str | None) -> str:
        p = self.app.params
        lines = [f"<b>챔피언 파라미터</b> <code>{F.esc(p.version)}</code> — {F.esc(p.note)}",
                 f"ensemble: " + ", ".join(f"{k}={v}" for k, v in p.ensemble.items())]
        for a, vals in p.alphas.items():
            if alpha and a != alpha:
                continue
            lines.append(f"<code>{F.esc(a)}</code>{'' if p.enabled.get(a, True) else ' (off)'}: " +
                         ", ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in vals.items()))
        return "\n".join(lines)

    def _trades_text(self, n: int) -> str:
        prim = self.app.primary_engine
        if not prim:
            return "-"
        trades = self.app.store.load_trades(engine=prim.name, limit=n)
        if not trades:
            return "거래 기록이 없습니다"
        lines = [f"<b>최근 {len(trades)}건</b> ({F.esc(prim.name)})"]
        for t in trades:
            icon = "✅" if t["pnl"] > 0 else "❌"
            lines.append(f"{icon} {fmt_ts(t['exit_time'], self.s.timezone, '%m-%d %H:%M')} {F.esc(t['symbol'])} {t['side']} "
                         f"<code>{F.esc(t['alpha'])}</code> {t['pnl']:+.2f} ({t['r_multiple']:+.2f}R) {F.esc(t['exit_reason'])}")
        return "\n".join(lines)


def _chunks(text: str, n: int) -> list[str]:
    if len(text) <= n:
        return [text]
    out, cur = [], ""
    for line in text.split("\n"):
        # A single line longer than n (LLM prose, big /universe) is hard-split so no chunk can exceed the
        # Telegram limit; an empty `cur` is never flushed (it used to produce an empty sendMessage).
        for piece in [line[i:i + n] for i in range(0, len(line), n)] or [line]:
            if len(cur) + len(piece) + 1 > n:
                if cur:
                    out.append(cur)
                cur = piece
            else:
                cur = f"{cur}\n{piece}" if cur else piece
    if cur:
        out.append(cur)
    return out
