"""Regression tests for heartless/web/static/index.html (round 19): XSS sink via innerHTML.

Finding: every table/list in the dashboard was rendered by assigning template strings to ``innerHTML`` without escaping.
Several interpolated values originate outside the bot (exchange error text replayed through ``error`` events, symbols from
exchangeInfo / reconciled positions, free-text ``reason`` / ``exit_reason`` columns), and the page runs with the owner's
auth cookie, so a crafted string could drive ``/api/mode``, ``/api/kill`` or ``/api/close_all``.

Two layers of coverage:
* static checks on the HTML source (always run) - an ``esc()`` helper exists and every string field named in the finding
  is wrapped by it, never interpolated raw;
* a behavioural check (runs when ``node`` is available) that executes the page's inline script against a minimal DOM stub
  and a fake ``fetch`` serving hostile payloads on every sink, then asserts no executable markup reaches any innerHTML.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "heartless" / "web" / "static" / "index.html"

PAYLOAD = '<img src=x onerror="fetch(\'/api/mode\',{method:\'POST\',body:\'{"mode":"live"}\'})">'
ATTR_PAYLOAD = 'BTC"><script>alert(1)</script>USDT'

# every server-supplied string the finding lists, as it appears inside the template literals
STRING_FIELDS = [
    "p.symbol", "p.status", "p.regime", "p.side", "p.alpha", "p.reason",
    "name", "e.params_version",
    "s.params_version", "c.name", "c.version", "c.source",
    "a", "order[+el.dataset.i]",
    "t.symbol", "t.side", "t.alpha", "t.exit_reason",
    "e.topic",
]


def _script() -> str:
    html = INDEX.read_text(encoding="utf-8")
    m = re.search(r"<script>(.*)</script>", html, re.S)
    assert m, "inline <script> block not found"
    return m.group(1)


# --- static source checks -----------------------------------------------------------------------------------------------

def test_esc_helper_defined():
    src = _script()
    assert re.search(r"const esc=s=>String\(s\?\?''\)\.replace\(/\[&<>\"'\]/g", src), "esc() helper missing or changed"


@pytest.mark.parametrize("field", STRING_FIELDS)
def test_string_fields_are_never_interpolated_raw(field):
    src = _script()
    raw = re.compile(r"\$\{\s*" + re.escape(field) + r"\s*\}")
    assert not raw.search(src), f"${{{field}}} is interpolated into innerHTML without esc()"
    assert f"esc({field})" in src, f"esc({field}) not present"


def test_alphas_tail_message_and_data_sym_are_escaped():
    src = _script()
    assert "esc(p.alphas.slice(1).join(', '))" in src
    assert 'data-sym="${esc(p.symbol)}"' in src
    # the whole event message expression (object branch incl. JSON.stringify fallback AND the plain-string branch)
    assert ("esc(typeof e.message==='object'?(e.message.message||e.message.reason||JSON.stringify(e.message)"
            ".slice(0,160)):e.message)") in src


# --- behavioural check under node ----------------------------------------------------------------------------------------

NODE = shutil.which("node")

STATUS = {
    "mode": "paper", "paused": False, "primary": "paper", "live_capable": False,
    "params_version": PAYLOAD, "health": {"market_stream": True},
    "today": {"n": 1, "win_rate": 50.0, "net": 1.0},
    "engines": {
        "paper": {"params_version": PAYLOAD, "equity": 1000.0, "realized_today": 1.0,
                  "positions": [{"symbol": ATTR_PAYLOAD, "side": PAYLOAD, "status": PAYLOAD, "regime": PAYLOAD,
                                 "alpha": PAYLOAD, "alphas": ["trend_pullback", PAYLOAD], "entry": 100.0,
                                 "mark": 101.0, "stop": 99.0, "tp": 103.0, "tp1_done": False, "unrealized": 1.0,
                                 "pnl_pct": 1.0, "r": 1.0, "reason": PAYLOAD}]},
        PAYLOAD: {"params_version": "v1", "equity": 1.0, "realized_today": 0.0, "positions": []},
    },
    "research": {"last_cycle": 1_700_000_000_000, "running": False, "next_due": 1_700_003_600_000},
    "challengers": [{"name": PAYLOAD, "version": PAYLOAD, "source": PAYLOAD, "started": 1_700_000_000_000}],
    "graduation": None,
    "alphas": [{"alpha": PAYLOAD, "trust": 1.2, "n": 3, "avg_r": 0.1}],
}
TRADES = [{"exit_time": 1_700_000_000_000, "symbol": PAYLOAD, "side": PAYLOAD, "alpha": PAYLOAD, "entry_price": 1.0,
           "exit_price": 1.1, "pnl": 0.1, "r_multiple": 0.5, "exit_reason": PAYLOAD}]
EVENTS = [
    {"ts": 1, "topic": PAYLOAD, "message": {"message": PAYLOAD}},            # error text with exchange msg
    {"ts": 2, "topic": "risk_event", "message": {"reason": PAYLOAD}},        # reason branch
    {"ts": 3, "topic": "universe", "message": {"added": [PAYLOAD]}},          # JSON.stringify fallback
    {"ts": 4, "topic": "error", "message": PAYLOAD},                          # plain-string message
]

DOM_STUB = r"""
const __sinks=[];
function __el(){return {_h:'',style:{},dataset:{i:'0'},className:'',textContent:'',clientWidth:600,value:'7',children:[],
  get innerHTML(){return this._h}, set innerHTML(v){this._h=String(v);__sinks.push(String(v))},
  appendChild(c){this.children.push(c);return c}, querySelector(){return __el()}, querySelectorAll(){return []},
  setAttribute(){}, getBoundingClientRect(){return {left:0,top:0,width:600,height:220}}, addEventListener(){}}}
globalThis.document={getElementById:()=>__el(),querySelector:()=>__el(),querySelectorAll:()=>[],createElement:()=>__el(),documentElement:{}};
globalThis.window={addEventListener(){}};
globalThis.getComputedStyle=()=>({getPropertyValue:()=>'#000'});
globalThis.setInterval=()=>0; globalThis.confirm=()=>false; globalThis.alert=()=>{}; globalThis.location={};
const __DATA=__DATA_JSON__;
globalThis.fetch=async(path)=>({status:200,json:async()=>__DATA[path.split('?')[0]]});
"""

TAIL = r"""
setTimeout(()=>{process.stdout.write(JSON.stringify({sinks:__sinks,
  esc:typeof esc==='function'?[esc(null),esc(undefined),esc(123),esc('<a href="x">&\'</a>')]:null}))},150);
"""


@pytest.mark.skipif(NODE is None, reason="node not available")
def test_render_functions_escape_hostile_server_strings(tmp_path):
    data = {"/api/status": STATUS, "/api/trades": TRADES, "/api/events": EVENTS, "/api/equity": [],
            "/api/pnl": {"stats": {"profit_factor": 1.0, "avg_r": 0.1}}}
    js = DOM_STUB.replace("__DATA_JSON__", json.dumps(data)) + _script() + TAIL
    script = tmp_path / "page.js"
    script.write_text(js, encoding="utf-8")
    proc = subprocess.run([NODE, str(script)], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    sinks = out["sinks"]
    assert sinks, "nothing was rendered"
    joined = "\n".join(sinks)

    # no executable markup from any payload survives into innerHTML
    assert "<img" not in joined
    assert 'onerror="fetch' not in joined  # the raw-quote form only exists when the payload was not escaped
    assert "<script" not in joined
    # attribute break-out: the symbol with a quote never terminates the data-sym attribute
    assert 'data-sym="BTC"' not in joined
    assert 'data-sym="BTC&quot;&gt;&lt;script&gt;' in joined
    # the escaped form reached every sink family (positions, engines, research, alpha chart, trades, events)
    escaped = "&lt;img src=x onerror=&quot;fetch(&#39;/api/mode&#39;"
    assert joined.count(escaped) >= 16, joined.count(escaped)
    # engine name key and alpha-chart label (svg innerHTML) are escaped too
    assert any(s.startswith("<td>🧪 &lt;img") for s in sinks), "engine name not escaped"
    assert any("<text" in s and 'text-anchor="end">&lt;img' in s for s in sinks), "alpha label not escaped"
    # every events branch (message / reason / JSON fallback / plain string) is escaped
    ev = [s for s in sinks if s.startswith("<div><span class=\"muted\">")]
    assert ev and ev[-1].count("&lt;img") == 5 and "<img" not in ev[-1]

    # esc() semantics: nullish -> '', numbers stringified, all five metacharacters replaced
    assert out["esc"] == ["", "", "123", "&lt;a href=&quot;x&quot;&gt;&amp;&#39;&lt;/a&gt;"]
