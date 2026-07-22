#!/usr/bin/env python3
"""Cosmic Mission Control — live dashboard over the T2Helix chronicle.

Zero dependencies (stdlib only). Read-only against every data source.

Sources:
  chronicle.db   insights, compass_log, pending_confirmations, goals
  ~/.cosmic_echo.jsonl          mission outcomes
  ~/.cosmic-cli/sessions/*.jsonl  per-mission step events

Run:  cosmic-cli dashboard   (or: python -m cosmic_cli.dashboard [port])
Open: http://localhost:4333

Auto-start: the cli group callback starts this server only for the subcommands
in main._DASHBOARD_SUBCOMMANDS — an allowlist, so anything else (the `gate` hook
above all) stays side-effect free. Opt out entirely with COSMIC_NO_DASHBOARD=1.
The launchd agent com.templetwo.cosmic-dashboard keeps it alive across reboots.
"""
import json
import sqlite3
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DB = Path.home() / ".claude/plugins/data/t2helix-templetwo-t2helix/chronicle.db"
ECHO = Path.home() / ".cosmic_echo.jsonl"
SESSIONS = Path.home() / ".cosmic-cli/sessions"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 4333


def q(sql, args=()):
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=2)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(sql, args).fetchall()]
    finally:
        con.close()


def tail_jsonl(path, n):
    if not path.exists():
        return []
    out = []
    for line in path.read_text(errors="replace").strip().splitlines()[-n:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def session_steps(n_files=3, n_steps=20):
    if not SESSIONS.is_dir():
        return []
    files = sorted(SESSIONS.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)[-n_files:]
    steps = []
    for f in files:
        for e in tail_jsonl(f, 40):
            e["_file"] = f.name.split("__")[0][:8]
            steps.append(e)
    return steps[-n_steps:]


def state():
    now = int(time.time() * 1000)
    day_start = now - (now % 86_400_000)  # UTC day; close enough for a pulse
    compass_total = {r["classification"]: r["n"] for r in q(
        "SELECT classification, COUNT(*) n FROM compass_log GROUP BY classification")}
    compass_today = {r["classification"]: r["n"] for r in q(
        "SELECT classification, COUNT(*) n FROM compass_log WHERE occurred_at>=? GROUP BY classification",
        (day_start,))}
    insights_today = q("SELECT COUNT(*) n FROM insights WHERE created_at>=?", (day_start,))[0]["n"]

    # activity: events per 10-min bucket, last 3 h (insights + compass verdicts)
    horizon = now - 3 * 3600 * 1000
    buckets = {}
    for r in q("SELECT created_at t FROM insights WHERE created_at>=?", (horizon,)):
        buckets[r["t"] // 600_000] = buckets.get(r["t"] // 600_000, 0) + 1
    for r in q("SELECT occurred_at t FROM compass_log WHERE occurred_at>=?", (horizon,)):
        buckets[r["t"] // 600_000] = buckets.get(r["t"] // 600_000, 0) + 1
    first = horizon // 600_000
    activity = [{"t": (b * 600_000), "n": buckets.get(b, 0)} for b in range(first, now // 600_000 + 1)]

    goal = q("SELECT * FROM goals ORDER BY last_referenced DESC LIMIT 1")
    missions = tail_jsonl(ECHO, 12)
    mc = sum(1 for m in tail_jsonl(ECHO, 500) if m.get("status") == "complete")
    mb = sum(1 for m in tail_jsonl(ECHO, 500) if m.get("status") == "blocked")

    return {
        "now": now,
        "compass_total": compass_total,
        "compass_today": compass_today,
        "insights_today": insights_today,
        "missions_complete": mc,
        "missions_blocked": mb,
        "activity": activity,
        "goal": goal[0] if goal else None,
        "feed": q("SELECT id, session_id, content, domain, layer, created_at FROM insights "
                  "ORDER BY id DESC LIMIT 25"),
        "compass": q("SELECT id, tool_name, action_summary, classification, rule_matched, reason, occurred_at "
                     "FROM compass_log ORDER BY id DESC LIMIT 25"),
        "pending": q("SELECT id, session_id, action_summary, reason, status, created_at, expires_at "
                     "FROM pending_confirmations WHERE expires_at>? ORDER BY id DESC LIMIT 10", (now,)),
        "missions": list(reversed(missions)),
        "steps": list(reversed(session_steps())),
    }


PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cosmic Mission Control</title>
<style>
  /* dark-committed cockpit; tokens from the validated reference palette (dark column) */
  :root {
    color-scheme: dark;
    --page:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --baseline:#383835; --ring:rgba(255,255,255,.10);
    --s1:#3987e5;                       /* series 1 · blue (dark step) */
    --good:#0ca30c; --warn:#fab219; --serious:#ec835a; --crit:#d03b3b;
  }
  * { box-sizing:border-box; margin:0; }
  body { background:var(--page); color:var(--ink); font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; padding:20px; }
  a { color:var(--s1); }
  header { display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; margin-bottom:18px; }
  h1 { font-size:19px; font-weight:650; letter-spacing:.2px; }
  .pulse { display:inline-block; width:9px; height:9px; border-radius:50%; background:var(--good); animation:pulse 1.6s ease-in-out infinite; vertical-align:1px; margin-right:7px; }
  @keyframes pulse { 50% { opacity:.25; } }
  .hmeta { color:var(--muted); font-size:12px; }
  .goal { color:var(--ink-2); font-size:12px; max-width:52ch; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }

  .tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; margin-bottom:12px; }
  .tile { background:var(--surface); border:1px solid var(--ring); border-radius:10px; padding:14px 16px; }
  .tile .label { color:var(--muted); font-size:11.5px; text-transform:uppercase; letter-spacing:.6px; }
  .tile .value { font-size:30px; font-weight:650; margin-top:2px; }
  .tile .sub { color:var(--ink-2); font-size:12px; margin-top:2px; }

  .chip { display:inline-flex; align-items:center; gap:5px; font-size:11.5px; padding:1px 8px; border-radius:999px; border:1px solid var(--ring); color:var(--ink-2); white-space:nowrap; }
  .chip .dot { width:8px; height:8px; border-radius:50%; flex:none; }
  .OPEN .dot { background:var(--good); } .PAUSE .dot { background:var(--warn); } .WITNESS .dot { background:var(--crit); }
  .complete .dot { background:var(--good); } .blocked .dot { background:var(--crit); }

  .chart { background:var(--surface); border:1px solid var(--ring); border-radius:10px; padding:14px 16px; margin-bottom:12px; }
  .chart h2, .panel h2 { font-size:12.5px; font-weight:600; color:var(--ink-2); text-transform:uppercase; letter-spacing:.6px; margin-bottom:10px; }
  .bars { display:flex; align-items:flex-end; gap:2px; height:72px; border-bottom:1px solid var(--baseline); }
  .bars .bar { flex:1; min-width:2px; background:var(--s1); border-radius:4px 4px 0 0; position:relative; }
  .bars .bar.zero { background:transparent; }
  .bars .bar:hover { filter:brightness(1.25); }
  .bars .bar:hover::after { content:attr(data-tip); position:absolute; bottom:calc(100% + 6px); left:50%; transform:translateX(-50%); background:var(--page); border:1px solid var(--ring); color:var(--ink); font-size:11px; padding:3px 8px; border-radius:6px; white-space:nowrap; z-index:3; }
  .axis { display:flex; justify-content:space-between; color:var(--muted); font-size:10.5px; margin-top:4px; font-variant-numeric:tabular-nums; }

  .cols { display:grid; grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); gap:12px; }
  .panel { background:var(--surface); border:1px solid var(--ring); border-radius:10px; padding:14px 16px; max-height:460px; overflow-y:auto; }
  .row { padding:8px 0; border-bottom:1px solid var(--grid); font-size:12.5px; color:var(--ink-2); overflow-wrap:anywhere; }
  .row:last-child { border-bottom:none; }
  .row .meta { color:var(--muted); font-size:11px; margin-top:2px; font-variant-numeric:tabular-nums; }
  .row.fresh { animation:fresh 2.5s ease-out; }
  @keyframes fresh { from { background:rgba(57,135,229,.16); } to { background:transparent; } }
  .tokenbar { height:4px; background:var(--grid); border-radius:2px; margin-top:6px; overflow:hidden; }
  .tokenbar i { display:block; height:100%; background:var(--warn); border-radius:2px; }
  .empty { color:var(--muted); font-size:12px; padding:8px 0; }
  footer { color:var(--muted); font-size:11px; margin-top:16px; }
</style></head><body>
<header>
  <h1><span class="pulse"></span>Cosmic Mission Control</h1>
  <span class="hmeta" id="clock"></span>
  <span class="goal" id="goal"></span>
</header>

<div class="tiles">
  <div class="tile"><div class="label">Compass verdicts · all time</div><div class="value" id="t-compass">–</div>
    <div class="sub" id="t-compass-sub"></div></div>
  <div class="tile"><div class="label">Verdicts today</div><div class="value" id="t-today">–</div>
    <div class="sub" id="t-today-sub"></div></div>
  <div class="tile"><div class="label">Chronicle entries today</div><div class="value" id="t-insights">–</div>
    <div class="sub">receipts written</div></div>
  <div class="tile"><div class="label">Missions (recent log)</div><div class="value" id="t-missions">–</div>
    <div class="sub" id="t-missions-sub"></div></div>
  <div class="tile"><div class="label">Gates awaiting a human</div><div class="value" id="t-pending">–</div>
    <div class="sub" id="t-pending-sub"></div></div>
</div>

<div class="chart">
  <h2>Chronicle + compass activity — events per 10 min, last 3 h</h2>
  <div class="bars" id="bars"></div>
  <div class="axis"><span id="ax0"></span><span id="ax1"></span></div>
</div>

<div class="cols">
  <div class="panel"><h2>Compass log — live verdicts</h2><div id="compass"></div></div>
  <div class="panel"><h2>Chronicle feed — receipts</h2><div id="feed"></div></div>
  <div class="panel"><h2>Pending gates + missions</h2><div id="gates"></div><div id="missions"></div></div>
</div>

<footer>read-only over chronicle.db · echo · sessions — polling every 2 s · OPEN ✓ allow · PAUSE ⏸ token gate · WITNESS ⛔ hard deny</footer>

<script>
const seen = { feed:new Set(), compass:new Set() };
let firstPaint = true;
const fmtT = ms => new Date(ms).toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'});
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const ICON = { OPEN:'✓', PAUSE:'⏸', WITNESS:'⛔', complete:'✓', blocked:'⛔' };
const chip = c => `<span class="chip ${esc(c)}"><span class="dot"></span>${ICON[c]||''} ${esc(c)}</span>`;

async function tick() {
  let s;
  try { s = await (await fetch('/api/state')).json(); }
  catch { document.getElementById('clock').textContent = 'reconnecting…'; return; }
  document.getElementById('clock').textContent = fmtT(s.now);
  document.getElementById('goal').textContent = s.goal ? ('goal · ' + (s.goal.goal || '')) : '';

  const ct = s.compass_total, cd = s.compass_today;
  const sum = o => (o.OPEN||0)+(o.PAUSE||0)+(o.WITNESS||0);
  document.getElementById('t-compass').textContent = sum(ct).toLocaleString();
  document.getElementById('t-compass-sub').innerHTML =
    `${chip('OPEN')} ${ct.OPEN||0} ${chip('PAUSE')} ${ct.PAUSE||0} ${chip('WITNESS')} ${ct.WITNESS||0}`;
  document.getElementById('t-today').textContent = sum(cd).toLocaleString();
  document.getElementById('t-today-sub').innerHTML =
    `${chip('OPEN')} ${cd.OPEN||0} ${chip('PAUSE')} ${cd.PAUSE||0} ${chip('WITNESS')} ${cd.WITNESS||0}`;
  document.getElementById('t-insights').textContent = s.insights_today.toLocaleString();
  document.getElementById('t-missions').textContent = s.missions_complete + s.missions_blocked;
  document.getElementById('t-missions-sub').innerHTML =
    `${chip('complete')} ${s.missions_complete} ${chip('blocked')} ${s.missions_blocked}`;
  document.getElementById('t-pending').textContent = s.pending.length;
  document.getElementById('t-pending-sub').textContent =
    s.pending.length ? 'confirm from a human seat' : 'threshold quiet';

  // activity bars — single series, hover tooltip per bar
  const max = Math.max(1, ...s.activity.map(a => a.n));
  document.getElementById('bars').innerHTML = s.activity.map(a =>
    `<div class="bar ${a.n ? '' : 'zero'}" style="height:${Math.max(a.n/max*100, a.n?4:0)}%"
      data-tip="${fmtT(a.t).slice(0,5)} — ${a.n} event${a.n===1?'':'s'}"></div>`).join('');
  document.getElementById('ax0').textContent = fmtT(s.activity[0]?.t ?? s.now).slice(0,5);
  document.getElementById('ax1').textContent = 'now';

  const paint = (elId, rows, key, render) => {
    document.getElementById(elId).innerHTML = rows.length
      ? rows.map(r => render(r, !firstPaint && !seen[key].has(r.id))).join('')
      : '<div class="empty">nothing yet — fly a mission</div>';
    rows.forEach(r => seen[key].add(r.id));
  };

  paint('compass', s.compass, 'compass', (r, fresh) => `
    <div class="row ${fresh?'fresh':''}">${chip(r.classification)} <b>${esc(r.tool_name)}</b>
      ${esc(r.action_summary).slice(0,160)}
      <div class="meta">${fmtT(r.occurred_at)}${r.rule_matched ? ' · rule: '+esc(r.rule_matched) : ''}</div></div>`);

  paint('feed', s.feed, 'feed', (r, fresh) => `
    <div class="row ${fresh?'fresh':''}"><b>#${r.id}</b> · ${esc(r.domain||'—')} · ${esc(r.layer)}<br>
      ${esc(r.content).slice(0,220)}
      <div class="meta">${fmtT(r.created_at)} · ${esc(r.session_id).slice(0,8)}</div></div>`);

  document.getElementById('gates').innerHTML = s.pending.length ? s.pending.map(p => {
    const left = Math.max(0, p.expires_at - s.now), total = p.expires_at - p.created_at;
    return `<div class="row">${chip('PAUSE')} ${esc(p.action_summary).slice(0,140)}
      <div class="meta">${esc(p.status)} · expires in ${Math.ceil(left/1000)}s · session ${esc(p.session_id).slice(0,12)}</div>
      <div class="tokenbar"><i style="width:${(left/total*100).toFixed(1)}%"></i></div></div>`;
  }).join('') : '<div class="empty">no gates pending</div>';

  document.getElementById('missions').innerHTML = s.missions.map(m => `
    <div class="row">${chip(m.status)} ${esc(m.directive||'').slice(0,120)}
      <div class="meta">${esc(m.model||'')} · ${m.steps ?? '?'} steps</div></div>`).join('');

  firstPaint = false;
}
tick(); setInterval(tick, 2000);
</script>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/state"):
            try:
                body = json.dumps(state()).encode()
                self._send(200, "application/json", body)
            except Exception as e:  # surface read errors to the client, keep serving
                self._send(500, "application/json", json.dumps({"error": str(e)}).encode())
        elif self.path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", PAGE.encode())
        else:
            self._send(404, "text/plain", b"not found")

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    if not DB.exists():
        sys.exit(f"chronicle.db not found at {DB}")
    print(f"Cosmic Mission Control · http://localhost:{PORT} · db={DB}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
