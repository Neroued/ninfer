#pragma once

#include <string_view>

namespace ninfer::supervisor {

inline constexpr std::string_view kDashboardHtml = R"HTML(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NInfer supervisor</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {
    --bg: #0c0e12;
    --panel: #141820;
    --ink: #e7ecf3;
    --muted: #8b95a7;
    --line: #2a3140;
    --ok: #3dd68c;
    --bad: #ff5d5d;
    --warn: #f5c542;
    --accent: #6ea8ff;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; background: var(--bg); color: var(--ink);
    font: 14px/1.4 "Segoe UI", system-ui, sans-serif; }
  header { display: flex; justify-content: space-between; align-items: baseline;
    padding: 16px 20px 8px; border-bottom: 1px solid var(--line); }
  h1 { font-size: 15px; letter-spacing: .16em; text-transform: uppercase; margin: 0; }
  .muted { color: var(--muted); }
  main { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; padding: 12px 20px 24px; }
  @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
  section { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 14px; }
  h2 { margin: 0 0 10px; font-size: 12px; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); }
  .row { display: flex; justify-content: space-between; gap: 12px; padding: 4px 0;
    border-bottom: 1px solid #1c2230; }
  .row:last-child { border-bottom: 0; }
  .num { font-variant-numeric: tabular-nums; font-family: Consolas, "Cascadia Mono", monospace; }
  .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; }
  .ok { background: #143525; color: var(--ok); }
  .bad { background: #3a1518; color: var(--bad); }
  .warn { background: #3a2f10; color: var(--warn); }
  .actions { display: flex; gap: 8px; }
  button { background: #1c2433; color: var(--ink); border: 1px solid var(--line);
    border-radius: 8px; padding: 6px 10px; cursor: pointer; }
  button:hover { border-color: var(--accent); }
  pre { margin: 0; max-height: 280px; overflow: auto; white-space: pre-wrap;
    font: 12px/1.35 Consolas, "Cascadia Mono", monospace; color: #c5d0e0; }
</style>
</head>
<body>
<header>
  <h1>NInfer supervisor</h1>
  <div id="mode" class="muted">loopback control surface · live SSE</div>
</header>
<main>
  <section>
    <h2>Engine</h2>
    <div class="row"><span>state</span><span id="state" class="pill warn">…</span></div>
    <div class="row"><span>health</span><span id="health" class="num">…</span></div>
    <div class="row"><span>pid</span><span id="pid" class="num">—</span></div>
    <div class="row"><span>uptime</span><span id="uptime" class="num">—</span></div>
    <div class="row"><span>restarts</span><span id="restarts" class="num">0</span></div>
    <div class="row"><span>last event</span><span id="event" class="muted">—</span></div>
    <div id="actions" class="actions" style="margin-top:12px">
      <button data-act="start">Start</button>
      <button data-act="stop">Stop</button>
      <button data-act="restart">Restart</button>
    </div>
  </section>
  <section>
    <h2>VRAM</h2>
    <div class="row"><span>adapter</span><span id="adapter" class="muted">—</span></div>
    <div class="row"><span>DXGI budget (system-wide WDDM pressure)</span><span id="budget" class="num">—</span></div>
    <div class="row"><span>device used (nvidia-smi)</span><span id="nvused" class="num">—</span></div>
    <div class="row"><span>device total (nvidia-smi)</span><span id="nvtotal" class="num">—</span></div>
    <div class="row"><span>supervisor process DXGI (not the engine)</span><span id="usage" class="num">—</span></div>
    <div class="row"><span>engine capacity (boot line)</span><span id="capline" class="muted">—</span></div>
    <div class="row"><span>admin tiers</span><span id="tiers" class="muted">—</span></div>
    <div class="row"><span>admin note</span><span id="adminnote" class="muted">—</span></div>
  </section>
  <section>
    <h2>Recent requests</h2>
    <div class="row"><span>done (window)</span><span id="rdone" class="num">—</span></div>
    <div class="row"><span>mean TTFT</span><span id="ttft" class="num">—</span></div>
    <div class="row"><span>mean decode</span><span id="decode" class="num">—</span></div>
    <div class="row"><span>reuse mix</span><span id="reuse" class="num">—</span></div>
    <div class="row"><span>log</span><span id="lognote" class="muted">—</span></div>
  </section>
  <section>
    <h2>Engine log tail</h2>
    <pre id="log">waiting…</pre>
  </section>
</main>
<script>
function gib(n){ if(!n) return "—"; return (n/1073741824).toFixed(2)+" GiB"; }
function pill(el, text, kind){ el.textContent=text; el.className="pill "+kind; }
function apply(s){
  const st=s.engine||{};
  const map={Stopped:"warn",Starting:"warn",Running:"ok",Stopping:"warn",BackingOff:"warn",Halted:"bad"};
  pill(document.getElementById("state"), st.state||"?", map[st.state]||"warn");
  document.getElementById("mode").textContent = s.monitor_only
    ? "monitor-only · no spawn/stop · live SSE"
    : "loopback control surface · live SSE";
  document.getElementById("actions").style.display = s.monitor_only ? "none" : "flex";
  document.getElementById("health").textContent = (s.health&&s.health.body)||"—";
  document.getElementById("pid").textContent = st.pid||"—";
  document.getElementById("uptime").textContent = st.uptime_s!=null ? st.uptime_s+" s" : "—";
  document.getElementById("restarts").textContent = st.restart_count||0;
  document.getElementById("event").textContent = st.last_event||"—";
  const d=s.dxgi||{};
  document.getElementById("adapter").textContent = d.adapter_name||d.error||"—";
  document.getElementById("budget").textContent = d.ok?gib(d.budget_bytes):"—";
  document.getElementById("usage").textContent = d.ok?gib(d.supervisor_usage_bytes):"—";
  const nv=s.nvidia_smi||{};
  document.getElementById("nvused").textContent = nv.ok?gib(nv.used_bytes):(nv.error||"—");
  document.getElementById("nvtotal").textContent = nv.ok?gib(nv.total_bytes):"—";
  document.getElementById("capline").textContent = s.engine_capacity_line ||
    (s.monitor_only ? "not in supervisor log (unmanaged); see admin tiers" : "waiting for engine boot line");
  const v=s.admin_vram;
  if(v && v.tiers){
    document.getElementById("tiers").textContent = v.tiers.map(t=>t.name+": "+gib(t.held_bytes)).join(" · ");
  } else { document.getElementById("tiers").textContent = "unavailable"; }
  document.getElementById("adminnote").textContent = s.admin_vram_note||"—";
  const r=s.requests||{};
  document.getElementById("rdone").textContent = r.done!=null?r.done:"—";
  document.getElementById("ttft").textContent = r.ttft_ms_mean? r.ttft_ms_mean.toFixed(0)+" ms":"—";
  document.getElementById("decode").textContent = r.decode_tok_s_mean? r.decode_tok_s_mean.toFixed(1)+" tok/s":"—";
  document.getElementById("reuse").textContent = "reset "+(r.reuse_full_reset||0)+" · append "+(r.reuse_append||0)+" · seed/restore "+(r.reuse_seed||0);
  document.getElementById("lognote").textContent = r.log_available? "ok" : (r.log_error||"not configured");
  document.getElementById("log").textContent = s.log_tail||"";
}
async function act(name){
  await fetch("/api/"+name,{method:"POST", headers:{"X-NInfer-Supervisor":"1"}});
}
document.querySelectorAll("button[data-act]").forEach(b=>b.onclick=()=>act(b.dataset.act));
const es=new EventSource("/api/events");
es.onmessage=e=>{ try{ apply(JSON.parse(e.data)); }catch(err){} };
fetch("/api/state").then(r=>r.json()).then(apply).catch(()=>{});
</script>
</body>
</html>
)HTML";

} // namespace ninfer::supervisor
