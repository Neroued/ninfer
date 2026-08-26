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
  .span2 { grid-column: 1 / -1; }
  .finding { border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; margin: 8px 0; }
  .finding .title { font-weight: 600; }
  .finding .stmt { margin: 6px 0; }
  .finding .meta { color: var(--muted); font-size: 12px; }
  .finding.unavailable { border-color: #3a2f10; }
  .finding.warning { border-color: var(--warn); }
  .finding.critical { border-color: var(--bad); }
  .finding pre { max-height: 140px; }
  svg.chart { width: 100%; height: 200px; display: block; background: #0c1018; border-radius: 6px; }
  .legend { display: flex; gap: 16px; margin-top: 8px; font-size: 12px; color: var(--muted); }
  .sw { display: inline-block; width: 12px; height: 3px; margin-right: 6px; vertical-align: middle; }
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
    <div class="row owned-only"><span>pid</span><span id="pid" class="num">—</span></div>
    <div class="row owned-only"><span>uptime</span><span id="uptime" class="num">—</span></div>
    <div class="row owned-only"><span>restarts</span><span id="restarts" class="num">0</span></div>
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
    <div class="row"><span>released now</span><span id="released" class="num">—</span></div>
    <div class="row"><span>time since release</span><span id="sincerel" class="num">—</span></div>
    <div class="row"><span>last vram action</span><span id="vramact" class="muted">—</span></div>
    <div class="row"><span>detector last ran</span><span id="detector" class="muted">—</span></div>
    <div class="row"><span>admin note</span><span id="adminnote" class="muted">—</span></div>
  </section>
  <section class="span2">
    <h2>VRAM + DXGI budget (raw 10 Hz · no smoothing)</h2>
    <svg id="vramchart" class="chart" viewBox="0 0 1000 200" preserveAspectRatio="none"></svg>
    <div class="legend">
      <span><i class="sw" style="background:#6ea8ff"></i>DXGI budget (WDDM pressure)</span>
      <span><i class="sw" style="background:#3dd68c"></i>nvidia-smi used (physical)</span>
      <span><i class="sw" style="background:#f5c542;height:12px;width:2px"></i>engine / admin-vram events</span>
    </div>
  </section>
  <section>
    <h2>Recent requests</h2>
    <div class="row"><span>done (window)</span><span id="rdone" class="num">—</span></div>
    <div class="row"><span>mean TTFT</span><span id="ttft" class="num">—</span></div>
    <div class="row"><span>mean decode</span><span id="decode" class="num">—</span></div>
    <div class="row"><span>reuse mix</span><span id="reuse" class="num">—</span></div>
    <div class="row"><span>MTP (captured)</span><span id="mtp" class="num">—</span></div>
    <div class="row"><span>log</span><span id="lognote" class="muted">—</span></div>
  </section>
  <section id="logsec">
    <h2>Engine log tail</h2>
    <pre id="log">waiting…</pre>
  </section>
  <section class="span2">
    <h2>Insights</h2>
    <p class="muted" id="insrc">same objects as GET /api/insights</p>
    <div id="insights"></div>
  </section>
</main>
<script>
function gib(n){ if(!n) return "—"; return (n/1073741824).toFixed(2)+" GiB"; }
function pill(el, text, kind){ el.textContent=text; el.className="pill "+kind; }
function drawSeries(ser){
  const svg=document.getElementById("vramchart");
  if(!svg) return;
  const b=ser&&ser.budget_bytes||[];
  const u=ser&&ser.nvidia_used_bytes||[];
  const t=ser&&ser.t_ms||[];
  const n=b.length;
  if(!n){ svg.innerHTML=""; return; }
  const W=1000,H=200,p=18;
  let ymax=1;
  for(let i=0;i<n;i++) ymax=Math.max(ymax, b[i]||0, u[i]||0);
  const t0=t[0]||0, t1=t[n-1]||t0;
  const span=Math.max(t1-t0,1);
  const x=i=>p+(W-2*p)*i/Math.max(n-1,1);
  const y=v=>H-p-(H-2*p)*(v||0)/ymax;
  const poly=arr=>arr.map((v,i)=>x(i).toFixed(1)+","+y(v).toFixed(1)).join(" ");
  let ev="";
  (ser.events||[]).forEach(e=>{
    const xx=p+(W-2*p)*((e.t_ms-t0)/span);
    if(xx<p-1||xx>W-p+1) return;
    const col=e.kind&&e.kind.indexOf("admin")>=0?"#f5c542":"#ff5d5d";
    ev+=`<line x1="${xx.toFixed(1)}" x2="${xx.toFixed(1)}" y1="${p}" y2="${H-p}" stroke="${col}" stroke-width="1"/>`;
  });
  svg.innerHTML=`<polyline fill="none" stroke="#6ea8ff" stroke-width="1.4" points="${poly(b)}"/>`+
    `<polyline fill="none" stroke="#3dd68c" stroke-width="1.4" points="${poly(u)}"/>`+ev+
    `<text x="${p}" y="12" fill="#8b95a7" font-size="11">${gib(ymax)}</text>`+
    `<text x="${p}" y="${H-4}" fill="#8b95a7" font-size="11">0</text>`;
}
function apply(s){
  const st=s.engine||{};
  const map={Stopped:"warn",Starting:"warn",Running:"ok",Stopping:"warn",BackingOff:"warn",Halted:"bad"};
  pill(document.getElementById("state"), st.state||"?", map[st.state]||"warn");
  document.getElementById("mode").textContent = s.monitor_only
    ? "monitor-only · no spawn/stop · live SSE"
    : "loopback control surface · live SSE";
  document.getElementById("actions").style.display = s.monitor_only ? "none" : "flex";
  document.querySelectorAll(".owned-only").forEach(el=>{
    el.style.display = s.monitor_only ? "none" : "";
  });
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
  const vc=s.vram_control||{};
  document.getElementById("released").textContent = vc.any_released ? "YES — seed store degraded" : "no";
  document.getElementById("sincerel").textContent =
    vc.since_release_s!=null ? vc.since_release_s+" s" : "—";
  document.getElementById("vramact").textContent =
    ((vc.last_transition||"")+" "+(vc.last_reason||"")).trim()||"—";
  document.getElementById("detector").textContent =
    vc.detector_last_ran_ms
      ? (vc.detector_age_s==0 ? "now" : vc.detector_age_s+" s ago")
      : "never — quiet is not the same as nothing happened";
  const r=s.requests||{};
  document.getElementById("rdone").textContent = r.done!=null?r.done:"—";
  document.getElementById("ttft").textContent = r.ttft_ms_mean? r.ttft_ms_mean.toFixed(0)+" ms":"—";
  document.getElementById("decode").textContent = r.decode_tok_s_mean? r.decode_tok_s_mean.toFixed(1)+" tok/s":"—";
  document.getElementById("reuse").textContent = "reset "+(r.reuse_full_reset||0)+" · append "+(r.reuse_append||0)+" · seed/restore "+(r.reuse_seed||0);
  if(r.mtp_drafted){
    const pct=((r.mtp_last_accept_rate||0)*100).toFixed(0);
    const pos=(r.mtp_accepted_per_position||[]).join(",");
    document.getElementById("mtp").textContent =
      pct+"% "+(r.mtp_accepted||0)+"/"+(r.mtp_drafted||0)+
      " · fallback "+(r.mtp_fallback_steps||0)+
      (pos? " · pos ["+pos+"]":"");
  } else { document.getElementById("mtp").textContent = "—"; }
  document.getElementById("lognote").textContent = r.log_available? "ok" : (r.log_error||"not configured");
  if(s.monitor_only){
    document.getElementById("log").textContent =
      "unmanaged engine: no child stdout. This is not an empty log.";
  } else {
    document.getElementById("log").textContent = s.log_tail||"";
  }
  const rep=s.insights||{};
  const src=rep.source||{};
  document.getElementById("insrc").textContent =
    "GET /api/insights · source "+(src.request_log||"?")+(src.path? " · "+src.path:"");
  const box=document.getElementById("insights");
  box.innerHTML="";
  (rep.insights||[]).forEach(it=>{
    const d=document.createElement("div");
    const avail=it.availability||"available";
    d.className="finding "+avail+" "+(it.severity||"");
    const over=it.measured_over||{};
    d.innerHTML = "<div class='title'>"+esc(it.title||it.id)+"</div>"+
      "<div class='stmt'>"+esc(it.statement||"")+"</div>"+
      "<div class='meta'>"+esc(it.id)+" · "+esc(avail)+" · "+esc(it.confidence||"")+
      " · measured_over requests="+(over.requests!=null?over.requests:"?")+
      (it.recommendation? " · "+esc(it.recommendation):"")+"</div>"+
      "<pre>"+esc(JSON.stringify(it.evidence||{},null,2))+"</pre>";
    box.appendChild(d);
  });
  drawSeries(s.series);
}
function esc(t){ return String(t==null?"":t).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c])); }
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
