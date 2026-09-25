/* Negotiation replay — shared by the main dashboard and the model-duel page.
   Expects globals: $, $$, fmt, signed, mean, cssVar, esc (defined here), tipOf,
   SCEN, META, uPlot, and fidPlot (null when the page has no fidelity curve). */
/* ---------- replay ---------- */
// Optional page hook: window.RFQ_PARTY_LABEL(e, party) names a side (the duel page
// shows the model); without it the role name is shown.
function partyLabel(e, p){
  return typeof window.RFQ_PARTY_LABEL === "function" ? window.RFQ_PARTY_LABEL(e, p) : p;
}
let uplots = [];
function destroyPlots(){
  // Replay plots only; the fidelity curve lives outside the replay card.
  uplots.filter(p=>p!==fidPlot).forEach(p=>{try{p.destroy()}catch(e){}});
  uplots = fidPlot ? [fidPlot] : [];
}
// uPlot renders at a fixed pixel width; without this a window resize leaves the
// canvas wider than its (now narrower) column and overflows the replay card.
let _plotResizeT = null;
window.addEventListener("resize", () => {
  clearTimeout(_plotResizeT);
  _plotResizeT = setTimeout(() => {
    uplots.forEach(u => {
      const host = u.root && u.root.parentElement;
      const w = host ? host.clientWidth : 0;
      if (w) { try { u.setSize({ width: Math.max(260, w), height: u.height }); } catch (e) {} }
    });
  }, 120);
});
function renderReplay(e){
  destroyPlots();
  const sc = SCEN[e.scenario_id];
  $("#replayTitle").textContent =
    `${e.strategy} vs ${e.opponent} · ${e.scenario_id} · agent = ${e.target_role} · opens ${e.first_speaker} · seed ${e.seed}`;
  $("#replayTitle").title = [
    [e.strategy, tipOf("strategy", e.strategy)],
    [e.opponent, tipOf("opponent", e.opponent)],
    [e.scenario_id, tipOf("scenario", e.scenario_id)],
  ].filter(([, t]) => t).map(([n, t]) => `${n}: ${t}`).join("\n");
  const body = $("#replayBody");
  body.innerHTML = `
    <div class="replay-grid">
      <div>
        <h3>Offers by round</h3>
        <div id="offerPlots"></div>
        <h3 style="margin-top:14px">Utility trajectory</h3>
        <div class="uholder" id="utilPlot"></div>
        <div class="transport">
          <button class="btn" id="playBtn">▶</button>
          <input type="range" id="scrub" min="0" max="${e.steps.length-1}" value="${e.steps.length-1}" />
          <span class="mono" id="scrubLbl"></span>
        </div>
        <div class="nowmsg" id="nowMsg"></div>
      </div>
      <div>
        <h3>Outcome</h3>
        <div class="kv" id="outcomeKV"></div>
        <div class="qform" id="qform"></div>
        <h3 style="margin-top:12px" id="logTitle">Action log</h3>
        <div id="overrideBanner"></div>
        <div class="log"><table><tbody id="logBody"></tbody></table></div>
      </div>
    </div>`;
  // outcome KV
  const prof = sc.parties[e.target_role];
  const U = e.utilities[e.target_role];
  const kv = $("#outcomeKV");
  const agr = e.agreement ? Object.entries(e.agreement).map(([k,v])=>`${k}=${v}`).join(", ") : "—";
  kv.innerHTML = `
    <div class="k">outcome</div><div class="v"><span class="badge ${e.outcome_kind}">${e.outcome_kind}</span></div>
    <div class="k">agreement</div><div class="v mono">${agr}</div>
    <div class="k">u · buyer</div><div class="v">${fmt(e.utilities.buyer,3)}</div>
    <div class="k">u · seller</div><div class="v">${fmt(e.utilities.seller,3)}</div>
    <div class="k">rounds to close</div><div class="v">${e.rounds_to_close}</div>
    <div class="k">latency</div><div class="v">${(e.latency_s*1000).toFixed(2)} ms</div>
    <div class="k">token cost</div><div class="v">${e.token_cost}</div>
    <div class="k">cost</div><div class="v">${e.cost_usd!=null ? "$"+e.cost_usd.toFixed(5)+" <span class=\"muted\">(real)</span>" : "<span class=\"muted\">not reported</span>"}</div>`
    + fidelityKV(e);
  // q formula
  const I=prof.ideal, d=prof.batna;
  const q = e.degenerate? null : Math.max(0, Math.min(1,(U-d)/(I-d)));
  $("#qform").innerHTML = e.no_zopa
    ? `no ZOPA: no outcome beats both BATNAs, so the correct result is a walk-away (q would be 0 for any strategy) → excluded from S_s, reported as walk✓ `
      + `(${e.validity && e.validity.correct_walk_away ? "correct walk-away ✓" : "no correct walk-away ✕"})`
    : e.degenerate
    ? `degenerate scenario (I = d = ${fmt(d,3)}) → q undefined, excluded`
    : `q = clip((U − d)/(I − d), 0, 1)<br>= clip((${fmt(U,3)} − ${fmt(d,3)})/(${fmt(I,3)} − ${fmt(d,3)}), 0, 1)`
      + ` = <b>${fmt(q,3)}</b> &nbsp;→&nbsp; S contribution ${fmt(100*q,1)}`;

  buildOfferPlots(e, sc);
  buildUtilPlot(e, sc);
  buildLog(e);
  const scrub=$("#scrub");
  scrub.oninput = () => setCursor(e, +scrub.value);
  let playing=null;
  $("#playBtn").onclick = () => {
    if (playing){ clearInterval(playing); playing=null; $("#playBtn").textContent="▶"; return; }
    $("#playBtn").textContent="⏸";
    if (+scrub.value>=e.steps.length-1) scrub.value=0;
    playing=setInterval(()=>{
      let v=+scrub.value; if (v>=e.steps.length-1){ clearInterval(playing); playing=null; $("#playBtn").textContent="▶"; return; }
      scrub.value=v+1; setCursor(e, v+1);
    }, 500);
  };
  setCursor(e, e.steps.length-1);
}

function fidelityKV(e){
  const f = e.fidelity;
  if (!f) return `<div class="k">strategy fidelity</div><div class="v muted">not checkable</div>`;
  const gaps = f.offers.map(o=>o.gap);
  const open = gaps.length ? signed(gaps[0],2) : "—";
  const bias = gaps.length ? signed(mean(gaps),2) : "—";
  const b = f.beta==null ? "—" : `${f.beta.toFixed(2)} <span class="muted">(ref ${f.ref_beta==null?"—":f.ref_beta.toFixed(2)})</span>`;
  return `<div class="k" title="agent offer − the strategy's offer, share of the BATNA→ideal range">fidelity · open / bias</div><div class="v">${open} / ${bias}</div>
    <div class="k" title="accepts the strategy would not make / accepts it would have made">fidelity · early✗ / missed✗</div><div class="v">${f.early} / ${f.missed} <span class="muted">of ${f.decisions}</span></div>
    <div class="k">fidelity · β fit</div><div class="v">${b}</div>`;
}

function stepSeries(e, sc){
  // Build per-issue offer series for buyer and seller across rounds.
  const rounds = [...new Set(e.steps.map(s=>s.round))].sort((a,b)=>a-b);
  const xs = rounds;
  const issues = sc.issues.map(i=>i.name);
  const series = {}; // issue -> {buyer:[], seller:[]}
  for (const iss of issues) series[iss] = {buyer:xs.map(()=>null), seller:xs.map(()=>null)};
  for (const s of e.steps){
    if (!s.outcome) continue;
    const xi = rounds.indexOf(s.round);
    for (const iss of issues){
      if (iss in s.outcome) series[iss][s.party][xi] = s.outcome[iss];
    }
  }
  // The tested agent's reference: the offer its strategy would have made each turn.
  const ref = {}; for (const iss of issues) ref[iss] = xs.map(()=>null);
  for (const o of (e.fidelity ? e.fidelity.offers : [])){
    const xi = rounds.indexOf(o.round);
    for (const iss of issues) if (xi>=0 && iss in o.ref_outcome) ref[iss][xi] = o.ref_outcome[iss];
  }
  return {xs, issues, series, ref};
}
let OFFER_CTX = null;
function buildOfferPlots(e, sc){
  const host=$("#offerPlots"); host.innerHTML="";
  const {xs, issues, series, ref} = stepSeries(e, sc);
  OFFER_CTX = {xs, issues, series, ref, plots:[]};
  const buyerC=cssVar("--accent"), sellerC=cssVar("--warn"), refC=cssVar("--muted");
  issues.forEach((iss, idx)=>{
    const div=document.createElement("div"); div.className="uholder"; div.style.marginTop = idx?"10px":"0";
    const cap=document.createElement("div"); cap.className="hint"; cap.textContent = iss + (sc.issues[idx].unit?` (${sc.issues[idx].unit})`:"");
    host.appendChild(cap); host.appendChild(div);
    const issueMeta = sc.issues[idx];
    const zopaVals = zopaRange(sc, iss);
    const opts = baseOpts(div.clientWidth||520, 140, {
      scales:{x:{time:false}, y:{range:(u,min,max)=>{
        const vals=issueMeta.values; return [Math.min(...vals), Math.max(...vals)];
      }}},
      series:[{}, mkSeries("buyer", buyerC), mkSeries("seller", sellerC),
        Object.assign(mkSeries("strategy ref", refC), {dash:[6,4]})],
      hooks: zopaVals ? {draw:[u=>drawBand(u, zopaVals[0], zopaVals[1])]} : {},
    });
    const data=[xs, series[iss].buyer, series[iss].seller, ref[iss]];
    const u=new uPlot(opts, data, div); uplots.push(u); OFFER_CTX.plots.push(u);
  });
  $("#offerPlots").insertAdjacentHTML("beforeend",
    `<div class="legend"><span style="--sw:${buyerC}">buyer offer</span><span style="--sw:${sellerC}">seller offer</span>${e.fidelity?`<span class="dash" style="--sw:${refC}">${e.target_role}'s strategy (${esc(e.strategy)}) would offer</span>`:""}<span style="--sw:var(--line-strong)">ZOPA band</span></div>`);
}
function mkSeries(label, color){
  return {label, stroke:color, width:2, points:{show:true, size:5, stroke:color, fill:color},
    spanGaps:true};
}
function zopaRange(sc, issue){
  if (!sc.zopa_outcomes || !sc.zopa_outcomes.length) return null;
  const vals = sc.zopa_outcomes.map(o=>o[issue]).filter(v=>v!=null);
  if (!vals.length) return null;
  return [Math.min(...vals), Math.max(...vals)];
}
function drawBand(u, lo, hi){
  const ctx=u.ctx; const x0=u.bbox.left, x1=u.bbox.left+u.bbox.width;
  const yLo=u.valToPos(lo, "y", true), yHi=u.valToPos(hi, "y", true);
  ctx.save(); ctx.fillStyle = cssVar("--accent-soft"); ctx.globalAlpha=0.7;
  ctx.fillRect(x0, Math.min(yLo,yHi), x1-x0, Math.abs(yHi-yLo)); ctx.restore();
}

let UTIL_CTX=null;
function buildUtilPlot(e, sc){
  const div=$("#utilPlot"); div.innerHTML="";
  const {xs} = stepSeries(e, sc);
  // running utility for each party from step utilities (last offer each round)
  const bu=xs.map(()=>null), su=xs.map(()=>null);
  const rounds=xs;
  for (const s of e.steps){
    if (!s.utilities) continue;
    const xi=rounds.indexOf(s.round);
    // record the acting party's utility of what's on the table
    bu[xi] = s.utilities.buyer; su[xi]=s.utilities.seller;
  }
  const buyerC=cssVar("--accent"), sellerC=cssVar("--warn"), refC=cssVar("--muted");
  const db=sc.parties.buyer.batna, ds=sc.parties.seller.batna;
  // The agent's reference offer, on the agent's own utility scale.
  const ag=sc.parties[e.target_role], ru=xs.map(()=>null);
  for (const o of (e.fidelity ? e.fidelity.offers : [])){
    const xi=rounds.indexOf(o.round); if (xi>=0) ru[xi] = ag.batna + o.ref_level*(ag.ideal-ag.batna);
  }
  const opts=baseOpts(div.clientWidth||520, 150, {
    scales:{x:{time:false}, y:{range:[0,1]}},
    series:[{}, mkSeries("u·buyer",buyerC), mkSeries("u·seller",sellerC),
      Object.assign(mkSeries("strategy ref",refC), {dash:[6,4]})],
    hooks:{draw:[u=>{ drawHLine(u, db, buyerC); drawHLine(u, ds, sellerC); }]},
  });
  const u=new uPlot(opts, [xs, bu, su, ru], div); uplots.push(u); UTIL_CTX={u, xs, ru};
  div.insertAdjacentHTML("afterend",
    `<div class="legend"><span style="--sw:${buyerC}">u·buyer</span><span style="--sw:${sellerC}">u·seller</span>${e.fidelity?`<span class="dash" style="--sw:${refC}">u·${e.target_role} of the strategy's offer</span>`:""}<span style="--sw:var(--muted)">thin dashed = BATNA</span></div>`);
}
function drawHLine(u, y, color){
  const ctx=u.ctx, x0=u.bbox.left, x1=u.bbox.left+u.bbox.width, yy=u.valToPos(y,"y",true);
  ctx.save(); ctx.strokeStyle=color; ctx.globalAlpha=.45; ctx.setLineDash([4,4]); ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(x0,yy); ctx.lineTo(x1,yy); ctx.stroke(); ctx.restore();
}

function baseOpts(w, h, extra){
  const grid={stroke:cssVar("--line"), width:1};
  const ax={stroke:cssVar("--muted"), grid, ticks:grid, font:"11px system-ui",
    size:34, labelSize:0};
  return Object.assign({
    width:Math.max(260,w), height:h, cursor:{y:false},
    legend:{show:false},
    axes:[Object.assign({}, ax, {size:24}), ax],
    scales:{x:{time:false}},
  }, extra);
}

function esc(s){ return String(s).replace(/[&<>"]/g, c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }
function fmtOutcome(o){
  return o ? Object.entries(o).map(([k,v])=>`${k}=${v}`).join(" ") : "—";
}
function adjustHtml(s){
  // What was adjusted (snap / shop floor) or errored, and why. Never a scripted
  // strategy substitution — the model's own decision, faithfully corrected.
  if (s.error){
    const wanted = s.intended_outcome ? ` (model sent ${esc(fmtOutcome(s.intended_outcome))})` : "";
    return `<div class="override err"><b>ERROR — episode excluded:</b> ${esc(s.error_reason||'unusable model reply')}${wanted}</div>`;
  }
  if (!s.adjusted) return "";
  const reason = s.adjust_reason || "adjusted";
  const wanted = s.intended_action
    ? `${s.intended_action}${s.intended_outcome?` · ${fmtOutcome(s.intended_outcome)}`:""}`
    : "";
  const played = `${s.action}${s.outcome?` · ${fmtOutcome(s.outcome)}`:""}`;
  let html = `<div class="override"><b>adjusted:</b> ${esc(reason)}`;
  if (wanted) html += `<div>model tried <span class="mono">${esc(wanted)}</span></div>`;
  html += `<div class="played">played <span class="mono">${esc(played)}</span></div></div>`;
  return html;
}
function fullResponseHtml(s){
  // Collapsible chain-of-thought and verbatim reply, so you can verify exactly
  // what the model produced this turn.
  let html = "";
  if (s.reasoning){
    html += `<details class="llm-detail"><summary>thinking (${s.reasoning.length.toLocaleString()} chars)</summary>`
      + `<pre class="llm-pre">${esc(s.reasoning)}</pre></details>`;
  }
  if (s.raw_response){
    html += `<details class="llm-detail"><summary>full response</summary>`
      + `<pre class="llm-pre">${esc(s.raw_response)}</pre></details>`;
  }
  return html;
}
function stepTag(s){
  if (s.error) return `<span class="fallback-tag" title="${esc(s.error_reason||'unusable reply')}">error</span>`;
  if (s.adjusted) return `<span class="fallback-tag" title="${esc(s.adjust_reason||'adjusted')}">adjusted</span>`;
  return "";
}
// What an agent said with a move: the public `message` (A2A channel — the other
// side read it) in quotes, and the private `rationale` (never shown to anyone
// but you) as a dimmer, labelled line.
function talkHtml(s){
  let h = "";
  if (s.message && s.message_withheld){
    h += `<div class="withheld"><b>withheld — not sent to the opponent</b> (a policy constraint fired on this move): <span class="say">${esc(s.message)}</span></div>`;
  } else if (s.message){
    const cut = s.message_truncated ? ` <span class="private" title="the other side received only the first part">(truncated on delivery)</span>` : "";
    h += `<div class="say">${esc(s.message)}</div>${cut}`;
  }
  if (s.rationale) h += `<div class="private"><b>private:</b> ${esc(s.rationale)}</div>`;
  return h;
}
function buildLog(e){
  const hasMsg = e.steps.some(s=>s.message || s.rationale);
  $("#logTitle").textContent = hasMsg ? "Action log & agent messages" : "Action log";
  const nAdj = e.steps.filter(s=>s.adjusted).length;
  let banner = "";
  if (e.errored) banner += `<div class="override-banner">⚠ episode EXCLUDED from scoring — the agent produced an unusable reply: ${esc(e.error_reason||'error')}.</div>`;
  const nWithheld = e.steps.filter(s=>s.message_withheld).length;
  if (nAdj){
    const kinds = {};
    e.steps.filter(s=>s.adjusted).forEach(s=>{ const k=s.adjust_kind||"other"; kinds[k]=(kinds[k]||0)+1; });
    const kindTxt = Object.entries(kinds).map(([k,n])=>`${k} ${n}`).join(", ");
    banner += `<div class="override-banner"><b>Policy constraints fired:</b> ${kindTxt}${nWithheld?` · ${nWithheld} message${nWithheld>1?"s":""} withheld from the opponent`:""}.</div>`;
  }
  if (nAdj) banner += `<div class="override-banner">${nAdj} of ${e.steps.length} moves were adjusted — an off-grid value snapped to the nearest legal tier, or a below-floor quote clamped up to the shop's reservation floor. No scripted values are ever substituted. Highlighted below.</div>`;
  $("#overrideBanner").innerHTML = banner;
  const tb=$("#logBody"); tb.innerHTML="";
  e.steps.forEach((s,i)=>{
    const tr=document.createElement("tr"); tr.dataset.i=i;
    const oc = s.outcome? Object.entries(s.outcome).map(([k,v])=>`${k}=${v}`).join(" "):"";
    const tag = stepTag(s);
    tr.innerHTML=`<td class="muted">${s.round}</td><td>${esc(partyLabel(e, s.party))}</td>
      <td><b>${s.action}</b>${tag}</td><td><span class="mono">${oc}</span>${talkHtml(s)}${adjustHtml(s)}</td>`;
    tr.onclick=()=>{ $("#scrub").value=i; setCursor(e,i); };
    tb.appendChild(tr);
  });
}
function setCursor(e, idx){
  const cur = e.steps[idx];
  $("#scrubLbl").textContent = `step ${idx+1}/${e.steps.length} · round ${cur.round}`;
  // current-turn message bubble
  const nm = $("#nowMsg");
  if (nm){
    const who = cur.party;
    const tag = stepTag(cur);
    const over = adjustHtml(cur);
    const full = fullResponseHtml(cur);
    if (cur.message || cur.rationale){
      nm.innerHTML = `<span class="who ${who}">${esc(partyLabel(e, who))}</span>
        <div class="bubble"><div class="act">${cur.action}${cur.outcome?` · ${Object.entries(cur.outcome).map(([k,v])=>k+"="+v).join(" ")}`:""} ${tag}</div>${talkHtml(cur)}${over}${full}</div>`;
    } else if (full){
      // No one-line rationale, but the model did reply (e.g. a malformed answer
      // that fell back) — show the raw reply/thinking so it can be inspected.
      nm.innerHTML = `<span class="who ${who}">${esc(partyLabel(e, who))}</span>
        <div class="bubble"><div class="act">${cur.action} ${tag}</div>${over}${full}</div>`;
    } else {
      const why = META.has_messages ? "no message this turn" : "scripted policy — no agent message";
      nm.innerHTML = `<span class="who ${who}">${esc(partyLabel(e, who))}</span><div class="bubble none">${why} · <b>${cur.action}</b>${tag}${over}</div>`;
    }
  }
  $$("#logBody tr").forEach(tr=>{
    const i=+tr.dataset.i; tr.className = i<idx?"past":i===idx?"now":"future";
    if (i===idx) tr.scrollIntoView({block:"nearest"});
  });
  // truncate uPlot data to idx (by round position)
  if (OFFER_CTX){
    const cutRound = e.steps[idx].round;
    const upto = OFFER_CTX.xs.map(r=>r<=cutRound);
    OFFER_CTX.issues.forEach((iss,pi)=>{
      const b=OFFER_CTX.series[iss].buyer.map((v,j)=>upto[j]?v:null);
      const s=OFFER_CTX.series[iss].seller.map((v,j)=>upto[j]?v:null);
      const r=OFFER_CTX.ref[iss].map((v,j)=>upto[j]?v:null);
      OFFER_CTX.plots[pi].setData([OFFER_CTX.xs, b, s, r]);
    });
  }
  if (UTIL_CTX){
    // rebuild util arrays up to cursor
    const cutRound=e.steps[idx].round;
    const bu=UTIL_CTX.xs.map(()=>null), su=UTIL_CTX.xs.map(()=>null);
    for (const s of e.steps){ if (s.round>cutRound || !s.utilities) continue;
      const xi=UTIL_CTX.xs.indexOf(s.round); bu[xi]=s.utilities.buyer; su[xi]=s.utilities.seller; }
    const ru=UTIL_CTX.ru.map((v,j)=>UTIL_CTX.xs[j]<=cutRound?v:null);
    UTIL_CTX.u.setData([UTIL_CTX.xs, bu, su, ru]);
  }
}

