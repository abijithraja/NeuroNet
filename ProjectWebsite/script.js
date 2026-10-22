/**
 * NEURONET — Interactive Workbench Engine
 * 20-node topology canvas + scenario simulation controller
 */

// ==========================================================================
// 1. SCENARIO DATA
// ==========================================================================
const SCENARIO_DATA = {
  congestion: {
    name: "Buffer Congestion Ramp",
    tag: "High Traffic Load",
    tagStyle: "background:rgba(245,158,11,0.1);color:#f59e0b;border-color:rgba(245,158,11,0.25)",
    description: "Link demand spikes to 120 % capacity. FIFO queue saturates; latency climbs +95 ms.",
    impactedNodes: [4, 5, 8, 9],
    impactedLinks: [[4,5],[5,8],[8,9]],
    recoveryLinks: [[4,12],[12,15],[15,9]],
    before: { latency:115, loss:5.2, throughput:48, congestion:0.92 },
    after:  { latency:22,  loss:0.2, throughput:92, congestion:0.18 },
    anomaly: { score:0.74, lstm:0.78, iso:0.68 },
    rl: { action:"REROUTE", confidence:88.4, qValue:"+38.5" },
    safety: { approved:true, status:"APPROVED", reason:"Confidence 88.4 % >= 75.0 % threshold. Zero thrashing risk." },
    shap: [
      { name:"Latency",    pct:48, color:"var(--rose)" },
      { name:"Congestion", pct:32, color:"var(--amber)" },
      { name:"Packet loss",pct:14, color:"var(--cyan)" },
      { name:"CPU load",   pct:6,  color:"var(--indigo)" }
    ],
    rationale: "Dueling DQN identified primary trunk buffer saturation (92 %). ECMP reroute dispatched via OpenFlow; latency recovered 93 ms, packet loss reduced 5.0 %."
  },
  link_failure: {
    name: "Physical Link Breakdown",
    tag: "Hardware Outage",
    tagStyle: "background:rgba(244,63,94,0.1);color:#f43f5e;border-color:rgba(244,63,94,0.25)",
    description: "Fibre cut on core inter-switch trunk. Throughput collapses to 3 Mbps; severe packet blackhole.",
    impactedNodes: [6,7],
    impactedLinks: [[6,7]],
    recoveryLinks: [[6,14],[14,7]],
    before: { latency:98, loss:9.4, throughput:3, congestion:0.88 },
    after:  { latency:24, loss:0.3, throughput:86, congestion:0.22 },
    anomaly: { score:0.86, lstm:0.89, iso:0.82 },
    rl: { action:"FAILOVER", confidence:92.1, qValue:"+42.1" },
    safety: { approved:true, status:"APPROVED", reason:"Critical link health < 0.20. Instant failover sanctioned." },
    shap: [
      { name:"Packet loss",  pct:46, color:"var(--rose)" },
      { name:"Link health",  pct:34, color:"var(--amber)" },
      { name:"Latency",      pct:14, color:"var(--cyan)" },
      { name:"Throughput",   pct:6,  color:"var(--indigo)" }
    ],
    rationale: "Interface health dropped to 12 %. Agent triggered hardware link failover to hot-standby interface, restoring 86 Mbps throughput in 5 ms."
  },
  attack_ddos: {
    name: "Volumetric DDoS SYN Flood",
    tag: "Security Disruption",
    tagStyle: "background:rgba(244,63,94,0.1);color:#f43f5e;border-color:rgba(244,63,94,0.25)",
    description: "Distributed SYN flood saturates line cards. Router CPU spikes to 97 %; extreme buffer exhaustion.",
    impactedNodes: [1,2,3,4],
    impactedLinks: [[1,2],[2,3],[3,4]],
    recoveryLinks: [[1,10],[10,11],[11,4]],
    before: { latency:142, loss:8.6, throughput:99, congestion:0.98 },
    after:  { latency:28,  loss:0.4, throughput:72, congestion:0.24 },
    anomaly: { score:0.92, lstm:0.95, iso:0.88 },
    rl: { action:"THROTTLE", confidence:86.8, qValue:"+35.4" },
    safety: { approved:true, status:"APPROVED", reason:"Autonomous rate-limiting approved to protect router control plane." },
    shap: [
      { name:"CPU utilisation", pct:44, color:"var(--rose)" },
      { name:"Packet loss",     pct:28, color:"var(--amber)" },
      { name:"Buffer load",     pct:18, color:"var(--cyan)" },
      { name:"Latency",         pct:10, color:"var(--indigo)" }
    ],
    rationale: "Router CPU reached 97 % saturation. Agent throttled low-priority background slice traffic, reducing CPU to 34 % and restoring critical queue flow."
  },
  routing_flap: {
    name: "BGP / OSPF Route Flapping",
    tag: "Routing Instability",
    tagStyle: "background:rgba(129,140,248,0.1);color:#818cf8;border-color:rgba(129,140,248,0.25)",
    description: "Route flapping causes path loops and extreme jitter. Telemetry variance creates diagnostic uncertainty.",
    impactedNodes: [10,11,12],
    impactedLinks: [[10,11],[11,12]],
    recoveryLinks: [],
    before: { latency:75, loss:3.4, throughput:55, congestion:0.65 },
    after:  { latency:75, loss:3.4, throughput:55, congestion:0.65 },
    anomaly: { score:0.58, lstm:0.62, iso:0.52 },
    rl: { action:"QOS_ADJUST", confidence:62.4, qValue:"+12.1" },
    safety: { approved:false, status:"ESCALATED", reason:"Confidence 62.4 % below required 75.0 % threshold. Escalated to Human NOC." },
    shap: [
      { name:"Latency jitter", pct:52, color:"var(--amber)" },
      { name:"Loss variance",  pct:28, color:"var(--cyan)" },
      { name:"Buffer variance",pct:20, color:"var(--indigo)" }
    ],
    rationale: "RL confidence (62.4 %) failed to clear the 75 % safety gate due to path instability. System deferred automated execution and escalated to the human NOC operator."
  },
  degradation: {
    name: "Optical Transponder Degradation",
    tag: "Physical Wear",
    tagStyle: "background:rgba(56,189,248,0.1);color:#38bdf8;border-color:rgba(56,189,248,0.25)",
    description: "Progressive laser ageing introduces creeping bit errors and micro-burst packet loss.",
    impactedNodes: [13,14,15],
    impactedLinks: [[13,14],[14,15]],
    recoveryLinks: [[13,17],[17,15]],
    before: { latency:46, loss:2.8, throughput:68, congestion:0.45 },
    after:  { latency:26, loss:0.4, throughput:75, congestion:0.20 },
    anomaly: { score:0.72, lstm:0.75, iso:0.68 },
    rl: { action:"TRAFFIC_SHAPE", confidence:79.5, qValue:"+28.2" },
    safety: { approved:true, status:"APPROVED", reason:"Confidence 79.5 % >= 75.0 %. Action cleared for dispatch." },
    shap: [
      { name:"Bit-error loss", pct:45, color:"var(--rose)" },
      { name:"Latency shift",  pct:30, color:"var(--amber)" },
      { name:"Congestion",     pct:25, color:"var(--cyan)" }
    ],
    rationale: "Token-bucket shaping smoothed burst transmission rates, preventing transceiver buffer drops while maintenance was scheduled."
  }
};

let currentScenarioKey = "congestion";
let isSimulating = false;

// ==========================================================================
// 2. 20-NODE TOPOLOGY CANVAS
// ==========================================================================
class TopologySimulator {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');
    this.nodes = [];
    this.links = [];
    this.particles = [];
    this.hoveredNode = null;
    this.draggedNode = null;
    this.activeScenario = SCENARIO_DATA[currentScenarioKey];
    this.initCanvasSize();
    this.generateTopology();
    this.attachEvents();
    this.animate();
  }

  initCanvasSize() {
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width  = rect.width  * window.devicePixelRatio;
    this.canvas.height = rect.height * window.devicePixelRatio;
    this.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    this.width  = rect.width;
    this.height = rect.height;
  }

  generateTopology() {
    this.nodes = [];
    this.links = [];
    const cx = this.width / 2;
    const cy = this.height / 2;

    const tiers = [
      { count:4,  radius:this.height*0.17, type:"Core",        color:"#38bdf8", size:9 },
      { count:6,  radius:this.height*0.31, type:"Aggregation", color:"#818cf8", size:7 },
      { count:10, radius:this.height*0.43, type:"Edge",        color:"#34d399", size:5.5 }
    ];

    let id = 0;
    tiers.forEach(tier => {
      for (let i = 0; i < tier.count; i++) {
        const angle = (i / tier.count) * Math.PI * 2 + (tier.count % 2 ? 0.3 : 0);
        const x = cx + Math.cos(angle) * tier.radius;
        const y = cy + Math.sin(angle) * tier.radius;
        this.nodes.push({ id: id++, x, y, radius: tier.size, type: tier.type, baseColor: tier.color, pulse: Math.random() * Math.PI });
      }
    });

    // Core full mesh (0-3)
    for (let i = 0; i < 4; i++)
      for (let j = i+1; j < 4; j++)
        this.links.push({ u:i, v:j });

    // Aggregation (4-9) to Core
    for (let i = 4; i < 10; i++) {
      this.links.push({ u:i, v:i % 4 });
      this.links.push({ u:i, v:(i+1) % 4 });
    }

    // Edge (10-19) to Aggregation
    for (let i = 10; i < 20; i++) {
      this.links.push({ u:i, v: 4 + (i % 6) });
      this.links.push({ u:i, v: 4 + ((i+1) % 6) });
    }

    // Packet particles
    this.particles = [];
    for (let i = 0; i < 32; i++) {
      const link = this.links[Math.floor(Math.random() * this.links.length)];
      this.particles.push({ link, progress: Math.random(), speed: 0.004 + Math.random() * 0.006 });
    }
  }

  setScenario(key) { this.activeScenario = SCENARIO_DATA[key]; }

  attachEvents() {
    window.addEventListener('resize', () => { this.initCanvasSize(); this.generateTopology(); });

    this.canvas.addEventListener('mousemove', e => {
      const rect = this.canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      let found = null;
      for (const n of this.nodes)
        if (Math.hypot(n.x - mx, n.y - my) < n.radius + 7) { found = n; break; }
      this.hoveredNode = found;
      this.canvas.style.cursor = found ? 'pointer' : 'default';
      if (this.draggedNode) { this.draggedNode.x = mx; this.draggedNode.y = my; }
    });

    this.canvas.addEventListener('mousedown', () => { if (this.hoveredNode) this.draggedNode = this.hoveredNode; });
    window.addEventListener('mouseup', () => { this.draggedNode = null; });
  }

  animate() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.width, this.height);

    const scen = this.activeScenario;
    const impacted  = scen ? scen.impactedLinks  : [];
    const recovery  = scen ? scen.recoveryLinks  : [];
    const impactSet = new Set(scen ? scen.impactedNodes : []);

    // Links
    this.links.forEach(link => {
      const u = this.nodes[link.u];
      const v = this.nodes[link.v];
      let isImpacted = false, isRecovery = false;
      for (const p of impacted)
        if ((p[0]===link.u&&p[1]===link.v)||(p[0]===link.v&&p[1]===link.u)) { isImpacted=true; break; }
      for (const p of recovery)
        if ((p[0]===link.u&&p[1]===link.v)||(p[0]===link.v&&p[1]===link.u)) { isRecovery=true; break; }

      ctx.beginPath();
      ctx.moveTo(u.x, u.y);
      ctx.lineTo(v.x, v.y);
      if (isImpacted) {
        ctx.strokeStyle = "rgba(244,63,94,0.65)";
        ctx.lineWidth = 2.2; ctx.setLineDash([4,4]);
      } else if (isRecovery) {
        ctx.strokeStyle = "rgba(52,211,153,0.7)";
        ctx.lineWidth = 2.0; ctx.setLineDash([]);
      } else {
        ctx.strokeStyle = "rgba(255,255,255,0.07)";
        ctx.lineWidth = 1.0; ctx.setLineDash([]);
      }
      ctx.stroke();
    });
    ctx.setLineDash([]);

    // Particles
    this.particles.forEach(p => {
      p.progress += p.speed;
      if (p.progress >= 1) p.progress = 0;
      const u = this.nodes[p.link.u];
      const v = this.nodes[p.link.v];
      const px = u.x + (v.x - u.x) * p.progress;
      const py = u.y + (v.y - u.y) * p.progress;
      ctx.beginPath();
      ctx.arc(px, py, 1.8, 0, Math.PI*2);
      ctx.fillStyle = "#38bdf8";
      ctx.shadowBlur = 5; ctx.shadowColor = "#38bdf8";
      ctx.fill();
      ctx.shadowBlur = 0;
    });

    // Nodes
    this.nodes.forEach(n => {
      n.pulse += 0.03;
      const faulty = impactSet.has(n.id);
      const color  = faulty ? "#f43f5e" : n.baseColor;
      const aura   = n.radius + Math.sin(n.pulse) * 2.5 + (faulty ? 3 : 0.5);

      ctx.beginPath();
      ctx.arc(n.x, n.y, aura, 0, Math.PI*2);
      ctx.fillStyle = faulty ? "rgba(244,63,94,0.15)" : "rgba(56,189,248,0.08)";
      ctx.fill();

      ctx.beginPath();
      ctx.arc(n.x, n.y, n.radius, 0, Math.PI*2);
      ctx.fillStyle = color;
      ctx.shadowBlur = faulty ? 12 : 7;
      ctx.shadowColor = color;
      ctx.fill();
      ctx.shadowBlur = 0;

      ctx.fillStyle = "#ffffff";
      ctx.font = "8px 'JetBrains Mono', monospace";
      ctx.textAlign = "center";
      ctx.fillText(`N${n.id}`, n.x, n.y - n.radius - 4);
    });

    // Hover tooltip
    if (this.hoveredNode) {
      const n = this.hoveredNode;
      const W=150, H=48;
      const bx = Math.min(this.width-W-10, Math.max(10, n.x - W/2));
      const by = Math.max(10, n.y - H - 16);
      ctx.fillStyle = "rgba(6,8,13,0.95)";
      ctx.strokeStyle = "rgba(56,189,248,0.35)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.roundRect(bx, by, W, H, 5);
      ctx.fill(); ctx.stroke();
      ctx.fillStyle = "#38bdf8";
      ctx.font = "bold 11px 'Space Grotesk', sans-serif";
      ctx.textAlign = "left";
      ctx.fillText(`Node ${n.id}  (${n.type})`, bx+10, by+17);
      const faulty = impactSet.has(n.id);
      ctx.fillStyle = faulty ? "#f43f5e" : "#34d399";
      ctx.font = "10px 'JetBrains Mono', monospace";
      ctx.fillText(faulty ? "STATUS: ANOMALY" : "STATUS: HEALTHY", bx+10, by+34);
    }

    requestAnimationFrame(() => this.animate());
  }
}

// ==========================================================================
// 3. SIMULATION CONTROLLER
// ==========================================================================
function updateSimulationUI(scenKey) {
  const scen = SCENARIO_DATA[scenKey];
  if (!scen) return;
  currentScenarioKey = scenKey;

  // Header
  document.getElementById('sim-scen-title').innerText = scen.name;
  document.getElementById('sim-scen-desc').innerText  = scen.description;
  const tagEl = document.getElementById('sim-scen-tag');
  tagEl.innerText = scen.tag;
  tagEl.setAttribute('style', scen.tagStyle + ';display:inline-block;font-family:var(--mono);font-size:10.5px;font-weight:600;text-transform:uppercase;letter-spacing:1px;padding:3px 10px;border-radius:4px;border:1px solid;');

  // Metric bars
  const b = scen.before, a = scen.after;
  document.getElementById('m-lat-before').innerText  = `${b.latency} ms`;
  document.getElementById('m-lat-after').innerText   = `${a.latency} ms`;
  document.getElementById('bar-lat-before').style.width = `${Math.min(100,(b.latency/160)*100)}%`;
  document.getElementById('bar-lat-after').style.width  = `${Math.min(100,(a.latency/160)*100)}%`;

  document.getElementById('m-loss-before').innerText = `${b.loss} %`;
  document.getElementById('m-loss-after').innerText  = `${a.loss} %`;
  document.getElementById('bar-loss-before').style.width = `${Math.min(100,(b.loss/10)*100)}%`;
  document.getElementById('bar-loss-after').style.width  = `${Math.min(100,(a.loss/10)*100)}%`;

  document.getElementById('m-tput-before').innerText = `${b.throughput} Mbps`;
  document.getElementById('m-tput-after').innerText  = `${a.throughput} Mbps`;
  document.getElementById('bar-tput-before').style.width = `${b.throughput}%`;
  document.getElementById('bar-tput-after').style.width  = `${a.throughput}%`;

  document.getElementById('m-cong-before').innerText = `${Math.round(b.congestion*100)} %`;
  document.getElementById('m-cong-after').innerText  = `${Math.round(a.congestion*100)} %`;
  document.getElementById('bar-cong-before').style.width = `${b.congestion*100}%`;
  document.getElementById('bar-cong-after').style.width  = `${a.congestion*100}%`;

  // Decision
  document.getElementById('v-action').innerText     = scen.rl.action;
  document.getElementById('v-confidence').innerText = `${scen.rl.confidence} %`;
  document.getElementById('v-qvalue').innerText     = `Q ${scen.rl.qValue}`;
  document.getElementById('v-rationale').innerText  = scen.rationale;

  const badge = document.getElementById('v-gate-badge');
  badge.innerText = scen.safety.status;
  badge.className = `gate-pill ${scen.safety.approved ? 'gate-pass' : 'gate-escalated'}`;

  const reasonEl = document.getElementById('v-gate-reason');
  reasonEl.innerText = scen.safety.reason;
  reasonEl.style.color = scen.safety.approved ? 'var(--green)' : 'var(--rose)';

  // SHAP bars
  const shapWrap = document.getElementById('shap-bars-wrap');
  shapWrap.innerHTML = '';
  scen.shap.forEach(item => {
    const row = document.createElement('div');
    row.className = 'mbar-group';
    row.innerHTML = `
      <div class="mbar-row">
        <span>${item.name}</span>
        <span style="color:${item.color};font-family:var(--mono);font-size:11.5px">${item.pct}%</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill" style="width:${item.pct}%;background:${item.color}"></div>
      </div>`;
    shapWrap.appendChild(row);
  });

  // Audit JSON
  const b2 = scen.before;
  const a2 = scen.after;
  const payload = {
    incident: scenKey,
    anomaly:  { ensemble_score: scen.anomaly.score, lstm_error: scen.anomaly.lstm, iso_forest: scen.anomaly.iso },
    decision: { action: scen.rl.action, confidence: +(scen.rl.confidence/100).toFixed(3), q_value: scen.rl.qValue },
    gate:     { verdict: scen.safety.status, threshold: 0.75 },
    executed: scen.safety.approved ? scen.rl.action : "ESCALATE_TO_HUMAN",
    delta:    { latency_ms: Math.round(b2.latency - a2.latency), throughput_mbps: Math.round(a2.throughput - b2.throughput) }
  };
  document.getElementById('json-audit-dump').innerText = JSON.stringify(payload, null, 2);

  if (window.topologySim) window.topologySim.setScenario(scenKey);
}

// Step-through animation
function runSequenceAnimation() {
  if (isSimulating) return;
  isSimulating = true;
  const steps = [1,2,3,4,5].map(i => document.getElementById(`t-step-${i}`));
  steps.forEach(s => s.classList.remove('active'));
  let cur = 0;
  const interval = setInterval(() => {
    if (cur > 0) steps[cur-1].classList.remove('active');
    if (cur < steps.length) {
      steps[cur].classList.add('active');
      cur++;
    } else {
      clearInterval(interval);
      steps.forEach(s => s.classList.add('active'));
      isSimulating = false;
    }
  }, 480);
}

// ==========================================================================
// 4. INIT
// ==========================================================================
document.addEventListener('DOMContentLoaded', () => {
  window.topologySim = new TopologySimulator('topology-canvas');

  document.querySelectorAll('.scen-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.scen-btn').forEach(b => b.classList.remove('scen-active'));
      btn.classList.add('scen-active');
      updateSimulationUI(btn.getAttribute('data-scen'));
      runSequenceAnimation();
    });
  });

  const runBtn = document.getElementById('btn-run-sim');
  if (runBtn) runBtn.addEventListener('click', runSequenceAnimation);

  updateSimulationUI('congestion');
  runSequenceAnimation();
});
