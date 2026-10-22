/**
 * NEURONET - Live 20-Node Control Plane Frontend Engine
 * 60 FPS HTML5 Canvas Topology + Real-Time REST Bridge
 */

// ============================================================================
// 1. STATE & GLOBAL CONFIG
// ============================================================================
let netState = {
  step: 0,
  scenario: "nominal",
  metrics: { latency_ms: 19.4, throughput_mbps: 94.2, packet_loss_pct: 0.12, congestion: 0.16, link_health: 0.98, cpu_util_pct: 24.5 },
  nodes: [],
  links: [],
  last_action: "MONITOR",
  last_decision: null
};

let autoPilot = false;
let autoPilotTimer = null;
let selectedNodeId = null;
let particles = [];

// ============================================================================
// 2. CANVAS SETUP & PARTICLE PHYSICS
// ============================================================================
const canvas = document.getElementById("topologyCanvas");
const ctx = canvas.getContext("2d");

// Particle System for 60 FPS traffic packet flows
class PacketParticle {
  constructor(link) {
    this.link = link;
    this.progress = Math.random();
    this.speed = 0.006 + Math.random() * 0.008;
    this.reverse = Math.random() > 0.5;
  }

  update() {
    if (this.link.status === "cut") {
      this.progress += this.speed * 0.3;
      if (this.progress > 0.5) this.progress = 0; // stall at cut point
      return;
    }
    this.progress += this.speed;
    if (this.progress > 1.0) {
      this.progress = 0;
      this.reverse = Math.random() > 0.5;
    }
  }

  draw(nodes) {
    const sNode = nodes[this.link.source];
    const tNode = nodes[this.link.target];
    if (!sNode || !tNode) return;

    const from = this.reverse ? tNode : sNode;
    const to = this.reverse ? sNode : tNode;

    const x = from.x + (to.x - from.x) * this.progress;
    const y = from.y + (to.y - from.y) * this.progress;

    ctx.save();
    if (this.link.status === "cut") {
      ctx.fillStyle = "#f43f5e";
      ctx.shadowColor = "#f43f5e";
      ctx.shadowBlur = 6;
      ctx.fillRect(x - 2, y - 2, 4, 4);
    } else if (this.link.status === "congested") {
      ctx.fillStyle = "#f59e0b";
      ctx.shadowColor = "#f59e0b";
      ctx.shadowBlur = 6;
      ctx.beginPath();
      ctx.arc(x, y, 2.8, 0, Math.PI * 2);
      ctx.fill();
    } else {
      ctx.fillStyle = "#ffffff";
      ctx.shadowColor = "#ffffff";
      ctx.shadowBlur = 4;
      ctx.beginPath();
      ctx.arc(x, y, 2.2, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }
}

function initParticles() {
  particles = [];
  if (!netState.links) return;
  netState.links.forEach(l => {
    const count = l.status === "congested" ? 4 : 2;
    for (let i = 0; i < count; i++) {
      particles.push(new PacketParticle(l));
    }
  });
}

// 60 FPS Render Loop
function renderCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const nodes = netState.nodes;
  const links = netState.links;

  if (nodes.length > 0 && links.length > 0) {
    // 1. Draw Links
    links.forEach(l => {
      const s = nodes[l.source];
      const t = nodes[l.target];
      if (!s || !t) return;

      ctx.save();
      ctx.beginPath();
      ctx.moveTo(s.x, s.y);
      ctx.lineTo(t.x, t.y);

      if (l.status === "cut") {
        ctx.strokeStyle = "rgba(244, 63, 94, 0.75)";
        ctx.lineWidth = 2.5;
        ctx.setLineDash([6, 6]);
      } else if (l.status === "congested") {
        ctx.strokeStyle = "rgba(245, 158, 11, 0.75)";
        ctx.lineWidth = 3.5;
        ctx.shadowColor = "rgba(245, 158, 11, 0.4)";
        ctx.shadowBlur = 8;
      } else {
        ctx.strokeStyle = "rgba(255, 255, 255, 0.12)";
        ctx.lineWidth = 1.6;
      }
      ctx.stroke();
      ctx.restore();
    });

    // 2. Draw Packet Particles
    particles.forEach(p => {
      p.update();
      p.draw(nodes);
    });

    // 3. Draw Nodes
    nodes.forEach(n => {
      const isSelected = selectedNodeId === n.id;
      let radius = n.tier === "core" ? 18 : (n.tier === "agg" ? 14 : 10);
      let color = n.tier === "core" ? "#ffffff" : (n.tier === "agg" ? "#a1a1aa" : "#22c55e");

      if (n.status === "critical") color = "#f43f5e";
      else if (n.status === "congested") color = "#f59e0b";
      else if (n.status === "warning") color = "#fbbf24";

      ctx.save();

      // Outer Halo for critical/congested nodes
      if (n.status !== "nominal") {
        ctx.beginPath();
        ctx.arc(n.x, n.y, radius + 8, 0, Math.PI * 2);
        ctx.fillStyle = n.status === "critical" ? "rgba(244, 63, 94, 0.25)" : "rgba(245, 158, 11, 0.25)";
        ctx.fill();
      }

      // Selection ring
      if (isSelected) {
        ctx.beginPath();
        ctx.arc(n.x, n.y, radius + 5, 0, Math.PI * 2);
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // Base Node
      ctx.beginPath();
      ctx.arc(n.x, n.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = "#000000";
      ctx.fill();
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = color;
      ctx.shadowColor = color;
      ctx.shadowBlur = 10;
      ctx.stroke();

      // Node Label
      ctx.fillStyle = "#fff";
      ctx.font = `${n.tier === 'core' ? '11px' : '9.5px'} 'JetBrains Mono', monospace`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(n.name, n.x, n.y);

      ctx.restore();
    });
  }

  requestAnimationFrame(renderCanvas);
}

// Canvas Click for Node Inspector
canvas.addEventListener("click", e => {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const clickX = (e.clientX - rect.left) * scaleX;
  const clickY = (e.clientY - rect.top) * scaleY;

  let found = null;
  netState.nodes.forEach(n => {
    const dist = Math.hypot(clickX - n.x, clickY - n.y);
    if (dist < 22) found = n;
  });

  const popup = document.getElementById("node-inspector");
  if (found) {
    selectedNodeId = found.id;
    document.getElementById("insp-name").innerText = `${found.name} (${found.tier.toUpperCase()})`;
    document.getElementById("insp-cpu").innerText = `${found.cpu}%`;
    document.getElementById("insp-buf").innerText = `${found.buffer}%`;
    document.getElementById("insp-status").innerText = found.status.toUpperCase();
    document.getElementById("insp-status").className = found.status === "nominal" ? "tag-insp-ok" : "tag-insp-err";
    popup.style.display = "block";
  } else {
    selectedNodeId = null;
    popup.style.display = "none";
  }
});

// ============================================================================
// 3. API COMMUNICATION & UI UPDATES
// ============================================================================

async function fetchStatus() {
  try {
    const res = await fetch("/api/status");
    if (!res.ok) return;
    const data = await res.json();
    netState = data;
    updateHUD(data.metrics);
    updateIncidentBadge(data.scenario);
    document.getElementById("val-step").innerText = data.step;

    if (particles.length === 0) {
      initParticles();
    }
  } catch (err) {
    console.warn("API connection pending...", err);
  }
}

function updateHUD(m) {
  if (!m) return;
  document.getElementById("hud-lat").innerText = m.latency_ms.toFixed(1);
  document.getElementById("hud-loss").innerText = m.packet_loss_pct.toFixed(2);
  document.getElementById("hud-tput").innerText = m.throughput_mbps.toFixed(1);
  document.getElementById("hud-cong").innerText = Math.round(m.congestion * 100);
  document.getElementById("hud-cpu").innerText = m.cpu_util_pct.toFixed(1);

  // Meter bars
  const setBar = (id, pct, fillClass) => {
    const el = document.getElementById(id);
    el.style.width = Math.min(100, Math.max(0, pct)) + "%";
    el.className = "hud-fill " + fillClass;
  };

  setBar("bar-lat", (m.latency_ms / 150) * 100, m.latency_ms > 70 ? "fill-rose" : (m.latency_ms > 35 ? "fill-amber" : "fill-green"));
  setBar("bar-loss", (m.packet_loss_pct / 8) * 100, m.packet_loss_pct > 3 ? "fill-rose" : "fill-green");
  setBar("bar-tput", m.throughput_mbps, m.throughput_mbps < 40 ? "fill-amber" : "fill-cyan");
  setBar("bar-cong", m.congestion * 100, m.congestion > 0.7 ? "fill-rose" : (m.congestion > 0.4 ? "fill-amber" : "fill-green"));
  setBar("bar-cpu", m.cpu_util_pct, m.cpu_util_pct > 80 ? "fill-rose" : "fill-cyan");
}

function updateIncidentBadge(scen) {
  const badge = document.getElementById("canvas-incident-badge");
  if (!scen || scen === "nominal") {
    badge.innerText = "NOMINAL";
    badge.className = "badge-nominal";
    document.querySelectorAll(".btn-fault").forEach(b => b.classList.remove("active-fault"));
  } else {
    badge.innerText = `INCIDENT: ${scen.toUpperCase()}`;
    badge.className = "badge-nominal badge-incident";
  }
}

// Fault Injection Trigger
document.querySelectorAll(".btn-fault").forEach(btn => {
  btn.addEventListener("click", async () => {
    const scenario = btn.getAttribute("data-fault");
    document.querySelectorAll(".btn-fault").forEach(b => b.classList.remove("active-fault"));
    btn.classList.add("active-fault");

    try {
      const res = await fetch("/api/fault", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scenario })
      });
      const data = await res.json();
      updateHUD(data.metrics);
      updateIncidentBadge(data.scenario);
      initParticles();

      // If Auto-Pilot is enabled, automatically trigger the agent after 1.2s delay
      if (autoPilot) {
        setTimeout(triggerAgentDecision, 1200);
      }
    } catch (err) {
      console.error("Fault injection failed:", err);
    }
  });
});

// Trigger Agent Decision
async function triggerAgentDecision() {
  const btn = document.getElementById("btn-decide");
  btn.disabled = true;
  btn.innerHTML = `<span>Evaluating Dueling DQN Policy...</span>`;

  try {
    const res = await fetch("/api/agent/decide", { method: "POST" });
    const data = await res.json();
    btn.disabled = false;
    btn.innerHTML = `
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>
      <span>Run AI Agent Inference (Auto-Heal)</span>
    `;

    if (data.decision) {
      renderDecisionUI(data.decision);
      updateHUD(data.metrics);
      updateIncidentBadge("nominal");
      initParticles();
      fetchAuditLogs();
    }
  } catch (err) {
    btn.disabled = false;
    btn.innerHTML = `<span>Error connecting to agent</span>`;
  }
}

document.getElementById("btn-decide").addEventListener("click", triggerAgentDecision);

function renderDecisionUI(dec) {
  document.getElementById("dec-action-name").innerText = dec.action;
  document.getElementById("dec-conf").innerText = `${Number(dec.confidence).toFixed(1)}%`;
  const qValNum = Number(dec.q_value);
  document.getElementById("dec-qval").innerText = `${qValNum > 0 ? '+' : ''}${qValNum.toFixed(2)}`;

  const gateBadge = document.getElementById("dec-gate-badge");
  gateBadge.innerText = dec.gate_status;
  gateBadge.className = `gate-badge ${dec.gate_status === 'APPROVED' ? 'gate-pass' : 'gate-escalate'}`;
  document.getElementById("dec-gate-desc").innerText = dec.gate_reason;

  // Render Q-values horizontal bars
  const qContainer = document.getElementById("q-bars-container");
  qContainer.innerHTML = "";
  if (dec.q_values) {
    Object.entries(dec.q_values).forEach(([act, val]) => {
      const isBest = act === dec.action;
      const numVal = Number(val);
      const item = document.createElement("div");
      item.className = `q-item ${isBest ? 'q-best' : ''}`;
      item.innerHTML = `<span>${act}</span><strong>${numVal > 0 ? '+' : ''}${numVal.toFixed(2)}</strong>`;
      qContainer.appendChild(item);
    });
  }

  // Render SHAP proxy bars
  const shapContainer = document.getElementById("shap-bars-container");
  shapContainer.innerHTML = "";
  if (dec.shap) {
    Object.entries(dec.shap).forEach(([feat, weight]) => {
      const row = document.createElement("div");
      row.innerHTML = `
        <div class="shap-row">
          <span>${feat}</span>
          <span style="font-family:var(--font-mono);color:var(--cyan)">${weight}%</span>
        </div>
        <div class="shap-track"><div class="shap-fill" style="width:${weight}%"></div></div>
      `;
      shapContainer.appendChild(row);
    });
  }
}

// Audit Logs Table
async function fetchAuditLogs() {
  try {
    const res = await fetch("/api/audit");
    if (!res.ok) return;
    const logs = await res.json();
    const tbody = document.getElementById("audit-tbody");
    if (logs.length === 0) return;

    tbody.innerHTML = "";
    logs.slice().reverse().forEach(log => {
      const tr = document.createElement("tr");
      const d = log.recovery_delta;
      const deltaText = d ? `-${d.latency_drop_ms}ms lat, +${d.throughput_gain_mbps}Mbps tput` : "-";
      const qv = Number(log.q_value);
      tr.innerHTML = `
        <td>${log.timestamp}</td>
        <td><span style="color:var(--amber)">${log.incident.toUpperCase()}</span></td>
        <td><strong style="color:#fff">${log.action}</strong></td>
        <td><span class="gate-badge ${log.gate_status === 'APPROVED' ? 'gate-pass' : 'gate-escalate'}" style="font-size:9.5px;padding:1px 5px">${log.gate_status}</span></td>
        <td>${Number(log.confidence).toFixed(1)}%</td>
        <td>${qv > 0 ? '+' : ''}${qv.toFixed(2)}</td>
        <td><span style="color:var(--green)">${deltaText}</span></td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Audit log fetch error:", err);
  }
}

// Header Action Buttons
document.getElementById("btn-tick").addEventListener("click", async () => {
  await fetch("/api/tick", { method: "POST" });
  fetchStatus();
});

document.getElementById("btn-reset").addEventListener("click", async () => {
  await fetch("/api/reset", { method: "POST" });
  fetchStatus();
  fetchAuditLogs();
});

document.getElementById("btn-clear-audit").addEventListener("click", () => {
  document.getElementById("audit-tbody").innerHTML = `<tr><td colspan="7" class="empty-audit">View cleared.</td></tr>`;
});

// Auto-Pilot Toggle
const btnAuto = document.getElementById("btn-autopilot");
btnAuto.addEventListener("click", () => {
  autoPilot = !autoPilot;
  if (autoPilot) {
    btnAuto.classList.add("active");
    btnAuto.querySelector("span:last-child").innerText = "Auto-Pilot: ON";
    autoPilotTimer = setInterval(() => {
      // Periodic autonomous monitoring tick
      fetch("/api/tick", { method: "POST" });
      fetchStatus();
    }, 1500);
  } else {
    btnAuto.classList.remove("active");
    btnAuto.querySelector("span:last-child").innerText = "Auto-Pilot: OFF";
    clearInterval(autoPilotTimer);
  }
});

// Initial Startup
fetchStatus();
fetchAuditLogs();
requestAnimationFrame(renderCanvas);

// Periodic sync every 1.5s
setInterval(fetchStatus, 1500);
