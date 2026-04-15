/**
 * dashboard.js — Cyberpunk LPR Dashboard client
 * Connects to Flask-SocketIO and updates the UI in real time.
 */

// ── Socket connection ──────────────────────────────────────────────────────
const socket = io({ transports: ["websocket", "polling"] });

// ── DOM refs ───────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const gaugeFill   = $("gaugeFill");
const gaugeVal    = $("gaugeVal");
const lifiState   = $("lifiState");
const dots        = [$("dot0"), $("dot1"), $("dot2")];
const terminal    = $("terminal");
const gateBar     = $("gateBar");
const gateDot     = $("gateDot");
const gateText    = $("gateText");
const gateTs      = $("gateTs");
const idCard      = $("idCard");
const idPlate     = $("idPlate");
const idStatus    = $("idStatus");
const idStamp     = $("idStamp");
const idTime      = $("idTime");
const idMethod    = $("idMethod");
const ocrVal      = $("ocrVal");
const streamPlate = $("streamPlate");
const clockEl     = $("clock");

// ── Live clock ─────────────────────────────────────────────────────────────
function tickClock() {
  const now = new Date();
  clockEl.textContent = now.toTimeString().slice(0, 8);
}
setInterval(tickClock, 1000);
tickClock();

// ── Terminal logger ─────────────────────────────────────────────────────────
const MAX_LOG_LINES = 120;

function appendLog(message, level = "info") {
  const ts = new Date().toTimeString().slice(0, 8);
  const line = document.createElement("div");
  line.className = `log-line ${level}`;
  line.innerHTML = `<span class="ts">[${ts}]</span><span class="msg">${escHtml(message)}</span>`;
  terminal.appendChild(line);
  // Trim old lines
  while (terminal.children.length > MAX_LOG_LINES) {
    terminal.removeChild(terminal.firstChild);
  }
  terminal.scrollTop = terminal.scrollHeight;
}

function escHtml(s) {
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

// ── SVG Gauge helper ────────────────────────────────────────────────────────
const CIRC = 2 * Math.PI * 50; // 314.16

function setGauge(intensity) {
  const pct = Math.min(intensity / 255, 1);
  const offset = CIRC * (1 - pct);
  gaugeFill.style.strokeDashoffset = offset.toFixed(1);
  gaugeVal.textContent = Math.round(intensity);

  if (pct > 0.78) {
    gaugeFill.style.stroke = "var(--green)";
    gaugeFill.style.filter = "drop-shadow(0 0 10px var(--green))";
  } else if (pct > 0.4) {
    gaugeFill.style.stroke = "var(--cyan)";
    gaugeFill.style.filter = "drop-shadow(0 0 6px var(--cyan))";
  } else {
    gaugeFill.style.stroke = "var(--red)";
    gaugeFill.style.filter = "drop-shadow(0 0 4px var(--red))";
  }
}

// ── Pulse dots ──────────────────────────────────────────────────────────────
function setPulseDots(count) {
  dots.forEach((d, i) => {
    if (i < count) {
      d.classList.add("active");
    } else {
      d.classList.remove("active");
    }
  });
}

// ── Gate Status ─────────────────────────────────────────────────────────────
function setGateStatus(status, plate, ts) {
  gateBar.classList.remove("open", "denied", "idle");
  if (status === "OPEN") {
    gateBar.classList.add("open");
    gateText.textContent = "GATE: OPEN";
    gateText.style.color = "var(--green)";
  } else if (status === "DENIED") {
    gateBar.classList.add("denied");
    gateText.textContent = "GATE: DENIED";
    gateText.style.color = "var(--red)";
  } else {
    gateBar.classList.add("idle");
    gateText.textContent = "GATE: IDLE";
    gateText.style.color = "var(--cyan)";
  }
  gateTs.textContent = ts || "";
}

// ── Vehicle ID Card ─────────────────────────────────────────────────────────
function showIdCard(plate, authorized, timestamp, method) {
  idPlate.textContent = plate || "—";
  idTime.textContent  = timestamp || "—";
  idMethod.textContent = method || "OCR";
  streamPlate.textContent = plate || "—";
  idStamp.className = "";

  if (authorized) {
    idStatus.textContent = "✅  ACCESS GRANTED";
    idStatus.style.color = "var(--green)";
    idStamp.textContent = "VERIFIED";
    idStamp.className = "id-card__stamp verified";
    idCard.classList.add("pop");
    setTimeout(() => idCard.classList.remove("pop"), 5000);
  } else {
    idStatus.textContent = "❌  ACCESS DENIED";
    idStatus.style.color = "var(--red)";
    idStamp.textContent = "DENIED";
    idStamp.className = "id-card__stamp denied";
    idCard.classList.remove("pop");
  }
}

// ════════════  SOCKET.IO EVENT HANDLERS  ════════════

// Initial state snapshot on connect / reconnect
socket.on("state_snapshot", data => {
  setGauge(data.intensity || 0);
  setPulseDots(data.pulse_count || 0);

  const state = data.lifi_state;
  lifiState.textContent = state ? "⬤ HIGH" : "⬤ LOW";
  lifiState.className = "lifi-state " + (state ? "high" : "low");

  if (data.gate_status) setGateStatus(data.gate_status, data.plate_text, data.last_gate_ts);
  if (data.plate_text)  streamPlate.textContent = data.plate_text;
  appendLog("[DASHBOARD] Connected to Edge-AI LPR server.", "info");
});

// Real-time Li-Fi updates (every frame)
socket.on("lifi_update", data => {
  setGauge(data.intensity);
  setPulseDots(data.pulse_count);

  const state = data.state;
  lifiState.textContent = state ? "⬤ HIGH" : "⬤ LOW";
  lifiState.className = "lifi-state " + (state ? "high" : "low");
});

// OCR result
socket.on("ocr_result", data => {
  ocrVal.textContent = data.plate_text || "—";
  ocrVal.style.color = data.authorized ? "var(--green)" : "var(--magenta)";
  streamPlate.textContent = data.plate_text || "—";
});

// Gate open / denied event
socket.on("gate_event", data => {
  setGateStatus(data.status, data.plate, data.timestamp);
  showIdCard(data.plate, data.status === "OPEN", data.timestamp,
             data.plate === "Li-Fi Override" || data.plate === "FORCE OVERRIDE" ? "Li-Fi" : "OCR");
  const lvl = data.status === "OPEN" ? "success" : "error";
  appendLog(`[GATE] ${data.status} — ${data.plate} @ ${data.timestamp}`, lvl);

  // Auto-reset gate indicator after 5 seconds (or on OPEN)
  if (data.status === "OPEN") {
    setTimeout(() => setGateStatus("IDLE", "", ""), 6000);
  }
});

// Log lines from AI pipeline
socket.on("log_line", data => {
  appendLog(data.message, data.level || "info");
});

// System reset
socket.on("system_reset", () => {
  setGauge(0);
  setPulseDots(0);
  lifiState.textContent = "⬤ LOW";
  lifiState.className = "lifi-state low";
  setGateStatus("IDLE", "", "");
  ocrVal.textContent = "—";
  streamPlate.textContent = "—";
  idPlate.textContent = "—";
  idStatus.textContent = "AWAITING SCAN";
  idStamp.className = "id-card__stamp";
  idCard.classList.remove("pop");
  appendLog("[SYSTEM] Reset triggered.", "info");
});

socket.on("connect", () => appendLog("[SOCKET] Connected.", "info"));
socket.on("disconnect", () => appendLog("[SOCKET] Disconnected.", "error"));

// ════════════  BUTTON HANDLERS  ════════════
$("btnForceOpen").addEventListener("click", async () => {
  appendLog("[OVERRIDE] Force-open command sent.", "success");
  await fetch("/force-open", { method: "POST" });
});

$("btnReset").addEventListener("click", async () => {
  appendLog("[SYSTEM] Reset command sent.", "info");
  await fetch("/system-reset", { method: "POST" });
});
