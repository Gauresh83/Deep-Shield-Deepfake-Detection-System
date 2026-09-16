/* DeepShield M3-ID — Dashboard Logic */

const API = "http://localhost:8000";

// ── Init ──────────────────────────────────────────────────────────────────────
(function init() {
  const user = requireAuth("login.html");
  if (!user) return;
  setUserUI(user);
  initMobileSidebar();

  // Set active nav
  document.querySelectorAll(".nav-item[data-page]").forEach(el => {
    el.classList.toggle("active", el.dataset.page === "dashboard");
  });

  // Set date in topbar
  const now = new Date();
  document.getElementById("topbar-sub").textContent =
    now.toLocaleDateString("en-IN", { weekday: "long", year: "numeric", month: "long", day: "numeric" });

  loadStats();
  renderModuleStatus();
  renderActivityChart();
  renderRecentScans();
  renderDashboardAlerts();
})();

// ── Recent alerts widget ─────────────────────────────────────────────────────
const ALERT_ICONS = {
  CRITICAL: `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`,
  HIGH: `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>`,
  MEDIUM: `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>`,
  LOW: `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><circle cx="12" cy="12" r="10"/><polyline points="20 6 9 17 4 12"/></svg>`,
  CLEAR: `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>`,
};
const ALERT_CLASS = { CRITICAL: "critical", HIGH: "high", MEDIUM: "medium", LOW: "low", CLEAR: "clear" };

async function renderDashboardAlerts() {
  const feed = document.getElementById("alert-feed");
  if (!feed) return;
  try {
    const d = await api.get("/api/alerts/?page=1&page_size=3");
    const alerts = d.alerts || [];
    if (!alerts.length) {
      feed.innerHTML = '<div style="padding:24px;text-align:center;color:var(--text-tertiary);font-size:13px">No alerts yet</div>';
      return;
    }
    feed.innerHTML = alerts.map(a => `
      <div class="alert-item">
        <div class="alert-icon-wrap ${ALERT_CLASS[a.severity] || "medium"}">${ALERT_ICONS[a.severity] || ALERT_ICONS.MEDIUM}</div>
        <div class="alert-body">
          <div class="alert-title">${a.severity} — ${a.title}</div>
          <div class="alert-desc">${a.message}</div>
          <div class="alert-meta"><span class="alert-time">${timeAgo(a.created_at)}</span></div>
        </div>
      </div>`).join("");
  } catch (err) {
    feed.innerHTML = '<div style="padding:24px;text-align:center;color:var(--text-tertiary);font-size:13px">Could not load alerts</div>';
  }
}

// ── Stats ─────────────────────────────────────────────────────────────────────
async function loadStats() {
  try {
    const d = await api.get("/api/analytics/overview");
    animateNumber(document.getElementById("s-total"), d.total_scans);
    animateNumber(document.getElementById("s-fakes"), d.fakes_caught);
    animateNumber(document.getElementById("s-susp"), d.suspicious);
    document.getElementById("s-acc").innerHTML = d.accuracy_rate + '<span style="font-size:18px;color:var(--text-tertiary)">%</span>';
  } catch (err) {
    animateNumber(document.getElementById("s-total"), 0);
    animateNumber(document.getElementById("s-fakes"), 0);
    animateNumber(document.getElementById("s-susp"), 0);
  }
}

// ── Module status ─────────────────────────────────────────────────────────────
const MODULE_META = [
  { key: "face", label: "Face Detection", desc: "EfficientNet-B4", color: "var(--accent)", acc: 91, icon: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="8" r="5"/><path d="M20 21a8 8 0 10-16 0"/></svg>` },
  { key: "voice", label: "Voice Analysis", desc: "AASIST / MFCC", color: "var(--success)", acc: 87, icon: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 1a3 3 0 00-3 3v8a3 3 0 006 0V4a3 3 0 00-3-3z"/><path d="M19 10v2a7 7 0 01-14 0v-2"/></svg>` },
  { key: "nlp", label: "Linguistic NLP", desc: "BERT Statistical", color: "var(--warning)", acc: 94, icon: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>` },
  { key: "fusion", label: "Risk Fusion", desc: "MLP + XAI", color: "var(--info)", acc: 98, icon: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>` },
];

function renderModuleStatus() {
  const list = document.getElementById("module-status-list");
  if (!list) return;
  list.innerHTML = MODULE_META.map(m => `
    <div style="display:flex;align-items:center;gap:12px">
      <div style="width:34px;height:34px;border-radius:var(--radius-md);background:var(--bg-subtle);border:1px solid var(--border);display:flex;align-items:center;justify-content:center;color:${m.color};flex-shrink:0">${m.icon}</div>
      <div style="flex:1;min-width:0">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px">
          <span style="font-size:12px;font-weight:600;color:var(--text-primary)">${m.label}</span>
          <span style="font-size:12px;font-weight:700;font-family:var(--font-mono);color:${m.color}">${m.acc}%</span>
        </div>
        <div class="score-bar-track">
          <div class="score-bar-fill" style="background:${m.color};width:0%" data-w="${m.acc}%"></div>
        </div>
      </div>
    </div>`).join("");

  // Animate bars
  setTimeout(() => {
    list.querySelectorAll(".score-bar-fill").forEach(el => { el.style.width = el.dataset.w; });
  }, 200);
}

// ── Activity chart ─────────────────────────────────────────────────────────────
async function renderActivityChart() {
  const ctx = document.getElementById("activity-chart");
  if (!ctx) return;
  const isDark = ["dark", "midnight"].includes(document.documentElement.getAttribute("data-theme"));
  const gridColor = isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.05)";
  const tickColor = isDark ? "#4F566B" : "#8B93A7";

  let labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
  let scans = [0, 0, 0, 0, 0, 0, 0];
  let fakes = [0, 0, 0, 0, 0, 0, 0];
  try {
    const d = await api.get("/api/analytics/weekly-activity");
    labels = d.days.map(x => x.date);
    scans = d.days.map(x => x.scans);
    fakes = d.days.map(x => x.fakes);
  } catch (err) { /* keep zeros if not available yet */ }

  new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        { label: "Scans", data: scans, borderColor: "var(--accent)", backgroundColor: "rgba(59,110,245,0.08)", borderWidth: 2, tension: 0.4, fill: true, pointBackgroundColor: "var(--accent)", pointRadius: 3 },
        { label: "Fakes", data: fakes, borderColor: "var(--danger)", backgroundColor: "rgba(239,68,68,0.05)", borderWidth: 2, tension: 0.4, fill: true, pointBackgroundColor: "var(--danger)", pointRadius: 3 },
      ]
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: tickColor, font: { family: "JetBrains Mono", size: 9 }, boxWidth: 10 } } },
      scales: {
        x: { ticks: { color: tickColor, font: { size: 9 } }, grid: { color: gridColor } },
        y: { ticks: { color: tickColor, font: { size: 9 } }, grid: { color: gridColor } },
      }
    }
  });
}

// ── Recent scans table ────────────────────────────────────────────────────────
let localScans = [];

async function renderRecentScans() {
  const tbody = document.getElementById("recent-scans-body");
  if (!tbody) return;
  try {
    const d = await api.get("/api/scans/?page=1&page_size=6");
    localScans = d.scans || [];
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;padding:32px;color:var(--text-tertiary)">Could not load scans</td></tr>`;
    return;
  }
  if (!localScans.length) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;padding:32px;color:var(--text-tertiary)">No scans yet</td></tr>`;
    return;
  }
  const modDots = (s) => [
    ["var(--accent)", s.face_score != null],
    ["var(--success)", s.voice_score != null],
    ["var(--warning)", s.nlp_score != null],
    ["var(--info)", true],
  ].map(([c, a]) => `<div style="width:7px;height:7px;border-radius:50%;background:${a ? c : "var(--border)"};flex-shrink:0"></div>`).join("");

  tbody.innerHTML = localScans.slice(0, 6).map(s => {
    const vc = { authentic: "authentic", suspicious: "suspicious", fake: "fake", pending: "pending" }[s.verdict] || "pending";
    const vl = { authentic: "Authentic", suspicious: "Suspicious", fake: "Fake", pending: "Pending" }[s.verdict] || "—";
    const sc = s.fusion_score != null ? Math.round(s.fusion_score) : null;
    return `<tr>
      <td>
        <div class="cell-primary" style="max-width:140px" class="truncate">${s.file_name || "scan"}</div>
        <div class="cell-mono" style="font-size:11px;color:var(--text-tertiary);margin-top:2px">${s.scan_type}</div>
      </td>
      <td><div style="display:flex;gap:4px;align-items:center">${modDots(s)}</div></td>
      <td class="cell-mono">${s.face_score != null ? Math.round(s.face_score) + "%" : "—"}</td>
      <td class="cell-mono">${s.voice_score != null ? Math.round(s.voice_score) + "%" : "—"}</td>
      <td class="cell-mono">${s.nlp_score != null ? Math.round(s.nlp_score) + "%" : "—"}</td>
      <td><span class="score-val ${sc != null ? scoreClass(sc) : ""}">${sc != null ? sc + "%" : "—"}</span></td>
      <td><span class="verdict-chip ${vc}">${vl}</span></td>
      <td class="cell-mono" style="font-size:11px;color:var(--text-tertiary)">${timeAgo(s.created_at)}</td>
    </tr>`;
  }).join("");
}

// ── Scan tabs ─────────────────────────────────────────────────────────────────
function switchTab(btn, tab) {
  document.querySelectorAll(".scan-tab").forEach(t => {
    t.classList.remove("active");
    t.style.borderBottom = "2px solid transparent";
    t.style.color = "";
  });
  btn.classList.add("active");
  btn.style.borderBottom = "2px solid var(--accent)";
  btn.style.color = "var(--accent)";
  document.querySelectorAll(".scan-content").forEach(c => c.style.display = "none");
  const el = document.getElementById(`tab-${tab}`);
  if (el) el.style.display = "block";
}

// ── Module toggles ────────────────────────────────────────────────────────────
function toggleMod(el) {
  el.classList.toggle("active");
  showToast(`${el.dataset.mod} module ${el.classList.contains("active") ? "enabled" : "disabled"}`, "info", 2000);
}

// ── File handling ─────────────────────────────────────────────────────────────
function handleFile(input, type) {
  const file = input.files[0];
  if (!file) return;
  const dz = document.getElementById(`dz-${type}`);
  const fp = document.getElementById(`fp-${type}`);
  if (dz) dz.style.display = "none";
  if (fp) {
    fp.style.display = "flex";
    document.getElementById(`fp-${type}-name`).textContent = file.name;
    document.getElementById(`fp-${type}-size`).textContent = formatBytes(file.size) + " · " + type.toUpperCase();
  }
}
function clearFile(type) {
  document.getElementById(`dz-${type}`).style.display = "block";
  document.getElementById(`fp-${type}`).style.display = "none";
  document.getElementById(`file-${type}`).value = "";
}

// ── Run scan ──────────────────────────────────────────────────────────────────
async function runScan() {
  const btn = document.getElementById("scan-btn");
  const activeTab = document.querySelector(".scan-tab.active");
  const tab = activeTab ? activeTab.dataset.tab : "video";

  let file = null, text = null, scanType = tab;
  if (tab === "video") file = document.getElementById("file-video").files[0];
  if (tab === "audio") { file = document.getElementById("file-audio").files[0]; scanType = "audio"; }
  if (tab === "text") text = document.getElementById("nlp-input").value.trim();

  if (!file && (!text || text.split(/\s+/).length < 5)) {
    showToast("Upload a file or enter at least 5 words of text", "warning");
    return;
  }

  const useFace = document.querySelector('.module-toggle[data-mod="face"]')?.classList.contains("active") ?? true;
  const useVoice = document.querySelector('.module-toggle[data-mod="voice"]')?.classList.contains("active") ?? true;
  const useNlp = document.querySelector('.module-toggle[data-mod="nlp"]')?.classList.contains("active") ?? true;

  btn.disabled = true;
  btn.innerHTML = `<div class="spinner" style="width:14px;height:14px"></div> Analysing...`;
  showToast("Running multi-modal analysis...", "info", 3000);

  try {
    const fd = new FormData();
    fd.append("scan_type", scanType);
    fd.append("face_enabled", useFace);
    fd.append("voice_enabled", useVoice);
    fd.append("nlp_enabled", useNlp);
    fd.append("fusion_enabled", true);
    if (file) fd.append("file", file);
    if (text) fd.append("input_text", text);

    const scan = await api.postForm("/api/scans/", fd);
    if (scan.detail) throw new Error(typeof scan.detail === "string" ? scan.detail : JSON.stringify(scan.detail));

    showScanResult({
      face: scan.face_score, voice: scan.voice_score, nlp: scan.nlp_score,
      fusion: Math.round(scan.fusion_score ?? 0), verdict: scan.verdict
    });

    loadStats();
    renderRecentScans();
    showToast("Scan complete!", "success");
  } catch (err) {
    showToast("Scan failed: " + err.message, "error", 4000);
  }

  btn.disabled = false;
  btn.innerHTML = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg> Run Analysis`;
}

function showScanResult({ face, voice, nlp, fusion, verdict }) {
  const modal = document.getElementById("scan-modal");
  const content = document.getElementById("scan-modal-content");
  const sc = scoreClass(fusion);
  const vc = { authentic: "authentic", suspicious: "suspicious", fake: "fake" }[verdict];
  const vLabel = { authentic: "Authentic — Identity Verified", suspicious: "Suspicious — Review Required", fake: "Deepfake Detected" }[verdict] || verdict;
  const color = { authentic: "var(--success)", suspicious: "var(--warning)", fake: "var(--danger)" }[verdict] || "var(--warning)";

  const rows = [["Face Authenticity", face, "var(--accent)"], ["Voice Match", voice, "var(--success)"], ["Linguistic Score", nlp, "var(--warning)"]]
    .filter(([, s]) => s != null)
    .map(([l, s, c]) => [l, Math.round(s), c]);

  content.innerHTML = `
    <h3 style="font-size:20px;font-weight:700;letter-spacing:-0.02em;color:var(--text-primary);margin-bottom:4px">Scan Complete</h3>
    <p style="font-size:12px;color:var(--text-tertiary);font-family:var(--font-mono);margin-bottom:24px">M3-ID Analysis · ${new Date().toLocaleTimeString()}</p>

    <div style="display:flex;flex-direction:column;gap:14px;margin-bottom:24px">
      ${rows.map(([l, s, c]) => `
        <div>
          <div style="display:flex;justify-content:space-between;margin-bottom:6px">
            <span style="font-size:12px;font-weight:600;color:var(--text-secondary)">${l}</span>
            <span style="font-size:13px;font-weight:700;font-family:var(--font-mono);color:${c}">${s}%</span>
          </div>
          <div class="score-bar-track"><div class="score-bar-fill" style="background:${c};width:${s}%"></div></div>
        </div>`).join("")}
    </div>

    <div style="background:var(--bg-subtle);border:1px solid var(--border);border-radius:var(--radius-lg);padding:20px;text-align:center;margin-bottom:20px">
      <div style="font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--text-tertiary);margin-bottom:8px">M3-ID Fusion Score</div>
      <div style="font-size:56px;font-weight:700;letter-spacing:-0.04em;color:${color};line-height:1">${fusion}%</div>
    </div>

    <div class="severity-badge ${verdict === "authentic" ? "CLEAR" : verdict === "suspicious" ? "MEDIUM" : "CRITICAL"}" style="width:100%;justify-content:center;margin-bottom:20px">
      ${vLabel}
    </div>

    <div style="display:flex;gap:10px">
      <button class="btn btn-secondary" style="flex:1" onclick="document.getElementById('scan-modal').style.display='none'">Close</button>
      <button class="btn btn-primary" style="flex:1" onclick="downloadReport(${fusion},'${verdict}')">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
        Download Report
      </button>
    </div>`;

  modal.style.display = "flex";
}

function downloadReport(fusion, verdict) {
  const txt = `DeepShield M3-ID — Scan Report\nGenerated: ${new Date().toLocaleString()}\n\nFusion Score: ${fusion}%\nVerdict: ${verdict.toUpperCase()}\n\nM3-ID Multi-Modal Identity Defender`;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([txt], { type: "text/plain" }));
  a.download = `M3ID_Report_${Date.now()}.txt`;
  a.click();
  showToast("Report downloaded", "success");
}

// Drop-and-drop init
initDropzone("dz-video", f => { const inp = document.getElementById("file-video"); const dt = new DataTransfer(); dt.items.add(f); inp.files = dt.files; handleFile(inp, "video"); });
initDropzone("dz-audio", f => { const inp = document.getElementById("file-audio"); const dt = new DataTransfer(); dt.items.add(f); inp.files = dt.files; handleFile(inp, "audio"); });

window.switchTab = switchTab;
window.toggleMod = toggleMod;
window.handleFile = handleFile;
window.clearFile = clearFile;
window.runScan = runScan;
window.downloadReport = downloadReport;
