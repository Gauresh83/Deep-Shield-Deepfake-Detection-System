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
    now.toLocaleDateString("en-IN", { weekday:"long", year:"numeric", month:"long", day:"numeric" });

  loadStats();
  renderModuleStatus();
  renderActivityChart();
  renderRecentScans();
})();

// ── Stats ─────────────────────────────────────────────────────────────────────
async function loadStats() {
  try {
    const token = getToken();
    if (token && !token.startsWith("demo_")) {
      const res = await fetch(`${API}/api/analytics/overview`, { headers: { Authorization: `Bearer ${token}` } });
      if (res.ok) {
        const d = await res.json();
        animateNumber(document.getElementById("s-total"), d.total_scans);
        animateNumber(document.getElementById("s-fakes"), d.fakes_caught);
        animateNumber(document.getElementById("s-susp"),  d.suspicious);
        return;
      }
    }
  } catch (_) {}
  // Demo fallback
  animateNumber(document.getElementById("s-total"), 142);
  animateNumber(document.getElementById("s-fakes"), 37);
  animateNumber(document.getElementById("s-susp"),  21);
}

// ── Module status ─────────────────────────────────────────────────────────────
const MODULE_META = [
  { key:"face",  label:"Face Detection",   desc:"EfficientNet-B4",    color:"var(--accent)",  acc:91, icon:`<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="8" r="5"/><path d="M20 21a8 8 0 10-16 0"/></svg>` },
  { key:"voice", label:"Voice Analysis",   desc:"AASIST / MFCC",      color:"var(--success)", acc:87, icon:`<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 1a3 3 0 00-3 3v8a3 3 0 006 0V4a3 3 0 00-3-3z"/><path d="M19 10v2a7 7 0 01-14 0v-2"/></svg>` },
  { key:"nlp",   label:"Linguistic NLP",   desc:"BERT Statistical",   color:"var(--warning)", acc:94, icon:`<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>` },
  { key:"fusion",label:"Risk Fusion",      desc:"MLP + XAI",          color:"var(--info)",    acc:98, icon:`<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>` },
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
function renderActivityChart() {
  const ctx = document.getElementById("activity-chart");
  if (!ctx) return;
  const isDark = ["dark","midnight"].includes(document.documentElement.getAttribute("data-theme"));
  const gridColor = isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.05)";
  const tickColor = isDark ? "#4F566B" : "#8B93A7";

  new Chart(ctx, {
    type: "line",
    data: {
      labels: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
      datasets: [
        { label:"Scans", data:[12,19,8,24,17,9,31], borderColor:"var(--accent)", backgroundColor:"rgba(59,110,245,0.08)", borderWidth:2, tension:0.4, fill:true, pointBackgroundColor:"var(--accent)", pointRadius:3 },
        { label:"Fakes", data:[2,4,1,7,3,1,8],      borderColor:"var(--danger)",  backgroundColor:"rgba(239,68,68,0.05)",  borderWidth:2, tension:0.4, fill:true, pointBackgroundColor:"var(--danger)",  pointRadius:3 },
      ]
    },
    options: {
      responsive:true,
      plugins:{ legend:{ labels:{ color:tickColor, font:{ family:"JetBrains Mono", size:9 }, boxWidth:10 } } },
      scales:{
        x:{ ticks:{ color:tickColor, font:{ size:9 } }, grid:{ color:gridColor } },
        y:{ ticks:{ color:tickColor, font:{ size:9 } }, grid:{ color:gridColor } },
      }
    }
  });
}

// ── Recent scans table ────────────────────────────────────────────────────────
const DEMO_SCANS = [
  { id:1, file_name:"interview_clip.mp4", scan_type:"video", face_score:22, voice_score:31, nlp_score:18, fusion_score:23, verdict:"fake",       created_at: new Date(Date.now()-120000).toISOString() },
  { id:2, file_name:"call_rec_04.wav",    scan_type:"audio", face_score:null, voice_score:41, nlp_score:35, fusion_score:38, verdict:"suspicious", created_at: new Date(Date.now()-900000).toISOString() },
  { id:3, file_name:"meeting_02.mp4",     scan_type:"video", face_score:92, voice_score:88, nlp_score:81, fusion_score:88, verdict:"authentic",  created_at: new Date(Date.now()-2700000).toISOString() },
  { id:4, file_name:"email_text.txt",     scan_type:"text",  face_score:null, voice_score:null, nlp_score:28, fusion_score:28, verdict:"fake",    created_at: new Date(Date.now()-7200000).toISOString() },
];

let localScans = [...DEMO_SCANS];

function renderRecentScans() {
  const tbody = document.getElementById("recent-scans-body");
  if (!tbody) return;
  if (!localScans.length) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;padding:32px;color:var(--text-tertiary)">No scans yet</td></tr>`;
    return;
  }
  const modDots = (s) => [
    ["var(--accent)",  s.face_score  != null],
    ["var(--success)", s.voice_score != null],
    ["var(--warning)", s.nlp_score   != null],
    ["var(--info)",    true],
  ].map(([c,a]) => `<div style="width:7px;height:7px;border-radius:50%;background:${a?c:"var(--border)"};flex-shrink:0"></div>`).join("");

  tbody.innerHTML = localScans.slice(0,6).map(s => {
    const vc = { authentic:"authentic", suspicious:"suspicious", fake:"fake", pending:"pending" }[s.verdict] || "pending";
    const vl = { authentic:"Authentic", suspicious:"Suspicious", fake:"Fake", pending:"Pending" }[s.verdict] || "—";
    const sc = s.fusion_score != null ? Math.round(s.fusion_score) : null;
    return `<tr>
      <td>
        <div class="cell-primary" style="max-width:140px" class="truncate">${s.file_name || "scan"}</div>
        <div class="cell-mono" style="font-size:11px;color:var(--text-tertiary);margin-top:2px">${s.scan_type}</div>
      </td>
      <td><div style="display:flex;gap:4px;align-items:center">${modDots(s)}</div></td>
      <td class="cell-mono">${s.face_score  != null ? Math.round(s.face_score)+"%" : "—"}</td>
      <td class="cell-mono">${s.voice_score != null ? Math.round(s.voice_score)+"%" : "—"}</td>
      <td class="cell-mono">${s.nlp_score   != null ? Math.round(s.nlp_score)+"%" : "—"}</td>
      <td><span class="score-val ${sc!=null ? scoreClass(sc) : ""}">${sc != null ? sc+"%" : "—"}</span></td>
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
  btn.disabled = true;
  btn.innerHTML = `<div class="spinner" style="width:14px;height:14px"></div> Analysing...`;
  showToast("Running multi-modal analysis...", "info", 3000);

  await new Promise(r => setTimeout(r, 2200));

  // Mock result
  const face  = Math.floor(Math.random() * 50) + 25;
  const voice = Math.floor(Math.random() * 50) + 20;
  const nlp   = Math.floor(Math.random() * 50) + 15;
  const fusion = Math.round((face * 0.35 + voice * 0.35 + nlp * 0.30));
  const verdict = fusion >= 70 ? "authentic" : fusion >= 45 ? "suspicious" : "fake";

  const newScan = { id: Date.now(), file_name: "scan_" + Date.now() + ".mp4", scan_type:"video", face_score:face, voice_score:voice, nlp_score:nlp, fusion_score:fusion, verdict, created_at: new Date().toISOString() };
  localScans.unshift(newScan);
  renderRecentScans();

  showScanResult({ face, voice, nlp, fusion, verdict });

  btn.disabled = false;
  btn.innerHTML = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg> Run Analysis`;
}

function showScanResult({ face, voice, nlp, fusion, verdict }) {
  const modal   = document.getElementById("scan-modal");
  const content = document.getElementById("scan-modal-content");
  const sc = scoreClass(fusion);
  const vc = { authentic:"authentic", suspicious:"suspicious", fake:"fake" }[verdict];
  const vLabel = { authentic:"Authentic — Identity Verified", suspicious:"Suspicious — Review Required", fake:"Deepfake Detected" }[verdict];
  const color = { authentic:"var(--success)", suspicious:"var(--warning)", fake:"var(--danger)" }[verdict];

  content.innerHTML = `
    <h3 style="font-size:20px;font-weight:700;letter-spacing:-0.02em;color:var(--text-primary);margin-bottom:4px">Scan Complete</h3>
    <p style="font-size:12px;color:var(--text-tertiary);font-family:var(--font-mono);margin-bottom:24px">M3-ID Analysis · ${new Date().toLocaleTimeString()}</p>

    <div style="display:flex;flex-direction:column;gap:14px;margin-bottom:24px">
      ${[["Face Authenticity", face, "var(--accent)"],["Voice Match", voice, "var(--success)"],["Linguistic Score", nlp, "var(--warning)"]].map(([l,s,c]) => `
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
