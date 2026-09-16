/* DeepShield M3-ID — Utilities */

/* ── Toast notifications ─────────────────────────────────────── */
const TOAST_ICONS = {
  success: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#12B76A" stroke-width="2.5" stroke-linecap="round"><polyline points="20 6 9 17 4 12"/></svg>`,
  error: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#EF4444" stroke-width="2.5" stroke-linecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>`,
  warning: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2.5" stroke-linecap="round"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`,
  info: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#0EA5E9" stroke-width="2.5" stroke-linecap="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>`,
};

function showToast(message, type = "info", duration = 4000, title = null) {
  let container = document.getElementById("toast-container");
  if (!container) {
    container = document.createElement("div");
    container.id = "toast-container";
    container.className = "toast-container";
    document.body.appendChild(container);
  }
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.innerHTML = `
    <div class="toast-icon">${TOAST_ICONS[type] || TOAST_ICONS.info}</div>
    <div class="toast-content">
      ${title ? `<div class="toast-title">${title}</div>` : ""}
      <div class="toast-msg">${message}</div>
    </div>
    <button class="toast-close" onclick="this.closest('.toast').remove()">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
    </button>`;
  container.appendChild(toast);
  setTimeout(() => {
    toast.classList.add("leaving");
    setTimeout(() => toast.remove(), 280);
  }, duration);
  return toast;
}

/* ── Animations ──────────────────────────────────────────────── */
function animateNumber(el, target, duration = 1600, decimals = 0) {
  const start = performance.now();
  const update = (now) => {
    const p = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - p, 3);
    const val = target * eased;
    el.textContent = decimals ? val.toFixed(decimals) : Math.floor(val);
    if (p < 1) requestAnimationFrame(update);
  };
  requestAnimationFrame(update);
}

function animateBar(el, targetPct, delay = 0) {
  el.style.width = "0%";
  setTimeout(() => { el.style.width = targetPct + "%"; }, delay);
}

/* ── Intersection observer for reveal animations ─────────────── */
function observeReveal(selector, cls = "page-enter") {
  const obs = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) { e.target.classList.add(cls); obs.unobserve(e.target); }
    });
  }, { threshold: 0.1 });
  document.querySelectorAll(selector).forEach(el => {
    el.style.opacity = "0";
    obs.observe(el);
  });
}

/* ── Mobile sidebar ──────────────────────────────────────────── */
function initMobileSidebar() {
  const sidebar = document.getElementById("sidebar");
  const overlay = document.getElementById("sidebar-overlay");
  const menuBtn = document.getElementById("mobile-menu-btn");
  if (!sidebar) return;
  function open() { sidebar.classList.add("open"); overlay && overlay.classList.add("visible"); }
  function close() { sidebar.classList.remove("open"); overlay && overlay.classList.remove("visible"); }
  menuBtn && menuBtn.addEventListener("click", open);
  overlay && overlay.addEventListener("click", close);
  window.openSidebar = open;
  window.closeSidebar = close;
}

/* ── Format helpers ──────────────────────────────────────────── */
function formatBytes(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / 1048576).toFixed(2) + " MB";
}
function timeAgo(date) {
  const diff = Date.now() - new Date(date).getTime();
  const s = Math.floor(diff / 1000);
  if (s < 60) return "just now";
  if (s < 3600) return Math.floor(s / 60) + "m ago";
  if (s < 86400) return Math.floor(s / 3600) + "h ago";
  return Math.floor(s / 86400) + "d ago";
}
function scoreColor(score) {
  if (score >= 70) return "var(--success)";
  if (score >= 45) return "var(--warning)";
  return "var(--danger)";
}
function scoreClass(score) {
  if (score >= 70) return "high";
  if (score >= 45) return "mid";
  return "low";
}
function verdictClass(v) {
  return { authentic: "success", suspicious: "warning", fake: "danger", pending: "neutral" }[v] || "neutral";
}

/* ── Auth helpers ────────────────────────────────────────────── */
function getUser() {
  const raw = sessionStorage.getItem("m3id_user") || localStorage.getItem("m3id_user");
  try { return raw ? JSON.parse(raw) : null; } catch { return null; }
}
function getToken() {
  return localStorage.getItem("m3id_access_token") || sessionStorage.getItem("m3id_access_token");
}
function requireAuth(redirectTo = "login.html") {
  const u = getUser();
  if (!u || !u.loggedIn) { window.location.href = redirectTo; return false; }
  return u;
}
function setUserUI(user) {
  document.querySelectorAll("[data-user-name]").forEach(el => el.textContent = user.name || "User");
  document.querySelectorAll("[data-user-avatar]").forEach(el => el.textContent = (user.avatar || user.name?.slice(0, 2) || "U").toUpperCase());
  document.querySelectorAll("[data-user-role]").forEach(el => el.textContent = user.role || "Researcher");
}
function logout() {
  if (!confirm("Are you sure you want to log out?")) return;
  ["m3id_user", "m3id_access_token", "m3id_refresh_token"].forEach(k => {
    localStorage.removeItem(k); sessionStorage.removeItem(k);
  });
  window.location.href = "../pages/login.html";
}

/* ── User menu dropdown ─────────────────────────────────────────── */
function toggleUserMenu(e) {
  e.stopPropagation();
  const menu = document.getElementById("user-dropdown-menu");
  if (!menu) return;
  const isOpen = menu.style.display === "block";
  menu.style.display = isOpen ? "none" : "block";
}
document.addEventListener("click", (e) => {
  const menu = document.getElementById("user-dropdown-menu");
  if (menu && menu.style.display === "block" && !menu.contains(e.target)) {
    menu.style.display = "none";
  }
});

/* ── Drag-drop ───────────────────────────────────────────────── */
function initDropzone(dzId, onFile) {
  const dz = document.getElementById(dzId);
  if (!dz) return;
  dz.addEventListener("dragover", e => { e.preventDefault(); dz.classList.add("dragging"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("dragging"));
  dz.addEventListener("drop", e => {
    e.preventDefault(); dz.classList.remove("dragging");
    const file = e.dataTransfer.files[0];
    if (file) onFile(file);
  });
}

window.showToast = showToast; window.animateNumber = animateNumber; window.animateBar = animateBar;
window.observeReveal = observeReveal; window.initMobileSidebar = initMobileSidebar;
window.formatBytes = formatBytes; window.timeAgo = timeAgo; window.scoreColor = scoreColor;
window.scoreClass = scoreClass; window.verdictClass = verdictClass;
window.getUser = getUser; window.getToken = getToken; window.requireAuth = requireAuth;
window.setUserUI = setUserUI; window.logout = logout; window.initDropzone = initDropzone;
window.toggleUserMenu = toggleUserMenu;
