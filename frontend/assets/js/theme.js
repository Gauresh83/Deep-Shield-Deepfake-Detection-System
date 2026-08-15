/* DeepShield M3-ID — Theme Manager */

const THEMES = ["light","dark","midnight","soft","system"];
const THEME_KEY = "m3id_theme";

function getSystemTheme() {
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function applyTheme(theme) {
  const resolved = theme === "system" ? getSystemTheme() : theme;
  document.documentElement.setAttribute("data-theme", resolved);
  document.documentElement.style.colorScheme = (resolved === "light" || resolved === "soft") ? "light" : "dark";
  // update active state in any open theme menus
  document.querySelectorAll(".theme-option").forEach(el => {
    el.classList.toggle("active", el.dataset.theme === theme);
  });
  // update icon in topbar button
  const btn = document.getElementById("themeBtn");
  if (btn) updateThemeIcon(btn, theme);
}

function updateThemeIcon(btn, theme) {
  const icons = {
    light:    `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>`,
    dark:     `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"/></svg>`,
    midnight: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>`,
    soft:     `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/></svg>`,
    system:   `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>`,
  };
  const iconWrap = btn.querySelector(".theme-btn-icon");
  if (iconWrap) iconWrap.innerHTML = icons[theme] || icons.system;
}

function setTheme(theme) {
  if (!THEMES.includes(theme)) theme = "system";
  localStorage.setItem(THEME_KEY, theme);
  applyTheme(theme);
  // close menu
  document.querySelectorAll(".theme-menu").forEach(m => m.classList.remove("open"));
}

function initTheme() {
  const saved = localStorage.getItem(THEME_KEY) || "system";
  applyTheme(saved);
  // watch system preference
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    const current = localStorage.getItem(THEME_KEY) || "system";
    if (current === "system") applyTheme("system");
  });
}

function buildThemeMenu(containerId) {
  const wrap = document.getElementById(containerId);
  if (!wrap) return;
  const current = localStorage.getItem(THEME_KEY) || "system";
  const items = [
    { value:"light",    label:"Light",    desc:"Clean & bright" },
    { value:"dark",     label:"Dark",     desc:"Easy on the eyes" },
    { value:"midnight", label:"Midnight", desc:"Deep space blue" },
    { value:"soft",     label:"Soft",     desc:"Warm & natural" },
    { value:"system",   label:"System",   desc:"Follows your OS" },
  ];
  wrap.innerHTML = items.map(t => `
    <div class="theme-option ${t.value === current ? "active":""}" data-theme="${t.value}" onclick="setTheme('${t.value}')">
      <div class="theme-dot ${t.value}"></div>
      <div style="flex:1">
        <div style="font-size:13px;font-weight:600;color:var(--text-primary)">${t.label}</div>
        <div style="font-size:11px;color:var(--text-tertiary)">${t.desc}</div>
      </div>
      <svg class="theme-check" style="display:${t.value===current?"block":"none"}" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>
    </div>`).join("");
}

function toggleThemeMenu(menuId) {
  const menu = document.getElementById(menuId);
  if (!menu) return;
  const isOpen = menu.classList.contains("open");
  // close all
  document.querySelectorAll(".theme-menu").forEach(m => m.classList.remove("open"));
  if (!isOpen) {
    buildThemeMenu(menuId + "-items");
    menu.classList.add("open");
  }
}

// close on outside click
document.addEventListener("click", e => {
  if (!e.target.closest(".theme-switcher")) {
    document.querySelectorAll(".theme-menu").forEach(m => m.classList.remove("open"));
  }
});

initTheme();
window.setTheme = setTheme;
window.toggleThemeMenu = toggleThemeMenu;
window.buildThemeMenu = buildThemeMenu;
