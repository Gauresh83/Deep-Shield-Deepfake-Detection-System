/* DeepShield M3-ID — Theme Manager */

const THEMES = ["dark", "midnight", "soft", "system"];
const THEME_KEY = "m3id_theme";

const THEME_ITEMS = [
  { value: "dark", icon: "🌙", label: "Dark", desc: "Easy on the eyes" },
  { value: "midnight", icon: "🌌", label: "Midnight", desc: "Deep space blue" },
  { value: "soft", icon: "☀️", label: "Soft", desc: "Warm & natural" },
  { value: "system", icon: "🖥️", label: "System", desc: "Follows your OS" },
];

function getSystemTheme() {
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function applyTheme(theme) {
  const resolved = theme === "system" ? getSystemTheme() : theme;
  document.documentElement.setAttribute("data-theme", resolved);
  document.documentElement.style.colorScheme = (resolved === "light" || resolved === "soft") ? "light" : "dark";
  // sync active state + aria-selected on any rendered menu items
  document.querySelectorAll(".theme-option").forEach(el => {
    const isActive = el.dataset.theme === theme;
    el.classList.toggle("active", isActive);
    el.setAttribute("aria-selected", isActive ? "true" : "false");
    const check = el.querySelector(".theme-check");
    if (check) check.style.display = isActive ? "flex" : "none";
  });
}

function setTheme(theme) {
  if (!THEMES.includes(theme)) theme = "system";
  localStorage.setItem(THEME_KEY, theme);
  applyTheme(theme);
  closeAllThemeMenus(/* restoreFocus */ true);
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
  wrap.innerHTML = THEME_ITEMS.map(t => `
    <div class="theme-option ${t.value === current ? "active" : ""}"
         data-theme="${t.value}"
         role="option"
         tabindex="0"
         aria-selected="${t.value === current ? "true" : "false"}"
         onclick="setTheme('${t.value}')"
         onkeydown="handleThemeOptionKeydown(event, '${t.value}')">
      <span class="theme-option-icon" aria-hidden="true">${t.icon}</span>
      <span class="theme-option-text">
        <span class="theme-option-name">${t.label}</span>
        <span class="theme-option-desc">${t.desc}</span>
      </span>
      <svg class="theme-check" style="display:${t.value === current ? "flex" : "none"}" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
    </div>`).join("");
}

function handleThemeOptionKeydown(e, value) {
  const options = Array.from(document.querySelectorAll(`#${e.currentTarget.closest(".theme-menu").id} .theme-option`));
  const idx = options.indexOf(e.currentTarget);
  if (e.key === "Enter" || e.key === " " || e.key === "Spacebar") {
    e.preventDefault();
    setTheme(value);
  } else if (e.key === "ArrowDown") {
    e.preventDefault();
    (options[idx + 1] || options[0]).focus();
  } else if (e.key === "ArrowUp") {
    e.preventDefault();
    (options[idx - 1] || options[options.length - 1]).focus();
  } else if (e.key === "Escape") {
    e.preventDefault();
    closeAllThemeMenus(true);
  } else if (e.key === "Tab") {
    // allow Tab to naturally move focus, but close the menu afterward
    closeAllThemeMenus(false);
  }
}

function toggleThemeMenu(menuId) {
  const menu = document.getElementById(menuId);
  if (!menu) return;
  const trigger = menu.previousElementSibling && menu.previousElementSibling.matches("button")
    ? menu.previousElementSibling
    : document.querySelector(`[aria-controls="${menuId}"]`);
  const isOpen = menu.classList.contains("open");
  closeAllThemeMenus(false);
  if (!isOpen) {
    buildThemeMenu(menuId + "-items");
    menu.classList.add("open");
    if (trigger) trigger.setAttribute("aria-expanded", "true");
    // move focus to the currently active option for keyboard users
    requestAnimationFrame(() => {
      const active = menu.querySelector(".theme-option.active") || menu.querySelector(".theme-option");
      if (active) active.focus();
    });
  }
}

function closeAllThemeMenus(restoreFocus) {
  document.querySelectorAll(".theme-menu.open").forEach(m => {
    m.classList.remove("open");
    const trigger = m.previousElementSibling && m.previousElementSibling.matches("button")
      ? m.previousElementSibling
      : document.querySelector(`[aria-controls="${m.id}"]`);
    if (trigger) {
      trigger.setAttribute("aria-expanded", "false");
      if (restoreFocus) trigger.focus();
    }
  });
}

// close on outside click
document.addEventListener("click", e => {
  if (!e.target.closest(".theme-switcher")) {
    closeAllThemeMenus(false);
  }
});

// close on Escape from anywhere (e.g. focus is still on the trigger button)
document.addEventListener("keydown", e => {
  if (e.key === "Escape") {
    const openMenu = document.querySelector(".theme-menu.open");
    if (openMenu) closeAllThemeMenus(true);
  }
});

initTheme();
window.setTheme = setTheme;
window.toggleThemeMenu = toggleThemeMenu;
window.buildThemeMenu = buildThemeMenu;
window.handleThemeOptionKeydown = handleThemeOptionKeydown;
