/* DeepShield M3-ID — API Client */
const API_BASE = "http://localhost:8000";

async function apiRequest(endpoint, options = {}) {
  const token = getToken();
  const headers = { "Content-Type": "application/json", ...(token && !token.startsWith("demo_") ? { Authorization: `Bearer ${token}` } : {}), ...(options.headers || {}) };
  try {
    const res = await fetch(API_BASE + endpoint, { ...options, headers });
    if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.detail || `HTTP ${res.status}`); }
    return res.json();
  } catch (err) { throw err; }
}

const api = {
  get:  (ep)       => apiRequest(ep),
  post: (ep, body) => apiRequest(ep, { method: "POST", body: JSON.stringify(body) }),
  del:  (ep)       => apiRequest(ep, { method: "DELETE" }),
  postForm: (ep, fd) => {
    const token = getToken();
    return fetch(API_BASE + ep, { method: "POST", headers: token && !token.startsWith("demo_") ? { Authorization: `Bearer ${token}` } : {}, body: fd }).then(r => r.json());
  },
  health: () => fetch(API_BASE + "/health").then(r => r.json()).catch(() => ({ status: "offline" })),
};
window.api = api;
