// Single place the frontend talks to the FastAPI backend.
// Empty string = same origin as whatever served this page. Since the backend
// (promptops/main.py) now serves these HTML/JS files itself (via StaticFiles),
// the frontend and API always share one origin, locally and once deployed --
// no URL to edit when you move from localhost to a public host.
const API_BASE = "";

// FastAPI error bodies look like {"detail": "..."} (or, for validation
// errors, {"detail": [...]}) -- pull the real reason out instead of just
// reporting the status code, so a Sarvam auth/credit/network failure shows
// up as an actual sentence instead of a bare "-> 502" no one can act on.
async function _errorDetail(r) {
  try {
    const body = await r.json();
    if (typeof body.detail === "string") return body.detail;
    if (body.detail) return JSON.stringify(body.detail);
  } catch (e) {
    // response wasn't JSON -- fall through to the generic message
  }
  return null;
}

async function apiGet(path) {
  const r = await fetch(`${API_BASE}${path}`);
  if (!r.ok) {
    const detail = await _errorDetail(r);
    throw new Error(detail ? `GET ${path} -> ${r.status}: ${detail}` : `GET ${path} -> ${r.status}`);
  }
  return r.json();
}

async function apiPost(path, body) {
  const r = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) {
    const detail = await _errorDetail(r);
    throw new Error(detail ? `POST ${path} -> ${r.status}: ${detail}` : `POST ${path} -> ${r.status}`);
  }
  return r.json();
}

async function apiDelete(path) {
  const r = await fetch(`${API_BASE}${path}`, { method: "DELETE" });
  if (!r.ok) {
    const detail = await _errorDetail(r);
    throw new Error(detail ? `DELETE ${path} -> ${r.status}: ${detail}` : `DELETE ${path} -> ${r.status}`);
  }
  return r.json();
}

// Make sure the demo data exists; seed it once if the DB is empty so the pages
// are never blank on a fresh backend. Returns the /prompts payload.
async function ensureSeeded() {
  let data = await apiGet("/prompts");
  if (!data.prompts || data.prompts.length === 0) {
    await apiPost("/demo/seed");
    data = await apiGet("/prompts");
  }
  return data;
}

// Map the backend's status vocabulary (green/amber/red) to the UI's label +
// Tailwind colour family, so both pages render statuses consistently.
const STATUS_UI = {
  green: { label: "HEALTHY", color: "emerald" },
  amber: { label: "FLAGGED", color: "amber" },
  red: { label: "BROKEN", color: "red" },
};
