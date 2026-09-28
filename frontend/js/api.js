// Thin client for the FastAPI backend (see src/flutter_ai/api/app.py).

async function request(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch (e) { /* keep status text */ }
    throw new Error(detail);
  }
  return res.json();
}

export const api = {
  health: () => request("/api/health"),
  presets: () => request("/api/presets"),
  scenarios: (preset) => request(`/api/scenarios?preset=${encodeURIComponent(preset || "uav_flap")}`),
  flutter: (preset) => request("/api/flutter", { method: "POST", body: JSON.stringify({ preset }) }),
  simulate: (body) => request("/api/simulate", { method: "POST", body: JSON.stringify(body) }),
};
