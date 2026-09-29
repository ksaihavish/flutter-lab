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
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) {  // pydantic validation errors: "field: message; ..."
        detail = body.detail.map((d) => `${(d.loc || []).filter((x) => x !== "body").join(".")}: ${d.msg}`).join("; ");
      } else detail = JSON.stringify(body.detail);
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
  // Guardian live view (src/flutter_ai/api/guardian.py): start a flight, poll it, play back the recording
  guardianOptions: () => request("/api/guardian/options"),
  guardianStart: (body) => request("/api/guardian/flights", { method: "POST", body: JSON.stringify(body) }),
  guardianFlight: (id) => request(`/api/guardian/flights/${encodeURIComponent(id)}`),
};
