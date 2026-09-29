// Animated wing sections: one lane per controller, drawn to true geometric scale.
// 3-DOF sections show the trailing-edge flap (solid) and the commanded flap angle (thin outline).
import { cssVar } from "./chart.js";

const T_NACA = 0.12;
const yt = (x) => 5 * T_NACA * (0.2969 * Math.sqrt(x) - 0.126 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4);

// NACA 0012 outline (chord-normalised, 0 = leading edge), optionally split at a hinge.
function naca(split = null) {
  const xs = Array.from({ length: 61 }, (_, i) => (1 - Math.cos((Math.PI * i) / 60)) / 2);
  if (split !== null && !xs.includes(split)) { xs.push(split); xs.sort((a, b) => a - b); }
  const outline = (x0, x1) => {
    const seg = xs.filter((x) => x >= x0 - 1e-12 && x <= x1 + 1e-12);
    return [...seg.map((x) => [x, yt(x)]), ...[...seg].reverse().map((x) => [x, -yt(x)])];
  };
  if (split === null) return { body: outline(0, 1), flap: null };
  return { body: outline(0, split), flap: outline(split, 1) };
}

// Ordinal mode meters: effort modes (simulator) and fault-tolerant modes (Guardian view)
const MODE_LEVEL = { efficient: 1, balanced: 2, aggressive: 3, emergency: 4, nominal: 1, robust: 2, cautious: 3, safe: 4 };

function nearest(ts, t) {
  let lo = 0, hi = ts.length - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (ts[mid] < t) lo = mid; else hi = mid; }
  return Math.abs(ts[lo] - t) <= Math.abs(ts[hi] - t) ? lo : hi;
}

export class Viewport {
  constructor(canvas) {
    this.canvas = canvas;
    this.data = null;
    this.streaks = Array.from({ length: 36 }, () => ({ x: Math.random(), y: Math.random(), len: 0.03 + Math.random() * 0.05 }));
    new ResizeObserver(() => this.render(this.time ?? 0, 0)).observe(canvas.parentElement);
  }

  setData(response, cases) {
    const p = response.params;
    const hingeN = p.flap_hinge === null || p.flap_hinge === undefined ? null : (1 + p.flap_hinge) / 2;
    this.shape = naca(hingeN);
    this.hingeN = hingeN;
    this.data = { response, cases };
    this.render(0, 0);
  }

  /** @param {number} t simulation time; @param {number} wallDt real seconds since last frame (flow streaks) */
  render(t, wallDt = 0, playSpeed = 0.25) {
    this.time = t;
    const wrap = this.canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const W = wrap.width, H = wrap.height;
    this.canvas.width = Math.round(W * dpr);
    this.canvas.height = Math.round(H * dpr);
    const ctx = this.canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const font = cssVar("--font");

    if (!this.data) {
      ctx.fillStyle = cssVar("--muted");
      ctx.font = `13px ${font}`;
      ctx.textAlign = "center";
      ctx.fillText(this.emptyText || "Run a simulation to animate the wing", W / 2, H / 2);
      return;
    }

    const { response, cases } = this.data;
    const p = response.params;
    const flap = p.n_dof === 3;
    const L = H / cases.length;
    const labelW = Math.min(210, W * 0.36);
    const areaW = W - labelW;
    const chordM = 2 * p.b;
    const C = Math.min(areaW * 0.42, 1.2 * L);   // chord in px
    const pxpm = C / chordM;                      // one scale for chord AND plunge
    const xEA = (1 + p.a) * p.b;                  // elastic axis from leading edge [m]
    const cx = labelW + areaW / 2;

    // flow streaks (speed capped so they stay visible in slow motion)
    const flowPx = Math.min(response.velocity * pxpm * playSpeed, 900);
    ctx.strokeStyle = cssVar("--grid");
    ctx.lineWidth = 1.5;
    for (const s of this.streaks) {
      s.x += (flowPx * wallDt) / areaW;
      if (s.x > 1.05) { s.x = -0.1; s.y = Math.random(); }
      const x0 = labelW + s.x * areaW, y0 = s.y * H;
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0 + s.len * areaW, y0); ctx.stroke();
    }

    // body-frame (m, y up, origin at EA) -> screen, for plunge h and pitch alpha
    const toScreen = (h, alpha, cy) => ([x, y]) => {
      const ca = Math.cos(-alpha), sa = Math.sin(-alpha);
      return [cx + (x * ca - y * sa) * pxpm, cy + h * pxpm - (x * sa + y * ca) * pxpm];
    };
    // chord-normalised outline point -> body frame, rotating flap points by beta about the hinge
    const bodyPt = ([xn, yn]) => [xn * chordM - xEA, yn * chordM];
    const flapPt = (beta) => ([xn, yn]) => {
      const hx = this.hingeN * chordM - xEA;
      const [x, y] = bodyPt([xn, yn]);
      const cb = Math.cos(-beta), sb = Math.sin(-beta);  // trailing-edge down = clockwise
      return [hx + (x - hx) * cb - y * sb, (x - hx) * sb + y * cb];
    };

    cases.forEach((c, lane) => {
      const r = c.result;
      const top = lane * L, cy = top + L / 2;
      const i = nearest(r.t, t);
      const failed = r.exceeded_limits && t >= r.failure_time - 1e-9;
      const h = r.h[i], alpha = r.alpha[i];
      const beta = flap ? r.beta[i] : 0;
      const betaCmd = flap && p.actuation === "flap_servo" ? r.u[0][i] : null;
      const color = cssVar(c.color);

      if (failed) { ctx.fillStyle = cssVar("--critical-wash"); ctx.fillRect(0, top, W, L); }
      if (lane > 0) {
        ctx.strokeStyle = cssVar("--border"); ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(0, top + 0.5); ctx.lineTo(W, top + 0.5); ctx.stroke();
      }

      // ---- label column
      ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
      ctx.fillStyle = color; ctx.fillRect(14, cy - 22, 16, 3);
      ctx.fillStyle = cssVar("--ink"); ctx.font = `600 13px ${font}`;
      this._fitText(ctx, c.label, 36, cy - 17, labelW - 44);
      ctx.font = `12px ${font}`; ctx.fillStyle = cssVar("--ink-2");
      if (failed) {
        ctx.fillStyle = cssVar("--critical");
        this._warnIcon(ctx, 14, cy - 6);
        ctx.fillText(`Limit exceeded at ${r.failure_time.toFixed(2)} s`, 34, cy + 5);
      } else if (c.case.supervisor !== "none") {
        const mode = r.mode[i];
        const text = `Mode: ${mode}`;
        ctx.fillText(text, 14, cy + 5);
        const lvl = MODE_LEVEL[mode] || 0;
        for (let k = 0; k < 4; k++) {
          ctx.fillStyle = k < lvl ? cssVar(`--seq-${k + 1}`) : cssVar("--grid");
          ctx.fillRect(14 + ctx.measureText(text).width + 8 + k * 6, cy - 4, 4, 10);
        }
      } else {
        ctx.fillText(c.case.controller === "open_loop" ? "No control" : "Fixed gains", 14, cy + 5);
      }
      ctx.fillStyle = cssVar("--ink-2");
      const deg = (v) => ((v * 180) / Math.PI).toFixed(1);
      ctx.fillText(`h ${(h * 1000).toFixed(1)} mm  α ${deg(alpha)}°${flap ? `  β ${deg(beta)}°` : ""}`, 14, cy + 23);

      // ---- structural limits
      const limY = p.h_limit * pxpm;
      ctx.strokeStyle = cssVar("--grid"); ctx.lineWidth = 1;
      for (const y of [cy - limY, cy + limY]) {
        if (y < top + 2 || y > top + L - 2) continue;
        ctx.beginPath(); ctx.moveTo(cx - C * 0.9, Math.round(y) + 0.5); ctx.lineTo(cx + C * 0.9, Math.round(y) + 0.5); ctx.stroke();
      }
      if (cy - limY > top + 12) {
        ctx.fillStyle = cssVar("--muted"); ctx.font = `11px ${font}`; ctx.textAlign = "right";
        ctx.fillText("structural limit", cx + C * 0.9, cy - limY - 3);
        ctx.textAlign = "left";
      }

      // ---- gust indicator
      const wg = r.w_gust[i] || 0;
      if (Math.abs(wg) > 0.05) {
        const len = Math.max(-L * 0.35, Math.min(L * 0.35, -wg * 8));
        ctx.strokeStyle = cssVar("--ink-2"); ctx.fillStyle = cssVar("--ink-2"); ctx.lineWidth = 1.5;
        const gx = labelW + 22;
        this._arrow(ctx, gx, cy, gx, cy + len);
        ctx.font = `11px ${font}`;
        ctx.fillText(`gust ${wg >= 0 ? "+" : ""}${wg.toFixed(1)} m/s`, gx + 8, cy + len / 2);
      }

      // ---- ghost at rest, spring, wing
      const rest = toScreen(0, 0, cy);
      ctx.strokeStyle = cssVar("--axis"); ctx.lineWidth = 1;
      this._path(ctx, this.shape.body.map(bodyPt).map(rest)); ctx.stroke();
      if (this.shape.flap) { this._path(ctx, this.shape.flap.map(bodyPt).map(rest)); ctx.stroke(); }

      const eaY = cy + h * pxpm;
      const mountY = top + 8;
      ctx.strokeStyle = cssVar("--muted"); ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(cx - 10, mountY); ctx.lineTo(cx + 10, mountY); ctx.stroke();
      this._spring(ctx, cx, mountY, eaY - 4);

      const now = toScreen(h, alpha, cy);
      const stroke = failed ? cssVar("--critical") : color;
      const drawPart = (pts) => {
        this._path(ctx, pts);
        ctx.globalAlpha = 0.22; ctx.fillStyle = stroke; ctx.fill();
        ctx.globalAlpha = 1; ctx.strokeStyle = stroke; ctx.lineWidth = 2; ctx.stroke();
      };
      drawPart(this.shape.body.map(bodyPt).map(now));
      if (this.shape.flap) {
        drawPart(this.shape.flap.map(flapPt(beta)).map(now));
        if (betaCmd !== null && !failed) {  // commanded flap: thin outline, shows servo lag
          ctx.globalAlpha = 0.6; ctx.strokeStyle = color; ctx.lineWidth = 1;
          this._path(ctx, this.shape.flap.map(flapPt(betaCmd)).map(now)); ctx.stroke();
          ctx.globalAlpha = 1;
        }
      }
      ctx.fillStyle = cssVar("--ink"); ctx.beginPath(); ctx.arc(cx, eaY, 3, 0, Math.PI * 2); ctx.fill();

      // ---- idealised force/moment actuators (2-DOF presets)
      if (p.actuation === "force_moment" && c.case.controller !== "open_loop" && !failed) {
        const uh = (r.u[0][i] || 0) / p.u_max[0];
        const ua = (r.u[1][i] || 0) / p.u_max[1];
        ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 2;
        if (Math.abs(uh) > 0.02) this._arrow(ctx, cx + C * 0.55, eaY, cx + C * 0.55, eaY + uh * L * 0.3);
        if (Math.abs(ua) > 0.02) {
          // nose-up moment (nose on the left) turns clockwise on screen
          const sweep = ua * Math.PI * 0.75;
          ctx.beginPath(); ctx.arc(cx, eaY, 16, -Math.PI / 2, -Math.PI / 2 + sweep, sweep < 0); ctx.stroke();
        }
      }
    });
  }

  _path(ctx, pts) {
    ctx.beginPath();
    pts.forEach(([x, y], k) => (k ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.closePath();
  }

  _spring(ctx, x, y0, y1) {
    const n = 8, amp = 5, len = y1 - y0;
    ctx.beginPath(); ctx.moveTo(x, y0);
    for (let k = 1; k < n; k++) ctx.lineTo(x + (k % 2 ? amp : -amp), y0 + (len * k) / n);
    ctx.lineTo(x, y1); ctx.stroke();
  }

  _arrow(ctx, x0, y0, x1, y1) {
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    const ang = Math.atan2(y1 - y0, x1 - x0), s = 6;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x1 - s * Math.cos(ang - 0.45), y1 - s * Math.sin(ang - 0.45));
    ctx.lineTo(x1 - s * Math.cos(ang + 0.45), y1 - s * Math.sin(ang + 0.45));
    ctx.closePath(); ctx.fill();
  }

  _warnIcon(ctx, x, y) {
    ctx.beginPath(); ctx.moveTo(x + 7, y); ctx.lineTo(x + 14, y + 13); ctx.lineTo(x, y + 13); ctx.closePath(); ctx.fill();
    ctx.fillStyle = cssVar("--surface"); ctx.fillRect(x + 6.2, y + 4, 1.6, 5); ctx.fillRect(x + 6.2, y + 10, 1.6, 1.6);
    ctx.fillStyle = cssVar("--critical");
  }

  _fitText(ctx, text, x, y, maxW) {
    let s = text;
    while (s.length > 3 && ctx.measureText(s).width > maxW) s = s.slice(0, -2);
    ctx.fillText(s === text ? s : s.trimEnd() + "…", x, y);
  }
}
