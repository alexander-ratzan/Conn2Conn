/* viewer_kit runtime: error line, data loading, decoding, formatting, tooltip, storage, canvas helpers.
   Inlined into every built viewer by viewer.py, after the data blocks and before the page's own script.
   Pages call `const D = Kit.load();` and then use the helpers below. */
"use strict";
const Kit = (() => {
  const $ = id => document.getElementById(id);

  // ---------- visible errors ----------
  function fail(msg) { const el = $("kit-err"); if (el) { el.hidden = false; el.textContent = "Viewer error: " + msg; } }
  addEventListener("error", e => fail(`${e.message} (line ${e.lineno})`));
  addEventListener("unhandledrejection", e => fail(String(e.reason)));

  // ---------- data ----------
  function unpack(v) {                       // {"__chunks__": [...]} (split base64 / long strings) -> one string
    if (v && typeof v === "object" && !Array.isArray(v)) {
      if (Array.isArray(v.__chunks__)) return v.__chunks__.join("");
      for (const k of Object.keys(v)) v[k] = unpack(v[k]);
    }
    return v;
  }
  function load() {                          // meta block + every data set block -> {...meta, sets: {key: set}}
    let meta;
    try { meta = JSON.parse($("kit-meta").textContent); }
    catch (e) { fail(`could not read the metadata block (${e.message}); rebuild the viewer`); throw e; }
    meta.sets = {};
    for (const el of document.querySelectorAll("script.kit-set")) {
      try { meta.sets[el.dataset.key] = unpack(JSON.parse(el.textContent)); }
      catch (e) { fail(`could not read data block ${el.dataset.key} (${e.message})`); throw e; }
    }
    return meta;
  }
  function bytes(s) { const b = atob(s), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return u; }
  const int16 = s => new Int16Array(bytes(s).buffer);
  const float32 = s => new Float32Array(bytes(s).buffer);

  // ---------- small utilities ----------
  const fin = v => typeof v === "number" && Number.isFinite(v);   // null (NaN from Python) is not finite
  const fmt = (v, d = 2) => fin(v) ? v.toFixed(d) : "–";
  const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  function fillSelect(el, items, value) {    // items: [[value, text], ...]
    el.innerHTML = items.map(([v, t]) => `<option value="${v}">${t}</option>`).join("");
    if (value != null) el.value = value;
  }
  const store = {                            // localStorage, never required (private windows, previews)
    get(key, fallback) { try { const v = localStorage.getItem(key); return v == null ? fallback : JSON.parse(v); } catch (e) { return fallback; } },
    set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch (e) { /* ignore */ } },
  };
  function onTheme(cb) { matchMedia("(prefers-color-scheme: dark)").addEventListener("change", cb); }

  // ---------- tooltip ----------
  const tip = {
    show(e, html) {
      const t = $("kit-tip"); if (!html) return tip.hide();
      t.innerHTML = html; t.style.display = "block";
      t.style.left = Math.min(e.clientX + 14, innerWidth - t.offsetWidth - 8) + "px";
      t.style.top = Math.min(e.clientY + 14, innerHeight - t.offsetHeight - 8) + "px";
    },
    hide() { $("kit-tip").style.display = "none"; },
  };

  // ---------- self-checks (page JS vs Python reference values) ----------
  const checks = [];
  function check(label, ok, detail = "") {
    checks.push({ label, ok, detail });
    const el = $("kit-checks"); if (!el) return;
    el.innerHTML = checks.map(c => `<div class="status"><span style="color:var(${c.ok ? "--good" : "--critical"})">${c.ok ? "✓" : "✗"}</span>` +
      `${c.label}: ${c.ok ? "pass" : "FAIL"}${c.detail ? " · " + c.detail : ""}</div>`).join("");
  }

  // ---------- canvas ----------
  function fit(canvas) {                     // size the backing store to the CSS box (HiDPI); returns a cleared context
    const dpr = window.devicePixelRatio || 1, w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); }
    const ctx = canvas.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
    return { ctx, w, h };
  }
  function niceTicks(lo, hi, n) {
    const span = hi - lo || 1, mag = 10 ** Math.floor(Math.log10(span / n));
    const step = [1, 2, 5, 10].map(f => f * mag).find(s => span / s <= n) || 10 * mag;
    const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10)); return out;
  }
  function range(values, padFrac = 0.06) {   // finite min / max, padded
    let lo = Infinity, hi = -Infinity;
    for (const a of values) if (fin(a)) { if (a < lo) lo = a; if (a > hi) hi = a; }
    if (lo === Infinity) return [0, 1];
    const pad = (hi - lo) * padFrac || 1; return [lo - pad, hi + pad];
  }
  function axes(ctx, box, xr, yr, opts = {}) {   // grid lines, tick labels, baseline, x label
    const { l, t, w, h } = box;
    ctx.font = "11px system-ui, sans-serif"; ctx.fillStyle = css("--text-secondary"); ctx.strokeStyle = css("--grid"); ctx.lineWidth = 1;
    for (const v of niceTicks(yr[0], yr[1], opts.ny || 3)) {
      const y = t + h - (v - yr[0]) / (yr[1] - yr[0]) * h; ctx.beginPath(); ctx.moveTo(l, y); ctx.lineTo(l + w, y); ctx.stroke();
      ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillText(opts.yfmt ? opts.yfmt(v) : v, l - 6, y);
    }
    if (opts.xticks !== false) for (const v of opts.xtickVals || niceTicks(xr[0], xr[1], opts.nx || 8)) {
      const x = l + (v - xr[0]) / (xr[1] - xr[0]) * w; ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillText(opts.xfmt ? opts.xfmt(v) : v, x, t + h + 5);
    }
    ctx.strokeStyle = css("--axis"); ctx.beginPath(); ctx.moveTo(l, t + h); ctx.lineTo(l + w, t + h); ctx.stroke();
    if (opts.xlabel) { ctx.textAlign = "right"; ctx.textBaseline = "top"; ctx.fillStyle = css("--text-muted"); ctx.fillText(opts.xlabel, l + w, t + h + 18); }
  }
  function hexRGB(h) { h = h.replace("#", ""); return [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16)); }
  function divergingLUT() {                  // 256 RGB triplets, -1 .. +1, from the --div-* tokens (neutral midpoint)
    const stops = [["--div-neg-2", -1], ["--div-neg-1", -0.5], ["--div-mid", 0], ["--div-pos-1", 0.5], ["--div-pos-2", 1]].map(([n, v]) => [v, hexRGB(css(n))]);
    const lut = new Uint8ClampedArray(256 * 3);
    for (let i = 0; i < 256; i++) {
      const v = i / 255 * 2 - 1; let k = 0; while (k < stops.length - 2 && v > stops[k + 1][0]) k++;
      const [v0, c0] = stops[k], [v1, c1] = stops[k + 1], f = (v - v0) / (v1 - v0);
      for (let c = 0; c < 3; c++) lut[i * 3 + c] = c0[c] + f * (c1[c] - c0[c]);
    }
    return lut;
  }
  function matrixImage(M, P, order, lim, lut) {  // P x P ImageData of M (row-major) in display order, clipped to ±lim
    const img = new ImageData(P, P), mid = hexRGB(css("--div-mid"));
    for (let a = 0; a < P; a++) for (let b = 0; b < P; b++) {
      const v = M[order[a] * P + order[b]], o = (a * P + b) * 4;
      if (!fin(v)) { img.data[o] = mid[0]; img.data[o + 1] = mid[1]; img.data[o + 2] = mid[2]; img.data[o + 3] = 255; continue; }
      const i = Math.max(0, Math.min(255, Math.round((Math.max(-lim, Math.min(lim, v)) / lim + 1) / 2 * 255)));
      img.data[o] = lut[i * 3]; img.data[o + 1] = lut[i * 3 + 1]; img.data[o + 2] = lut[i * 3 + 2]; img.data[o + 3] = 255;
    }
    return img;
  }
  function colorbar(canvas, lut) {
    const ctx = canvas.getContext("2d"), W = canvas.width, H = canvas.height, img = ctx.createImageData(W, H);
    for (let x = 0; x < W; x++) { const i = Math.round(x / (W - 1) * 255);
      for (let y = 0; y < H; y++) { const o = (y * W + x) * 4; img.data[o] = lut[i * 3]; img.data[o + 1] = lut[i * 3 + 1]; img.data[o + 2] = lut[i * 3 + 2]; img.data[o + 3] = 255; } }
    ctx.putImageData(img, 0, 0);
  }

  return { $, fail, load, bytes, int16, float32, fin, fmt, css, fillSelect, store, onTheme, tip, check, checks,
           fit, niceTicks, range, axes, hexRGB, divergingLUT, matrixImage, colorbar };
})();
