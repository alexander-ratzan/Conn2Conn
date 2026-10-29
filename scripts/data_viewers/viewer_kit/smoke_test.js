// viewer_kit smoke test: run a built viewer's scripts under node with a permissive DOM stub.
//   node smoke_test.js <page.html>
// Fails (exit 1) on an exception, on text in #kit-err, or on a failed Kit.check. Drawing and layout are not tested.
"use strict";
const fs = require("fs");
const vm = require("vm");

const html = fs.readFileSync(process.argv[2], "utf8");
const scripts = [...html.matchAll(/<script([^>]*)>([\s\S]*?)<\/script>/g)].map(m => ({ attrs: m[1], text: m[2] }));
const attr = (a, k) => (a.match(new RegExp(`${k}="([^"]*)"`)) || [])[1];

// initial control values, as a browser would have them (first / selected option, value=, checked)
const initial = {};
for (const m of html.matchAll(/<select[^>]*id="([^"]+)"[^>]*>([\s\S]*?)<\/select>/g)) {
  const opts = [...m[2].matchAll(/<option([^>]*)>([^<]*)/g)].map(o => ({ v: attr(o[1], "value") ?? o[2], sel: /selected/.test(o[1]) }));
  if (opts.length) initial[m[1]] = { value: (opts.find(o => o.sel) || opts[0]).v };
}
for (const m of html.matchAll(/<input([^>]*)>/g)) {
  const id = attr(m[1], "id"); if (!id) continue;
  initial[id] = { value: attr(m[1], "value") ?? "", checked: /\schecked/.test(m[1]) };
}

function stub(name, init = {}) {
  const store = { ...init };
  const fn = function () { return stub(name + "()"); };
  return new Proxy(fn, {
    get(t, k) {
      if (k in store) return store[k];
      if (k === Symbol.toPrimitive) return hint => (hint === "number" ? 0 : "");
      if (k === Symbol.iterator) return function* () {};
      if (k === "then") return undefined;
      if (["clientWidth", "offsetWidth", "width"].includes(k)) return 800;
      if (["clientHeight", "offsetHeight", "height"].includes(k)) return 400;
      if (["value", "textContent", "innerHTML"].includes(k)) return "";
      if (k === "checked" || k === "hidden") return false;
      if (k === "length") return 0;
      if (k === "dataset" || k === "style") return (store[k] = {});
      if (k === "querySelectorAll") return () => [];
      if (k === "getContext") return () => (store.__ctx ||= stub(name + ".ctx"));
      if (k === "createImageData") return (w, h) => new ImageData(w, h);
      return stub(name + "." + String(k));
    },
    set(t, k, v) {
      store[k] = v;
      if ((k === "innerHTML") && store.value === undefined) {      // a filled <select> takes its first option's value
        const o = String(v).match(/<option[^>]*value="([^"]*)"/); if (o) store.value = o[1];
      }
      return true;
    },
  });
}
class ImageData { constructor(w, h) { this.width = w; this.height = h; this.data = new Uint8ClampedArray(w * h * 4); } }

const elements = {};
const jsonById = {}, sets = [];
for (const s of scripts) {
  if (!/application\/json/.test(s.attrs)) continue;
  const id = attr(s.attrs, "id"), key = attr(s.attrs, "data-key");
  const el = { textContent: s.text, dataset: { key }, id };
  if (id) jsonById[id] = el; else if (/kit-set/.test(s.attrs)) sets.push(el);
}
const document = {
  getElementById: id => jsonById[id] || (elements[id] ||= stub("#" + id, initial[id] || {})),
  querySelectorAll: sel => (sel === "script.kit-set" ? sets : []),
  querySelector: sel => stub(sel),
  createElement: tag => stub("<" + tag + ">"),
  createElementNS: (ns, tag) => stub("<" + tag + ">"),
  documentElement: stub("html"),
  body: stub("body"),
  addEventListener() {},
};
const errors = [];
const ctx = {
  document, console, ImageData, atob: s => Buffer.from(s, "base64").toString("binary"),
  getComputedStyle: () => ({ getPropertyValue: () => "#808080" }),
  matchMedia: () => ({ matches: false, addEventListener() {} }),
  addEventListener: (type, cb) => { if (type === "error") ctx.__onerror = cb; },
  requestAnimationFrame: cb => setTimeout(cb, 0), setTimeout, clearTimeout,
  localStorage: { getItem: () => null, setItem() {} },
  innerWidth: 1200, innerHeight: 800, devicePixelRatio: 1,
  Math, JSON, Uint8Array, Int16Array, Float32Array, Float64Array, Uint8ClampedArray, Array, Object, Number, String, Set, Map,
};
ctx.window = ctx;
vm.createContext(ctx);

const t0 = Date.now();
let n = 0;
for (const s of scripts) {
  if (/application\/json/.test(s.attrs) || !s.text.trim()) continue;
  try { vm.runInContext(s.text, ctx, { filename: `script${n}` }); } catch (e) { errors.push(`script ${n}: ${e.stack.split("\n").slice(0, 3).join(" | ")}`); }
  n++;
}
setTimeout(() => {
  const err = elements["kit-err"] && elements["kit-err"].textContent;
  if (err) errors.push("#kit-err: " + err);
  const checks = vm.runInContext("typeof Kit === 'undefined' ? [] : Kit.checks", ctx);
  for (const c of checks) { console.log(`  check ${c.ok ? "pass" : "FAIL"}: ${c.label}${c.detail ? " · " + c.detail : ""}`); if (!c.ok) errors.push("check failed: " + c.label); }
  if (errors.length) { console.error(errors.join("\n")); process.exit(1); }
  console.log(`smoke test OK: ${process.argv[2]} (${n} scripts, ${sets.length} data sets, ${Date.now() - t0} ms)`);
}, 50);
