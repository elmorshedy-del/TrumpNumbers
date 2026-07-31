/* Minimal SVG chart helpers. No dependencies, no build step.
 *
 * Mark specs follow the project's dataviz rules: thin marks, 4px rounded
 * data-ends anchored to the baseline, 2px lines, >=8px markers, a 2px surface
 * gap between adjacent fills, recessive grid, and a hover layer on every plot.
 */

const NS = "http://www.w3.org/2000/svg";

function el(name, attrs = {}, parent = null) {
  const n = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) {
    if (v !== null && v !== undefined) n.setAttribute(k, v);
  }
  if (parent) parent.appendChild(n);
  return n;
}

function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/* ---------- shared tooltip ---------- */

let tip;
function tooltip() {
  if (!tip) {
    tip = document.createElement("div");
    tip.className = "tooltip";
    document.body.appendChild(tip);
  }
  return tip;
}

export function showTip(evt, title, rows) {
  const t = tooltip();
  t.innerHTML =
    `<div class="t-title">${title}</div>` +
    rows.map((r) => `<div class="t-row">${r}</div>`).join("");
  t.style.opacity = "1";
  const pad = 14;
  let x = evt.clientX + pad;
  let y = evt.clientY + pad;
  const box = t.getBoundingClientRect();
  if (x + box.width > window.innerWidth - 8) x = evt.clientX - box.width - pad;
  if (y + box.height > window.innerHeight - 8) y = evt.clientY - box.height - pad;
  t.style.left = `${x}px`;
  t.style.top = `${y}px`;
}

export function hideTip() {
  if (tip) tip.style.opacity = "0";
}

/* ---------- scales & frame ---------- */

function niceTicks(min, max, count = 5) {
  const span = max - min || 1;
  const raw = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm >= 7.5 ? 10 : norm >= 3.5 ? 5 : norm >= 1.5 ? 2 : 1) * mag;
  const out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push(v);
  return out;
}

function frame(mount, { height = 240, padL = 44, padR = 14, padT = 12, padB = 30 } = {}) {
  mount.innerHTML = "";
  const width = Math.max(mount.clientWidth || 640, 280);
  const svg = el("svg", {
    viewBox: `0 0 ${width} ${height}`,
    width: "100%",
    height,
    role: "img",
  }, mount);
  return { svg, width, height, padL, padR, padT, padB, iw: width - padL - padR, ih: height - padT - padB };
}

function yAxis(f, min, max, fmt = (v) => v) {
  const ticks = niceTicks(min, max, 4);
  for (const t of ticks) {
    const y = f.padT + f.ih - ((t - min) / (max - min || 1)) * f.ih;
    el("line", {
      x1: f.padL, x2: f.padL + f.iw, y1: y, y2: y,
      stroke: css("--grid"), "stroke-width": 1,
    }, f.svg);
    const label = el("text", {
      x: f.padL - 8, y: y + 4, "text-anchor": "end",
      fill: css("--text-muted"), "font-size": 11,
    }, f.svg);
    label.textContent = fmt(t);
  }
  return ticks;
}

/* ---------- bar chart (magnitude, one series) ---------- */

export function barChart(mount, data, opts = {}) {
  const {
    height = 230, color = "--series-1", valueKey = "value", labelKey = "label",
    refLine = null, refLabel = "", tipTitle = (d) => d[labelKey],
    tipRows = (d) => [`${d[valueKey]}`], xTickEvery = null, yFmt = (v) => v,
  } = opts;
  if (!data.length) return;

  const f = frame(mount, { height });
  const max = Math.max(...data.map((d) => d[valueKey]), refLine || 0) * 1.08 || 1;
  yAxis(f, 0, max, yFmt);

  // 2px surface gap between adjacent bars; 4px rounded top anchored to baseline.
  const slot = f.iw / data.length;
  const bw = Math.max(slot - 2, 1);
  const fill = css(color);

  data.forEach((d, i) => {
    const h = Math.max((d[valueKey] / max) * f.ih, d[valueKey] > 0 ? 1.5 : 0);
    const x = f.padL + i * slot + 1;
    const y = f.padT + f.ih - h;
    const r = Math.min(4, bw / 2, h);
    const rect = el("path", {
      d: `M${x},${f.padT + f.ih} V${y + r} a${r},${r} 0 0 1 ${r},-${r} H${x + bw - r} a${r},${r} 0 0 1 ${r},${r} V${f.padT + f.ih} Z`,
      fill, opacity: 0.92,
    }, f.svg);
    const hit = el("rect", {
      x: f.padL + i * slot, y: f.padT, width: slot, height: f.ih, fill: "transparent",
    }, f.svg);
    const on = (e) => { rect.setAttribute("opacity", "1"); showTip(e, tipTitle(d), tipRows(d)); };
    hit.addEventListener("mousemove", on);
    hit.addEventListener("mouseenter", on);
    hit.addEventListener("mouseleave", () => { rect.setAttribute("opacity", "0.92"); hideTip(); });
  });

  if (refLine !== null) {
    const y = f.padT + f.ih - (refLine / max) * f.ih;
    el("line", {
      x1: f.padL, x2: f.padL + f.iw, y1: y, y2: y,
      stroke: css("--series-2"), "stroke-width": 2, "stroke-dasharray": "5 4",
    }, f.svg);
    if (refLabel) {
      const t = el("text", {
        x: f.padL + f.iw, y: y - 6, "text-anchor": "end",
        fill: css("--series-2"), "font-size": 11, "font-weight": 600,
      }, f.svg);
      t.textContent = refLabel;
    }
  }

  const every = xTickEvery || Math.ceil(data.length / 8);
  data.forEach((d, i) => {
    if (i % every) return;
    const t = el("text", {
      x: f.padL + i * slot + bw / 2, y: f.height - 10, "text-anchor": "middle",
      fill: css("--text-muted"), "font-size": 11,
    }, f.svg);
    t.textContent = d[labelKey];
  });
}

/* ---------- weekly history + forecast fan ---------- */

export function forecastChart(mount, history, forecast, opts = {}) {
  const { height = 280 } = opts;
  if (!history.length) return;

  const f = frame(mount, { height, padR: 74 });
  const fanX = history.length; // forecast sits one slot past the last week
  const n = fanX + 1;

  const vals = history.map((d) => d.total);
  const hi = forecast ? forecast.quantiles["0.975"] : Math.max(...vals);
  const lo = forecast ? forecast.quantiles["0.025"] : Math.min(...vals);
  const yMax = Math.max(...vals, hi) * 1.06;
  const yMin = Math.min(...vals, lo) * 0.92;

  yAxis(f, yMin, yMax);
  const X = (i) => f.padL + (i / (n - 1)) * f.iw;
  const Y = (v) => f.padT + f.ih - ((v - yMin) / (yMax - yMin || 1)) * f.ih;

  // Interval bands, widest first. Deliberately drawn as an asymmetric fan:
  // the distribution is lopsided and a symmetric error bar would misrepresent it.
  if (forecast) {
    const q = forecast.quantiles;
    const bands = [
      ["0.025", "0.975", "--band-weak"],
      ["0.1", "0.9", "--band-strong"],
    ];
    for (const [a, b, tone] of bands) {
      el("path", {
        d: `M${X(fanX - 1)},${Y(history[history.length - 1].total)} L${X(fanX)},${Y(q[b])} L${X(fanX)},${Y(q[a])} Z`,
        fill: css(tone),
      }, f.svg);
    }
  }

  const line = history.map((d, i) => `${i ? "L" : "M"}${X(i)},${Y(d.total)}`).join(" ");
  el("path", { d: line, fill: "none", stroke: css("--series-1"), "stroke-width": 2,
    "stroke-linejoin": "round", "stroke-linecap": "round" }, f.svg);

  if (forecast) {
    const last = history[history.length - 1];
    el("path", {
      d: `M${X(fanX - 1)},${Y(last.total)} L${X(fanX)},${Y(forecast.median)}`,
      fill: "none", stroke: css("--series-2"), "stroke-width": 2, "stroke-dasharray": "5 4",
    }, f.svg);
    // >=8px marker, with a 2px surface ring so it reads over the band.
    el("circle", { cx: X(fanX), cy: Y(forecast.median), r: 5.5,
      fill: css("--series-2"), stroke: css("--surface-1"), "stroke-width": 2 }, f.svg);
    const lab = el("text", { x: X(fanX) + 10, y: Y(forecast.median) + 4,
      fill: css("--text-primary"), "font-size": 12, "font-weight": 640 }, f.svg);
    lab.textContent = Math.round(forecast.median);
  }

  history.forEach((d, i) => {
    const hit = el("circle", { cx: X(i), cy: Y(d.total), r: 9, fill: "transparent" }, f.svg);
    hit.addEventListener("mousemove", (e) =>
      showTip(e, `Week ending ${d.week_ending}`, [`<b>${d.total}</b> posts`]));
    hit.addEventListener("mouseleave", hideTip);
  });

  const every = Math.ceil(history.length / 6);
  history.forEach((d, i) => {
    if (i % every) return;
    const t = el("text", { x: X(i), y: f.height - 10, "text-anchor": "middle",
      fill: css("--text-muted"), "font-size": 11 }, f.svg);
    t.textContent = d.week_ending.slice(5);
  });
}

/* ---------- horizontal ranked bars (leaderboard) ---------- */

export function rankedBars(mount, rows, opts = {}) {
  const { valueKey = "value", labelKey = "label", height = null,
          colorFor = () => "--series-1", fmt = (v) => v.toFixed(2), tipRows = null } = opts;
  if (!rows.length) return;

  const rowH = 24;
  const h = height || rows.length * rowH + 22;
  const f = frame(mount, { height: h, padL: 148, padT: 6, padB: 16, padR: 54 });
  const max = Math.max(...rows.map((r) => r[valueKey])) * 1.02 || 1;

  rows.forEach((r, i) => {
    const y = f.padT + i * rowH;
    const w = Math.max((r[valueKey] / max) * f.iw, 1.5);
    const bh = rowH - 8; // 2px+ gap between adjacent fills
    const rad = Math.min(4, w, bh / 2);
    el("path", {
      d: `M${f.padL},${y} H${f.padL + w - rad} a${rad},${rad} 0 0 1 ${rad},${rad} V${y + bh - rad} a${rad},${rad} 0 0 1 -${rad},${rad} H${f.padL} Z`,
      fill: css(colorFor(r)), opacity: 0.92,
    }, f.svg);

    const nameEl = el("text", { x: f.padL - 9, y: y + bh / 2 + 4, "text-anchor": "end",
      fill: css("--text-primary"), "font-size": 12 }, f.svg);
    nameEl.textContent = r[labelKey];

    // Direct value label — also the relief for the light-mode contrast WARN.
    const vEl = el("text", { x: f.padL + w + 7, y: y + bh / 2 + 4,
      fill: css("--text-secondary"), "font-size": 11.5 }, f.svg);
    vEl.textContent = fmt(r[valueKey]);

    if (tipRows) {
      const hit = el("rect", { x: 0, y, width: f.width, height: rowH, fill: "transparent" }, f.svg);
      hit.addEventListener("mousemove", (e) => showTip(e, r[labelKey], tipRows(r)));
      hit.addEventListener("mouseleave", hideTip);
    }
  });
}

/* ---------- ACF / PIT style bars with a reference band ---------- */

export function stemChart(mount, values, opts = {}) {
  const { height = 200, band = null, labels = null, tipTitle = (i) => `lag ${i}`,
          color = "--series-1", startIndex = 0 } = opts;
  if (!values.length) return;

  const f = frame(mount, { height });
  const maxAbs = Math.max(...values.map(Math.abs), band || 0) * 1.2 || 1;
  const min = Math.min(0, -maxAbs);
  yAxis(f, min, maxAbs, (v) => v.toFixed(2));

  const Y = (v) => f.padT + f.ih - ((v - min) / (maxAbs - min || 1)) * f.ih;
  const slot = f.iw / values.length;

  if (band) {
    el("rect", { x: f.padL, y: Y(band), width: f.iw, height: Math.abs(Y(-band) - Y(band)),
      fill: css("--series-2"), opacity: 0.10 }, f.svg);
  }
  el("line", { x1: f.padL, x2: f.padL + f.iw, y1: Y(0), y2: Y(0),
    stroke: css("--text-muted"), "stroke-width": 1 }, f.svg);

  values.forEach((v, i) => {
    const x = f.padL + i * slot + slot / 2;
    const significant = band ? Math.abs(v) > band : true;
    el("line", { x1: x, x2: x, y1: Y(0), y2: Y(v),
      stroke: css(significant ? color : "--text-muted"),
      "stroke-width": Math.min(slot * 0.5, 7), "stroke-linecap": "round",
      opacity: significant ? 0.92 : 0.45 }, f.svg);

    const hit = el("rect", { x: f.padL + i * slot, y: f.padT, width: slot, height: f.ih,
      fill: "transparent" }, f.svg);
    hit.addEventListener("mousemove", (e) =>
      showTip(e, tipTitle(i + startIndex), [`<b>${v.toFixed(3)}</b>`,
        band ? (significant ? "outside the noise band" : "within noise") : ""].filter(Boolean)));
    hit.addEventListener("mouseleave", hideTip);
  });

  const every = Math.ceil(values.length / 10);
  values.forEach((_, i) => {
    if (i % every) return;
    const t = el("text", { x: f.padL + i * slot + slot / 2, y: f.height - 10,
      "text-anchor": "middle", fill: css("--text-muted"), "font-size": 11 }, f.svg);
    t.textContent = labels ? labels[i] : i + startIndex;
  });
}

/* ---------- calibration scatter: sharpness vs coverage ---------- */

export function calibrationScatter(mount, rows, opts = {}) {
  const { height = 300 } = opts;
  if (!rows.length) return;

  const f = frame(mount, { height, padL: 52, padB: 42, padR: 20 });
  const xs = rows.map((r) => r.coverage80);
  const ys = rows.map((r) => r.width80);
  const xMin = Math.min(...xs, 0.5), xMax = Math.max(...xs, 0.95);
  const yMin = 0, yMax = Math.max(...ys) * 1.12;

  yAxis(f, yMin, yMax);
  const X = (v) => f.padL + ((v - xMin) / (xMax - xMin || 1)) * f.iw;
  const Y = (v) => f.padT + f.ih - ((v - yMin) / (yMax - yMin || 1)) * f.ih;

  // The target: an 80% interval should contain the truth 80% of the time.
  el("line", { x1: X(0.8), x2: X(0.8), y1: f.padT, y2: f.padT + f.ih,
    stroke: css("--series-2"), "stroke-width": 2, "stroke-dasharray": "5 4" }, f.svg);
  const tl = el("text", { x: X(0.8), y: f.padT - 2, "text-anchor": "middle",
    fill: css("--series-2"), "font-size": 11, "font-weight": 600 }, f.svg);
  tl.textContent = "calibrated (0.80)";

  for (const t of [0.5, 0.6, 0.7, 0.8, 0.9]) {
    if (t < xMin || t > xMax) continue;
    const lab = el("text", { x: X(t), y: f.height - 22, "text-anchor": "middle",
      fill: css("--text-muted"), "font-size": 11 }, f.svg);
    lab.textContent = t.toFixed(2);
  }
  const xlab = el("text", { x: f.padL + f.iw / 2, y: f.height - 4, "text-anchor": "middle",
    fill: css("--text-muted"), "font-size": 11.5 }, f.svg);
  xlab.textContent = "coverage of the 80% interval  →  honest at 0.80";

  rows.forEach((r) => {
    const off = Math.abs(r.coverage80 - 0.8);
    const tone = off <= 0.07 ? "--series-3" : off <= 0.14 ? "--series-1" : "--series-2";
    el("circle", { cx: X(r.coverage80), cy: Y(r.width80), r: 5.5, fill: css(tone),
      stroke: css("--surface-1"), "stroke-width": 2 }, f.svg);
    const hit = el("circle", { cx: X(r.coverage80), cy: Y(r.width80), r: 12, fill: "transparent" }, f.svg);
    hit.addEventListener("mousemove", (e) => showTip(e, r.model, [
      `coverage of 80% interval: <b>${(r.coverage80 * 100).toFixed(0)}%</b>`,
      `mean interval width: <b>${r.width80.toFixed(0)}</b> posts`,
      `CRPS: <b>${r.crps.toFixed(2)}</b>`,
    ]));
    hit.addEventListener("mouseleave", hideTip);
  });
}

export function onResize(fn) {
  let t;
  window.addEventListener("resize", () => { clearTimeout(t); t = setTimeout(fn, 160); });
}
