/* Charts for the credit dashboard.
 *
 * Two forms only, each chosen for the job its data does:
 *
 *   trendLine   - change over time for ONE metric. A single series, so no
 *                 legend: the title names it. 2px stroke, >=8px markers,
 *                 crosshair and tooltip on hover.
 *   riskBars    - counts by risk signal. This is STATUS, not category, so it
 *                 uses the reserved good/warning/serious/critical steps and
 *                 every bar carries its label -- a status colour never carries
 *                 meaning alone.
 *
 * Palettes were validated with the dataviz validator against this surface
 * (#141922), not eyeballed. The three categorical slots pass all-pairs in dark
 * mode; the status four are the documented fixed steps, all clear of 3:1 on
 * this surface, and paired with text labels because warning and serious sit
 * close together by design.
 *
 * No dual axes anywhere: two measures of different scale get two charts.
 */

const VIZ = {
  surface: '#141922',
  grid: 'rgba(139,152,169,.16)',
  ink: '#e6edf3',
  dim: '#8b98a9',
  series: ['#3987e5', '#d95926', '#199e70'],   // categorical, validated all-pairs
  status: {                                     // reserved, never used as series
    healthy: '#0ca30c',
    watch: '#fab219',
    elevated_risk: '#ec835a',
    severe_risk: '#d03b3b',
    insufficient_evidence: '#8b98a9',
  },
};

function svgEl(name, attrs = {}) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', name);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

/* Line chart for one metric over time. Undefined periods break the line
 * rather than being interpolated across -- a gap in the filings is a fact
 * about the data, and joining over it would invent a value. */
function trendLine(host, { points, metric, direction }) {
  host.innerHTML = '';
  const usable = points.filter(p => p.value !== null);
  if (usable.length < 2) {
    host.innerHTML = `<p class="note">Not enough visible periods to plot ${metric}.</p>`;
    return;
  }
  const W = host.clientWidth || 640, H = 220;
  const M = { t: 18, r: 18, b: 34, l: 52 };
  const iw = W - M.l - M.r, ih = H - M.t - M.b;
  const vals = usable.map(p => p.value);
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (lo === hi) { lo -= 1; hi += 1; }
  const pad = (hi - lo) * 0.12; lo -= pad; hi += pad;
  const x = i => M.l + (points.length === 1 ? iw / 2 : (i / (points.length - 1)) * iw);
  const y = v => M.t + ih - ((v - lo) / (hi - lo)) * ih;

  const svg = svgEl('svg', { width: '100%', height: H, viewBox: `0 0 ${W} ${H}`,
                             role: 'img', 'aria-label': `${metric} over time` });

  // recessive grid + axis labels
  for (let g = 0; g <= 3; g++) {
    const v = lo + (hi - lo) * (g / 3), yy = y(v);
    svg.appendChild(svgEl('line', { x1: M.l, x2: W - M.r, y1: yy, y2: yy,
                                    stroke: VIZ.grid, 'stroke-width': 1 }));
    const t = svgEl('text', { x: M.l - 8, y: yy + 4, fill: VIZ.dim,
                              'font-size': 10, 'text-anchor': 'end' });
    t.textContent = Math.abs(v) >= 1000 ? v.toExponential(1) : v.toFixed(2);
    svg.appendChild(t);
  }

  // segments, broken across undefined periods
  let run = [];
  const flush = () => {
    if (run.length > 1) {
      svg.appendChild(svgEl('polyline', {
        points: run.map(([i, v]) => `${x(i)},${y(v)}`).join(' '),
        fill: 'none', stroke: VIZ.series[0], 'stroke-width': 2,
        'stroke-linejoin': 'round', 'stroke-linecap': 'round',
      }));
    }
    run = [];
  };
  points.forEach((p, i) => { p.value === null ? flush() : run.push([i, p.value]); });
  flush();

  // markers, with a surface ring so overlaps stay legible
  points.forEach((p, i) => {
    if (p.value === null) return;
    svg.appendChild(svgEl('circle', { cx: x(i), cy: y(p.value), r: 4.5,
      fill: VIZ.series[0], stroke: VIZ.surface, 'stroke-width': 2 }));
  });

  // direct-label the endpoints only -- never a number on every point
  const last = points.map((p, i) => [i, p]).filter(([, p]) => p.value !== null).pop();
  if (last) {
    const [i, p] = last;
    const lab = svgEl('text', { x: Math.min(x(i) + 8, W - M.r), y: y(p.value) - 9,
      fill: VIZ.ink, 'font-size': 11, 'text-anchor': 'end' });
    lab.textContent = p.value.toFixed(2);
    svg.appendChild(lab);
  }

  // x labels: first, middle, last only, to avoid collision
  [0, Math.floor((points.length - 1) / 2), points.length - 1].forEach(i => {
    if (i < 0 || !points[i]) return;
    const t = svgEl('text', { x: x(i), y: H - 10, fill: VIZ.dim, 'font-size': 10,
      'text-anchor': i === 0 ? 'start' : i === points.length - 1 ? 'end' : 'middle' });
    t.textContent = points[i].period_end.slice(0, 7);
    svg.appendChild(t);
  });

  host.appendChild(svg);

  // hover: crosshair + tooltip, hit target wider than the mark
  const tip = document.createElement('div');
  tip.className = 'tip'; tip.style.display = 'none';
  host.appendChild(tip);
  const rule = svgEl('line', { stroke: VIZ.dim, 'stroke-width': 1,
    'stroke-dasharray': '3 3', y1: M.t, y2: M.t + ih, opacity: 0 });
  svg.appendChild(rule);
  svg.addEventListener('mousemove', ev => {
    const box = svg.getBoundingClientRect();
    const px = (ev.clientX - box.left) * (W / box.width);
    let best = 0, bd = Infinity;
    points.forEach((p, i) => { const d = Math.abs(x(i) - px); if (d < bd) { bd = d; best = i; } });
    const p = points[best];
    rule.setAttribute('x1', x(best)); rule.setAttribute('x2', x(best));
    rule.setAttribute('opacity', 1);
    tip.style.display = 'block';
    tip.style.left = Math.min(x(best) * (box.width / W) + 12, box.width - 130) + 'px';
    tip.style.top = '10px';
    tip.innerHTML = `<b>${p.period_end}</b><br>${metric} `
      + (p.value === null ? '<i>not computable</i>' : p.value.toFixed(4));
  });
  svg.addEventListener('mouseleave', () => {
    rule.setAttribute('opacity', 0); tip.style.display = 'none';
  });

  if (direction) {
    const cap = document.createElement('p');
    cap.className = 'note';
    cap.textContent = `Direction over the visible history: ${direction}.`
      + (points.some(p => p.value === null) ? ' Gaps are periods the metric could not be computed.' : '');
    host.appendChild(cap);
  }
}

/* Horizontal bars, counts by risk signal. Status colours + a text label on
 * every bar, so the colour is never the only carrier of meaning. */
function riskBars(host, counts) {
  host.innerHTML = '';
  const order = ['healthy', 'watch', 'elevated_risk', 'severe_risk', 'insufficient_evidence'];
  const rows = order.filter(k => counts[k]).map(k => [k, counts[k]]);
  if (!rows.length) { host.innerHTML = '<p class="note">Nothing assessed yet.</p>'; return; }
  const max = Math.max(...rows.map(r => r[1]));
  const wrap = document.createElement('div');
  wrap.className = 'bars';
  rows.forEach(([k, n]) => {
    const row = document.createElement('div');
    row.className = 'bar-row';
    row.innerHTML = `<span class="bar-label">${k.replace(/_/g, ' ')}</span>`
      + `<span class="bar-track"><span class="bar-fill" style="width:${(n / max) * 100}%;`
      + `background:${VIZ.status[k]}"></span></span>`
      + `<span class="bar-val">${n}</span>`;
    row.title = `${n} filer(s) assessed ${k.replace(/_/g, ' ')}`;
    wrap.appendChild(row);
  });
  host.appendChild(wrap);
}
