/* Corporate Distress Assessment screen.
 *
 * A diagnosis of filings already made. Three rules hold throughout, and each
 * is structural rather than a line in the disclaimer:
 *
 *   Nothing is called "current". Every state carries the period it came from
 *   and, where the filing is old, how old. Diebold at a 2024 prediction date
 *   is reading a 2022 balance sheet, and a reader must see that.
 *
 *   Reported and calculated are separate tables. A filed figure can be checked
 *   against the document; a ratio can only be checked by trusting the formula.
 *   Mixing them would present both as equally verifiable.
 *
 *   Zone names describe a condition, never an outcome. "Elevated stress" is a
 *   statement about a balance sheet. "Likely to fail" is a forecast, and this
 *   screen has no basis for one.
 *
 * Colour follows the reserved status scale: amber and crimson appear only
 * against a limit the filed figures actually breach, never as decoration.
 */

const ZONE_COLOR = {
  stable:       '#0ca30c',
  monitored:    '#fab219',
  elevated:     '#ec835a',
  severe:       '#d03b3b',
  insufficient: '#8794a8',
};
const ZONE_ORDER = ['stable', 'monitored', 'elevated', 'severe'];

const money = v => {
  if (v === null || v === undefined) return '—';
  const a = Math.abs(v);
  if (a >= 1e9) return (v / 1e9).toFixed(2) + 'bn';
  if (a >= 1e6) return (v / 1e6).toFixed(1) + 'm';
  if (a >= 1e3) return (v / 1e3).toFixed(0) + 'k';
  return v.toFixed(0);
};
const ratio = v => (v === null || v === undefined) ? '—'
  : (Math.abs(v) >= 1000 ? v.toExponential(2) : v.toFixed(2));

/* Segmented linear gauge. The active segment is filled; the rest are outlines,
 * so the reading is legible without relying on hue alone. */
function stressGauge(d) {
  const active = ZONE_ORDER.indexOf(d.zone);
  const segs = ZONE_ORDER.map((z, i) => {
    const on = i === active;
    const col = ZONE_COLOR[z];
    const lbl = d.zones.find(x => x.key === z)?.label || z;
    return `<div class="seg${on ? ' on' : ''}" style="--c:${col}">
        <div class="seg-bar"></div><div class="seg-lbl">${lbl}</div></div>`;
  }).join('');
  const note = d.zones.find(x => x.key === d.zone)?.note || '';
  const breached = d.calculated.filter(c => c.breached).length;
  const computable = d.calculated.filter(c => c.computable).length;
  return `<div class="card">
    <h3>Condition as of ${d.period_end || d.as_of}</h3>
    <div class="gauge">${segs}</div>
    <div class="gauge-read">
      <span class="zone-chip" style="--c:${ZONE_COLOR[d.zone]}">${d.zone_label}</span>
      <span class="gauge-sub">${breached} of ${computable} conventional limits breached
        in the figures filed for the period ending ${d.period_end || '—'}.</span>
    </div>
    <p class="note">${note} This is a reading of filings already made, assessed
      as of ${d.as_of}. It is not a projection.</p>
    ${d.stale ? `<p class="note warnnote">The newest usable filing is
      ${d.filing_age_days} days old. Every figure on this screen describes that
      vintage, and the position may since have changed in either direction.</p>` : ''}
  </div>`;
}

/* Reported facts and calculated indices, deliberately side by side but never
 * merged -- the distinction is the point. */
function factorMatrix(d) {
  const rep = d.reported.map(r => `<tr><td>${r.label}</td>
      <td class="num">${r.computable ? money(r.value) : '<span class="dim">not tagged</span>'}</td></tr>`).join('');
  const calc = d.calculated.map(c => `<tr>
      <td>${c.label}</td>
      <td class="num ${c.breached ? 'sev' : ''}">${c.computable ? ratio(c.value) : '<span class="dim">n/a</span>'}</td>
      <td class="num dim">${c.threshold}</td>
      <td>${c.breached ? '<span class="pill breach">breached</span>'
          : c.computable ? '<span class="pill within">within</span>' : ''}</td></tr>`).join('');
  return `<div class="grid2">
    <div class="card">
      <h3>Reported figures <span class="tier backtested">as filed</span></h3>
      <table><tr><th>Line item</th><th style="text-align:right">Value (USD)</th></tr>${rep}</table>
      <p class="note">Taken directly from the company's own filing for the period
        ending ${d.period_end || '—'}. Each can be checked against the document.
        &ldquo;Not tagged&rdquo; means the filer did not report that item in
        machine-readable form, which is itself a finding.</p>
    </div>
    <div class="card">
      <h3>Solvency factors <span class="tier context-only">calculated</span></h3>
      <table><tr><th>Measure</th><th style="text-align:right">Value</th>
        <th style="text-align:right">Limit</th><th>State</th></tr>${calc}</table>
      <p class="note">Arithmetic performed on the figures at left, against
        conventional credit levels rather than thresholds fitted to any outcome.
        Fitting them would make this a model of past bankruptcies rather than a
        reading of the balance sheet.</p>
    </div>
  </div>`;
}

/* Vertical timeline: the diagnosis as it would have read at each earlier
 * period, recomputed from what was public then. */
function filingTimeline(d) {
  if (!d.timeline.length) return '';
  const rows = d.timeline.map(t => `
    <div class="tl-row">
      <div class="tl-date">${t.as_of}</div>
      <div class="tl-mark"><span style="--c:${ZONE_COLOR[t.zone]}"></span></div>
      <div class="tl-body">
        <div class="tl-zone" style="color:${ZONE_COLOR[t.zone]}">
          ${(d.zones.find(z => z.key === t.zone) || {}).label || t.zone}</div>
        <div class="tl-note">${t.breached} of ${t.computable} limits breached
          in the figures public by that date</div>
      </div>
    </div>`).join('');
  return `<div class="card">
    <h3>Retrospective filing timeline</h3>
    <div class="timeline">${rows}
      <div class="tl-row current">
        <div class="tl-date">${d.period_end || d.as_of}</div>
        <div class="tl-mark"><span style="--c:${ZONE_COLOR[d.zone]}"></span></div>
        <div class="tl-body">
          <div class="tl-zone" style="color:${ZONE_COLOR[d.zone]}">${d.zone_label}</div>
          <div class="tl-note">latest period visible as of ${d.as_of}</div>
        </div>
      </div>
    </div>
    <p class="note">Each reading was recomputed from filings public at that date,
      not by applying today's view to older numbers. That is what makes the
      series a record rather than a reconstruction.</p>
  </div>`;
}

function companyHeader(d) {
  return `<div class="card idhead">
    <div>
      <div class="idname">${d.name || 'CIK ' + d.cik}</div>
      <div class="idmeta">
        <span>CIK <b>${String(d.cik).padStart(10, '0')}</b></span>
        ${d.ticker ? `<span>Ticker <b>${d.ticker}</b></span>` : ''}
        ${d.exchange ? `<span>Listed <b>${d.exchange}</b></span>` : ''}
        ${d.state ? `<span>Incorporated <b>${d.state}</b></span>` : ''}
      </div>
      <div class="idsic">SIC ${d.sic} &mdash; ${d.sic_description}</div>
    </div>
    <div class="idzone">
      <div class="idzone-l">Assessed as of</div>
      <div class="idzone-v">${d.as_of}</div>
    </div>
  </div>`;
}

function renderDiagnostic(host, d) {
  if (d.error) {
    const list = (d.candidates || []).map(c => `<tr class="clickable" data-pick="${c.cik}">
        <td>${c.name}</td><td>${c.ticker}</td><td class="num">${c.cik}</td></tr>`).join('');
    host.innerHTML = `<div class="card blocked"><h3>No diagnosis</h3>
      <p class="err">${d.error}</p>${list ? `<table style="margin-top:12px">
      <tr><th>Company</th><th>Ticker</th><th style="text-align:right">CIK</th></tr>
      ${list}</table>` : ''}</div>`;
    return;
  }
  host.innerHTML = companyHeader(d) + stressGauge(d) + factorMatrix(d) + filingTimeline(d)
    + `<div class="card disclaimer">
        This tool provides an analytical diagnosis of past and present financial
        filings. It does not predict future performance, market viability, or
        guarantee bankruptcy outcomes.
      </div>`;
}
