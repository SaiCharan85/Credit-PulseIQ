"""Measure what the filing-text signals add, before spending a backtest on them.

An agent run over 200 cases costs hours. The signals it would read are
deterministic, so their predictive content can be measured directly and far
more cheaply. If they add nothing here they will add nothing through the agent,
and if they add a lot, the agent's job is to not squander it.

This is a *diagnostic*, not a result: a deterministic signal model is not the
agent. Any fitted combination uses the same temporal fold discipline as
``run_fusion``, because a weight fitted on the cases it is scored on would make
the number meaningless.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

from agents.llm import load_env_file
from data.edgar import EdgarClient
from data.signals import filing_events, latest_report, scan_report_text
from models.hazard import HazardBaseline, roc_auc
from models.panel import load_panel, split_by_date


def collect(cik: int, as_of: date, edgar: EdgarClient, read_text: bool) -> dict:
    """Every signal for one filer at one date, or explicit unavailability."""
    out = {
        "cik": cik,
        "as_of": as_of.isoformat(),
        "index_ok": False,
        "text_ok": False,
        "late_filing": 0,
        "covenant": 0,
        "delisting": 0,
        "auditor_change": 0,
        "restatement": 0,
        "impairment": 0,
        "going_concern": 0,
        "material_weakness": 0,
    }
    try:
        index = edgar.filing_index(cik)
    except Exception:  # noqa: BLE001
        return out
    out["index_ok"] = True
    codes = {"2.04": "covenant", "3.01": "delisting", "4.01": "auditor_change",
             "4.02": "restatement", "2.06": "impairment", "late_filing": "late_filing"}
    for event in filing_events(index, as_of):
        key = codes.get(event.code)
        if key:
            out[key] = 1
    if read_text:
        report = latest_report(index, as_of)
        if report is not None:
            try:
                text = edgar.fetch_filing_document(
                    cik, report["accession"], report["primary_document"]
                )
            except Exception:  # noqa: BLE001
                return out
            found = scan_report_text(text)
            out["text_ok"] = True
            out["going_concern"] = int(found["going_concern_doubt"])
            out["material_weakness"] = int(found["material_weakness"])
    return out


FEATURES = (
    "late_filing", "covenant", "delisting", "auditor_change",
    "restatement", "impairment", "going_concern", "material_weakness",
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panel", type=Path, default=Path("data/cache/panel.csv"))
    ap.add_argument("--cutoff", type=date.fromisoformat, default=date(2024, 6, 1))
    ap.add_argument("--max-negatives", type=int, default=100)
    ap.add_argument("--out", type=Path, default=Path("data/cache/signal_probe.csv"))
    ap.add_argument("--no-text", action="store_true", help="metadata only; skips document fetches")
    args = ap.parse_args(argv)

    load_env_file()
    from evals.backtest import stratified_sample

    panel = load_panel(args.panel)
    train, test = split_by_date(panel, args.cutoff)
    sample = stratified_sample(test, args.max_negatives)
    cases = sample.cases
    edgar = EdgarClient()

    rows = []
    for i, case in enumerate(cases, 1):
        rec = collect(case.cik, case.observation_date, edgar, not args.no_text)
        rec["label"] = case.label
        rows.append(rec)
        if i % 20 == 0:
            print(f"  {i}/{len(cases)}", file=sys.stderr)

    with args.out.open("w", newline="", encoding="utf8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"signals -> {args.out}", file=sys.stderr)

    y = [r["label"] for r in rows]
    print(f"\n=== per-signal prevalence ({len(rows)} cases, {sum(y)} positives) ===", file=sys.stderr)
    print(f"{'signal':<20} {'in failures':>12} {'in survivors':>13} {'lift':>7}", file=sys.stderr)
    npos, nneg = sum(y), len(y) - sum(y)
    for f in FEATURES:
        p = sum(r[f] for r in rows if r["label"] == 1) / max(1, npos)
        q = sum(r[f] for r in rows if r["label"] == 0) / max(1, nneg)
        lift = p / q if q else float("inf")
        print(f"{f:<20} {p:>11.1%} {q:>12.1%} {lift:>7.1f}x", file=sys.stderr)

    # Fold discipline, same as run_fusion: fit early, score late.
    order = sorted(range(len(rows)), key=lambda i: (cases[i].observation_date, cases[i].cik))
    rows = [rows[i] for i in order]
    cases = [cases[i] for i in order]
    y = [r["label"] for r in rows]
    cut = int(len(rows) * 0.5)
    while cut < len(rows) and cases[cut].observation_date == cases[cut - 1].observation_date:
        cut += 1

    from sklearn.linear_model import LogisticRegression

    x = [[r[f] for f in FEATURES] for r in rows]
    if len(set(y[:cut])) < 2 or len(set(y[cut:])) < 2:
        print("\na fold is single-class; cannot score", file=sys.stderr)
        return 0
    lr = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
    lr.fit(x[:cut], y[:cut])
    sig = [float(p) for p in lr.predict_proba(x[cut:])[:, 1]]

    hazard = HazardBaseline().fit(train)
    haz = hazard.predict_proba(cases)[cut:]
    ev_y = y[cut:]

    print(f"\n=== eval fold: {len(ev_y)} cases, {sum(ev_y)} positives ===", file=sys.stderr)
    print(f"  text/event signals alone : AUC {roc_auc(ev_y, sig):.4f}", file=sys.stderr)
    print(f"  hazard alone             : AUC {roc_auc(ev_y, haz):.4f}", file=sys.stderr)
    import math

    def lg(p, e=1e-6):
        p = max(e, min(1 - e, p))
        return math.log(p / (1 - p))

    both = LogisticRegression(C=1.0, max_iter=1000)
    both.fit(
        [[lg(a), lg(b)] for a, b in zip(
            [float(p) for p in lr.predict_proba(x[:cut])[:, 1]],
            hazard.predict_proba(cases)[:cut], strict=True)],
        y[:cut],
    )
    fused = [float(p) for p in both.predict_proba(
        [[lg(a), lg(b)] for a, b in zip(sig, haz, strict=True)])[:, 1]]
    print(f"  signals + hazard         : AUC {roc_auc(ev_y, fused):.4f}", file=sys.stderr)
    print(
        "\nThis is the ceiling the agent is being handed, not the agent's score.",
        file=sys.stderr,
    )
    print(json.dumps({f: round(c, 3) for f, c in zip(FEATURES, lr.coef_[0], strict=True)}), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
