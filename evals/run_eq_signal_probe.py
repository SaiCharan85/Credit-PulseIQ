"""Do filing-text signals predict restatements? (SPEC 12, phase 4)

The narrow-panel probe reached 0.867, but that number was in-sample and drawn
from the distress-enriched watchlist -- the same two conditions that produced a
0.714 ratio baseline which collapsed to 0.579 once the universe was widened. So
it is re-run here properly: wide point-in-time universe, fitted on the training
window, scored on the test window, never both.

Deterministic throughout. This measures whether the *information* is there
before any agent is built to read it.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from datetime import date
from pathlib import Path

from agents.llm import load_env_file
from data.edgar import EdgarClient
from data.signals import filing_events, latest_report, scan_report_text
from models.earnings import load_eq_panel
from models.hazard import roc_auc
from models.panel import split_by_date

FEATURES = (
    "late_filing",
    "auditor_change",
    "delisting",
    "covenant",
    "impairment",
    "material_weakness",
    "going_concern",
)
_CODE = {
    "late_filing": "late_filing",
    "4.01": "auditor_change",
    "3.01": "delisting",
    "2.04": "covenant",
    "2.06": "impairment",
}


def signals_for(edgar: EdgarClient, cik: int, as_of: date) -> dict[str, int]:
    out = dict.fromkeys(FEATURES, 0)
    try:
        index = edgar.filing_index(cik)
    except Exception:  # noqa: BLE001
        return out
    for event in filing_events(index, as_of, 540):
        key = _CODE.get(event.code)
        if key:
            out[key] = 1
    report = latest_report(index, as_of)
    if report is not None:
        try:
            text = edgar.fetch_filing_document(
                cik, report["accession"], report["primary_document"]
            )
        except Exception:  # noqa: BLE001
            return out
        found = scan_report_text(text)
        out["material_weakness"] = int(found["material_weakness"])
        out["going_concern"] = int(found["going_concern_doubt"])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panel", type=Path, default=Path("data/cache/eq_panel_wide.csv"))
    ap.add_argument("--cutoff", type=date.fromisoformat, default=date(2023, 7, 1))
    ap.add_argument("--per-fold-positives", type=int, default=200)
    ap.add_argument("--per-fold-negatives", type=int, default=300)
    ap.add_argument("--out", type=Path, default=Path("data/cache/eq_signal_probe.csv"))
    args = ap.parse_args(argv)

    load_env_file()
    rows = load_eq_panel(args.panel)
    train, test = split_by_date(rows, args.cutoff)

    def take(fold, seed):
        pos = [r for r in fold if r.label == 1]
        neg = [r for r in fold if r.label == 0]
        random.Random(seed).shuffle(pos)
        random.Random(seed).shuffle(neg)
        return pos[: args.per_fold_positives] + neg[: args.per_fold_negatives]

    tr, te = take(train, 0), take(test, 1)
    print(
        f"probing {len(tr)} train rows ({sum(r.label for r in tr)} pos) "
        f"and {len(te)} test rows ({sum(r.label for r in te)} pos)",
        file=sys.stderr,
    )

    edgar = EdgarClient()
    data = []
    every = tr + te
    for n, r in enumerate(every, 1):
        rec = signals_for(edgar, r.cik, r.observation_date)
        rec.update(cik=r.cik, as_of=r.observation_date.isoformat(), label=r.label,
                   fold="train" if n <= len(tr) else "test")
        data.append(rec)
        if n % 100 == 0:
            print(f"  {n}/{len(every)}", file=sys.stderr)

    with args.out.open("w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)
    print(f"-> {args.out}", file=sys.stderr)

    trd = [d for d in data if d["fold"] == "train"]
    ted = [d for d in data if d["fold"] == "test"]
    npos = sum(d["label"] for d in ted)
    nneg = len(ted) - npos
    print(f"\n{'signal':<20}{'in restaters':>13}{'in clean':>10}{'lift':>8}", file=sys.stderr)
    for f in FEATURES:
        p = sum(d[f] for d in ted if d["label"] == 1) / max(1, npos)
        q = sum(d[f] for d in ted if d["label"] == 0) / max(1, nneg)
        lift = p / q if q else float("inf")
        print(f"{f:<20}{p:>12.1%}{q:>10.1%}{lift:>7.1f}x", file=sys.stderr)

    from sklearn.linear_model import LogisticRegression

    xtr = [[d[f] for f in FEATURES] for d in trd]
    ytr = [d["label"] for d in trd]
    xte = [[d[f] for f in FEATURES] for d in ted]
    yte = [d["label"] for d in ted]
    lr = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced").fit(xtr, ytr)
    p_te = [float(p) for p in lr.predict_proba(xte)[:, 1]]
    p_tr = [float(p) for p in lr.predict_proba(xtr)[:, 1]]
    print(f"\n  text signals, in-sample  : AUC {roc_auc(ytr, p_tr):.4f}", file=sys.stderr)
    print(f"  text signals, OUT-OF-SAMPLE: AUC {roc_auc(yte, p_te):.4f}  <- the honest one",
          file=sys.stderr)
    print("  ratio baseline for contrast: AUC 0.579", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
