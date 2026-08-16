"""Fit and evaluate probability recalibration on the deterministic baselines.

Three-way temporal split: the model fits on the earliest window, the calibrator
on a later slice of training data it has never seen, and the test set stays
untouched. Calibrating on the model's own training rows would learn its
training-set optimism and report calibration that does not hold.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from models.calibration import (
    IdentityCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    max_calibration_error,
    reliability_table,
    split_for_calibration,
)
from models.hazard import expected_calibration_error, roc_auc
from models.panel import load_panel, split_by_date


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--panel", type=Path, default=Path("data/cache/panel.csv"))
    parser.add_argument("--cutoff", type=date.fromisoformat, default=date(2024, 6, 1))
    parser.add_argument("--calibration-cutoff", type=date.fromisoformat, default=date(2023, 6, 1))
    args = parser.parse_args(argv)

    rows = load_panel(args.panel)
    if not rows:
        print(f"no panel at {args.panel}; run `python -m models.run_baseline --rebuild`", file=sys.stderr)
        return 1

    train, test = split_by_date(rows, args.cutoff)
    fit_rows, calib_rows = split_for_calibration(train, args.calibration_cutoff)
    print(
        f"model fold {len(fit_rows)} rows ({sum(r.label for r in fit_rows)} pos) | "
        f"calibration fold {len(calib_rows)} rows ({sum(r.label for r in calib_rows)} pos) | "
        f"test {len(test)} rows ({sum(r.label for r in test)} pos)",
        file=sys.stderr,
    )
    if not fit_rows or not calib_rows:
        print("empty fold; adjust --calibration-cutoff", file=sys.stderr)
        return 1

    from models.hazard import HazardBaseline

    model = HazardBaseline().fit(fit_rows)
    calib_scores = model.predict_proba(calib_rows)
    calib_labels = [r.label for r in calib_rows]
    test_scores = model.predict_proba(test)
    test_labels = [r.label for r in test]

    print("\n=== hazard baseline: recalibration on the test split ===", file=sys.stderr)
    header = f"{'calibrator':12}{'ECE':>9}{'MCE':>9}{'AUC':>9}"
    print(header, file=sys.stderr)
    print("-" * len(header), file=sys.stderr)

    results = {}
    for calibrator in (IdentityCalibrator(), PlattCalibrator(), IsotonicCalibrator()):
        try:
            calibrator.fit(calib_scores, calib_labels)
            mapped = calibrator.transform(test_scores)
        except Exception as exc:  # noqa: BLE001
            print(f"{calibrator.name:12} failed: {exc}", file=sys.stderr)
            continue
        ece = expected_calibration_error(test_labels, mapped)
        mce = max_calibration_error(mapped, test_labels)
        auc = roc_auc(test_labels, mapped)
        results[calibrator.name] = mapped
        print(
            f"{calibrator.name:12}{ece:>9.4f}{mce:>9.4f}{auc:>9.4f}",
            file=sys.stderr,
        )

    print(
        "\nAUC is unchanged by construction: both maps are monotonic, so they "
        "move probabilities without reordering anything.",
        file=sys.stderr,
    )

    best = min(
        (n for n in results if n != "identity"),
        key=lambda n: expected_calibration_error(test_labels, results[n]) or 1.0,
        default=None,
    )
    if best:
        print(f"\n  reliability, uncalibrated vs {best}:", file=sys.stderr)
        print(f"    {'bin':>12}{'n':>7}{'stated':>9}{'observed':>10}", file=sys.stderr)
        for label, scores in (("raw", results["identity"]), (best, results[best])):
            print(f"    -- {label}", file=sys.stderr)
            for row in reliability_table(scores, test_labels, bins=5):
                print(
                    f"    [{row['bin_low']:.1f}-{row['bin_high']:.1f}){row['n']:>7}"
                    f"{row['stated']:>9.3f}{row['observed']:>10.3f}",
                    file=sys.stderr,
                )

    print(
        "\nCalibrated to this universe's base rate, which is enriched far above "
        "the population rate (Zmijewski 1984). Absolute probabilities do not "
        "transfer to the wider population.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
