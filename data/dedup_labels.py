"""One bankruptcy, one label -- even when two registrants filed for it.

A reorganisation can leave two CIKs reporting the same petition. QVC Group,
Inc. and *Old* QVC Group, Inc. both filed an item 1.03 for the same Chapter 11
on 2026-04-16, and both became positives. One event, counted twice.

The temptation is to treat every shared filing date as a duplicate. That is
wrong and the data says so: of twelve dates carrying more than one label, only
one is a genuine double-count. The rest are companies that happened to file on
the same day -- Tupperware and BurgerFi on 2024-09-17, Chord Energy and
Lonestar Resources on 2020-09-30. Collapsing those would delete real
bankruptcies and quietly shrink the positive class.

So the test is **same date AND the same underlying company**, judged by name
after stripping the noise that makes one entity look like two: the corporate
suffix, and the "Old"/"New" prefix a reorganisation adds. Two unrelated names
on one date survive; "QVC Group" and "Old QVC Group" do not.

Nothing here guesses. A pair that only matches on date is reported and kept.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

#: Corporate furniture that differs between registrants of the same business.
_SUFFIX = re.compile(
    r"\b(inc|incorporated|corp|corporation|co|company|llc|l\.l\.c|lp|l\.p|plc|"
    r"ltd|limited|holdings?|group|the|sa|nv|ag|trust|partners|"
    r"international|intl)\b\.?",
    re.I,
)

#: What a reorganisation prepends to the shell left behind. "Old QVC Group" and
#: "QVC Group" are the same business on either side of a plan.
_REORG_PREFIX = re.compile(r"^\s*(old|new|reorganized|successor)\s+", re.I)


def normalise(name: str) -> str:
    """A name reduced to the part that identifies the business."""
    out = _REORG_PREFIX.sub("", name or "")
    out = _SUFFIX.sub(" ", out)
    out = re.sub(r"[^a-z0-9 ]+", " ", out.lower())
    return " ".join(out.split())


def same_business(a: str, b: str) -> bool:
    """Whether two registrant names denote one business.

    Containment, not equality: "QVC" and "QVC Group" normalise to "qvc" and
    "qvc", but "Spirit Aviation" and "CareMax" share nothing. A two-character
    floor stops initialisms collapsing everything into everything.
    """
    x, y = normalise(a), normalise(b)
    if not x or not y or min(len(x), len(y)) < 3:
        return False
    return x == y or x in y or y in x


@dataclass(frozen=True)
class Duplicate:
    """One event recorded under two registrants."""

    event_date: str
    keep_cik: int
    keep_name: str
    drop_cik: int
    drop_name: str

    def __str__(self) -> str:
        return (
            f"{self.event_date}  keep {self.keep_cik} {self.keep_name!r}  "
            f"drop {self.drop_cik} {self.drop_name!r}"
        )


def find_duplicates(
    rows: list[dict], names: dict[int, str], signal: str = "chapter11_petition"
) -> list[Duplicate]:
    """Pairs that record one bankruptcy twice.

    The survivor is the registrant *without* the reorganisation prefix -- the
    entity that carries on. Where neither has one, the lower CIK wins purely so
    the choice is deterministic and a rerun does not flip it.
    """
    by_date: dict[str, list[int]] = {}
    for row in rows:
        if (row.get("signal") or "").strip() != signal:
            continue
        by_date.setdefault(row.get("event_date", ""), []).append(int(row["cik"]))

    found: list[Duplicate] = []
    for event_date, ciks in sorted(by_date.items()):
        for i, a in enumerate(sorted(set(ciks))):
            for b in sorted(set(ciks))[i + 1 :]:
                na, nb = names.get(a, ""), names.get(b, "")
                if not (na and nb) or not same_business(na, nb):
                    continue
                a_is_shell = bool(_REORG_PREFIX.match(na))
                b_is_shell = bool(_REORG_PREFIX.match(nb))
                if a_is_shell and not b_is_shell:
                    keep, drop = (b, nb), (a, na)
                elif b_is_shell and not a_is_shell:
                    keep, drop = (a, na), (b, nb)
                else:
                    keep, drop = (a, na), (b, nb)
                found.append(Duplicate(event_date, keep[0], keep[1], drop[0], drop[1]))
    return found


def apply_dedup(path: Path, duplicates: list[Duplicate], out: Path) -> int:
    """Write the events file without the dropped registrants' petitions.

    Only the duplicated *petition* rows go. A dropped registrant's other events
    -- a delisting, a covenant breach -- describe things that genuinely
    happened to that filer and are left alone.
    """
    drop = {(d.drop_cik, d.event_date) for d in duplicates}
    kept, removed = [], 0
    with path.open(encoding="utf8") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        for row in reader:
            key = (int(row["cik"]), row.get("event_date", ""))
            if key in drop and (row.get("signal") or "") == "chapter11_petition":
                removed += 1
                continue
            kept.append(row)
    with out.open("w", encoding="utf8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept)
    return removed
