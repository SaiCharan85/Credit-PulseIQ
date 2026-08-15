"""Deterministic peer groups and peer-relative percentiles (SPEC 3).

Peer construction is pure: it takes a SIC map and metric values in, and returns
a comparison out. No network, no LLM, no hidden state -- so a peer percentile in
a memo can be reproduced exactly from the audit trail.

Two controls matter here:

* **Vintage pinning.** Peer values must be computed under the same as-of cutoff
  as the target. Comparing a company's 2024 leverage against peers' 2026 figures
  would leak the future into a "peer-relative" signal, which is the subtle way
  lookahead sneaks back in after the obvious paths are closed.
* **Honest thinness.** Small universes produce small groups. A percentile
  against two peers is noise, so groups below ``min_peers`` are returned as
  ``insufficient`` rather than dressed up as a statistic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from compute.provenance import ComputedValue

LEVEL_SIC4 = "sic4"
LEVEL_SIC3 = "sic3"
LEVEL_SIC2 = "sic2"
LEVEL_INSUFFICIENT = "insufficient"

DEFAULT_MIN_PEERS = 3


def normalize_sic(sic: str | int | None) -> str:
    """SIC codes are 4-digit and lose leading zeros when stored as integers."""
    if sic is None or sic == "":
        return ""
    return str(sic).strip().zfill(4)


class PeerGroup(BaseModel):
    """A resolved comparison set and the rule that produced it."""

    model_config = ConfigDict(frozen=True)

    target_cik: int
    members: list[int] = Field(default_factory=list)
    level: str = LEVEL_INSUFFICIENT
    sic_prefix: str = ""
    notes: list[str] = Field(default_factory=list)

    @property
    def is_usable(self) -> bool:
        return self.level != LEVEL_INSUFFICIENT

    @property
    def size(self) -> int:
        return len(self.members)


class PeerComparison(BaseModel):
    """Where a company sits in its peer distribution for one metric."""

    model_config = ConfigDict(frozen=True)

    metric: str
    target_cik: int
    value: float | None
    percentile: float | None
    peer_median: float | None
    peer_count: int
    group: PeerGroup
    period_end: date | None = None
    as_of: date | None = None
    notes: list[str] = Field(default_factory=list)

    @property
    def is_usable(self) -> bool:
        return self.percentile is not None


def build_peer_group(
    target_cik: int,
    sic_by_cik: Mapping[int, str | int | None],
    min_peers: int = DEFAULT_MIN_PEERS,
    exclude: Sequence[int] = (),
) -> PeerGroup:
    """Widen from SIC-4 to SIC-3 to SIC-2 until ``min_peers`` are found.

    Widening is recorded in ``level`` so a consumer can discount a comparison
    made against a broad 2-digit division. Real SIC assignments are uneven --
    Sunrun is tagged 3690 (electronic equipment) while Sunnova is 4931
    (electric services) despite both being residential solar -- so exact-code
    matching alone would leave most groups empty.
    """
    target_sic = normalize_sic(sic_by_cik.get(target_cik))
    excluded = set(exclude) | {target_cik}
    if not target_sic:
        return PeerGroup(
            target_cik=target_cik,
            level=LEVEL_INSUFFICIENT,
            notes=["target has no SIC code"],
        )

    for level, width in ((LEVEL_SIC4, 4), (LEVEL_SIC3, 3), (LEVEL_SIC2, 2)):
        prefix = target_sic[:width]
        members = sorted(
            cik
            for cik, sic in sic_by_cik.items()
            if cik not in excluded and normalize_sic(sic).startswith(prefix)
        )
        if len(members) >= min_peers:
            notes = []
            if level != LEVEL_SIC4:
                notes.append(f"widened to {level} ({prefix}) to reach {min_peers} peers")
            return PeerGroup(
                target_cik=target_cik,
                members=members,
                level=level,
                sic_prefix=prefix,
                notes=notes,
            )

    widest = sorted(
        cik
        for cik, sic in sic_by_cik.items()
        if cik not in excluded and normalize_sic(sic).startswith(target_sic[:2])
    )
    return PeerGroup(
        target_cik=target_cik,
        members=widest,
        level=LEVEL_INSUFFICIENT,
        sic_prefix=target_sic[:2],
        notes=[f"only {len(widest)} peers at sic2; need {min_peers}"],
    )


def percentile_rank(value: float, peer_values: Sequence[float]) -> float | None:
    """Percentile of ``value`` within ``peer_values``, 0-100.

    Uses the mid-rank convention: ``(below + 0.5 * ties) / n * 100``. Ties get
    half credit so that identical values do not resolve to 0 or 100 depending
    on comparison order. The method is stated because "percentile" is ambiguous
    across libraries and the number ends up in a memo.
    """
    if not peer_values:
        return None
    below = sum(1 for p in peer_values if p < value)
    ties = sum(1 for p in peer_values if p == value)
    return (below + 0.5 * ties) / len(peer_values) * 100.0


def median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def compare_to_peers(
    metric: str,
    target: ComputedValue,
    peer_values: Mapping[int, ComputedValue],
    group: PeerGroup,
    as_of: date | None = None,
) -> PeerComparison:
    """Percentile-rank ``target`` against peer values from the same vintage.

    Peers whose ``as_of`` is not strictly before ``as_of`` are dropped, not
    silently included: a peer figure that was not public at prediction time is
    lookahead wearing a peer-comparison costume.
    """
    notes: list[str] = list(group.notes)
    usable: list[float] = []
    dropped_future = 0
    for cik in group.members:
        cv = peer_values.get(cik)
        if cv is None or not cv.is_defined:
            continue
        if as_of is not None and (cv.as_of is None or cv.as_of >= as_of):
            dropped_future += 1
            continue
        usable.append(cv.value)  # type: ignore[arg-type]

    if dropped_future:
        notes.append(f"dropped {dropped_future} peer(s) not yet public as of {as_of}")

    if target.value is None:
        notes.append("target metric undefined")
        return PeerComparison(
            metric=metric,
            target_cik=group.target_cik,
            value=None,
            percentile=None,
            peer_median=median(usable),
            peer_count=len(usable),
            group=group,
            period_end=target.period_end,
            as_of=as_of,
            notes=notes,
        )

    if len(usable) < DEFAULT_MIN_PEERS or not group.is_usable:
        notes.append(f"insufficient peers with data ({len(usable)}); percentile withheld")
        return PeerComparison(
            metric=metric,
            target_cik=group.target_cik,
            value=target.value,
            percentile=None,
            peer_median=median(usable),
            peer_count=len(usable),
            group=group,
            period_end=target.period_end,
            as_of=as_of,
            notes=notes,
        )

    return PeerComparison(
        metric=metric,
        target_cik=group.target_cik,
        value=target.value,
        percentile=percentile_rank(target.value, usable),
        peer_median=median(usable),
        peer_count=len(usable),
        group=group,
        period_end=target.period_end,
        as_of=as_of,
        notes=notes,
    )
