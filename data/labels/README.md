# Label sets

Real outcomes only. No synthetic data, no proxy labels dressed up as real ones
(README, "Scope discipline"). Labels are the entire edge, so provenance and
dating matter more here than anywhere else in the repo.

## Two dates, always

| Column | Meaning |
|---|---|
| `event_date` | When it happened (bankruptcy petition date) |
| `as_of_date` | When it became public (the disclosing filing's EDGAR date) |

They differ, sometimes materially: Bed Bath & Beyond petitioned 2023-04-23 and
disclosed 2023-04-24; Hertz petitioned 2020-05-22 and disclosed 2020-05-26.
Lead time is measured to `event_date`; visibility is governed by `as_of_date`.

Only ground truth and its provenance are stored. Name, ticker, SIC and size are
resolved from EDGAR at load time because they are point-in-time attributes that
drift — see `data/watchlist.py`.

## `chapter11.csv` — 25 events

Every row passed **three independent checks**, because no single signal is
trustworthy:

1. **Structured** — the 8-K carries item code `1.03` ("Bankruptcy or
   Receivership") in EDGAR submissions metadata.
2. **Textual** — the filing body contains Chapter 11 language ("voluntary
   petition", "chapter 11").
3. **Subject** — the language describes *this filer's own* petition, not a
   counterparty's, a subsidiary receivership, or a Chapter 7 liquidation.

Each check caught errors the others missed:

- Text-search alone produced false positives at a high rate — WEX, Howard
  Hughes, SM Energy, Kennedy-Wilson, Amneal and Opendoor all match
  "Bankruptcy or Receivership" as boilerplate item-header text while being
  perfectly solvent. 31 of 89 size-eligible candidates were rejected this way.
- Item codes alone are **filer-supplied and miscoded**. J.C. Penney's earliest
  `1.03` (2014-01-28) is a shareholder rights plan, six years before its actual
  bankruptcy; Granite Construction's 2026 `1.03` contains no Chapter 11 language
  at all. Deriving `event_date` from item codes would have injected a six-year
  lead-time error into the headline metric.
- Cumulus Media's only `1.03` is from its *2017* bankruptcy, correctly excluded
  from the current window.
- Canoo and Sonder filed **Chapter 7**, not 11 — a different event type.

`date_basis` records how `event_date` was established: `petition_date_from_8k_text`
where the petition date was parsed from the filing, `8k_filing_date_fallback`
otherwise. The fallback is conservative (it can only *understate* lead time, by
the disclosure lag of a few days) and never manufactures lookahead.

### Cohorts

| Cohort | Count | Window |
|---|---|---|
| `recent_2025_2026` | 15 | primary evaluation set |
| `prior_2021_2024` | 8 | secondary |
| `historical_pre_2021` | 2 | ~10% tail (Hertz, Sears) |

The historical tail is tagged separately so the backtest can include or exclude
it: pre-2021 filings sit in a different XBRL-coverage and rate environment, and
mixing eras silently is how a backtest flatters itself.

### Regenerating

```bash
python -m data.discover --start 2025-01-01 --end 2026-08-14
```

Discovery is reproducible but its output is **not** written straight to this
file. Candidates are reviewed before they become labels.

## `aaer.csv` — schema only, deliberately empty

The earnings-quality leg is gated behind the distress leg producing a real
calibration curve (PROMPT hard rule 8), so this file carries its schema and no
rows. An empty label set is the honest Phase 1–2 state; a fabricated one would
defeat the point of the project.

When it is populated, candidate sources are SEC's AAER index
(`sec.gov/enforcement-litigation/accounting-auditing-enforcement-releases`) and
the Bao et al. accounting-fraud feature+label set. Both are keyed by company
name, not CIK, so the linkage step needs the same three-signal discipline
applied above.

`misstatement_start` / `misstatement_end` bound the fiscal window the
misstatement covered. They matter: an AAER released in 2024 about 2019 conduct
must be evaluated against 2019 filings, and the release date is what governs
visibility.

## Other sources considered

| Source | Adds | Cost |
|---|---|---|
| EDGAR 8-K 1.03 | disclosure date, CIK-native | in use |
| CourtListener API (free) | true petition dates, chapter, docket | name→CIK linkage |
| EDGAR Form 25-NSE | delistings — the intermediate distress enrichment SPEC §7 asks for | CIK-native |
| FJC Integrated Database | all federal bankruptcy filings | debtor-name keyed |
| LoPucki BRD | curated large-cap Chapter 11 | coverage ended ~2022 |

EDGAR remains the spine because it is the only source natively keyed to CIK,
which is what the financials are keyed to.
