# Results

Two backtests over the **same** 200 test cases (100 Chapter 11 filings, 100
survivors), prediction dates from 2024-07-01, temporal cutoff 2024-06-01,
`gemma-4-31b-it` in a ReAct loop. The only difference between them is which
tools the agent could call, so any change is attributable to the tools and not
to the sample.

## Headline

| arm | AUC | false-confidence | ECE | F1 |
| --- | --- | --- | --- | --- |
| Backtest 1 — agent, financial ratios only | 0.881 | 0.250 | 0.135 | 0.753 |
| **Backtest 2 — agent, + filing-text tools** | **0.965** | **0.080** | **0.081** | **0.883** |
| Tier 0 — Altman Z'' | 0.885 | - | 0.141 | - |
| Tier 1 — discrete-time hazard | 0.966 | - | 0.101 | - |

Paired bootstrap, 8,000 resamples, 200 identical cases:

```
tools effect    : +0.0841   95% CI [+0.0484, +0.1246]   -> REAL
agent vs hazard : -0.0017   95% CI [-0.0262, +0.0225]   -> tie
```

**Giving the agent evidence a ratio model cannot read is worth +0.084 AUC.**
That closed a clear loss to the hazard baseline into a statistical tie, and cut
the catastrophic-error rate by a factor of three.

**It does not beat the baseline.** The point estimate is fractionally below and
the interval spans zero. Parity with an auditable explanation is the claim the
evidence supports; superiority is not.

## Why it improved — the mechanism

| | count |
| --- | --- |
| bankruptcies missed in backtest 1, caught in backtest 2 | 27 |
| bankruptcies caught in backtest 1, missed in backtest 2 | 0 |
| of the 27 rescued, carrying a text/event signal | 26 (96%) |

No trade-off: 27 misses became catches and nothing regressed. The agent's
failure mode was never subtle reasoning error, it was *blindness* -- it spent
~12 steps investigating a set of inputs that did not contain the answer for a
quarter of the bankruptcies. You cannot reason your way to a covenant breach
from a current ratio.

The control matters as much as the effect. CIK 1498710 moved 0.42 -> 0.48 with
no signals found: where there was nothing new to read, the verdict barely
changed. Had the tools merely made the model more alarmist, every score would
have risen and precision would have collapsed. Instead precision *rose*, 0.795
-> 0.870. Scores moved where evidence existed and nowhere else.

Representative rescues:

| company | before | after | what it found |
| --- | --- | --- | --- |
| CIK 1584754 | 0.05 | 0.95 | delisting notice, going-concern doubt |
| CIK 867773 | 0.27 | 0.98 | late filing, covenant breach, delisting, restatement, going concern, material weakness |
| CIK 1391127 | 0.38 | 0.98 | six signals |

CIK 867773 is the case worth pointing at: the hazard model scored it **0.47, a
pass**, while six separate alarms sat in filings public before the prediction
date.

---

# Backtest 1 — the agent's own reasoning, measured

200 test cases (100 Chapter 11 filings, 100 survivors), prediction dates from
2024-07-01, temporal cutoff 2024-06-01. `gemma-4-31b-it` driving a ReAct loop
over typed tools. The agent had **no access** to any statistical model
(`baseline_seen=0` on every row), so this is its own reasoning and nothing else.

## Headline

| arm | AUC | ECE |
| --- | --- | --- |
| Agent (ReAct, own reasoning) | 0.881 | 0.135 |
| Tier 0 — Altman Z'' | 0.885 | 0.141 |
| **Tier 1 — discrete-time hazard** | **0.966** | 0.101 |

**The agent does not beat the hazard baseline.** It ties Altman within noise and
trails the hazard model by 0.085 AUC. Baselines are scored on the *identical*
graded cases, matched on `(cik, as_of)` — an earlier version scored them on the
full split while the agent's AUC covered only its non-abstained cases, which let
it drop the cases it found hardest.

## Other measurements

| | |
| --- | --- |
| median lead time | 161 days |
| false-confidence rate | 0.250 (25/100 failures called healthy/watch at conf >= 0.7) |
| abstentions | 2 (both protocol failures — it never *chose* to abstain) |
| verification failures | 0 (no fabricated numbers survived the critic) |
| mean steps per case | 11.0 |

Calibration is systematically optimistic: in every band below 0.8 the observed
failure rate exceeds the stated risk (stated 0.31 -> observed 0.66 in the
0.2-0.4 band). The agent under-calls distress, and does so confidently.

## What beats the baseline

Not the agent. Deterministic signals read out of filing *text and metadata* —
absent from the ratio panel by construction — do:

| signal | in failures | in survivors | lift |
| --- | --- | --- | --- |
| delisting notice (8-K 3.01) | 58% | 0% | inf |
| going-concern doubt | 54% | 2% | 27x |
| material weakness | 38% | 6% | 6.3x |
| late filing (NT 10-K/Q) | 34% | 1% | 34x |
| auditor change (8-K 4.01) | 18% | 3% | 6.0x |
| restatement (8-K 4.02) | 17% | 0% | inf |
| covenant breach (8-K 2.04) | 6% | 0% | inf |
| material impairment (8-K 2.06) | 5% | 8% | 0.6x (useless) |

On a held-out temporal fold (67 cases, 30 positives):

| arm | AUC |
| --- | --- |
| signals alone | 0.887 |
| hazard alone | 0.957 |
| **signals + hazard** | **0.982** |

Paired bootstrap, 4,000 resamples: **+0.0252, 95% CI [+0.0018, +0.0615]**, the
lower bound clear of zero. This is a real improvement over the baseline.

For contrast, fusing the *agent* with the hazard model gave +0.0042 with a CI of
[-0.0091, +0.0199] — noise — and the parameter-free rank-average control came in
*below* hazard alone, which is the tell that the fitted gain came from the fold
rather than the data.

## Honest limits

* The universe is enriched with bankrupt filers far above the population base
  rate (Zmijewski 1984). Absolute AUCs are optimistic; the *ordering* between
  arms is what transfers.
* Lead times on positives are median 166 days but the 10th percentile is 43 and
  the minimum 9. A near-term tail makes some cases easier than a real screening
  application would be.
* `signals + hazard` is a deterministic model, not the agent. It establishes
  that the information beats the baseline, not that the agent can exploit it.
  Backtest 2 tests that.
* 100 positives is a small sample. Every AUC here carries wide error bars, which
  is why the fusion claims are bootstrapped rather than asserted.

## Files

| file | contents |
| --- | --- |
| `backtest1_agent_200cases.csv` | ratios only: per-case verdicts, scores, tool sequences |
| `backtest2_agent_200cases.csv` | + filing-text tools: same, plus rationale and cited evidence |
| `backtest1_run.log`, `backtest2_run.log` | full harness output including reliability curves |
| `signal_probe_200cases.csv` | the eight text/event signals per case |

---

# Phase 4 — earnings quality: a measured negative

The second leg does not work, and that is the finding. Four independent
approaches were tried against 891 test positives with a clean leak canary, and
none beats coin-flipping by a useful margin.

| approach | out-of-sample AUC |
| --- | --- |
| Tier 0 — Beneish M (published, fitted to nothing) | 0.512 |
| Tier 1 — fitted logistic on financial ratios | 0.579 |
| Tier 1 — plus Beneish decomposed into its terms | 0.585 |
| filing-text and 8-K event signals | 0.579 |
| **leak canary (shuffled labels)** | **0.526 — clean** |

## Labels

8-K item 4.02, "non-reliance on previously issued financial statements", rather
than AAER. An AAER names an individual as often as an issuer, has no CIK, and
states its misstatement window only as prose inside a PDF. Item 4.02 is filed by
the company with a structured item code. The trade is stated rather than hidden:
this detects *accounting problems*, including honest error, not fraud.

**887 verified companies** from 2,008 raw events. The exclusions matter more than
the inclusions: 810 events were SPAC regulatory reclassifications, in two waves.
The April 2021 warrant reclassification (598) is well known. The second -- Class
A shares moved to temporary equity in late 2021 (212) -- uses no warrant language
and passes a warrant filter untouched. After the first rule alone, 2021 still
held 299 events against a 2019-2020 baseline near 65; after both, 141 against
145 and 152 in 2022-2023. A spike is contamination, a sustained level shift is
plausibly real, so the filtering stopped there rather than tuning until the
years matched.

## Two results that were wrong before the universe was widened

Both came from the 354-company distress watchlist, and both looked good:

| | narrow (28 test positives) | wide (891) |
| --- | --- | --- |
| fitted ratios | 0.714 | **0.579** |
| filing-text signals | 0.867 (in-sample) | **0.579** |

The text collapse is the more instructive one. In the watchlist, going-concern
doubt appeared in 15.8% of non-restaters; across all annual filers it appears in
**41.7%**. The market by filer count is dominated by micro-caps where that
language is routine, so a signal that discriminated sharply against healthy
large-cap peers says almost nothing in the real population.

## Why the missing-data explanation is ruled out

Beneish M is computable on 17% of wide-panel rows: it returns nothing unless all
eight terms resolve, so one untagged line item discards the other seven. The
terms were therefore registered individually, lifting coverage to 35-69% each.
AUC moved 0.579 to 0.585 and not one component reached the strongest covariates.
The signal is not hidden behind absent tags. It is absent.

## Attempt six: engineered features. No effect.

Resampling was not tried, and should not be: AUC is a ranking metric and is
essentially invariant to class balance, `class_weight="balanced"` is already
applied, and resampling redistributes information rather than creating it.

Feature engineering was tried, because all five earlier attempts used
point-in-time *levels*. Three families were added -- industry-adjusted
percentiles within (observation date, 2-digit SIC), year-over-year change per
firm, and trailing volatility -- 27 features in total.

The comparison is *paired* rather than absolute, because five approaches had
already been scored on this fold and there is no untouched slice to retreat to:
everything after 2023-07-01 was inside the test fold of all five, and the panel
cannot be extended forward because a 12-month label window needs outcomes that
have not happened yet. Both arms therefore fit on the same window and score the
same rows, differing only in the feature set. Fold reuse moves them together
and cannot manufacture a difference.

| arm | AUC |
| --- | --- |
| base covariates | 0.5852 |
| base + 27 engineered features | 0.5843 |
| **difference** | **-0.0008, 95% CI [-0.0116, +0.0097]** |

The interval is tight around zero, so this is a clear null rather than an
underpowered one. Industry adjustment was the strongest prior hypothesis --
raw accruals are structurally different across sectors and the accounting
literature adjusts for it as a matter of course -- and it moved the number by
less than one part in a thousand.

Six approaches, one conclusion. No seventh was attempted; each further cut of
the same data buys a multiple-comparisons problem that costs more credibility
than any number it could produce.

## Attempt seven: regulator and governance signals. A real but marginal effect.

The first six attempts all drew from filed ratios and eight event flags. This
one used information none of them had: SEC staff comment letters
(`UPLOAD`/`CORRESP`), executive departures (8-K item 5.02), and the filer's own
prior restatement history. Run on the income-decreasing label at filers over
$100M -- 217 test positives.

| arm | AUC |
| --- | --- |
| base covariates | 0.6053 |
| **+ oversight signals** | **0.6400** |
| difference | **+0.0348**, 95% CI [+0.0125, +0.0572] |
| same, Bonferroni-corrected for seven attempts | CI [+0.0064, +0.0644] |

**The interval excludes zero even after correcting for every attempt made on
this leg.** It is the first robust signal the earnings leg has produced.

### A quarter of the first result was my own leakage

The initial run reported +0.0433. It was inflated by a feature bug flagged
before the verdict arrived: labels come from `first_event_per_company`, so a
positive row sits in the twelve months before a company's *first* restatement
and by construction has no prior. The prevalence table showed it plainly --
prior restatements appeared in 17.7% of all rows but only 7.8% of restaters,
an inverted lift. The model was free to learn "no prior restatement therefore
likelier to restate", which is the label construction talking, not the data.

Dropping the three prior-restatement features costs 0.0099 of the 0.0433. What
remains survives that removal, a Bonferroni correction, and re-fitting with
scaled features after the first fit failed to converge.

### Why it is still marginal

Pre-committed before the run: an effect below +0.05 is reported and not built
on. +0.0348 is below it, and 0.640 sits at the bottom edge of the 0.65-0.75
the published literature reaches. It is a real effect on a model that is still
not useful.

The prevalence table explains why it cannot be larger. Comment letters appear
in 52.1% of all rows and 67.7% of restaters; officer events in 82.2% and 90.3%.
These are routine corporate events, not warnings -- the same shape as
going-concern language appearing in 41.7% of non-restaters. A signal present on
half the population cannot separate a 3% positive class by much, however real
its association.

### What would be needed

Testing repeat offending properly requires labelling *every* restatement rather
than the first per filer, which changes the label definition and therefore the
question. That is a new study, not attempt eight.

## Scope

Under the honest-scoping rule this leg is **demo-grade, not a backtested
predictor**. It is not fused into the graded signal. Knowing where the system
does not work is part of the deliverable, and the published literature agrees the
ceiling here is low -- restatement prediction lands near 0.65-0.75 at best,
because a large share of restatements are errors rather than patterns.

No investigator was built for this leg. The eval-first order meant the baselines
were measured first, so nothing was spent on an agent for a leg with no signal
to find.

| file | contents |
| --- | --- |
| `eq_baselines.log` | full baseline output including the leak canary |
| `eq_signal_probe.csv` | per-row filing-text signals, train and test folds |
| `eq_engineered.log` | the paired engineered-feature comparison |
| `eq_oversight.log` | the oversight-signal comparison and signal prevalence |
