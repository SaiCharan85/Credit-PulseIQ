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
| `backtest1_agent_200cases.csv` | per-case verdicts, scores, tool sequences |
| `backtest1_run.log` | full harness output including reliability curve |
| `signal_probe_200cases.csv` | the eight text/event signals per case |
