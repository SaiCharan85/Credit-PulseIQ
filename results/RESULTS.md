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
