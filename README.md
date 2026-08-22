# CreditPulse IQ

**A corporate credit-distress agent whose reasoning is backtested against real bankruptcy and accounting-fraud outcomes — with lookahead controls, deterministic number-verification, and a tracked false-confidence rate.**

CreditPulse IQ monitors public companies from their SEC filings and produces an evidence-cited, confidence-scored credit-risk memo for a human analyst. It is a monitoring and analysis tool, not an autonomous decision-maker: it never approves a loan, never places a trade, and never states a number it did not deterministically compute and verify.

The project is deliberately built **eval-first**. The evaluation harness is the deliverable; the agent is what slots into it.

---

## Problems this solves

**For a credit analyst.** Monitoring doesn't scale — no one reads every filing for hundreds of names. Distress is fragmented across documents: leverage in the balance sheet, covenant trouble in an 8-K, doubt in an audit opinion, a missed deadline in an NT 10-K. Detection comes too late, because bankruptcy *is* the end of the story. Filings aren't comparable — revenue has five-plus XBRL tags and interest expense is untagged on 39% of filers. And confidence is uncalibrated: "elevated risk" with no idea whether 80% confidence is right 80% of the time.

**For the AI engineering.** These are the harder half, and the reason the repo is shaped as it is:

| Problem | Mechanism |
|---|---|
| LLMs are unreliable at arithmetic | The LLM never computes; it calls typed tools |
| LLMs fabricate plausible numbers | The verifier re-derives every cited figure and hard-fails what it can't reproduce |
| Lookahead leakage invalidates most finance backtests | Enforced at the data layer *and* re-checked at the verify boundary |
| Agentic systems ship unevaluated | Eval-first: 270 tests before a single model call |
| Ground truth is assumed rather than verified | Four-signal labels; 363 text false-positives and 37 miscoded item codes rejected |
| Survivorship bias | CIK-pinned identity; survivors verified as still filing |
| Rare-event class imbalance | A severity ladder, so early warning is measurable at several thresholds |
| Reward hacking the grader | Separate judge; retries triggered only by deterministic checks |
| Silent failure on missing data | Undefined-not-wrong, and missingness carried as a *feature* (the data is MNAR) |
| "Is the agent any good?" is unanswerable | Deterministic baselines it has to beat |

**What it deliberately does not solve.** No credit decisions (advisory only), no private companies, no fraud oracle, covenant leg demo-grade, context leg never moves the graded number.

The sharpest framing: *most "agentic finance" demos can't tell you whether their agent is right. This one is built so that question has an answer.*

---

## Why this exists (and what makes it different)

Most "agentic finance" projects demo well on three hand-picked tickers and were never evaluated at scale. CreditPulse IQ is built around the three things that are hard and rarely done:

1. **Backtested reasoning against real labels.** The distress and earnings-quality investigators are scored against actual outcomes — Chapter 11 filings and SEC Accounting & Auditing Enforcement Releases (AAERs) — using strict as-of data cutoffs so the agent never sees information that did not exist at prediction time. Reported with precision/recall, **lead time**, calibration, and a first-class **false-confidence rate** (high-confidence "healthy" on a name that later failed).

2. **Deterministic verification.** The LLM never does arithmetic. Every ratio, trend, and peer comparison is computed in Python from raw filing values; a verifier recomputes any figure the agent cites and hard-fails the response if it cannot be reproduced. This makes the output auditable and the evals gradeable.

3. **Honest scoping.** Two legs (distress, earnings-quality) rest on real public labels and are backtested. The covenant leg and the context/retrieval leg are explicitly **not** backtested predictors — they are labeled as demo-grade / context-only rather than oversold. Knowing where a system does *not* work is part of the deliverable.

---

## Headline results

L3 backtest, distress leg. 200 firm-period cases (100 Chapter 11 positives + 100 sampled negatives), strict as-of cutoffs, temporal hold-out from 2024-06-01, `gemma-4-31b-it` driving the ReAct loop. Two arms over the **same 200 cases**, differing only in which tools the agent could call.

| Metric | Agent, ratios only | **Agent, + filing-text tools** | Altman Z'' | Hazard (Shumway) |
|---|---|---|---|---|
| AUC | 0.881 | **0.965** | 0.885 | 0.966 |
| Calibration (ECE) | 0.135 | **0.081** | 0.141 | 0.101 |
| Precision / recall | 0.795 / 0.714 | **0.870 / 0.897** | — | — |
| Median lead time | 161 days | **155 days** | — | — |
| **False-confidence rate** | 0.250 (25/100) | **0.080 (8/100)** | — | — |
| Verification failures | 0 | **0** | — | — |

Paired bootstrap, 8,000 resamples, identical cases:

```
tools effect    : +0.0841   95% CI [+0.0484, +0.1246]   -> REAL
agent vs hazard : -0.0017   95% CI [-0.0262, +0.0225]   -> tie
```

**The finding: giving the agent evidence a ratio model cannot read is worth +0.084 AUC.** That is the largest measured effect in the project and its interval is nowhere near zero. It converted a clear loss against the hazard baseline into a statistical tie, and cut catastrophic errors threefold.

**The agent does not beat the baseline, and against a properly specified one it loses.** This is a correction to an earlier version of this README, which reported a tie and stopped there.

Every model in this project was `LogisticRegression(C=1.0)` -- the sklearn default, never tuned, never a different family. A baseline nobody tried to make strong is not a baseline. Tuning a gradient-boosted model on an inner validation split carved from the training window, and scoring the test fold once:

| arm, identical 169 cases | AUC |
|---|---|
| Altman Z'' | 0.8801 |
| **Agent (ReAct + filing-text tools)** | **0.9634** |
| hazard, logistic C=1.0 *(the originally published baseline)* | 0.9748 |
| **hazard, tuned LightGBM** | **0.9884** |

```
agent - logistic : -0.0113  95% CI [-0.0377, +0.0120]  tie
agent - GBM      : -0.0249  95% CI [-0.0520, -0.0036]  agent is worse
```

So the honest sentence is: **the agent ties an untuned logistic hazard model and is beaten by a tuned gradient-boosted one.** For producing a ranked score, the statistical model wins, and it wins by more than we first reported.

What this does *not* touch is the tools result above. That is an agent-against-itself paired comparison on identical cases, so a stronger baseline cannot move it: giving the agent evidence a ratio model cannot read is still worth +0.084, and false-confidence still falls threefold.

**Where the gain came from.** 27 bankruptcies missed in the first arm are caught in the second, **none regressed**, and 26 of the 27 carried a filing-text signal. The agent's failure mode was never faulty reasoning -- it was blindness. It spent ~12 steps investigating a feature set that did not contain the answer for a quarter of the failures. You cannot reason your way to a covenant breach from a current ratio.

The control matters as much as the effect: where no signal was found the verdict barely moved (CIK 1498710, 0.42 -> 0.48), and precision *rose* 0.795 -> 0.870. Had the tools merely made the model more alarmist, every score would have risen and precision would have collapsed.

**Zero verification failures** across both arms. Not one fabricated figure reached an output; every cited number was independently recomputed from its provenance.

Reproduce with `python -m evals.run_l3 --agent react --max-negatives 100`. Full numbers and limits in [`results/RESULTS.md`](results/RESULTS.md); methodology in [`SPEC.md`](SPEC.md) §7.

### The second leg does not work, and that is reported

The earnings-quality leg was built to the same standard and **failed**. Four independent approaches against 891 test positives with a clean leak canary:

| approach | out-of-sample AUC |
|---|---|
| Beneish M (published) | 0.512 |
| fitted logistic on ratios | 0.579 |
| plus Beneish decomposed into its terms | 0.585 |
| filing-text and 8-K event signals | 0.579 |
| leak canary (shuffled labels) | 0.526 — clean |

Two earlier numbers looked promising and were wrong: ratios at 0.714 and text at 0.867, both artefacts of a distress-enriched universe and in-sample fitting. Widening to a 12,570-filer point-in-time universe killed both. The leg is **demo-grade, excluded from the graded signal**, and no investigator was built for it — the eval-first order meant the baselines were measured before anything was spent on an agent with nothing to find.

### Does giving the agent the baseline help? No.

A second arm let the agent consult the fitted hazard model through a
`get_model_score` tool, framed as one opinion to weigh rather than an answer to
copy. It copied it.

| | |
|---|---|
| Called `get_model_score` | **143 / 143 cases** |
| Correlation, agent score vs baseline | **0.967** |
| Cases differing by more than 0.25 | 6 (4%) |

On the same 143 cases: **baseline alone AUC 0.980, agent alone 0.972, the two
averaged 0.979.** The agent reproduces the baseline slightly worse than the
baseline reproduces itself, and blending them beats neither. Where the two
disagree by more than 0.2 (7 cases), the agent is closer to the truth 3 times
and the baseline 4 -- indistinguishable from chance.

The tool is now absent from the schema list rather than merely discouraged, and
the fusion arm refuses to run on any prediction with `baseline_seen=1`, because
fusing the agent with a model it already read is circular.

This only became visible because per-case results record whether the tool was
called and what the agent scored. The aggregate metrics were identical either
way: an agent that reasons its way to 0.97 and an agent that copies its way
there produce the same summary table.

### A result that only appeared because abstention was instrumented

The first full run reported AUC 0.965 -- apparently matching the hazard model. It was wrong. 151 of 200 cases (75.5%) had abstained, **every one a step-budget exhaustion and not a single honest judgment**: the model averaged 13.3 of 14 steps, investigating thoroughly and getting cut off before it could conclude. The metrics rested on the 49 cases that happened to finish, a self-selected quarter.

Splitting abstentions into *honest* versus *protocol failure* is what surfaced it. Without that split it would have read as admirable caution and shipped as a headline number.

### Three grading bugs that flattered the agent

Found and fixed before the final numbers, each documented in the commit history:

- **Baselines were scored on the full split** while the agent's AUC covered only its non-abstained cases -- letting it drop the hardest cases while the baselines, which never abstain, got no such filter. Baselines are now matched on `(cik, as_of)`.
- **The signal-to-probability mapping used direction only**, scoring "watch at 0.9" identically to "healthy at 0.9". Four ordinal verdicts collapsed to two numbers.
- **The LLM cache keyed on model and conversation but not the tool list**, so toggling a tool while leaving the prompt untouched would have replayed answers generated with the other tool set.

---

## What exists today

All five build phases are complete. **526 tests pass, all offline** -- the ReAct loop is tested against a scripted model, so agentic behaviour, the guard gate and memo assembly are all verified without an endpoint or a network.

| phase | status |
|---|---|
| 1. Ground truth + scaffold | complete |
| 2. Deterministic spine + L0 | complete |
| 3. Distress investigator, fully backtested | complete — **the deliverable** |
| 4. Earnings-quality leg | complete **as a measured negative**; demo-grade |
| 5. Orchestrator, guardrails, cited memo | complete, honestly scoped to one green leg |

The eval ladder is populated end to end:

| | covers | files |
|---|---|---|
| L0 | deterministic ratio, score, trend and peer arithmetic | 15 |
| L1 | XBRL extraction from real filings | 1 |
| L2 | peer construction and trends at the tool boundary, incl. peer as-of leakage | 1 |
| L3 | investigator vs real Chapter 11 and restatement outcomes | 3 |
| L4 | end-to-end, filer facts in -> cited memo out | 1 |
| L5 | guardrails and adversarial: decision framing, fabricated figures, abstention | 1 |

| | |
|---|---|
| Distress universe | **369 CIK-pinned filers** — 156 Chapter 11, 213 still filing |
| Chapter 11 labels | **156**, every one verified by four independent checks |
| Graded distress events | **3,005** across four severity tiers |
| Earnings-quality universe | **12,570** point-in-time annual filers, no survivorship bias |
| Restatement labels | **887** verified, after excluding 810 SPAC regulatory reclassifications |
| Deterministic metrics | 36, each with a hand-computed L0 assertion |
| Tests | **526** |

Labels by cohort, and the ladder by tier:

| Cohort | Count | | Tier | Events | Filers |
|---|---|---|---|---|---|
| `recent_2025_2026` | 29 | | `default` | 156 | 156 |
| `prior_2021_2024` | 112 | | `near_default` | 863 | 252 |
| `historical_pre_2021` | 15 (9.6%) | | `stress` | 776 | 186 |
| | | | `early_warning` | 1,210 | 246 |

At 156 positives a recall estimate carries a 95% CI of roughly ±8pp — enough to report. Of those 156, **129 (83%) did not survive**: 92 deregistered via Form 15, 37 went dark, 21 emerged.

### Findings that shaped the design

Each came out of real data and is pinned by a test:

- **Ticker identity is survivorship-biased.** SEC's `company_tickers.json` lists only currently-listed filers, so every bankrupt name is missing from it — and `BBBY` now resolves to Overstock, which bought the brand out of bankruptcy. The universe is keyed to CIK, never ticker.
- **No single bankruptcy signal is trustworthy.** Full-text search flags solvent companies (WEX, SM Energy, Opendoor) on boilerplate; filer-supplied item codes are miscoded — J.C. Penney's earliest Item 1.03 is a **2014 shareholder rights plan**, six years before its bankruptcy.
- **A truthful filing can still be about someone else.** FirstEnergy Corp ($42bn, healthy) filed an Item 1.03 describing its *subsidiary's* Chapter 11. Genuine filers join themselves to the debtors with a conjunction ("the Company **and** certain of its subsidiaries…filed"); parents use an appositive ("each a subsidiary **of** X, filed"). That distinction is the fourth check.
- **Distress erases the tag you need.** Sleep Number's final pre-bankruptcy 10-K reports no long-term debt at all; its entire $588m is reclassified to `DebtCurrent`. Naive extraction leaves leverage undefined on exactly the companies the model exists to catch.
- **Missingness is not random.** Data goes missing *because* of distress, which makes it MNAR — so imputation would erase the signal, and absence is carried as an explicit feature instead.

### Deterministic baselines (the bar the agent must beat)

Without a baseline an L3 precision figure is uninterpretable. If the ReAct investigator can't beat a 1980-vintage logit on the same as-of data, that is a result worth reporting honestly.

The hazard framing is deliberate: Shumway (2001) showed that static one-observation-per-firm models are biased, and it handles the fact that survivors are **censored** rather than proven negatives.

### Calibration

Measuring ECE says a model is overconfident; it does not fix it. Monotonic recalibration is fitted on a **separate temporal fold** — the model trains on the earliest window, the calibrator on a later slice it never saw, and the test set stays untouched:

| Calibrator | ECE | AUC |
|---|---|---|
| None | 0.1653 | 0.9393 |
| Platt | 0.0431 | 0.9393 |
| **Isotonic** | **0.0250** | 0.9386 |

**6.6× better calibrated with the ranking intact** — AUC holding constant is the correctness check, since both maps are monotonic. The raw model stated 0.091 where the event occurred 0.3% of the time.

Holding out a calibration fold costs some discrimination (AUC 0.957 → 0.939, fitting on 3,209 rows instead of 4,365). Stated rather than hidden.

Panel: **5,582 firm-period observations**, 354 filers, 500 positives, 1-year horizon. Split **temporally** at 2024-06-01 (a random split would leak, since a firm's adjacent quarters are near-identical). Test set: 1,217 rows, 100 positives, 8.2% base rate.

| Model | Test AUC | P@10 | P@25 | ECE |
|---|---|---|---|---|
| Tier 0 — Altman Z'' (unfitted) | 0.879 | 0.600 | 0.640 | 0.311 |
| **Tier 1 — discrete-time hazard** | **0.957** | **0.700** | **0.760** | 0.160 |
| *Leak canary — labels shuffled* | *0.488* | *0.300* | *0.200* | — |

**The leak canary is the number that matters most.** Permuting labels collapses the model to chance, which is what says the signal is in the data rather than in the pipeline. It runs as `--shuffle-labels`.

Read these honestly:

- **Calibration is poor (ECE 0.16).** `class_weight="balanced"` deliberately inflates probabilities away from the 8.2% base rate, and the base rate itself swings by year (2.9% in 2021 → 15.0% in 2023 → 6.6% in 2025). Discrimination is the comparable quantity; these are not population default rates (Zmijewski 1984 — the universe is enriched far above the true ~0.5%).
- **Coefficients are not individually interpretable.** `current_ratio` and `quick_ratio` come out with large opposite signs — textbook multicollinearity between near-identical covariates.
- **Test AUC exceeds train AUC (0.957 vs 0.897).** With 100 test positives that is ~2 standard errors, and the periods have different base rates; it is not treated as a real improvement.
- **Missingness helps, but modestly.** Metrics alone reach 0.937; missingness indicators alone reach 0.740 with P@25 of just 0.20. Ablation confirms the model reads distress rather than proxying sector — a real risk, since survivors are missing `Liabilities` (and hence Altman/Ohlson) far more often than distressed filers (46% vs 19%), largely a REIT/financial composition effect.

---

### The distress investigator (Phase 3)

A real ReAct loop, not one prompted call: it hypothesises, calls a typed tool, reads the structured result, chooses the next call *from that result*, and terminates on its own judgment — including at *insufficient evidence*.

The division of control is the design:

| The loop owns | The model owns |
|---|---|
| Tool dispatch and argument validation | Which tool to call next |
| The step budget, and forced abstention | When it has enough evidence |
| The deterministic critic and bounded retries (≤2) | Signal, confidence, residual |

**The as-of date is not a tool argument.** A `ToolBox` is constructed for one filer at one prediction date, so the model cannot request data it should not see — lookahead is unrepresentable rather than merely guarded against.

Tools are declared through the API's native function-calling interface, so arguments arrive provider-validated. A JSON-in-text fallback remains for models without it.

Live example — Sleep Number at 2026-03-01, three months before it filed:

```
tools : available_periods → get_metric ×3 → check_threshold → get_prior_distress_events → get_trend
signal: severe_risk (0.9)   verification: passed   retries: 0
cited : current_ratio 0.20, quick_ratio 0.086, equity −$451.6m, interest_coverage 0.47
```

### What the capability floor actually is

Eight models were tried against this loop. The failures were as informative as the successes:

| Model | Outcome |
|---|---|
| Qwen2.5-0.5B / 1.5B (local CPU) | Too weak — 1.5B **fabricated a metric it never fetched**; the critic caught it |
| `openai/gpt-oss-20b` | Harmony control tokens corrupt content *and* tool names on every configuration |
| `llama-3.1-8b-instant` | Loops without deciding — 12 calls, repeating tools, never concludes |
| `llama-3.3-70b-versatile` | **Works** — but 100K tokens/day ≈ 6 cases |
| `gemini-3.7-flash` | Works — but 20 requests/day |
| `mistral-small-3.2-24b` | **Works well** — 10 tool calls, correct call, verification passed |
| `gemma-4-31b-it` | **Works** — 14,400 requests/day; the run model |

Two conclusions worth stating. First, the floor is real: below roughly 8B the model cannot hold a multi-step tool protocol, and at 1.5B it invents figures — which is exactly what the numeric guard exists to catch, and did. Second, **free-tier quotas vary by ~700×** between models on the same provider, and that, not capability, was the binding constraint throughout.

---

## Architecture

**Pattern: orchestrator–workers.** A coordinator routes to specialized worker agents and fuses their outputs; workers do not talk to each other (orchestrated, not collaborative — chosen for auditability). Each worker is a **ReAct** tool-use loop. The whole thing is wrapped in a **deterministic critic–reviser** verification stage.

```
watchlist & trigger  →  data plane  →  deterministic compute  →  orchestrator
                                                                      │
              ┌───────────────────┬──────────────────┬────────────────┤
              ▼                   ▼                  ▼                ▼
          distress        earnings quality       covenant        context
       (Ch. 11 labels)     (AAER labels)       (demo-grade)   (retrieval, context-only)
              └───────────────────┴───────────┬──────┴────────────────┘
                                              ▼
                                      verify & fuse  →  cited risk memo + audit trail
```

- **Data plane** — pulls and structures SEC filings (EDGAR XBRL + exhibits) for a watchlist; triggers on new filings or quarterly.
- **Deterministic compute** — ratios, trends, peer comparisons as pure functions. **No LLM.**
- **Orchestrator** — reads deterministic signals and decides which workers to run and how deep; a clean balance sheet gets a shallow pass, a leverage spike triggers the deeper loops.
- **Workers (ReAct loops):**
  - *Distress* — financial-deterioration investigation. Backtested vs Chapter 11. **Reference implementation — build first.**
  - *Earnings quality* — accrual/red-flag investigation. Backtested vs AAER.
  - *Covenant* — extracts terms + computes headroom. **Demo-grade** (thin public data, proxy labels only).
  - *Context/retrieval* — findings-triggered retrieval over related filings/news/peers to explain a signal. **Context-only:** contributes narrative, never moves the graded numeric signal, every claim cited and groundedness-checked. Out of the outcome backtest.
- **Verify & fuse** — recomputes every cited figure, fuses signals into one graded assessment, escalates to *insufficient evidence* rather than guessing.
- **Output** — an evidence-cited, confidence-scored memo with a full audit trail, for human review.

Model serving via vLLM; a **separate** judge model scores memo quality (never the generating model).

**Governed by design.** Guardrails are the enforcement pillar of a governance-aware design, not the whole of it: the cited audit trail (auditability), the human-in-the-loop memo (oversight), the online judge and drift alarms (monitoring), and the documented limitations (honesty) are the rest. Because the system is advisory and never makes a credit decision, it sits outside fair-lending and high-risk-decisioning regimes by design, while mapping cleanly onto model-risk-management and NIST AI RMF expectations. See [`SPEC.md`](SPEC.md) § 10.

---

## Evaluation (tooling per layer)

The eval harness is the point. Tooling is matched to what each layer actually is — not everything is an "LLM eval."

| Layer | What it tests | Kind | Tooling | Status |
|---|---|---|---|---|
| **L0** | Deterministic ratio/trend/peer math | Unit test | pytest | ✅ |
| **L1** | XBRL extraction accuracy | Unit test | pytest | ✅ |
| **L2** | Peer-group + trend correctness | Unit test | pytest | ✅ |
| **L3** | Investigator diagnosis vs real labels | **Outcome-based backtest** | **custom harness** (pytest-driven) | built, running |
| **L4** | End-to-end (name → memo), cost/latency | Outcome + behavioral | custom harness + DeepEval | |
| **L5** | Guardrails / adversarial; retrieval groundedness | LLM-as-judge / behavioral | DeepEval; RAGAS (context leg) | |
| Online | Live drift, groundedness, calibration | LLM-as-judge + tracing | Langfuse / Opik | |

The headline is the **custom outcome-based backtest** (L3) with lookahead controls — that is the differentiator. RAGAS/DeepEval/Langfuse are the supporting cast for the groundedness, guardrail, and monitoring layers.

---

## Feedback loops (three timescales)

- **Within-request — critic–reviser (deterministic critic).** A hard check fails (unreproducible figure, scope, staleness) → the specific defect is fed back for a **bounded** retry → on exhaustion, **abstain** (*insufficient evidence*). Never loops against the LLM judge (that would reward-hack the grader).
- **Near-real-time — judge → human.** Soft judge flags (low groundedness/calibration) route to human review and lower confidence; they do **not** trigger automated retries.
- **Over-time — active-learning flywheel.** Human accept/edit/override and accrued real outcomes become new eval examples and labels for the narrow classifiers. Not RLHF: no reward model, no weight updates on the agent.

---

## Scope discipline (what this deliberately does not do)

- **No synthetic training data.** Real labels are the entire edge. `data/labels/aaer.csv` ships as a schema with zero rows rather than a fabricated set.
- **No auto-decisioning.** Human-in-the-loop memo only; no approve/deny, no trade signals.
- **Covenant stays demo-grade; context leg is context-only.** Neither is sold as a backtested predictor.
- **The agent is not fine-tuned.** A strong instruct model is orchestrated; only the narrow classifiers are later fine-tune candidates (supervised LoRA/QLoRA, not RLHF).
- **Public companies only.** The valuable private/middle-market segment needs private data this project intentionally does not fake.
- **Agency is confined to the workers.** Compute, verification, and the input gate are deliberately plain code.

---

## Repository layout

| Path | Contents | State |
|---|---|---|
| [`data/`](data/) | EDGAR client, watchlist, label sets, discovery, severity ladder, outcomes, as-of dating | built |
| [`compute/`](compute/) | Deterministic ratios, trends, peer groups, composite scores, provenance (no LLM) | built |
| [`verify/`](verify/) | Numeric recomputation, lookahead + staleness guards | built |
| [`models/`](models/) | Firm-period panel, hazard + Altman baselines, evaluation metrics | built |
| [`evals/`](evals/) | L0–L2 suite, real-filing fixtures | built |
| [`agents/`](agents/) | Distress investigator (ReAct loop), typed tools, deterministic critic, model clients | built |
| `agents/` (rest) | Orchestrator + earnings-quality / covenant / context workers | Phase 4–5 |
| `serving/` | vLLM config + judge model | Phase 3 |
| `feedback/` | Input-quality gate + online judge + flywheel | Phase 5 |

See [`SPEC.md`](SPEC.md) for the full design and [`PROMPT.md`](PROMPT.md) for the build brief.

---

## Quickstart

```bash
pip install -e ".[dev]"

# L0-L2 run offline from committed fixtures
pytest -q

# Anything hitting SEC needs a declaring User-Agent (SEC requires contact info)
export CREDITPULSE_SEC_UA="CreditPulse IQ research you@example.edu"

# Rebuild the label set end to end
python -m data.discover --start 2021-01-01 --end 2026-08-14   # candidates for review
python -m data.promote_labels --candidates data/labels/candidates_chapter11.csv
python -m data.build_watchlist --survivors 250                # sector-matched survivors
python -m data.distress_events --since 2015-01-01             # the severity ladder
python -m data.outcomes                                       # survived or not

# Fit the deterministic baselines the agent has to beat
python -m models.run_baseline

# L3 backtest. The rule-based control needs no model at all:
python -m evals.run_l3 --agent rules --max-negatives 300

# The ReAct investigator needs an OpenAI-compatible endpoint (vLLM or hosted).
# A ~20B mixture-of-experts is the default: few active parameters, so it runs
# at small-model cost. Measured floor is below it -- 0.5B could not hold the
# tool protocol, 1.5B fabricated a metric it had never fetched.
export CREDITPULSE_LLM_BASE_URL="https://api.groq.com/openai/v1"
export CREDITPULSE_LLM_MODEL="openai/gpt-oss-20b"
export CREDITPULSE_LLM_API_KEY="..."
python -m evals.run_l3 --agent react --max-negatives 300

# The judge must be a different model (hard rule 7); cross-provider is safest.
export CREDITPULSE_JUDGE_MODEL="gemini-2.5-flash"
```

Responses are cached by model and conversation, so re-running a backtest after
changing only the grading code costs nothing and reproduces exactly.

---

## Data sources

All public and free:

- **SEC EDGAR** — XBRL company facts (`data.sec.gov/api/xbrl`), submissions metadata, filing documents, and full-text search (`efts.sec.gov`).
- **Bankruptcy labels** — Chapter 11 petitions established from 8-K Item 1.03 filings, as-of dated, three-signal verified.
- **AAER dataset** — SEC Accounting & Auditing Enforcement Releases as misstatement labels (schema in place; population gated behind Phase 3).

Other sources evaluated and their trade-offs are recorded in [`data/labels/README.md`](data/labels/README.md).

---

## Build sequence (eval-first — do not reorder)

1. ✅ Ground truth + scaffold; EDGAR data plane; label sets with as-of dating.
2. ✅ Deterministic spine + L0 tests gating in CI. No LLM yet.
3. ✅ **Distress investigator as a real ReAct loop + its L3 backtest.** Produced a real calibration curve; filing-text tools measured at +0.084 AUC.
4. ✅ Earnings-quality leg vs restatement labels, reusing the harness — **measured, failed, reported**. Demo-grade, excluded from the graded signal.
5. ✅ Assemble: orchestrator, guardrails, memo, L4 and L5. Scoped to one green leg rather than faking a fusion. Feedback layer not built.

## Status

**Complete as an eval-first study.** One backtested leg at parity with a strong statistical baseline, one leg measured and honestly declared unusable, the full L0–L5 ladder, and every number reproducible from the committed harness.

Known gaps, stated rather than hidden:

- **The feedback layer is not built** (SPEC §9) — no online judge, no drift monitoring. It was always the last item and it needs production traffic to mean anything.
- **The covenant leg does not exist.** It was demo-grade by design and nothing was built.
- **Absolute probabilities do not transfer.** The universe is enriched with distressed filers far above the population base rate (Zmijewski 1984); ranking transfers, calibrated probabilities do not.
- **The agent has never chosen to abstain.** All abstentions across 200 cases were protocol failures. "It will tell you when it does not know" is an unexercised claim.
- **One model, one window.** Validated on `gemma-4-31b-it` over 2024–2025. Any swap needs re-measuring — which the harness makes cheap, and which is the point of having built it first.

## License

TBD (MIT suggested for a portfolio repo).
