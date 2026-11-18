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
| Agentic systems ship unevaluated | Eval-first: the harness came before the agent, and now carries 1,412 tests |
| Ground truth is assumed rather than verified | Four-signal labels; 363 text false-positives and 37 miscoded item codes rejected |
| Survivorship bias | CIK-pinned identity; survivors verified as still filing |
| Rare-event class imbalance | A severity ladder, so early warning is measurable at several thresholds |
| Reward hacking the grader | Separate judge; retries triggered only by deterministic checks |
| Silent failure on missing data | Undefined-not-wrong, and missingness carried as a *feature* (the data is MNAR) |
| "Is the agent any good?" is unanswerable | Deterministic baselines it has to beat |

**What it deliberately does not solve.** No credit decisions (advisory only), no private companies, no fraud oracle, covenant leg demo-grade, retrieved text never supplies a figure.

The sharpest framing: *most "agentic finance" demos can't tell you whether their agent is right. This one is built so that question has an answer.*

---

## Why this exists (and what makes it different)

Most "agentic finance" projects demo well on three hand-picked tickers and were never evaluated at scale. CreditPulse IQ is built around the three things that are hard and rarely done:

1. **Backtested reasoning against real labels.** The distress and earnings-quality investigators are scored against actual outcomes — Chapter 11 filings and SEC Accounting & Auditing Enforcement Releases (AAERs) — using strict as-of data cutoffs so the agent never sees information that did not exist at prediction time. Reported with precision/recall, **lead time**, calibration, and a first-class **false-confidence rate** (high-confidence "healthy" on a name that later failed).

2. **Deterministic verification.** The LLM never does arithmetic. Every ratio, trend, and peer comparison is computed in Python from raw filing values; a verifier recomputes any figure the agent cites and hard-fails the response if it cannot be reproduced. This makes the output auditable and the evals gradeable.

3. **Honest scoping.** Two legs (distress, earnings-quality) rest on real public labels and are backtested. Filing-text retrieval is backtested as an input to the distress leg, never as a source of figures. The covenant leg is explicitly demo-grade rather than oversold. Knowing where a system does *not* work is part of the deliverable.

---

## Headline results

L3 backtest, distress leg. 200 firm-period cases (100 Chapter 11 positives + 100 sampled negatives), strict as-of cutoffs, temporal hold-out from 2024-06-01, `gemma-4-31b-it` driving the ReAct loop. Two arms over the **same 200 cases**, differing only in which tools the agent could call.

| Metric | Agent, ratios only | **Agent, + filing-text tools** | Altman Z'' | Hazard (Shumway) |
|---|---|---|---|---|
| AUC | 0.881 | **0.963** | 0.885 | 0.966 |
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

**The agent ties a properly specified baseline -- which took two corrections to establish.** The first comparison was against an untuned `LogisticRegression`. A tuned LightGBM then appeared to beat the agent, but only because the panel repeats each firm quarterly: the fitted models had already trained on every graded company (13.4 prior observations each). Holding test firms out entirely (group-aware splitting) removes that memorisation:

| identical 169 graded cases | AUC |
|---|---|
| Agent (ReAct + filing-text tools) | 0.9634 |
| LightGBM, regularised, group-aware | 0.9768 |
| *LightGBM, leaky split* | *0.9919* |

```
agent - GBM (group-aware, regularised) : -0.0133   95% CI [-0.0421, +0.0124]   -> tie
```

Two caveats travel with those levels. Roughly **0.09 AUC of every fitted baseline was firm memorisation**, and the most recent window is easy: over 25 group-aware random splits the GBM averages 0.9169 [0.8916, 0.9518], so the serving ranker ships with 0.9169 rather than the single-fold number. The paired comparison is unaffected, because the window inflation is shared. (The agent ranks by its calibrated `risk_probability`; resampled by firm it is 0.9651 [0.9337, 0.9874].)

**Where the gain came from.** 27 bankruptcies missed without the filing-text tools are caught with them, none regressed, and 26 of the 27 carried a filing-text signal. The failure mode was blindness, not reasoning -- you cannot reason your way to a covenant breach from a current ratio. Precision *rose* (0.795 -> 0.870), so the tools did not simply make the agent more alarmist.

**Zero verification failures** across both arms: every cited figure was independently recomputed from its provenance.

Reproduce with `python -m evals.run_l3 --agent react --max-negatives 100`. Full numbers and limits in [`results/RESULTS.md`](results/RESULTS.md); methodology in [`SPEC.md`](SPEC.md) §7.

### One model call instead of ten: single-pass investigation with filing-text RAG

The ReAct loop spends ~10 model calls deciding which deterministic tool to run next. Every one of those tools is deterministic, so the single-pass investigator runs them all in code, adds passages **retrieved from the filings themselves**, and asks the model once. Figures, the critic and the as-of rule are unchanged; any quoted passage must appear verbatim in the passage it cites.

A/B on a stratified 199 cases (99 bankruptcies), `gemma-4-31b-it`, identical cases and grading:

| Metric | ReAct tool loop | **Single-pass + RAG** | Hazard |
|---|---|---|---|
| AUC (abstentions scored 0.5) | 0.974 | **0.964** | 0.969 |
| Bankruptcies called healthy | 8 / 99 | **0 / 99** | — |
| Recall | 0.896 | **0.989** | — |
| Base-rate-corrected precision | **0.462** | 0.315 | — |
| Median lead time | 148 days | **165 days** | — |
| Model calls per case | ~10 | **~1** | — |
| Verification failures | 0 | **0** | — |

```
single - loop : -0.0093   paired-bootstrap 95% CI [-0.0287, +0.0092]   -> tie
```

**Same ranking quality at a tenth of the model calls**, with a different error profile: single-pass missed no bankruptcy and warns earlier, at the cost of more false alarms.

The retrieval design is under [Architecture](#filing-text-retrieval-rag). It is evaluated against deterministic ground truth (no judge), 158 cases:

| Warning | Recall@5 | MRR@10 | nDCG@10 |
|---|---|---|---|
| Going concern | 0.93 | 0.94 | 0.75 |
| Material weakness (filing-vocabulary queries + section filter) | 0.88 | 0.76 | 0.67 |
| Material weakness, + cross-encoder rerank (offline only) | 0.96 | 0.81 | — |
| Covenant trouble | ~0.4 | ~0.37 | ~0.25 |

**Which setting the A/B used.** The single-pass arm retrieves with the plain-language baseline: five standard questions, hybrid search, three passages each (`agents/single_pass.py`). The tuned material-weakness setting (0.88, up from 0.67 for the baseline) and the rerank are retrieval results, not part of the A/B; adopting them in the investigator would need the A/B re-run.

The cross-encoder (`bge-reranker-v2-m3`) reads query and passage together, which lifts material weakness to 0.96. It is measured, not served: on CPU it costs minutes per case against milliseconds for hybrid search. On covenant trouble it reaches 0.58 only when it rescores the whole candidate pool (~4.5 min per case); small rerankers (MiniLM-L-6/L-12) do not separate "in compliance" from "not in compliance" (0.33-0.38).

Store coverage is 100%: every warning the filing contains is indexed, so every miss is a ranking miss. Covenant retrieval is the open extension -- embeddings rank "in compliance with our covenants" beside "not in compliance". It does not affect the headline; the fix is a deterministic covenant scan like the going-concern one (see [`results/RESULTS.md`](results/RESULTS.md), Phase 5).

Reproduce with `python -m evals.run_l3 --agent single --store cloud --max-negatives 100`, then `python -m evals.compare_arms` and `python -m evals.run_rag_recall --store cloud`.

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

All six build phases are complete. **1,412 tests pass, all offline** -- the ReAct loop is tested against a scripted model, so agentic behaviour, the guard gate and memo assembly are all verified without an endpoint or a network.

| phase | status |
|---|---|
| 1. Ground truth + scaffold | complete |
| 2. Deterministic spine + L0 | complete |
| 3. Distress investigator, fully backtested | complete — **the deliverable** |
| 4. Earnings-quality leg | complete **as a measured negative**; demo-grade |
| 5. Orchestrator, guardrails, cited memo | complete, honestly scoped to one green leg |
| 6. Filing-text RAG + single-pass investigator | complete as a backtested design; not yet wired into the app |

The eval ladder is populated end to end:

| | covers | files |
|---|---|---|
| L0 | deterministic arithmetic, PDF and form parsing, the read-only verifier, the ordinal risk score, label dedup, tracing, retry parsing, score nulls | 29 |
| L1 | XBRL extraction from real filings | 1 |
| L2 | peer construction and trends at the tool boundary, incl. peer as-of leakage | 1 |
| L3 | investigator vs real Chapter 11 outcomes; the LangGraph pipeline | 4 |
| L4 | end-to-end, filer facts in -> cited memo out | 1 |
| L5 | guardrails and adversarial: decision framing, fabricated figures, citation provenance, context poisoning, answer shape, follow-ups, partial answers | 13 |

Eleven harnesses run against a live model rather than a fixture. Each of the
last three was added because a defect reached a reader that every existing eval
passed cleanly:

| | asks | current |
|---|---|---|
| `run_l3` | does the agent beat the baselines? | 0.963, ties a 0.977 GBM |
| `run_response_eval` | are answers true and safe? | 39/39 over 8 categories |
| `run_answer_quality` | do answers take the shape asked for? | 18/18 over 13 intents |
| `run_figure_audit` | does every number reproduce from EDGAR? | 142/142 to 1e-9 |
| `run_hallucination_probe` | can a figure be baited out of it? | 0 leaked, 0 false refusals |
| `run_fairness_decay` | does it hold across subgroups? | found, then withdrew, a false gap |
| `run_label_crosscheck` | are the labels right? | 8/8 against court dockets |
| `run_evidence_recall` | did it find the signals present? | 84%, 0 never-looked |
| `run_signal_probe` | which signals move the verdict? | the mechanism behind +0.084 |

| | |
|---|---|
| Distress universe | **368 CIK-pinned filers** — 155 Chapter 11, 213 still filing |
| Chapter 11 labels | **155**, verified four ways in-repo and spot-checked against court dockets |
| Graded distress events | **3,004** across four severity tiers |
| Earnings-quality universe | **12,570** point-in-time annual filers, no survivorship bias |
| Restatement labels | **887** verified, after excluding 810 SPAC regulatory reclassifications |
| Deterministic metrics | 36, each with a hand-computed L0 assertion |
| Filing-text chunks indexed | **~139k** from 875 10-K/10-Q filings |
| Tests | **1,412** |

Labels by cohort, and the ladder by tier:

| Cohort | Count | | Tier | Events | Filers |
|---|---|---|---|---|---|
| `recent_2025_2026` | 29 | | `default` | 156 | 156 |
| `prior_2021_2024` | 112 | | `near_default` | 863 | 252 |
| `historical_pre_2021` | 15 (9.6%) | | `stress` | 776 | 186 |
| | | | `early_warning` | 1,210 | 246 |

At 156 positives a recall estimate carries a 95% CI of roughly ±8pp — enough to report. Of those 156, **129 (83%) did not survive**: 92 deregistered via Form 15, 37 went dark, 21 emerged.

### The stack, named

| layer | what runs |
|---|---|
| **Orchestration** | LangGraph `StateGraph` — six nodes, a conditional guard edge, optional checkpointing |
| **Agent** | hand-rolled ReAct loop over typed tools (served); single-pass investigator over the same tools plus retrieved passages (backtested); deterministic critic with ≤2 retries |
| **Verification** | read-only recomputation from raw XBRL at 1e-9; citation binding on measure *and* period |
| **Retrieval** | Qdrant hybrid search (bge-small + BM25, RRF), store-enforced as-of filter |
| **Ranking** | LightGBM, group-aware, served as an ordinal percentile beside the agent |
| **Evals** | six-level ladder, 1,412 tests, adversarial probes, subgroup fairness, resampled AUC, retrieval Recall/MRR/nDCG |
| **Observability** | Langfuse spans, scores and experiments (A/B arms, retrieval variants), SSE step streaming, per-memo audit trail |

### What a reader actually gets

Two models, two jobs, measured separately and shown side by side.

| | job | resampled |
|---|---|---|
| **GBM ranker** | orders filers against each other; drives the watchlist | 0.9187, 25 group-aware splits [0.8885, 0.9394] |
| **ReAct agent** | investigates one filer and writes the cited memo | 0.9651, firms resampled [0.9337, 0.9874] |

They tie on discrimination, so neither overrules the other. What separates them
is that the GBM emits a number and cannot say why, while every figure the agent
quotes carries the measure and period it came from. **Where the two disagree the
UI says so** rather than averaging them -- a filer the covariates rank high and
the narrative reads healthy is the most interesting row on the screen.

Around that:

* **Every figure is verifiable.** `evals/run_figure_audit.py` goes back to
  EDGAR, re-applies the as-of filter, recomputes, and compares against what is
  on screen across four surfaces. Currently **142/142 to 1e-9**.
* **Numbers cannot be invented.** An adversarial probe of twelve questions
  designed to bait a figure the system does not have leaks **0**, with **0**
  false refusals on a control group -- the control matters, because refusing
  everything scores a perfect zero and is useless.
* **Answers take the shape of the question.** Length, prose vs points vs
  sections vs tables, and charts are each chosen from the question rather than
  fixed in the prompt.
* **Names resolve by ranking, not by rule.** BM25 over names and tickers picks the best candidate in the whole question and rejects weak matches on score; a left-to-right heuristic had matched *Canaan Inc.* to a typo'd "Can you". Tickers and English words that are also company names ("any", "value") only match when capitalised.
* **Follow-ups work.** The last three turns are carried, scoped to one filer at
  one date, and the conversation persists until the reader clears it. Ask about
  a different company and it says where to go instead of guessing.
* **A fairness claim that did not survive being checked.** A 0.20 AUC gap on thinly-disclosed filers was an artefact of ranking that subgroup by `risk_score` instead of the headline `risk_probability`; on the reported metric there is no gap (0.976 vs 0.964). The UI warning was withdrawn, and views that ranked by `risk_score` now use the calibrated score.

### Findings that shaped the design

Each came out of real data and is pinned by a test:

- **Ticker identity is survivorship-biased.** SEC's `company_tickers.json` lists only currently-listed filers, so every bankrupt name is missing from it — and `BBBY` now resolves to Overstock, which bought the brand out of bankruptcy. The universe is keyed to CIK, never ticker.
- **No single bankruptcy signal is trustworthy.** Full-text search flags solvent companies (WEX, SM Energy, Opendoor) on boilerplate; filer-supplied item codes are miscoded — J.C. Penney's earliest Item 1.03 is a **2014 shareholder rights plan**, six years before its bankruptcy.
- **A truthful filing can still be about someone else.** FirstEnergy Corp ($42bn, healthy) filed an Item 1.03 describing its *subsidiary's* Chapter 11. Genuine filers join themselves to the debtors with a conjunction ("the Company **and** certain of its subsidiaries…filed"); parents use an appositive ("each a subsidiary **of** X, filed"). That distinction is the fourth check.
- **Distress erases the tag you need.** Sleep Number's final pre-bankruptcy 10-K reports no long-term debt at all; its entire $588m is reclassified to `DebtCurrent`. Naive extraction leaves leverage undefined on exactly the companies the model exists to catch.
- **Missingness is not random.** Data goes missing *because* of distress, which makes it MNAR — so imputation would erase the signal, and absence is carried as an explicit feature instead.

### Deterministic baselines and calibration

The agent is graded against baselines on the same as-of data. A discrete-time hazard (Shumway 2001) treats survivors as **censored** rather than proven negatives. Panel: 5,582 firm-quarters, 354 filers, 500 positives, split **temporally** at 2024-06-01 (test: 1,217 rows, 100 positives, 8.2% base rate).

| Model | Test AUC | P@10 | P@25 | ECE |
|---|---|---|---|---|
| Tier 0 — Altman Z'' (unfitted) | 0.879 | 0.600 | 0.640 | 0.311 |
| **Tier 1 — discrete-time hazard** | **0.957** | **0.700** | **0.760** | 0.160 |
| *Leak canary — labels shuffled* | *0.488* | *0.300* | *0.200* | — |

**The leak canary matters most**: permuting labels collapses the model to chance, so the signal is in the data, not the pipeline (`--shuffle-labels`). Isotonic recalibration fitted on a separate temporal fold cuts ECE from 0.165 to **0.025** with ranking intact (AUC 0.939). Probabilities do not transfer to the population -- the universe is enriched far above the ~0.5% true rate (Zmijewski 1984) -- rankings do. Full caveats in [`SPEC.md`](SPEC.md) §7.1.

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

**Orchestration: a LangGraph state machine around a hand-rolled ReAct loop.**

The pipeline is declared as a graph, so the stages, the state each carries, and
the branch at the guard gate are explicit rather than implied by call order:

```
                 ┌──────────────┐
   CIK + date ──►│  load_facts  │  EDGAR fetch, then the as-of filter. Nothing
                 └──────┬───────┘  after this point can see a later filing.
              error ────┴──── ok
                ↓            ↓
              END      ┌──────────┐
                       │  triage  │  How deep to look; sets the step budget.
                       └────┬─────┘
                            ↓
                  ┌──────────────────┐
                  │   investigate    │ ◄── the ReAct loop, ONE node
                  │                  │     tool → observe → choose → repeat,
                  └────────┬─────────┘     critic-checked, ≤2 bounded retries
                           ↓
                     ┌──────────┐
                     │   rank   │  GBM over the same facts. Never gates.
                     └────┬─────┘
                          ↓
                     ┌──────────┐
                     │  guard   │  numeric · lookahead · scope · calibration
                     └────┬─────┘
                may_ship ─┴─ blocked
                     ↓         ↓
               ┌──────────┐   END  ← an outcome, not an error path
               │ assemble │
               └────┬─────┘
                    ▼   cited memo + ranked percentile + audit trail
```

Three things the graph buys that the previous function did not. **Checkpointing:**
a cold investigation is ~130s of sequential model calls, and with a thread id it
resumes at the node it stopped on instead of paying that again. **A declared
branch:** a blocked memo was an early return; it is now an edge that says the
guard fired, which is a different statement from an error. **Streaming:**
`graph.stream()` emits one update per node, which the SSE endpoint had been
assembling by hand from a callback.

The API serves through this graph. `serve.py` builds the investigator, hands
it to `agents.graph.run`, and the facts it already loaded are passed in rather
than refetched -- a second EDGAR round trip against a rate-limited endpoint
buys nothing. The filing-text tools and the context notes travel as graph
state, because losing them silently would cost the +0.084 AUC those tools buy
while the assessment still looked fine.

**The ReAct loop stays inside a single node, deliberately.** The step budget, the
forced abstention and the bounded critic retry are the behaviours the 0.963
backtest measured across 200 cases. Re-expressing them as graph nodes would be a
rewrite of the thing under measurement, and the figure would stop describing what
ships. `evals/test_l3_graph.py` fails if a later refactor crosses that line.

**Underlying pattern: orchestrator–workers.** A coordinator routes to specialized worker agents and fuses their outputs; workers do not talk to each other (orchestrated, not collaborative — chosen for auditability). Each worker is a **ReAct** tool-use loop. The whole thing is wrapped in a **deterministic critic–reviser** verification stage.

```
watchlist & trigger  →  data plane  →  deterministic compute  →  orchestrator
                                                                      │
              ┌───────────────────┬──────────────────┬────────────────┤
              ▼                   ▼                  ▼                ▼
          distress        earnings quality       covenant     filing-text RAG
       (Ch. 11 labels)     (AAER labels)       (demo-grade)   (feeds distress;
                                                                 never a figure)
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
  - *Filing-text RAG* — retrieves what the filings *say* (going-concern doubt, liquidity plans, control weaknesses, debt terms) for the distress investigator. **Never a source of figures**: numbers come only from XBRL through the compute layer. Backtested in the single-pass arm; not yet used by the served app or its chat.
- **Verify & fuse** — recomputes every cited figure, fuses signals into one graded assessment, escalates to *insufficient evidence* rather than guessing.
- **Output** — an evidence-cited, confidence-scored memo with a full audit trail, for human review.

Model serving via vLLM; a **separate** judge model scores memo quality (never the generating model).

### Filing-text retrieval (RAG)

Used by the single-pass investigator, which replaces the ReAct loop's ~10 tool-choosing calls with one call over a precomputed evidence pack plus retrieved passages. Backtested; the served app still runs the ReAct loop.

```
INGEST  (offline, resumable, memory-guarded)        QUERY  (per case: one filer, one prediction date)

EDGAR 10-K / 10-Q HTML                               5 standard credit questions
   │  html -> text; tables dropped                       │
   ▼                                                     ▼
section splitter                                  ┌───────────────────────────────────┐
   │  contents page skipped; Items 1A, 7, 7A,      │ FILTER FIRST, built by the store: │
   │  8, 9A ... labelled                           │   cik = filer                     │
   ▼                                               │   as_of - 18 months <= filed      │
recursive splitter: 320 tokens, 64 overlap         │   filed <= as_of                  │
   │  sanitised; injection-flagged                 └─────────────────┬─────────────────┘
   ▼                                                                 ▼
embed: bge-small dense (384-d) + BM25 sparse       dense ──┐
   │  local ONNX, no torch                                 ├── RRF fusion -> dedupe by text hash
   ▼                                               sparse ─┘                │
Qdrant `filing_chunks`                                                      ▼
   payload: cik, form, filed, section, chunk_no    re-check each hit: filed > as_of -> LookaheadError
                                                                            ▼
                                                   P1..P10 + evidence pack (same ToolBox figures)
                                                                            ▼
                                                   ONE model call -> critic + verbatim-quote check
                                                                            ▼   (<=2 retries, then abstain)
                                                                       cited memo
```

| Choice | Why |
|---|---|
| **As-of filter built by the store** | The caller passes a date and cannot loosen it. A 10-K filed after the petition says "Chapter 11" plainly -- leakage in words is harder to spot than in a ratio, so every hit is re-checked and a future chunk fails the run loudly. |
| **Filter first, then rank** | One filer before one date is a few thousand chunks. With payload indexes Qdrant scans that set exactly, avoiding the recall loss approximate (HNSW) search suffers under strict filters. |
| **Hybrid dense + BM25** | Embeddings catch paraphrase; BM25 catches terms of art ("going concern", "forbearance"). Fused by reciprocal rank, so the two scores need no calibration against each other. |
| **Section-aware chunks** | A query can search only where a disclosure lives -- this lifted material-weakness Recall@5 from 0.67 to 0.88. |
| **Dedupe by text hash** | Risk factors are copied quarter to quarter; five copies of one paragraph are one result. |
| **Verbatim quotes only** | The critic verifies a number by recomputing it; a sentence can only be verified by finding it, so every quote must appear in the passage it cites. |

**Governed by design.** Guardrails are the enforcement pillar of a governance-aware design, not the whole of it: the cited audit trail (auditability), the human-in-the-loop memo (oversight), the online judge and drift alarms (monitoring), and the documented limitations (honesty) are the rest. Because the system is advisory and never makes a credit decision, it sits outside fair-lending and high-risk-decisioning regimes by design, while mapping cleanly onto model-risk-management and NIST AI RMF expectations. See [`SPEC.md`](SPEC.md) § 10.

---

## Evaluation (tooling per layer)

The eval harness is the point. Tooling is matched to what each layer actually is — not everything is an "LLM eval."

| Layer | What it tests | Kind | Tooling | Status |
|---|---|---|---|---|
| **L0** | Deterministic ratio/trend/peer math | Unit test | pytest | ✅ |
| **L1** | XBRL extraction accuracy | Unit test | pytest | ✅ |
| **L2** | Peer-group + trend correctness | Unit test | pytest | ✅ |
| **L3** | Investigator diagnosis vs real labels; tool loop vs single-pass A/B | **Outcome-based backtest** | **custom harness** (pytest-driven) | ✅ |
| **L4** | End-to-end (facts in → cited memo out), answer shape | Outcome + behavioral | custom harness (`test_l4_*`, `run_answer_quality`) | ✅ |
| **L5** | Guardrails / adversarial: decision framing, baited figures, citation provenance, context poisoning | Deterministic + behavioral | custom harness (13 test files, `run_hallucination_probe`, `run_figure_audit`) | ✅ |
| **RAG** | Retrieval: Recall@5, MRR@10, precision@5, nDCG@10, coverage; verbatim-quote check; answer relevance | Deterministic ground truth, no judge | custom (`evals/run_rag_recall.py`) | ✅ |
| Tracking | A/B arms and retrieval variants as experiments on shared datasets | Experiment tracking | Langfuse | ✅ |
| Online | Live drift, groundedness, calibration | LLM-as-judge + tracing | Langfuse | not built |

The headline is the **custom outcome-based backtest** (L3) with lookahead controls — that is the differentiator.

**Deterministic wherever possible.** Every check here has an exact answer — a figure either recomputes from XBRL or it does not, a quote either appears in its passage or it does not, a retrieved chunk either matches the warning's ground truth or it does not. A judge is reserved for what nothing exact can grade, and is never the generating model. The one open gap, prose faithfulness, needs a validated fact-checker rather than a generic judge.

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
| [`agents/`](agents/) | Distress investigator (ReAct loop), single-pass investigator, typed tools, deterministic critic, model clients with shared rate pacing | built |
| [`rag/`](rag/) | Filing-text chunking, embeddings, Qdrant store with enforced as-of filter, resumable ingestion | built |
| `agents/` (rest) | Orchestrator + earnings-quality / covenant / context workers | Phase 4–5 |
| `serving/` | vLLM config + judge model | Phase 3 |
| `feedback/` | Input-quality gate + online judge + flywheel | Phase 5 |

See [`SPEC.md`](SPEC.md) for the full design.

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

# Filing-text RAG. Credentials go in .env (CREDITPULSE_QDRANT_URL / _API_KEY), never in code.
pip install -e ".[rag]"
python -m rag.ingest --all --store cloud                    # resumable; skips filings already stored
python -m evals.run_rag_recall --store cloud                # Recall@5, MRR, precision@5, nDCG@10
python -m evals.run_l3 --agent single --store cloud --max-negatives 100 --workers 4 --tpm 15000
python -m evals.compare_arms data/cache/l3_react_full.csv data/cache/l3_single.csv
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
6. ✅ **Filing-text RAG + single-pass investigator, A/B'd against the tool loop.** Tie on AUC at a tenth of the model calls; retrieval measured per warning type.

## Status

**Complete as an eval-first study.** One backtested leg at parity with a strong statistical baseline -- in two designs, a tool loop and a single-pass RAG investigator, tied with each other -- one leg measured and honestly declared unusable, the full L0–L5 ladder, and every number reproducible from the committed harness.

Known gaps, stated rather than hidden:

- **The feedback layer is not built** (SPEC §9) — no online judge, no drift monitoring. It was always the last item and it needs production traffic to mean anything.
- **The covenant-analysis leg was not built.** It would read credit-agreement exhibits and was scoped demo-grade, never gating the assessment. (Separate from covenant-trouble *retrieval*, below.)
- **Absolute probabilities do not transfer.** The universe is enriched with distressed filers far above the population base rate (Zmijewski 1984); ranking transfers, calibrated probabilities do not.
- **The agent has never chosen to abstain.** All abstentions across 200 cases were protocol failures. "It will tell you when it does not know" is an unexercised claim.
- **Covenant-trouble retrieval is weak** (Recall@5 ~0.4). An open extension: a deterministic covenant scan is the planned fix.
- **Prose faithfulness is not reported.** A small NLI checker mislabelled correct, figure-heavy claims, so it was disabled. Figures are verified by the critic and quotes by exact match; free-text claims need a validated checker first.
- **One model, one window.** Validated on `gemma-4-31b-it` over 2024–2025. Any swap needs re-measuring — which the harness makes cheap, and which is the point of having built it first.

## License

Copyright (c) 2026 Sai Charan. Licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE).

You may read, run, modify and share this code for noncommercial purposes -- personal
study, research, education, evaluation. Selling it, or using it in a commercial product
or service, is not permitted without a separate written license from the author.
