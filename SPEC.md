# CreditPulse IQ — build spec

This is the design document. It exists so the build stays eval-first and honestly scoped, and so every engineering tradeoff has a stated reason (useful for the repo and for interviews).

Guiding principle: **the eval harness is the deliverable; the agent slots into it.** Build the thing that proves it works before the thing that does the work.

---

## 1. Scope

**In scope.** Monitoring public U.S. companies for credit distress from SEC filings, producing an evidence-cited, confidence-scored risk memo for a human. Two backtested reasoning legs (distress, earnings-quality) plus one demo-grade leg (covenant).

**Out of scope, deliberately.** Auto-decisioning of any kind; loan origination; trade signals; private/middle-market names (no data without a partner); synthetic data; fine-tuning the agent's reasoning loop.

**Success metric.** Not revenue, not a demo — a defensible backtest: precision/recall, lead time, calibration, and false-confidence rate on real outcomes, with lookahead controls that survive scrutiny.

---

## 2. Data plane

**Input.** SEC EDGAR — XBRL financial statements (structured), the text of 10-K/10-Q filings (for retrieval, §5.1) and filing exhibits (credit agreements, for the covenant leg). Bulk download for backfill; REST API for incremental.

**Universe.** Start small (a few hundred names) spanning failed and surviving companies so the backtest has positives and negatives.

**Labels.**
- *Distress:* Chapter 11 filings and exchange delistings, each with a **filing/effective date**.
- *Earnings quality:* AAER misstatement events, each with the **period the misstatement covered** and the **enforcement release date**.

**As-of dating is mandatory and enforced in code.** Every record carries the date it became public. The backtest may only feed the agent data whose as-of date precedes the prediction date. This is the single most important correctness control in the project — lookahead leakage invalidates every result.

### 2.1 Identity is keyed to CIK, never ticker

Established during Phase 1 and non-negotiable thereafter. Tickers are reused and company names are overwritten:

- SEC's `company_tickers.json` contains **only currently-listed** filers. Every bankrupt name in our universe is absent from it, so any universe built from that file is survivorship-biased by construction.
- `BBBY` today resolves to CIK 1130713 — Overstock, which bought the brand out of bankruptcy. The company that actually filed Chapter 11 is CIK 886158, now named "20230930-DK-Butterfly-1, Inc."
- J.C. Penney is "Old COPPER Company, Inc."; Rite Aid is "NEW RITE AID, LLC"; Chesapeake Energy is "EXPAND ENERGY Corp".

The watchlist therefore stores CIK and cohort only. Name, ticker, SIC and size are point-in-time attributes resolved from EDGAR at load time.

### 2.2 The 1,000-filing cap

EDGAR's submissions API returns only the most recent 1,000 filings inline; older filings live in overflow files under `filings.files`. **37 of our 50 names exceed the cap** (PACCAR alone has 4,493 filings). Any code reading `filings.recent` alone silently loses history on three-quarters of the universe. `EdgarClient.filing_index` walks the overflow.

### 2.3 Label discovery: four signals, because no single one is trustworthy

`data/discover.py` sweeps EDGAR for Chapter 11 candidates and requires all four:

1. **Full-text search** over 8-K filings — high recall, poor precision.
2. **Structured item code `1.03`** ("Bankruptcy or Receivership") from submissions metadata — filer-supplied and therefore miscoded.
3. **Filing text** describing a Chapter 11 petition — excluding Chapter 7 liquidations.
4. **Registrant is a debtor** — excluding parents reporting a *subsidiary's* case.

Each check catches errors the others miss. Over the full 2018–2026 sweep: **1,136 candidates → 268 confirmed → 156 promoted.**

Concrete rejections:

- **Text alone:** WEX, Howard Hughes, SM Energy, Kennedy-Wilson, Amneal and Opendoor match the boilerplate while solvent — 31 of 89 size-eligible 2025–26 candidates.
- **Item code alone:** J.C. Penney's earliest `1.03` is a **2014 shareholder rights plan**, six years before its bankruptcy; Granite Construction's 2026 `1.03` contains no Chapter 11 language at all. Deriving `event_date` from item codes would have injected a six-year lead-time error into the headline metric.
- **Wrong chapter:** Canoo and Sonder filed Chapter 7.
- **Wrong company:** FirstEnergy Corp ($42bn, healthy) filed an item 1.03 describing FirstEnergy Solutions' case. The filing is entirely truthful, just not about the filer.

**Check 4 had to be textual, not behavioural.** A first attempt inferred it from post-petition filing behaviour ("did it keep reporting normally?") and was wrong: it flagged Diebold Nixdorf, Core Scientific, CBL and Ferrellgas — all genuine filers that reorganised and resumed reporting — which would have systematically deleted the *successful reorganisations* from the positive class. The correct discriminator is grammatical: genuine filers join themselves to the debtors with a conjunction ("the Company **and** certain of its subsidiaries…filed"), while parents use an appositive ("each a subsidiary **of** X, filed").

Affiliates co-filing one case (iHeartMedia filed as three registrants) are collapsed, so one credit event contributes one positive.

Discovery emits **candidates for review**, never labels. Promotion (`data/promote_labels.py`) is a separate, scripted step.

### 2.4 The severity ladder

Chapter 11 is the cleanest label and the rarest. `data/distress_events.py` extracts the intermediate events that precede it, from structured item codes and form types, giving **3,005 dated events** over the universe:

| Tier | Signals | Events | Filers |
|---|---|---|---|
| `default` | Chapter 11 petition (from the verified label set) | 156 | 156 |
| `near_default` | Debt acceleration (8-K 2.04), delisting (Form 25/25-NSE) | 863 | 252 |
| `stress` | Listing-rule failure (3.01), restatement (4.02) | 776 | 186 |
| `early_warning` | Late filing (NT 10-K/Q), auditor change (4.01), impairment (2.06) | 1,210 | 246 |

**Tiers are scored separately and never pooled.** A late filing is noisy and routine for some filers; a Chapter 11 is unambiguous. Pooling would inflate the headline number. Signals common at healthy companies (auditor changes, restructuring charges) are flagged `noisy` so a consumer can require corroboration.

Note that the terminal tier is *not* derived from item code 1.03 — that is precisely the code filers miscode. It comes from the four-signal label set.

### 2.5 Post-bankruptcy outcomes

`data/outcomes.py` records whether each filer survived, inferred from post-petition filing behaviour. **129 of 156 (83%) did not**: 92 deregistered (Form 15), 37 went dark, 21 emerged, 6 in process.

This does not change the prediction target — the event predicted is the petition, which is the same whether the filer later emerges or liquidates (probability of default, not loss given default). It is recorded because it is cheap, CIK-native, and lets results be sliced by severity.

---

## 3. Deterministic compute (no LLM)

Pure functions over raw filing values: leverage, coverage, liquidity, margin trends, accrual measures, peer-relative percentiles. Everything numeric downstream comes from here or is rejected.

- Inputs are raw XBRL line items (with MDRM-style tags preserved for provenance).
- Every output carries provenance: which line items, which periods, which formula.
- Peer groups are constructed deterministically (size/sector) and pinned to as-of vintages.

This layer is built and tested (L0) **before any agent exists**, and gates in CI from day one.

### 3.1 Design rules established in Phase 2

**Formulas take primitive filing values only, never other computed ratios.** Chaining derived values would mean auditing one number required auditing a tree of intermediates. Flat formulas mean every figure traces in one hop to tagged line items in a specific filing, and the verifier can re-execute it against those raw values alone.

**Undefined rather than wrong.** Division by zero returns `None`, never `inf` and never `0`. Distressed companies routinely have zero equity or zero interest expense; `inf` poisons comparisons silently and `0` reads as "no leverage" — the direction that produces false confidence.

**A computed value's `as_of` is the latest filing date among its inputs.** A ratio built from a balance sheet filed in March and earnings filed in May is not knowable until May.

**Absence is assumed zero only where genuinely implied, and always recorded.** `long_term_debt` is in that set because distress itself removes the tag: once maturities accelerate, debt is reclassified as current. Sleep Number's final pre-bankruptcy 10-K tags no long-term debt at all — its entire $588m sits in `DebtCurrent`. Treating that as missing would leave leverage undefined on exactly the companies the distress leg exists to catch. A sibling guard prevents assuming *all* debt is zero when no debt tag resolves.

**Altman Z''**, not Z. The original needs market value of equity, which is not in the filings and would need a price feed with its own as-of problems. Z'' substitutes book equity over total liabilities, so the whole score comes from one filing.

---

## 4. Orchestrator

Reads the deterministic signals and decides which investigators to run and how deep. It does **not** compute anything and does **not** produce the final numbers.

**Routing policy (sketch).**
- Compute a cheap triage score from deterministic signals.
- Clean profile → shallow pass, distress investigator only, low depth.
- Elevated leverage / deteriorating coverage / liquidity stress → deep distress loop + earnings-quality loop.
- Accrual anomalies or restatement flags → earnings-quality loop prioritized.
- Covenant leg runs only when a credit-agreement exhibit is present (demo-grade, never gates the overall assessment).

The routing depends on intermediate findings — that branching is what makes this genuinely agentic rather than a fixed pipeline.

---

## 5. Investigators (tool-contracts)

Each investigator is an agentic loop: hypothesize → call a tool → observe → decide next step → loop on residual → terminate (including at *insufficient evidence*). All tools are typed; the agent gets tools, never the raw database.

Shared tools:

| Tool | Signature | Returns |
|---|---|---|
| `get_line_item` | `(cik, tag, period)` | raw XBRL value + provenance |
| `get_trend` | `(cik, metric, n_periods)` | time series (from compute layer) |
| `get_peer_comparison` | `(cik, metric, period)` | value vs peer distribution |
| `recompute` | `(formula, inputs)` | deterministic result (the agent's only arithmetic) |
| `check_threshold` | `(metric, value)` | pass/flag against reference thresholds |

**Distress investigator** — financial-deterioration loop (leverage, coverage, liquidity, margin). Backtested against Chapter 11. This is the first and reference implementation.

**Earnings-quality investigator** — accrual anomalies, revenue-recognition red flags, restatement/related-party threads. Backtested against AAER. Reuses the shared harness.

**Covenant investigator** — extracts covenant terms from the credit-agreement exhibit, computes current headroom against the actual financials. **Demo-grade:** no large public labeled covenant-extraction set and only proxy breach labels (waiver 8-Ks, going-concern language), so it is presented as extraction + headroom, not a validated breach predictor.

**Single-pass investigator** — the distress investigator's alternative design. Every tool the ReAct loop chooses between is deterministic, so code runs them all up front (`build_pack`: metrics with thresholds, trends, peers, prior events, filing events, going-concern scan) and adds passages retrieved from the filings (§5.1). One model call, then the **unchanged** critic and bounded retry. One check is added: every quoted passage must appear verbatim in the passage it cites. It lives beside the critic, not in it, so the critic stays byte-for-byte what earlier backtests measured.

**Output schema (per investigator).** A typed object: signal, confidence, evidence (cited line items + recomputed figures), and a residual/unexplained field that can trigger escalation. The single-pass investigator adds cited passage ids.

### 5.1 Filing-text retrieval

Retrieval supplies what the filings *say* — going-concern doubt, liquidity plans, control weaknesses, debt terms — never a figure. Numbers come only from XBRL through §3.

**Ingestion (`rag/ingest.py`).** 10-K/10-Q HTML → text with tables dropped → section splitter (detects and skips the contents page; labels Items 1A, 7, 7A, 8, 9A and the rest) → recursive splitter, 320 tokens with 64 overlap → sanitised, injection-flagged chunks. Resumable (skips stored accessions) and memory-guarded. Current store: 875 filings, 124 filers, ~139k chunks.

**Index.** Qdrant collection `filing_chunks`: `bge-small-en-v1.5` dense vectors (384-d, local ONNX) and BM25 sparse vectors, with payload indexes on `cik`, `filed_ord`, `section`, `form`, `accession`. Local mode for development, cloud for the full store; credentials only in `.env`.

**Query (`FilingStore.search`).**
- *The as-of filter is built by the store, not the caller.* `search` takes the prediction date as a required argument and constructs `cik = filer AND as_of − 18 months ≤ filed ≤ as_of` itself. Every returned hit is checked again; a future chunk raises `LookaheadError`. A filing made after the petition states the outcome in words, which is harder to spot than a leaked ratio, so this fails loudly rather than producing a quietly excellent backtest.
- *Filter first, then rank.* One filer before one date is a few thousand chunks; with payload indexes Qdrant scans that set exactly instead of using approximate (HNSW) search, which loses recall under strict filters.
- *Hybrid.* Dense and BM25 candidates (40 each) fused by reciprocal rank, so their scores need no calibration against each other. Duplicate text (risk factors copied quarter to quarter) is collapsed by hash.
- *Section filters* are optional per query; the material-weakness queries search the controls item and risk factors only.

**Use.** The single-pass investigator asks five standard credit questions (three passages each, ten at most) and passes them as `P1..Pn`, fenced as untrusted company text that is data, never instructions.

---

## 6. Verify & fuse

- **Recompute-check:** the verifier independently recomputes every figure the agent cites. Disagreement → downgrade confidence and flag; unreproducible figure → hard fail (see guardrails).
- **Fusion:** combine investigator signals into one graded risk assessment with an overall confidence.
- **Escalation:** if residual/unexplained signal is high, terminate at *insufficient evidence* rather than forcing a rating.

Output: a cited, confidence-scored memo + full audit trail (every tool call, every recomputed number, every decision).

---

## 7. Eval ladder (L0–L5)

The core of the project. Layers gate progressively; lower layers run in CI.

| Layer | Tests | LLM? |
|---|---|---|
| **L0** | Deterministic ratio/trend/peer math vs known values | No |
| **L1** | XBRL extraction accuracy (raw line items) | No |
| **L2** | Peer-group construction + trend correctness | No |
| **L3** | Investigator diagnosis vs real labels (Ch. 11, AAER) | Yes |
| **L4** | End-to-end (name → memo), cost/latency | Yes |
| **L5** | Guardrails / adversarial | Yes |

**L3 is the differentiator. Metrics:**
- Precision / recall against real outcomes.
- **Lead time** — how many quarters before the event the risk was flagged.
- **False-confidence rate** — high-confidence "healthy" on a name that later failed. Tracked as a first-class, catastrophic-error metric.
- **Calibration** — does stated confidence match empirical outcome frequency (ECE + reliability curve)?

**Backtest protocol.**
- Strict as-of cutoffs enforced in code (§2).
- Class imbalance is real: frame as early-warning / watch-list, and enrich positives with intermediate distress signals (delistings, going-concern opinions, enforcement actions) rather than only terminal events.
- **Agent-trajectory eval**, not just final-answer: did it call the right tools, in a sensible order, recompute correctly, and escalate when it should?

### 7.1 Deterministic baselines (built)

An L3 precision figure is uninterpretable without a baseline. If the ReAct investigator cannot beat a 1980-vintage logit on the same as-of data, that is a finding to report, not to hide.

| Tier | Model | Status |
|---|---|---|
| 0 | Altman Z'', Ohlson O — unfitted reference | built (`compute/`) |
| 1 | Discrete-time hazard, Shumway (2001) | built (`models/`) |
| 2 | GBM / RUSBoost challenger | documented, not built |

`models/panel.py` builds the firm-period panel: 5,582 observations over 354 filers, quarterly 2021–2025, 1-year horizon, 500 positives. Three controls are load-bearing — as-of features (`as_of_view` per row), post-event rows dropped (predicting an event that already happened is not prediction), and missingness carried as an explicit feature rather than imputed.

Results on a **temporal** hold-out at 2024-06-01 (test: 1,217 rows, 100 positives, 8.2% base rate):

| Model | Test AUC | P@10 | P@25 | ECE |
|---|---|---|---|---|
| Altman Z'' | 0.879 | 0.600 | 0.640 | 0.311 |
| Hazard | 0.957 | 0.700 | 0.760 | 0.160 |
| Labels shuffled (canary) | 0.488 | 0.300 | 0.200 | — |

**Known weaknesses, stated:**

- *Calibration.* ECE 0.16 is poor. `class_weight="balanced"` pushes probabilities off the base rate, and the base rate itself moves by year (2.9% → 15.0% → 6.6%). Fixing this needs recalibration on a held-out fold, not tuning.
- *Collinearity.* `current_ratio` and `quick_ratio` receive large offsetting coefficients; individual effects are not interpretable.
- *Test > train AUC* (0.957 vs 0.897) is within ~2 standard errors at 100 positives and is not claimed as an improvement.
- *Sector confound.* Survivors lack a tagged `Liabilities` far more often than distressed filers (46% vs 19%), so missingness indicators risk proxying sector. An ablation bounds this: metrics alone reach 0.937, missingness alone only 0.740 with P@25 of 0.20. The signal is predominantly financial, but sector controls belong in Phase 3.

**Leak canary.** `--shuffle-labels` permutes labels and refits. A sound pipeline collapses to chance; this one does (0.488). This is the cheapest defence against the cardinal sin and should be run whenever the panel changes.

**Statistical power is a gate, not an afterthought.** 25 positives gives a recall estimate with a 95% CI of roughly ±20pp, and leaves ECE reliability bins too sparse to read. The universe must reach ~100 positives and a few hundred survivors before L3 numbers are reported as results rather than as a smoke test. Discovery already surfaced 58 confirmed positives in 2025–26 alone, so this is a scaling step, not a research problem.

### 7.2 Single-pass vs tool loop, and retrieval (built)

A/B on 199 stratified cases (99 bankruptcies), identical cases, grading and critic:

| | ReAct tool loop | Single-pass + retrieval |
|---|---|---|
| AUC (abstentions scored 0.5) | 0.974 | 0.964 |
| Paired-bootstrap difference | | −0.009, 95% CI [−0.029, +0.009] — tie |
| Bankruptcies called healthy | 8 / 99 | 0 / 99 |
| Base-rate-corrected precision | 0.46 | 0.32 |
| Model calls per case | ~10 | ~1 |

Retrieval is scored without a judge. Ground truth for each (filer, as-of, warning) is the set of stored chunks matching the deterministic warning patterns; queries are plain language, so lexical overlap alone cannot carry the result. Metrics: Recall@5, MRR@10, precision@5, nDCG@10, and **coverage** reported separately (a coverage miss is an ingestion fault no ranking can fix). Going concern: Recall@5 0.93, nDCG 0.75. Material weakness with section-filtered queries: 0.88 / 0.67. Covenant trouble: ~0.4 / ~0.25 — see §13. Results are uploaded to Langfuse as experiments on shared datasets so variants line up case by case.

**Infrastructure findings.** Two backtests pacing separately against one provider quota overshoot it, so pacing is shared through a file ledger across threads and processes. A rate-limit retry window of ~1 minute recorded 55 refused calls as agent abstentions; rate limits and 503s now retry for ~10 minutes, and an infrastructure failure is never graded as a judgment.

**Online eval (production).** An LLM-as-judge (separate model) scores live memos on groundedness, calibration, and scope where the outcome is not yet known — this watches quality *drift* over time, complementing the offline backtest.

---

## 8. Guardrails (L5)

Enforced at the verify step, before any memo ships:

- **Numeric-verification guard** — block any response containing a figure the verifier cannot reproduce. Hard fail.
- **Calibration guard** — cap stated confidence when residual is high; *insufficient evidence* is a valid terminal state.
- **Scope guard** — output is "elevated risk," never "sell this position" or "deny this loan." Refuse/redirect decision framing.
- **Data-freshness guard** — surface missing/stale filings rather than silently reasoning on incomplete data.

---

## 9. Feedback layer

Two gates plus the flywheel. Note: mostly deterministic plumbing + one judge — **not** a new agent.

**Input-quality gate** (between data plane and compute): completeness, schema/type validation, staleness, malformed XBRL, peer-group integrity, extraction-confidence floors. Bad data is quarantined and flagged before it reaches the investigators. Deterministic; reserve a thin LLM slice only for genuinely ambiguous reads (e.g. does a restatement change how a number should be interpreted).

**Online judge** (on output): see §7 online eval.

**The flywheel** (the part worth getting excited about):
- *Human review as ground truth:* log every accept/edit/override; these become new eval examples and classifier training data. Prioritize labeling the cases where the agent was least confident or the judge flagged them (active learning).
- *Outcome accrual:* as real outcomes land (a monitored name files Ch. 11, an AAER drops), auto-score the historical prediction and update the L3 dashboards, with drift alarms.
- *Data-quality signals:* a rising extraction-failure rate on a filer/form type flags the parser, not the model.

---

## 10. Governance posture

Guardrails are the *enforcement* pillar of governance, not the whole of it. Governance is the wider system — accountability, auditability, human oversight, monitoring, and documented limitations — that decides what the guardrails should be and demonstrates they work. CreditPulse IQ is **governance-aware by design**: several of its core features already are the governance surface, they just weren't labeled as such. This section names them and states which regimes the system does and does not fall under. It is not a compliance-theater add-on and does not expand scope.

**The five pillars, mapped to what already exists:**

| Governance pillar | Where it lives in CreditPulse IQ |
|---|---|
| Enforcement | Guardrail suite — numeric-verification, calibration, scope, freshness (§8) |
| Auditability | Cited memo + full audit trail: every tool call, recomputed figure, and decision (§6) |
| Human oversight | Human-in-the-loop by design; no auto-decisioning; memo is advisory (§1, §8 scope guard) |
| Monitoring | Online judge for drift + L3 dashboards with drift alarms (§7, §9) |
| Documented limitations | Honest-scoping written into the spec — covenant demo-grade, label incompleteness, class imbalance (§13) |

**Regimes it does *not* fall under (by design).** Because the system is monitoring/advisory and never makes or materially drives a credit decision, it sits outside the heaviest regimes: fair-lending law (ECOA/Reg B adverse-action, disparate impact) applies to credit *decisions*, not analytical monitoring; and it is not a high-risk automated decision system under frameworks like the EU AI Act. The scope guard (§8) is what keeps it on this side of the line — the moment it started outputting decisions, those regimes would attach.

**Regimes it *is* informed by.** Any model a financial institution relies on carries model-risk-management expectations (independent validation, documentation, and ongoing monitoring — the spirit of SR 11-7) and maps cleanly onto the NIST AI Risk Management Framework's govern/map/measure/manage functions. This is exactly why the backtest validation, the audit trail, and the documented limitations are load-bearing rather than decorative: they are the artifacts a model-governance review asks for.

**Accountability (stated, not enforced by code).** The memo is a decision *input* for a named human reviewer who remains accountable for any action taken; the system does not act. Model changes should pass the L0–L5 ladder before shipping (validation gate), and the false-confidence rate is the designated kill-signal metric — a sustained regression is grounds to pull the system, not tune around it.

**Deliberate non-goal.** This does not make CreditPulse IQ an "AI governance framework," and the project is not renamed or repositioned as one. The honest claim is narrower and stronger: a monitoring agent built as a *governed asset* — enforced, audited, overseen, monitored, and honest about its limits — rather than unaccountable code.

---

## 11. Serving

- **Agent model:** a strong open instruct model served via vLLM (OpenAI-compatible endpoint, continuous batching for the backtest sweeps, guided decoding for the typed output schemas). Not fine-tuned.
- **Judge model:** a separate model (free-tier hosted is fine) — never grade with the generating model.
- **Rate limits are the binding constraint on free tiers**, not capability. Calls are paced proactively under the provider's tokens-per-minute ceiling (shared across processes), responses are cached by model and conversation so reruns cost nothing, and a stall watchdog restarts a silent run from the cache.
- **Optional later:** LoRA/QLoRA fine-tune of the *narrow classifiers* (earnings-quality, distress) where AAER and bankruptcy labels are abundant — runs on free Colab/Kaggle GPU. Not the agent.

---

## 12. Build sequence (eval-first)

1. **Ground truth + scaffold.** Repo, EDGAR data plane for a small universe, both label sets with as-of dating baked in from line one. ✅
2. **Deterministic spine + L0.** Compute layer as pure functions; L0 conformance evals gating in CI. No LLM yet. ✅
3. **Distress investigator, fully backtested.** One investigator end-to-end (tools, vLLM loop, verification, schema) + its L3 backtest with lookahead controls, lead-time, and false-confidence rate. Do not proceed until it produces a real calibration curve. ✅
4. **Earnings-quality investigator.** Second leg against AAER, reusing the harness the first leg forced you to build. ✅ measured and failed; demo-grade.
5. **Assemble + scope honestly.** Orchestrator; guardrails; covenant attached as labeled demo; cited memo. ✅ scoped to one green leg; feedback layer not built.
6. **Filing-text retrieval + single-pass investigator.** Ingestion, store-enforced as-of retrieval, retrieval evals, and an A/B against the tool loop. ✅ tie at a tenth of the model calls; not yet wired into serving.

Each phase is shippable and demonstrable on its own. The README leads with the L3 numbers, not the architecture.

---

## 13. Known risks / honest caveats

- **Covenant is demo-grade** — stated, not hidden.
- **Class imbalance** — terminal events are rare; framed as early-warning, positives enriched with intermediate distress signals.
- **Label incompleteness (AAER)** — only *caught* misstatements are labeled; reported as "elevated misstatement risk," not a fraud oracle.
- **Lookahead leakage** — the cardinal sin; enforced in code via as-of cutoffs, verified in the backtest harness.
- **Public-company coverage** — the underserved private/middle-market segment is intentionally out of scope; not faked with synthetic data.
- **Universe size (current)** — 25 positives is enough to build and debug the harness, not enough to report L3 numbers. See §7.
- **Survivors are censored, not proven negatives** — a name with no event on record may simply not have failed *yet*.
- **Covenant-trouble retrieval is weak** (Recall@5 ~0.4). Embeddings rank "we were in compliance with our covenants" beside "we were not in compliance", and two query rewrites did not fix it; a graded key (stated breach 2, covenant stress 1) measures it fairly. Planned fix: a deterministic covenant scan in the evidence pack, as the going-concern scan already is, then a cross-encoder re-ranker for covenant queries. It does not limit the headline — single-pass missed no bankruptcy without it.
- **Prose faithfulness is not measured.** A small NLI model labelled correct, figure-heavy claims as contradicted, so it measured itself, not the agent, and was disabled. Figures are verified by the critic and quotes by exact match; free-text claims need a checker validated against hand-labelled claims first.
- **Retrieval is backtested, not served.** The app and its chat modes still run the ReAct investigator; the single-pass investigator is not yet wired into the graph.

### 13.1 Measured metric coverage

Filers do not tag uniformly, so some metrics are simply unavailable. Measured over an 18-filer sample of the watchlist at each filer's latest fiscal year:

| Metric | Coverage | Cause |
|---|---|---|
| `interest_coverage` | **56%** | `interest_expense` untagged on 39% of filers |
| `gross_margin` | 61% | `cost_of_revenue` untagged on 33% |
| `altman_z_double_prime` | 72% | depends on `total_liabilities` |
| `liabilities_to_assets` | 83% | `Liabilities` untagged on 17% |
| most others | 83–100% | |

Two coverage gaps were left open on purpose, because closing them would trade a visible hole for an invisible error:

- **Interest expense.** The common alternatives on filers that skip the standard tags are *net* interest (`InterestIncomeExpenseNonoperatingNet`, `InterestIncomeExpenseNet`) or the cash-flow item `InterestPaidNet`. Net tags fold interest income in and flip sign between filers. A wrongly-signed coverage ratio reads as *healthy* — the one direction that must never happen silently. Only the unambiguous `InterestExpenseOther` was added.
- **Total liabilities.** Deriving it as assets minus equity is exact only when noncontrolling interests are zero, and NCI is present across much of this universe.

Interest coverage is the most diagnostic distress ratio, so 56% is a real constraint on the distress leg and the investigator must be able to reason without it — that is part of why *insufficient evidence* is a first-class terminal state (§6). Raising coverage safely (sign-aware handling of net interest tags) is tracked work, not a silent default.
