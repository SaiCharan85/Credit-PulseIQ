# CreditPulse IQ — build brief

Paste this as the opening message in the repo root. It is written to enforce the *eval-first* build order and the design's non-negotiables, so the build goes deep on one proven leg before going wide.

---

## Opening prompt (paste this)

> You are helping me build **CreditPulse IQ**, a corporate credit-distress monitoring agent. Read `README.md` and `SPEC.md` in full before writing any code — they are the source of truth, and this brief does not repeat their detail.
>
> **The single most important rule: this is an eval-first project. The evaluation harness is the deliverable; the agent slots into it. Build depth on one backtested investigator before any breadth. Do not scaffold all four workers at once.**
>
> Work through the build sequence in `SPEC.md` §12, one phase at a time. After each phase, stop, run the tests, and summarize what's proven before proposing the next phase. Do not jump ahead.
>
> Phases 1–2 are done: the EDGAR data plane, the deterministic spine, and L0–L2. Start at **Phase 3**: build the **distress investigator as a genuine ReAct loop** (SPEC §5) — it must plan, call typed tools, observe results, decide the next call based on findings, and terminate on its own judgment (including at *insufficient evidence*). Then build its **L3 backtest** (SPEC §7) with strict as-of cutoffs, precision/recall, lead time, and false-confidence rate.

---

## Hard rules (repeat these; do not let them drift)

1. **The LLM never does arithmetic.** Every number comes from `compute/` (deterministic) and is re-checked by `verify/recompute.py`. If an agent needs a figure, it calls a tool; it does not compute it in the prompt.
2. **Workers must be real ReAct loops, not single prompted calls.** A worker implemented as one big "analyze this" LLM call is a bug — it defeats the entire agentic design. The loop selects tools dynamically based on prior observations.
3. **Agency is confined to the workers and the orchestrator.** `compute/`, `verify/`, the input gate, and label handling are plain deterministic code. Do not turn them into agents.
4. **Lookahead leakage is the cardinal sin.** The backtest may only feed data whose `as_of_date` precedes the prediction date. Enforce it in code (`data/facts.py::visible_as_of`, `as_of_view`) and assert it in the backtest harness.
5. **Never loop the agent against the LLM judge.** Within-request retries are triggered only by deterministic checks (unreproducible figure, scope, staleness), are bounded (≤2), and terminate in abstention. Soft judge flags route to a human, not a retry.
6. **Honest scoping is enforced in output.** The covenant leg is demo-grade; the context/retrieval leg is context-only and must never move the graded numeric signal. Label them as such in the memo schema.
7. **Separate judge model.** Never grade an output with the model that generated it.
8. **Build order is fixed.** No earnings-quality, covenant, or context worker until the distress investigator produces a real calibration curve on the backtest.

## Rules earned during Phases 1–2 (do not relearn these the hard way)

9. **Identity is CIK, never ticker.** Tickers are reused after bankruptcy and SEC's ticker map omits delisted filers entirely.
10. **No single source establishes a label.** Chapter 11 requires item code + filing text + own-petition subject. Discovery emits candidates; a human promotes them.
11. **Undefined beats wrong.** Division by zero returns `None`, never `inf` or `0`. Missing inputs produce a stated reason, not a substituted default.
12. **Assumed values must be visible in provenance.** Where absence genuinely implies zero, the synthetic tag `ABSENT:assumed_zero` records it.
13. **Read the overflow.** EDGAR's submissions API caps `recent` at 1,000 filings; 37 of our first 50 names exceeded it.
14. **A truthful filing can be about someone else.** Chapter 11 labels need a fourth check: the *registrant* must be among the debtors. Genuine filers use a conjunction ("the Company **and** its subsidiaries…filed"); parents use an appositive ("each a subsidiary **of** X, filed"). Never infer this from post-petition filing behaviour — that deletes the firms that reorganised successfully.
15. **Run the leak canary.** `python -m models.run_baseline --shuffle-labels` must collapse to AUC ~0.5. Run it whenever the panel changes; it is the cheapest defence against rule 4.
16. **Beat the baseline or say so.** Tier 0 (Altman Z'') and Tier 1 (discrete-time hazard) are built and scored. The investigator's L3 numbers are reported *against* them, not in isolation.
17. **Do not impute.** Missing data here is MNAR — it goes missing for reasons correlated with the outcome. Carry `<metric>__missing` indicators instead, and check by ablation whether they are proxying sector rather than distress.

---

## Tooling per layer (don't reach for the wrong one)

- L0–L2 (deterministic): **pytest**. These are unit tests, not "LLM evals." ✅ built
- L3–L4 (reasoning vs real labels): a **custom backtest harness** — this is the core; no library does it.
- L5 + online (groundedness, guardrails, drift): **DeepEval**; **RAGAS** only for the context/retrieval leg's groundedness; **Langfuse/Opik** for tracing and online monitoring.

## Stack

- Python ≥ 3.10, Pydantic v2 for all typed schemas (tool I/O, investigator output, memo).
- vLLM for the agent model (OpenAI-compatible endpoint); a separate hosted model as judge.
- Keep secrets in a gitignored `.env` / `.keys.json` — never commit keys. SEC requires a declaring User-Agent: `CREDITPULSE_SEC_UA`.

## Definition of done for the first milestone

A running distress investigator (real ReAct loop) that, over the watchlist with as-of cutoffs enforced, produces an L3 backtest report with precision/recall, median lead time, and false-confidence rate — and a calibration curve. Numbers go in the README headline table. That milestone, alone, is the portfolio-worthy artifact; everything else is expansion on top of it.

**Before reporting those numbers as results**, scale the universe to ~100 positives and a few hundred survivors (SPEC §7). At 25 positives the harness is provably working but the metrics are not yet reportable.
