"""Web front end for the assessment pipeline.

    python serve.py            then open http://127.0.0.1:8000

A thin layer over ``Orchestrator``. It adds no logic and no numbers of its own:
the JSON it returns is the same ``Memo`` the CLI renders, so the browser cannot
show anything the harness did not produce and verify.

Two behaviours are deliberate and worth not "fixing" later:

**A blocked memo renders as a block, not as an error page.** When the numeric
or scope guard fails, the UI shows what was withheld and why. That is the
system working, and hiding it behind a 500 would misrepresent a guard as a
crash.

**The rule-based control is the default.** It needs no model, no key and no
quota, so the app is usable the moment it starts. It is also a naive threshold
model that scores 0.885 AUC against the ReAct agent's 0.965, and the UI says so
next to the result -- a reader should not mistake the floor for the system.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path
from queue import Queue
from threading import Thread

from fastapi import FastAPI, UploadFile
from fastapi import File as _File
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from agents import tracing
from agents.diagnostic import ZONES
from agents.diagnostic import build as build_diagnostic
from agents.earnings_notes import earnings_notes
from agents.explain import explain, is_conceptual
from agents.graph import run as graph_run
from agents.llm import load_env_file
from agents.orchestrator import Assessment, Orchestrator
from agents.outcomes_lookup import load_outcomes, survived_note
from agents.schemas import SIGNAL_ORDER
from agents.screening import build as build_screen
from agents.screening import is_screening
from data.company_search import load_directory, resolve
from data.edgar import EdgarClient
from models import ranker

app = FastAPI(title="CreditPulse IQ", docs_url="/api/docs")
INDEX = Path(__file__).parent / "web" / "index.html"

#: Measured on the 200-case L3 backtest. Shown in the UI so the number a user
#: is looking at arrives with the evidence for how much to trust it.
ARM_AUC = {"rules": 0.885, "react": 0.965}
HAZARD_AUC = 0.966

#: Evidence density, reported without a reliability claim attached.
#:
#: This carried a warning that thinly-disclosed filers were ranked at 0.763
#: against 0.964 for the rest. That gap was an artifact: it came from ranking
#: the subgroup by ``risk_score`` while the published figure ranks by
#: ``risk_probability``. On the metric the product actually reports there is no
#: gap -- 0.976 on sparse disclosure against 0.964 -- and the warning was
#: teaching readers to discount the cases where the system is fine.
#:
#: The count stays, because how much a filer disclosed is worth knowing on its
#: own. The claim about reliability is gone until something measures one.
SPARSE_EVIDENCE_MAX = 2

#: Questions about what actually happened, answerable from the labels.
ASKS_OUTCOME = re.compile(
    r"\bdid (?:it|the |this )?\w* ?(?:survive|fail|go bankrupt|go under|make it)\b"
    r"|\bwhat happened\b|\bdid they (?:survive|fail|file)\b"
    r"|\bis (?:it|the company|this company) still (?:around|trading|alive|in business)\b"
    r"|\bwas (?:the |this )?(?:assessment|model|call|prediction) (?:right|correct|wrong)\b"
    r"|\bactual(?:ly)? (?:outcome|happen)"
    r"|\bdid (?:it|they|the company) file\b",
    re.I,
)

#: Portfolio cap. Each name is a real EDGAR fetch against a rate-limited API.
MAX_BATCH = 40

#: Module-level singleton so the dependency is not constructed in a default
#: argument (ruff B008).
UPLOAD = _File(...)


class AssessRequest(BaseModel):
    #: Either a CIK or a name/ticker. Nobody knows CIK 867773; they know
    #: "Diebold" or "DBD", and requiring the identifier made the tool
    #: unusable for the person it is for.
    cik: int | None = None
    query: str = ""
    as_of: date
    agent: str = "rules"


class BatchRequest(BaseModel):
    ciks: list[int] = []
    #: Free-text names or tickers, resolved before assessment. Anything
    #: ambiguous is reported rather than guessed.
    names: list[str] = []
    as_of: date
    agent: str = "rules"


class AskRequest(BaseModel):
    assessment: dict
    question: str
    #: Which body of evidence the question is allowed to draw on.
    #:
    #: One chat box silently doing three different jobs was the problem. A
    #: question about the loaded company, a comparison across several, and a
    #: general finance question have different grounding available, and
    #: answering all three from one memo meant two of them were refused with a
    #: message about the third. Naming the mode makes the scope visible to the
    #: reader instead of a surprise.
    #:
    #: ``company``  -- the loaded assessment only. Strictest, fully cited.
    #: ``compare``  -- several filers, each assessed before being discussed.
    #: ``general``  -- concepts and definitions, no company claims at all.
    mode: str = "company"
    #: Prior exchanges in this chat, so a follow-up has an antecedent. The
    #: client must clear it when the company or as-of date changes -- carrying
    #: one filer's figures into a question about another is the grounding
    #: failure this system exists to prevent, arriving by the side door. The
    #: server enforces that below rather than trusting the client.
    history: list[dict] = []
    #: Optional portfolio context for screening questions. Without it a
    #: cross-company question says what it would need rather than answering
    #: from the model's own knowledge, which nothing could check.
    ciks: list[int] = []


#: The front end is edited constantly during development and the browser
#: caches it aggressively -- several rounds of "I don't see the change" were
#: a stale cache rather than a bug. Served no-store so a reload is always the
#: current file.
NO_STORE = {"Cache-Control": "no-store, must-revalidate", "Pragma": "no-cache"}


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(INDEX.read_text(encoding="utf8"), headers=NO_STORE)


@app.get("/pro", response_class=HTMLResponse)
def pro() -> HTMLResponse:
    """The three-panel analyst terminal: watchlist, cited memo, provenance."""
    return HTMLResponse(
        (Path(__file__).parent / "web" / "pro.html").read_text(encoding="utf8"),
        headers=NO_STORE,
    )


@app.get("/diagnostic.js")
def diagnosticjs():
    from fastapi.responses import Response

    return Response(
        (Path(__file__).parent / "web" / "diagnostic.js").read_text(encoding="utf8"),
        media_type="application/javascript",
        headers=NO_STORE,
    )


@app.get("/chart.js")
def chartjs():
    from fastapi.responses import Response

    return Response(
        (Path(__file__).parent / "web" / "chart.js").read_text(encoding="utf8"),
        media_type="application/javascript",
        headers=NO_STORE,
    )


@app.get("/api/search")
def company_search(q: str) -> JSONResponse:
    """Name or ticker to CIK. Ambiguity is returned, never resolved silently."""
    try:
        directory = load_directory()
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": f"company directory unavailable: {exc}"}, status_code=503)
    match, candidates = resolve(q, directory)
    return JSONResponse(
        {
            "query": q,
            "resolved": {"cik": match.cik, "name": match.name, "ticker": match.ticker,
                         "reason": match.reason} if match else None,
            "candidates": [
                {"cik": c.cik, "name": c.name, "ticker": c.ticker, "reason": c.reason}
                for c in candidates
            ],
        }
    )


#: Metrics offered as trend lines. Each is a single series, so the chart
#: needs no legend -- its title names it.
TREND_METRICS = (
    "current_ratio", "quick_ratio", "liabilities_to_assets", "interest_coverage",
    "net_margin", "return_on_assets", "ocf_to_debt", "altman_z_double_prime",
)


@app.get("/api/trend")
def trend(cik: int, as_of: date, metric: str = "current_ratio", periods: int = 8) -> JSONResponse:
    """One metric across the filer's visible history, oldest first.

    As-of filtered like everything else: a point exists only if the filing it
    came from was public by the prediction date, so the line a reader sees is
    the line that existed then.
    """
    from compute.lineitems import FactIndex, annual_period_ends
    from compute.trends import build_trend
    from data.facts import as_of_view

    if metric not in TREND_METRICS:
        return JSONResponse({"error": f"unknown metric '{metric}'"}, status_code=400)
    load_env_file()
    try:
        facts = EdgarClient().facts(cik)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=400)

    view = FactIndex(as_of_view(facts, as_of))
    ends = annual_period_ends(view)[:periods]
    if not ends:
        return JSONResponse({"cik": cik, "metric": metric, "points": [],
                             "note": "no annual period visible at this date"})
    built = build_trend(metric, view, sorted(ends))
    return JSONResponse({
        "cik": cik,
        "metric": metric,
        "as_of": as_of.isoformat(),
        "direction": built.direction,
        "points": [
            {"period_end": p.period_end.isoformat(),
             "value": float(p.value) if p.is_defined else None}
            for p in built.points
        ],
        "note": "; ".join(built.notes),
    })


@app.get("/api/diagnostic")
def diagnostic(as_of: date, cik: int | None = None, query: str = "") -> JSONResponse:
    """A dated diagnosis of filings already made. Never a projection."""
    load_env_file()
    if cik is None:
        if not query.strip():
            return JSONResponse({"error": "give a CIK, name or ticker"}, status_code=400)
        try:
            match, candidates = resolve(query, load_directory())
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"error": f"lookup failed: {exc}"}, status_code=503)
        if match is None:
            return JSONResponse(
                {
                    "error": (f"'{query}' matches {len(candidates)} filers; choose one"
                              if candidates else f"no SEC filer matches '{query}'"),
                    "candidates": [
                        {"cik": c.cik, "name": c.name, "ticker": c.ticker, "reason": c.reason}
                        for c in candidates
                    ],
                },
                status_code=409 if candidates else 404,
            )
        cik = match.cik

    edgar = EdgarClient()
    try:
        facts = edgar.facts(cik)
        subs = edgar.submissions(cik)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": f"could not load CIK {cik}: {exc}"}, status_code=400)

    d = build_diagnostic(cik, as_of, facts, subs)
    return JSONResponse({
        "cik": d.cik,
        "name": d.name,
        "ticker": d.ticker,
        "exchange": d.exchange,
        "sic": d.sic,
        "sic_description": d.sic_description,
        "state": d.state,
        "as_of": d.as_of.isoformat() if d.as_of else None,
        "period_end": d.period_end.isoformat() if d.period_end else None,
        "filing_age_days": d.filing_age_days,
        "stale": d.stale,
        "zone": d.zone,
        "zone_label": d.zone_label,
        "zones": [{"key": k, "label": lbl, "note": n} for k, lbl, n in ZONES],
        "reported": [
            {"key": r.key, "label": r.label, "value": r.value, "computable": r.computable}
            for r in d.reported
        ],
        "calculated": [
            {"key": r.key, "label": r.label, "value": r.value, "computable": r.computable,
             "threshold": r.threshold, "breached": r.breached}
            for r in d.calculated
        ],
        "timeline": [
            {"as_of": t.as_of.isoformat(),
             "period_end": t.period_end.isoformat() if t.period_end else None,
             "zone": t.zone, "breached": t.breached, "computable": t.computable}
            for t in d.timeline
        ],
    })


#: The ladder as the data plane defines it, worst first. These are graded
#: signals already public, not predictions: a tier says what was filed, and
#: the as-of date says when it became visible.
LADDER_TIERS = ("default", "near_default", "stress", "early_warning")


@app.get("/api/ladder")
def ladder(as_of: date, limit: int = 60, tier: str = "") -> JSONResponse:
    """Distress events visible on or before ``as_of``, worst tier first.

    Filtered by ``as_of`` like everything else. An event dated after the
    chosen date is withheld even though it exists in the file, so the feed
    shows the watchlist as it stood, not as it reads today.
    """
    import csv as _csv

    path = Path("data/labels/distress_events.csv")
    if not path.exists():
        return JSONResponse({"as_of": as_of.isoformat(), "rows": [], "counts": {}})

    names: dict[int, str] = {}
    try:
        for row in load_directory():
            names[row["cik"]] = row["name"]
    except Exception:  # noqa: BLE001 - a missing directory costs names, not rows
        pass

    rows, counts = [], dict.fromkeys(LADDER_TIERS, 0)
    for r in _csv.DictReader(path.open(encoding="utf8")):
        try:
            visible = date.fromisoformat(r["as_of_date"].strip())
        except (KeyError, ValueError):
            continue
        if visible > as_of:
            continue
        t = (r.get("tier") or "").strip()
        if t not in counts:
            continue
        counts[t] += 1
        if tier and t != tier:
            continue
        cik = int(str(r["cik"]).strip().lstrip("0") or 0)
        rows.append({
            "cik": cik,
            "name": names.get(cik, ""),
            "tier": t,
            "signal": r.get("signal", ""),
            "event_date": r.get("event_date", ""),
            "as_of_date": r.get("as_of_date", ""),
            "form": r.get("source_form", ""),
            "noisy": str(r.get("noisy", "")).strip().lower() in ("1", "true"),
        })

    order = {t: i for i, t in enumerate(LADDER_TIERS)}
    rows.sort(key=lambda r: (order.get(r["tier"], 9), r["as_of_date"]), reverse=False)
    rows.sort(key=lambda r: order.get(r["tier"], 9))
    return JSONResponse({
        "as_of": as_of.isoformat(),
        "counts": counts,
        "total_visible": sum(counts.values()),
        "rows": rows[:limit],
    })


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "arms": ARM_AUC, "hazard_baseline_auc": HAZARD_AUC}


@app.post("/api/assess")
def assess(req: AssessRequest) -> JSONResponse:
    payload, status = _assess(req)
    return JSONResponse(payload, status_code=status)


@app.get("/api/assess/stream")
def assess_stream(
    as_of: date,
    cik: int | None = None,
    query: str = "",
    agent: str = "react",
) -> StreamingResponse:
    """The same assessment, emitting each investigation step as it lands.

    A cold ReAct run is a couple of minutes of sequential model calls, and no
    amount of tuning removes that: every step is chosen from the result of the
    one before, so the loop cannot be parallelised without becoming a fixed
    pipeline. What *can* be removed is the silence. A reader watching the
    system open a filing, read the auditor's language and check a covenant is
    reading a progress bar made of evidence, and is far better placed to judge
    the answer than one who stared at a spinner for the same two minutes.

    Server-sent events rather than a websocket: this is one-directional, and
    SSE reconnects on its own.
    """
    req = AssessRequest(cik=cik, query=query, as_of=as_of, agent=agent)
    queue: Queue = Queue()

    def on_step(index: int, step: dict) -> None:
        queue.put(("step", {"index": index, **step}))

    def work() -> None:
        try:
            payload, status = _assess(req, on_step=on_step)
            queue.put(("done", {"status": status, **payload}))
        except Exception as exc:  # noqa: BLE001 - a crash must close the stream
            queue.put(("done", {"status": 500, "error": f"{type(exc).__name__}: {exc}"}))
        finally:
            queue.put((None, None))

    Thread(target=work, daemon=True).start()

    def events():
        while True:
            kind, data = queue.get()
            if kind is None:
                return
            yield f"event: {kind}\ndata: {json.dumps(data, default=str)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={**NO_STORE, "X-Accel-Buffering": "no"},
    )


def _assess(req: AssessRequest, on_step=None) -> tuple[dict, int]:
    """Shared body of the plain and streaming assessment endpoints."""
    load_env_file()
    with tracing.span(
        f"assess:{req.agent}", kind="agent",
        input={"cik": req.cik, "query": req.query, "as_of": str(req.as_of),
               "agent": req.agent},
    ) as _trace:
        payload, status = _assess_traced(req, on_step, _trace)
        tracing.update(_trace, output={
            "signal": (payload.get("memo") or {}).get("signal"),
            "shipped": payload.get("shipped"),
            "blocked_reason": payload.get("blocked_reason"),
            "error": payload.get("error"),
        })
        memo = payload.get("memo") or {}
        # The scores worth tracking across runs, not just reading one trace at
        # a time: did the guard let it out, did it conclude or run out of
        # steps, and how hard did it have to work.
        tracing.score("shipped", bool(payload.get("shipped")))
        if memo:
            tracing.score("signal", str(memo.get("signal")))
            tracing.score("confidence", float(memo.get("confidence") or 0.0))
            tracing.score("tool_calls", float(memo.get("tool_calls") or 0))
            failed = sum(1 for c in memo.get("audit_trail", []) if c.get("ok") is False)
            tracing.score("tool_failures", float(failed),
                          "tool calls that errored during the investigation")
        elif payload.get("blocked_reason"):
            tracing.score("guard_block", payload["blocked_reason"][:120])
        tracing.flush()
        return payload, status


def _assess_traced(req: AssessRequest, on_step, _trace) -> tuple[dict, int]:
    """The assessment itself. Split out so the trace wraps every exit path."""
    cik = req.cik
    resolved_note = ""
    if cik is None:
        if not req.query.strip():
            return {"error": "give a CIK, company name or ticker"}, 400
        try:
            match, candidates = resolve(req.query, load_directory())
        except Exception as exc:  # noqa: BLE001
            return {"error": f"company lookup failed: {exc}"}, 503
        if match is None:
            return {
                "error": (
                    f"'{req.query}' matches {len(candidates)} filers; choose one"
                    if candidates
                    else f"no SEC filer matches '{req.query}'"
                ),
                "candidates": [
                    {"cik": c.cik, "name": c.name, "ticker": c.ticker, "reason": c.reason}
                    for c in candidates
                ],
            }, (409 if candidates else 404)
        cik = match.cik
        resolved_note = f"{match.name} ({match.ticker})" if match.ticker else match.name

    edgar = EdgarClient()
    try:
        facts = edgar.facts(cik)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"could not load filings for CIK {cik}: {exc}"}, 400
    if not facts:
        return {"error": f"no XBRL facts for CIK {cik}"}, 404

    extra: dict = {}
    if req.agent == "react":
        from agents.distress import DistressInvestigator
        from agents.llm import CachingClient, RateLimitedClient, default_client, preflight

        try:
            client = RateLimitedClient(inner=default_client())
            ok, detail = preflight(client)
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, str(exc)
        if not ok:
            return {"error": f"no model endpoint configured: {detail}"}, 503
        def _traced_step(index: int, step: dict) -> None:
            # A tool observation per step is what turns the trace from "one
            # slow request" into the branching investigation it actually is.
            with tracing.span(step.get("tool", "tool"), kind="tool",
                              input=step.get("arguments")) as sp:
                tracing.update(sp, output={"step": index})
            if on_step is not None:
                on_step(index, step)

        investigator = DistressInvestigator(
            CachingClient(inner=client), on_step=_traced_step
        )
        try:
            extra["filing_index"] = edgar.filing_index(cik)
            extra["fetch_document"] = lambda a, d: edgar.fetch_filing_document(cik, a, d)
        except Exception:  # noqa: BLE001
            pass
    else:
        from agents.rulebased import RuleBasedInvestigator

        investigator = RuleBasedInvestigator()

    # Earnings-quality observations ride along as context. Measured at
    # 0.51-0.61 AUC over six approaches, so they never move the graded signal.
    notes = earnings_notes(facts, req.as_of)
    # Through the graph. Facts are passed in rather than refetched -- they were
    # loaded above to answer other questions, and a second EDGAR round trip
    # against a rate-limited endpoint buys nothing.
    state = graph_run(
        investigator, cik, req.as_of, facts=facts, context_notes=notes, **extra
    )
    if state.get("error"):
        return {"error": state["error"]}, 400
    result = Assessment(
        cik=cik, as_of=req.as_of, triage=state["triage"],
        output=state.get("output"), guards=state.get("guards"),
        memo=state.get("memo"), blocked_reason=state.get("blocked_reason", ""),
    )

    payload: dict = {
        "cik": cik,
        "company": resolved_note,
        "as_of": req.as_of.isoformat(),
        "agent": req.agent,
        "arm_auc": ARM_AUC.get(req.agent),
        "hazard_baseline_auc": HAZARD_AUC,
        # The as-of view, not the whole fetch. This is printed as "figures
        # available", and reporting the unfiltered total said 36,659 for a date
        # at which 16,123 were visible -- twice the truth, on the one number
        # whose job is to say how much the system could see.
        "n_facts": _visible_fact_count(facts, req.as_of),
        "triage": {
            "depth": result.triage.depth,
            "steps": result.triage.steps,
            "reasons": result.triage.reasons,
            "latest_period_end": result.triage.latest_period_end.isoformat()
            if result.triage.latest_period_end
            else None,
        },
        "shipped": result.shipped,
    }
    if not result.shipped:
        payload["blocked_reason"] = result.blocked_reason
        return payload, 200

    memo = result.memo
    payload["memo"] = {
        "signal": memo.signal,
        "confidence": memo.confidence,
        "risk_score": memo.risk_score,
        "summary": memo.summary,
        "residual": memo.residual,
        "limitations": memo.limitations,
        "routing": memo.routing,
        "tool_calls": memo.tool_calls,
        "sections": [
            {
                "title": s.title,
                "tier": s.tier,
                "body": s.body,
                "moves_the_signal": s.moves_the_signal,
                "evidence": [
                    {
                        "metric": e.metric,
                        "value": e.value,
                        "period_end": e.period_end.isoformat() if e.period_end else None,
                        "note": e.note,
                    }
                    for e in s.evidence
                ],
            }
            for s in memo.sections
        ],
        "audit_trail": memo.audit_trail,
        "text": memo.render(),
    }
    # The filer's own recorded history, attached to every assessment. A clean
    # reading of post-reorganisation accounts is only confusing when the
    # reorganisation is invisible.
    history = _history_section(filer_events(cik, req.as_of), req.as_of)
    if history:
        payload["memo"]["sections"].append(history)

    future = _future_date_note(req.as_of)
    if future:
        payload["memo"]["limitations"] = [future, *payload["memo"]["limitations"]]

    # How much to trust this particular reading. Attached to the payload rather
    # than buried in an eval script: the subgroup where the headline AUC stops
    # applying is the one an analyst most needs warning about.
    n_evidence = sum(
        1
        for s in payload["memo"]["sections"]
        for e in s.get("evidence", [])
        if e.get("value") is not None
    )
    # The statistical ranker, alongside the agent's reading. Two jobs: the GBM
    # sorts the queue, the agent explains one filer. They tie on
    # discrimination, so neither overrules the other -- and where they
    # disagree that is surfaced rather than averaged away.
    ranking = ranker.rank(cik, req.as_of, facts)
    if ranking is not None:
        payload["ranked"] = {
            "score": ranking.score,
            "percentile": ranking.percentile,
            "trained_through": ranking.trained_through,
            "resampled_auc": ranking.resampled_auc,
            "resampled_range": list(ranking.resampled_range),
            "disagreement": ranker.disagreement(
                (payload.get("memo") or {}).get("signal", ""), ranking.percentile
            ),
        }

    payload["reliability"] = {
        "evidence_count": n_evidence,
        "sparse": n_evidence <= SPARSE_EVIDENCE_MAX,
        # No reliability claim. The one that was here did not survive being
        # measured against the metric the product reports.
        "note": "",
    }
    return payload, 200


def _assess_one(edgar: EdgarClient, investigator, cik: int, as_of: date) -> dict:
    """One row of a portfolio run. A filer that cannot be loaded is reported
    as such rather than dropped -- a portfolio view that silently omits the
    names it could not read is worse than one that says so."""
    try:
        facts = edgar.facts(cik)
    except Exception as exc:  # noqa: BLE001
        return {"cik": cik, "status": "unavailable", "detail": str(exc)[:120]}
    if not facts:
        return {"cik": cik, "status": "unavailable", "detail": "no XBRL facts"}
    # A row identified only by CIK is unreadable. The directory lookup is
    # cached and costs nothing next to the EDGAR fetch above.
    name = ""
    try:
        for row in load_directory():
            if row["cik"] == cik:
                name = row["name"]
                break
    except Exception:  # noqa: BLE001 - a missing directory costs a name, not a row
        pass
    result = Orchestrator(investigator).run(cik, as_of, facts)
    if not result.shipped:
        return {"cik": cik, "status": "blocked", "detail": result.blocked_reason[:160]}
    return {
        "cik": cik,
        "status": "ok",
        "name": name,
        "signal": result.memo.signal,
        # risk_score orders thinly-disclosed filers poorly -- that part of the
        # fairness finding was real even though the headline claim was not --
        # so a calibrated probability is preferred for ranking where one exists.
        "risk_score": result.memo.risk_score,
        "rank_by": (
            result.memo.risk_probability
            if getattr(result.memo, "risk_probability", None) is not None
            else result.memo.risk_score
        ),
        "confidence": result.memo.confidence,
        "depth": result.triage.depth,
        "limitations": len(result.memo.limitations),
        "tool_calls": result.memo.tool_calls,
    }


@app.post("/api/batch")
def batch(req: BatchRequest) -> JSONResponse:
    """Assess a portfolio. Capped, because each name is a real EDGAR fetch."""
    from agents.rulebased import RuleBasedInvestigator

    if req.agent == "react":
        return JSONResponse(
            {"error": "batch runs the deterministic control only; "
                      "the ReAct arm costs ~2 minutes and a model call per name"},
            status_code=400,
        )
    load_env_file()
    ciks = list(req.ciks)
    unresolved: list[str] = []
    if req.names:
        directory = load_directory()
        for raw in req.names:
            match, _ = resolve(raw, directory)
            if match is None:
                unresolved.append(raw)
            else:
                ciks.append(match.cik)
    ciks = ciks[:MAX_BATCH]
    edgar = EdgarClient()
    investigator = RuleBasedInvestigator()
    rows = [_assess_one(edgar, investigator, c, req.as_of) for c in ciks]

    def rank_key(row: dict) -> tuple:
        """Order by risk, falling back to the ordinal signal.

        The rule-based control emits no risk_score, so ranking on it alone put
        every row at the same value and the table claimed an order it did not
        have. The signal is always present and is itself ordered, so it carries
        the ranking when the score is absent.
        """
        if row.get("status") != "ok":
            return (-1, -1.0, 0.0)
        severity = SIGNAL_ORDER.index(row["signal"]) if row["signal"] in SIGNAL_ORDER else -1
        score = row.get("risk_score")
        return (severity, score if score is not None else -1.0, row.get("confidence", 0.0))

    ranked = sorted(rows, key=rank_key, reverse=True)
    return JSONResponse(
        {
            "as_of": req.as_of.isoformat(),
            "requested": len(ciks) + len(unresolved),
            "assessed": sum(1 for r in rows if r["status"] == "ok"),
            "truncated": len(req.ciks) + len(req.names) > MAX_BATCH,
            "unresolved": unresolved,
            "rows": ranked,
        }
    )


@app.post("/api/upload")
async def upload(file: UploadFile = UPLOAD) -> JSONResponse:
    """Read CIKs out of an uploaded CSV or text file.

    Parsing only -- it returns the identifiers it found so the caller can
    review them before spending fetches. Uploading a list and having it
    silently run is how a typo becomes two hundred EDGAR requests.
    """
    raw = (await file.read()).decode("utf8", errors="replace")
    found: list[int] = []
    seen: set[int] = set()
    for line in raw.splitlines():
        for token in re.split(r"[,;	|]", line):
            token = token.strip().strip('"').lstrip("0")
            if token.isdigit() and 1 <= len(token) <= 10:
                value = int(token)
                if value not in seen:
                    seen.add(value)
                    found.append(value)
    return JSONResponse(
        {
            "filename": file.filename,
            "ciks": found[:MAX_BATCH],
            "found": len(found),
            "truncated": len(found) > MAX_BATCH,
        }
    )


@app.post("/api/scan")
async def scan(file: UploadFile = UPLOAD) -> JSONResponse:
    """Scan an uploaded filing for going-concern and material-weakness language.

    Deterministic -- regex over the stripped text, no model. The same code the
    agent's check_going_concern tool uses, so what a user sees here is what the
    investigator would see.
    """
    from data import formtype, pdftext
    from data.signals import scan_report_text

    data = await file.read()
    if not data.strip():
        return JSONResponse({"error": "empty file"}, status_code=400)

    extra: dict[str, object] = {}
    doc: pdftext.PdfText | None = None
    if pdftext.is_pdf(data):
        doc = pdftext.extract(data)
        if doc.error:
            return JSONResponse({"error": doc.error}, status_code=400)
        # A scan with no text layer must not come back clean. Every pattern
        # fails to match the empty string, so reporting "not found" here would
        # manufacture a false negative on the strongest signal we have -- on a
        # page that may say "substantial doubt" in plain sight.
        if not doc.has_text_layer:
            return JSONResponse(
                {
                    "error": (
                        f"This PDF has no text layer -- {doc.n_pages} page(s) of images "
                        "with no readable characters. Scanning it would report "
                        "'not found' for every signal, which would be wrong rather "
                        "than reassuring. Run it through OCR, or upload the HTML "
                        "version from EDGAR."
                    ),
                    "filename": file.filename,
                    "pages": doc.n_pages,
                    "text_layer": False,
                },
                status_code=422,
            )
        raw = doc.text
        extra = {
            "format": "pdf",
            "pages": doc.n_pages,
            "text_layer": True,
            "image_pages": doc.image_pages,
        }
    else:
        raw = data.decode("utf8", errors="replace")
        extra = {"format": "html/text"}
        if not raw.strip():
            return JSONResponse({"error": "empty file"}, status_code=400)

    # Label the document before reporting on it. "Not found" in a Form 4 means
    # the form cannot carry the disclosure, not that the company is clean, and
    # showing the same two words for both readings is a false all-clear.
    from data.signals import strip_html

    form = formtype.identify(raw if doc is not None else strip_html(raw))
    extra["form_code"] = form.code
    extra["form_name"] = form.name
    extra["form_carries_financials"] = form.carries_financials
    if form.identified and not form.carries_financials:
        extra["form_note"] = form.instead

    found = scan_report_text(raw)
    if doc is not None:
        # Page numbers turn a quote into a citation a reader can go and check.
        # A quote carries context either side of the matched phrase, so it often
        # straddles a break -- report the span rather than sending the reader to
        # the page before the sentence they wanted.
        for key in ("going_concern_quote", "material_weakness_quote"):
            span = doc.locate_span(str(found.get(key) or ""))
            if span is not None:
                first, last = span
                extra[key.replace("_quote", "_page")] = (
                    str(first) if first == last else f"{first}–{last}"
                )

    return JSONResponse(
        {
            "filename": file.filename,
            "characters": len(raw),
            **extra,
            **found,
            "note": "deterministic phrase match over the filing text; no model involved",
        }
    )


#: How many metric series to load for a trend question. Enough to show the
#: shape of the position without burying the memo's own findings in rows.
MAX_TREND_SERIES = 4


def _trend_series(cik: int, as_of: date, assessment: dict) -> list[dict]:
    """Real multi-period series for the measures this memo already cites.

    Scoped to what the memo cites on purpose. Loading every chartable metric
    would put figures in front of the model that the assessment never
    considered, and the answer would drift from explaining the finding to
    narrating a data dump. As-of filtered by the same code path as the memo,
    so a series can never contain a period the memo could not see.
    """
    from compute.lineitems import FactIndex, annual_period_ends
    from compute.trends import build_trend
    from data.facts import as_of_view

    cited: list[str] = []
    for section in (assessment.get("memo") or {}).get("sections", []):
        for e in section.get("evidence", []):
            m = e.get("metric")
            if m in TREND_METRICS and m not in cited:
                cited.append(m)
    if not cited:
        return []

    try:
        facts = EdgarClient().facts(cik)
    except Exception:  # noqa: BLE001 - no history is a degraded answer, not an error
        return []
    view = FactIndex(as_of_view(facts, as_of))
    ends = sorted(annual_period_ends(view)[:6])
    if len(ends) < 2:
        return []

    out = []
    for metric in cited[:MAX_TREND_SERIES]:
        built = build_trend(metric, view, ends)
        points = [
            {"period_end": p.period_end.isoformat(), "value": p.value} for p in built.points
        ]
        if sum(1 for p in points if p["value"] is not None) >= 2:
            out.append({"metric": metric, "direction": built.direction, "points": points})
    return out


def filer_events(cik: int, as_of: date, limit: int = 12) -> list[dict]:
    """This filer's own recorded distress events, on or before ``as_of``.

    Same file the severity ladder reads, filtered to one CIK. As-of filtered
    like everything else, so an event the prediction date could not have seen
    stays hidden.
    """
    import csv as _csv

    path = Path("data/labels/distress_events.csv")
    if not path.exists():
        return []
    out = []
    for r in _csv.DictReader(path.open(encoding="utf8")):
        try:
            if int(r["cik"]) != cik:
                continue
            visible = date.fromisoformat(r["as_of_date"].strip())
        except (KeyError, ValueError):
            continue
        if visible > as_of:
            continue
        out.append({
            "event_date": r.get("event_date", "").strip(),
            "signal": (r.get("signal") or "").strip(),
            "tier": (r.get("tier") or "").strip(),
            "form": (r.get("form") or "").strip(),
        })
    out.sort(key=lambda e: e["event_date"])
    return out[-limit:]


def _visible_fact_count(facts, as_of: date) -> int:
    """How many facts were public on the prediction date."""
    from data.facts import as_of_view

    try:
        return len(as_of_view(facts, as_of))
    except Exception:  # noqa: BLE001 - a count is not worth failing an assessment
        return len(facts)


#: How far ahead of today an as-of date may sit before it is worth saying so.
#: A day or two is a typo or a timezone; four years is a misunderstanding, and
#: the reader should be told which they have.
FUTURE_DATE_SLACK_DAYS = 2


def _future_date_note(as_of: date) -> str:
    """A limitation line when the prediction date has not happened yet.

    Not refused. Asking about today is the ordinary case and the boundary is
    fuzzy, so this goes where every other thing a reader should weigh already
    lives -- in the memo's limitations, next to staleness.
    """
    ahead = (as_of - date.today()).days
    if ahead <= FUTURE_DATE_SLACK_DAYS:
        return ""
    return (
        f"The prediction date {as_of} is {ahead} days in the future. Nothing "
        f"has been withheld because nothing filed after it exists yet, so this "
        f"is a reading of everything on file rather than a point-in-time view. "
        f"If you meant a past date, the as-of filter is doing nothing here."
    )


def _history_section(events: list[dict], as_of: date) -> dict:
    """The filer's own event history, as a context-only memo section.

    Why this exists. Chord Energy assessed at 2024-07-01 comes back *healthy*,
    and the severity ladder shows it at *default* in 2020. Both are correct:
    it filed Chapter 11 on 2020-09-30 as Oasis Petroleum, emerged, and merged
    with Whiting in 2022. The assessment is reading post-reorganisation
    accounts.

    But the product never put those two facts on the same page, so an analyst
    saw a clean bill of health for a company they knew had defaulted, and had
    no way to reconcile it. The reconciliation is not a caveat -- it is the
    most interesting thing about the filer.

    Context-only: a past default is history, and letting it move a signal about
    present accounts would be the model reasoning from the label instead of
    from the filings.
    """
    if not events:
        return {}
    worst = next((e for e in events if e["tier"] == "default"), None)
    lines = [
        f"  {e['event_date']}  {e['signal'].replace('_', ' ')} "
        f"({e['form'] or 'filing'})"
        for e in events
    ]
    lead = (
        f"This filer has a recorded Chapter 11 petition on "
        f"{worst['event_date']}. The assessment above reads the accounts "
        f"visible at {as_of}, which are later than that event -- a company "
        f"that defaulted and reorganised can read as sound afterwards, and "
        f"both statements are true."
        if worst
        else f"Recorded distress events for this filer, all before {as_of}."
    )
    return {
        "title": "Recorded events for this filer",
        "tier": "context-only",
        # Composed here from the labelled event file, not written by the model,
        # so its dates may be quoted like any other verified figure.
        "generated_by": "system",
        "body": lead + "\n" + "\n".join(lines),
        "moves_the_signal": False,
        "evidence": [],
    }


#: A question asking what a term *means*, as opposed to one asking about the
#: filer that happens to lack a pronoun.
#:
#: The first version tested "conceptual grammar and no pronoun", which routed
#: two ordinary company questions to the general-definitions path: "explain the
#: liquidity position in about 250 words" came back as a textbook definition of
#: liquidity, and "what did the auditor say" came back as "I answer general
#: questions here". Plenty of real questions about a filer name no pronoun. The
#: frame has to be definitional, not merely impersonal.
_ASKS_A_DEFINITION = re.compile(
    r"\bwhat (?:is|are) (?:a|an)\b"
    r"|\bwhat does .{0,40}\b(?:mean|measure|tell|indicate|refer to)\b"
    r"|\bwhat do .{0,40}\bmean\b"
    r"|\bdefine\b|\bmeaning of\b|\bdifference between\b"
    r"|\bmean anyway\b|\bwhat exactly is\b"
    r"|\bhow (?:does|do) .{0,30}\bwork\b",
    re.I,
)

#: A reference to the filer currently loaded. Its presence turns a
#: grammatically general question ("how leveraged is it") into a specific one.
_REFERS_TO_LOADED = re.compile(
    r"\b(it|its|it's|this|these|they|their|them|the company|the filer|"
    r"the business|the issuer|here|this one|that)\b",
    re.I,
)

#: Words that look like company names but are ordinary question vocabulary.
#: Without this every sentence starting "What" resolves to some filer.
#: Sentence-opening words and question vocabulary that capitalise innocently.
#: Kept, but it is the second line of defence -- the length floor and the cue
#: requirement below do the real work, because no stopword list survives typos.
_NOT_A_COMPANY = {
    "what", "why", "how", "when", "where", "which", "who", "is", "does", "the",
    "this", "that", "it", "its", "and", "or", "but", "compare", "explain",
    "show", "give", "tell", "ok", "so", "chapter", "should", "would", "could",
    "can", "cna", "cant", "did", "will", "please", "also", "then",
    "here", "there", "about", "with", "from", "into", "your", "their",
}

#: The shortest a bare capitalised word may be before it is treated as a
#: company. "Can" and "Cna" are three letters and both resolve against the SEC
#: directory; nothing shorter than four is worth the false positives.
_MIN_NAME_CHARS = 4

#: A phrase that announces a company is coming. When one is present the phrase
#: after it is what the reader named, which is how "compare this to Valaris"
#: finds Valaris instead of a typo earlier in the sentence.
_COMPANY_CUE = re.compile(
    r"\b(?:compared?\s+(?:it\s+|this\s+|them\s+)?(?:to|with|against)"
    r"|versus|vs\.?|against|relative to|next to|alongside"
    r"|how (?:does|do|is|are)|what about|how about|and also)\s+",
    re.I,
)


def _resolve_phrase(phrase: str, directory, loaded_cik: int) -> str:
    """A directory match for one phrase, if it is a different filer.

    Kept for the exact-name path; BM25 handles ranked matching in
    ``_names_another_company``.
    """
    if len(phrase) < _MIN_NAME_CHARS:
        return ""
    if all(w.lower() in _NOT_A_COMPANY for w in phrase.split()):
        return ""
    match, _ = resolve(phrase, directory)
    return match.name if match and match.cik != loaded_cik else ""


def _companies_in_question(question: str, exclude: int | None = None,
                           limit: int = 3) -> list[tuple[int, str]]:
    """Every filer the question names, resolved from the raw text.

    Case-insensitive and typo-tolerant because it goes through BM25 rather than
    a capitalisation rule: "comapre this with valaris" resolves Valaris Ltd, and
    the old heuristic -- capitalised words only -- could not see it at all.

    Returns at most ``limit``: a question naming five companies is a screening
    request, and answering it as a comparison would quietly drop most of them.
    """
    from data.bm25 import index

    found: list[tuple[int, str]] = []
    seen: set[int] = {exclude} if exclude else set()
    try:
        hits = index().search(question, limit=limit + len(seen) + 2)
    except Exception:  # noqa: BLE001 - resolution failing costs a name, not the answer
        return []
    for hit in hits:
        if hit.cik in seen or not hit.is_confident:
            continue
        seen.add(hit.cik)
        found.append((hit.cik, hit.name))
        if len(found) >= limit:
            break
    return found


def _general_mode_pointer(question: str) -> str:
    """Where the company half of a general-mode question can be answered.

    Naming a company in general mode is not a reason to say nothing. The
    concept still gets explained; this appends where the rest of the question
    is answerable, instead of "I answer general questions here" and a dead end.
    """
    named = _companies_in_question(question)
    if not named:
        return ""
    who = " and ".join(name for _cik, name in named)
    return (
        f"\n\nYou named {who}. General mode makes no claim about any company, "
        "so nothing here is grounded in their filings. Ask the same question in "
        "Compare and each one is assessed from its own filings first."
    )


def _names_another_company(question: str, loaded_cik: int) -> str:
    """A company the reader named that is not the one loaded.

    Conservative by construction. A false positive costs a real answer -- the
    reader is redirected instead of served -- so anything ambiguous resolves to
    silence and the question is answered normally.
    """
    try:
        directory = load_directory()
    except Exception:  # noqa: BLE001
        return ""

    def _candidates(text: str) -> list[str]:
        words = [w.strip(".,?!'\"") for w in text.split()]
        keep = [w for w in words if w[:1].isupper()]
        out: list[str] = []
        for n in (3, 2):          # multi-word first: two capitals in a row are
            for i in range(len(keep) - n + 1):   # rarely an accident
                out.append(" ".join(keep[i : i + n]))
        out.extend(w for w in keep if len(w) >= _MIN_NAME_CHARS)
        return out

    # Extraction is a rule; ranking is BM25. Feeding the whole sentence to the
    # index found Here Group Ltd for "here" and Wheels Up for "up", so the
    # index only ever sees phrases that plausibly name a company -- and then
    # scores them, which is what the old positional scan could not do.
    from data.bm25 import index

    ix = index()
    after_cue = [
        phrase
        for cue in _COMPANY_CUE.finditer(question)
        for phrase in _candidates(question[cue.end() :])
    ]
    # A phrase the reader pointed at with a cue outranks one merely present.
    hit = ix.best_in(question, after_cue, exclude_cik=loaded_cik)
    if hit:
        return hit.name
    hit = ix.best_in(question, _candidates(question), exclude_cik=loaded_cik)
    if hit:
        return hit.name

    # Exact-name fallback for filers the ranked index scores weakly -- a
    # single distinctive token can fall below the evidence floor.
    for phrase in after_cue or _candidates(question):
        found = _resolve_phrase(phrase, directory, loaded_cik)
        if found:
            return found
    return ""


#: The reader asking for two filers side by side, rather than merely mentioning
#: another company in passing.
_WANTS_COMPARISON = re.compile(
    r"\bcompare|\bcomparison|\bversus\b|\bvs\.?\b|\bagainst\b"
    r"|\bbetter or worse\b|\bhow does .{0,30}stack up|\bside by side\b"
    r"|\brelative to\b|\bwho is (?:worse|better|riskier|safer)\b",
    re.I,
)


def _compare_with(cik: int, other_name: str, as_of: date, question: str,
                  assessment: dict) -> dict | None:
    """Assess the named filer and put the two side by side.

    Returns None on any failure, so a comparison we cannot make falls through
    to the redirect rather than becoming an error. The other company is
    resolved, fetched and assessed here -- it is never discussed on the
    strength of the model having heard of it.
    """
    from agents.rulebased import RuleBasedInvestigator
    from agents.screening import build as build_screen

    try:
        directory = load_directory()
        match, _ = resolve(other_name, directory)
        if match is None or match.cik == cik:
            return None
        edgar = EdgarClient()
        rows = [
            _assess_one(edgar, RuleBasedInvestigator(), c, as_of)
            for c in (cik, match.cik)
        ]
        # _assess_one resolves a name; make sure the loaded filer carries the
        # one already on screen rather than falling back to a bare CIK.
        for row in rows:
            if row.get("cik") == cik and assessment.get("company"):
                row["name"] = assessment["company"]
    except Exception:  # noqa: BLE001 - a failed comparison degrades to advice
        return None
    if any(r.get("status") != "ok" for r in rows):
        return None

    screened = build_screen(question, rows, as_of)
    loaded = assessment.get("company") or f"CIK {cik}"
    return {
        "question": question,
        "answer": (
            f"Both filers assessed from their own filings as at {as_of}, then "
            f"compared -- neither is described from memory.\n\n"
            f"{screened.text}"
        ),
        "allowed": True,
        "reason": "compared_on_request",
        "mode": "company",
        "compared": [loaded, match.name],
        "rows": screened.rows,
        "caveats": screened.caveats,
        "ungrounded_numbers": [],
    }


#: Asking where this filer sits against the population. Answerable from the
#: ranking already attached to the assessment, and previously unreachable
#: because the company redirect fired on the typo first.
_ASKS_RANK = re.compile(
    r"\bpercentile\b|\bwhat percent\b|\brank(?:s|ed|ing)?\b"
    r"|\bcompared with (?:the )?(?:other|rest|population|peers)\b"
    r"|\bwhere does (?:it|this) (?:sit|stand|fall)\b",
    re.I,
)


def _placement(pct: float) -> str:
    """Where the filer sits, in words. "Above 100%" is arithmetically true for
    the top-ranked name and reads like a bug."""
    if pct >= 99.5:
        return ("The statistical ranker places this filer at the very top "
                "of the population it was trained on.")
    if pct <= 0.5:
        return ("The statistical ranker places this filer at the very bottom "
                "of the population it was trained on.")
    return f"The statistical ranker places this filer above {pct:.0f}% of the population it was trained on."


def _rank_answer(assessment: dict) -> str:
    """Plain-English placement, or an honest note that there is none."""
    ranked = assessment.get("ranked") or {}
    pct = ranked.get("percentile")
    if pct is None:
        return (
            "No statistical ranking was produced for this filer -- the ranker "
            "needs enough reported history to build a feature row, and this "
            "one did not have it. The assessment above still stands on its own."
        )
    lo, hi = (ranked.get("resampled_range") or [0, 0])[:2]
    return (
        f"{_placement(pct)}\n\n"
        f"That is an ordering, not a probability. It says where this company "
        f"sits relative to others, and nothing about how likely it is to fail: "
        f"the ranker scores {ranked.get('resampled_auc', 0):.3f} AUC across 25 "
        f"group-aware resamples [{lo:.3f}, {hi:.3f}], and that ordering holds "
        f"while the absolute level does not transfer.\n\n"
        f"The investigation is a separate reading, and where the two disagree "
        f"the assessment says so rather than averaging them."
    )


def _compute_on_request(cik: int, as_of: date, question: str, assessment: dict) -> dict:
    """Compute registered formulas the question names but the memo omitted.

    Returned as an ordinary memo section so it flows through every existing
    control unchanged: it is rendered into the context, its values join the
    trusted set, and the citation check can bind a figure to it. Marked
    ``on-request`` so a reader can see which figures the assessment reached for
    on its own and which one they asked for.

    Same as-of view as the assessment, so a figure computed on request can no
    more see past the prediction date than one the agent found itself.
    """
    # Both registries, explicitly. compute.ratios holds 21 formulas and
    # compute.scores the other 15 (Piotroski, Ohlson, the Beneish components),
    # so importing only the first makes routing depend on whether something
    # else happened to import the second first -- and a question about the
    # F-score would silently find nothing on a cold process.
    import compute.scores  # noqa: F401
    from agents.qa import requested_metrics
    from compute.lineitems import FactIndex, annual_period_ends
    from compute.provenance import FORMULAS
    from compute.ratios import compute_metric
    from data.facts import as_of_view

    already = {
        e.get("metric")
        for s in (assessment.get("memo") or {}).get("sections", [])
        for e in s.get("evidence", [])
    }
    wanted = [m for m in requested_metrics(question, set(FORMULAS)) if m not in already]
    if not wanted:
        return {}

    try:
        facts = EdgarClient().facts(cik)
    except Exception:  # noqa: BLE001 - no extra figure is a worse answer, not an error
        return {}
    view = FactIndex(as_of_view(facts, as_of))
    ends = annual_period_ends(view)
    if not ends:
        return {}

    evidence = []
    for metric in wanted[:6]:
        try:
            cv = compute_metric(metric, view, ends[0])
        except Exception:  # noqa: BLE001
            continue
        # A measure that cannot be computed is reported as such rather than
        # dropped: "the filer did not tag what this needs" is an answer.
        evidence.append({
            "metric": metric,
            "value": float(cv.value) if cv.is_defined else None,
            "period_end": ends[0].isoformat(),
            "note": "on-request" if cv.is_defined else "not computable",
        })
    if not evidence:
        return {}
    return {
        "title": "Computed on request",
        "tier": "backtested",
        "body": "",
        "moves_the_signal": False,
        "evidence": evidence,
    }


@app.post("/api/ask")
def ask_question(req: AskRequest) -> JSONResponse:
    """Grounded Q&A over one finished assessment."""
    from agents.llm import CachingClient, RateLimitedClient, default_client, preflight
    from agents.qa import ask, wants_trends

    load_env_file()
    try:
        client = RateLimitedClient(inner=default_client())
        ok, detail = preflight(client)
    except Exception as exc:  # noqa: BLE001
        ok, detail = False, str(exc)
    if not ok:
        return JSONResponse({"error": f"no model endpoint configured: {detail}"}, status_code=503)

    # A concept question needs no assessment. Refusing "what is a covenant
    # breach?" because no memo covers it protects nothing and makes the tool
    # useless; the grounding rule is about claims on a *named company*.
    if not req.assessment.get("memo") and not is_conceptual(req.question):
        from agents.explain import _SPECIFIC, REDIRECT_TO_ASSESSMENT

        if _SPECIFIC.search(req.question):
            return JSONResponse({
                "question": req.question,
                "answer": REDIRECT_TO_ASSESSMENT,
                "allowed": True,
                "reason": "needs_assessment",
                "ungrounded_numbers": [],
            })

    if is_conceptual(req.question) and not req.assessment.get("memo"):
        result = explain(CachingClient(inner=client), req.question)
        return JSONResponse({
            "question": req.question,
            "answer": (
                result.text
                if not _companies_in_question(req.question)
                else result.text + "\n\n" + (
                    "You named "
                    + " and ".join(n for _c, n in _companies_in_question(req.question))
                    + ". General mode makes no claim about any company, so switch to "
                    "This company or Compare and the same question is answered from "
                    "their filings."
                )
            ),
            "allowed": result.allowed,
            "reason": result.reason,
            "mode": "general",
            "ungrounded_numbers": [],
        })

    cik = req.assessment.get("cik")
    as_of_raw = req.assessment.get("as_of")
    as_of = date.fromisoformat(as_of_raw) if as_of_raw else None

    # "Did it survive?" is answerable from the outcome labels and was being
    # refused because the *assessment* cannot see past its own date. That is
    # the model's constraint, not the reader's: a 2025 bankruptcy is history
    # to someone asking now. The investigator still never sees this.
    if ASKS_OUTCOME.search(req.question) and cik and as_of:
        return JSONResponse({
            "question": req.question,
            "answer": survived_note(int(cik), as_of, load_outcomes()),
            "allowed": True,
            "reason": "outcome_from_labels",
            "ungrounded_numbers": [],
        })

    # A different company, named in the middle of a conversation about this
    # one. The model already refuses -- correctly, it has no data on the other
    # filer -- but "the assessment does not contain information about Apple" is
    # a dead end. A reader who names a company wants that company, so say where
    # to get it rather than only what is missing.
    # A compound question gets a compound answer.
    #
    # The routing here was a chain of early returns, so "at what percentile
    # does it rank, and also compare this to Valaris" answered the percentile
    # and dropped the comparison without saying so. People do not ask one thing
    # at a time, and silently serving half a question is worse than refusing
    # it: the reader has no way to know the rest was ignored.
    #
    # Each part is answered by whatever owns it, and the parts are composed.
    if req.mode == "company" and cik:
        parts: list[tuple[str, str]] = []
        handled = False

        if _ASKS_RANK.search(req.question) and req.assessment.get("ranked"):
            parts.append(("Where it ranks", _rank_answer(req.assessment)))
            handled = True

        # A definitional aside is a part like any other, so it composes: a
        # reader can ask what a term means *and* something about the filer in
        # one breath. The compound rewrite had dropped this branch entirely,
        # and the answer-quality matrix caught it on its first run.
        if (
            _ASKS_A_DEFINITION.search(req.question)
            and not _REFERS_TO_LOADED.search(req.question)
        ):
            explained = explain(CachingClient(inner=client), req.question)
            if explained.allowed and explained.text:
                parts.append(("What that means", explained.text))
                handled = True

        # BM25 over the raw question, not capitalised words. The heuristic it
        # replaces could not see "comapre this with valaris" at all -- lowercase
        # and misspelt is how people actually type.
        named = _companies_in_question(req.question, exclude=int(cik), limit=1)
        other = named[0][1] if named else ""
        if other:
            compared = _compare_with(int(cik), other, as_of or date.today(),
                                     req.question, req.assessment)
            if compared:
                parts.append((f"Compared with {other}", compared["answer"]))
            else:
                parts.append((
                    f"About {other}",
                    f"This chat is scoped to "
                    f"{req.assessment.get('company') or f'CIK {cik}'}, and "
                    f"{other} could not be assessed from its filings just now. "
                    f"Search for it on the left to open it directly."
                ))
            handled = True

        if handled:
            # Whatever is left of the question, if anything, still goes to the
            # model. A reader who asks for a rank *and* an explanation should
            # get both, not a rank and silence.
            leftover = _ASKS_RANK.sub(" ", req.question)
            if other:
                leftover = leftover.replace(other, " ")
            if len(leftover.split()) >= 6:
                try:
                    extra = ask(CachingClient(inner=client), req.assessment,
                                req.question)
                    if extra.allowed and extra.text:
                        parts.append(("On the filings", extra.text))
                except Exception:  # noqa: BLE001 - a missing part, not a failure
                    pass
            body = "\n\n".join(
                f"## {title}\n{text}" if len(parts) > 1 else text
                for title, text in parts
            )
            return JSONResponse({
                "question": req.question,
                "answer": body,
                "allowed": True,
                "reason": "composed" if len(parts) > 1 else parts[0][0].lower(),
                "mode": "company",
                "parts": [t for t, _ in parts],
                "ungrounded_numbers": [],
            })

    # ---- explicit modes -------------------------------------------------
    #
    # The reader has said what kind of question this is, so honour it rather
    # than inferring. General mode in particular has to be reachable on demand:
    # "what is a covenant breach" is a fair question that the company-scoped
    # path will always answer with "that is not in the assessment".
    if req.mode == "general":
        result = explain(CachingClient(inner=client), req.question)
        return JSONResponse({
            "question": req.question,
            # A company named in general mode is not a reason to say nothing.
            # The concept still gets explained; the reader is then told where
            # the company half of their question can actually be answered.
            "answer": result.text + _general_mode_pointer(req.question),
            "allowed": result.allowed,
            "reason": result.reason or "general",
            "mode": "general",
            "ungrounded_numbers": [],
            "note": (
                "General mode explains concepts. It makes no claim about any "
                "company, because nothing here could verify one."
            ),
        })

    if req.mode == "compare":
        # Every filer named is assessed before it is discussed. The alternative
        # -- letting the model compare from recollection -- produces confident
        # prose about companies nothing in this system ever looked at.
        # Filers named in the sentence count as staged. Telling a reader who
        # wrote "compare this with Valaris" to go and add Valaris somewhere
        # else is the software asking them to do its job.
        named = [c for c, _n in _companies_in_question(req.question, exclude=cik)]
        targets = list(dict.fromkeys(
            [*(req.ciks or []), *([int(cik)] if cik else []), *named]
        ))
        if len(targets) < 2:
            return JSONResponse({
                "question": req.question,
                "answer": (
                    "Comparison needs two filers and I could identify "
                    f"{len(targets)}. Name the other company in the question -- "
                    "\"compare this with Valaris\" is enough -- or stage it in the "
                    "Compare view. Either way each is assessed from its own "
                    "filings before anything is said about it; nothing here is "
                    "compared from memory."
                ),
                "allowed": True,
                "reason": "compare_needs_two",
                "mode": "compare",
                "ungrounded_numbers": [],
            })
        from agents.rulebased import RuleBasedInvestigator

        rows = [
            _assess_one(EdgarClient(), RuleBasedInvestigator(), c, as_of or date.today())
            for c in targets[:MAX_BATCH]
        ]
        screened = build_screen(req.question, rows, as_of or date.today())
        return JSONResponse({
            "question": req.question,
            "answer": screened.text,
            "allowed": True,
            "reason": "compare",
            "mode": "compare",
            "caveats": screened.caveats,
            "rows": screened.rows,
            "ungrounded_numbers": [],
        })

    # Cross-company questions cannot be served from one memo.
    if is_screening(req.question):
        if not req.ciks:
            return JSONResponse({
                "question": req.question,
                "answer": (
                    "That question spans companies and this panel is scoped to one "
                    "assessment. Load a portfolio and it will be answered from "
                    "measured assessments of those filers, not from general "
                    "knowledge, because nothing outside a verified assessment can "
                    "be checked."
                ),
                "allowed": True,
                "reason": "screening_needs_portfolio",
                "ungrounded_numbers": [],
            })
        from agents.rulebased import RuleBasedInvestigator

        rows = [
            _assess_one(EdgarClient(), RuleBasedInvestigator(), c, as_of or date.today())
            for c in req.ciks[:MAX_BATCH]
        ]
        screened = build_screen(req.question, rows, as_of or date.today())
        return JSONResponse({
            "question": req.question,
            "answer": screened.text,
            "allowed": True,
            "reason": "screening",
            "caveats": screened.caveats,
            "rows": screened.rows,
            "ungrounded_numbers": [],
        })

    # Load the history *before* answering when the question needs it.
    #
    # Order matters and it is the whole mitigation. The memo carries one figure
    # per measure, so a question about movement has no true answer available
    # from it -- and a model told to write at length with nothing further to
    # say is a model being invited to invent. Fetching the real series first
    # means "explain in more depth" is served by more evidence rather than
    # more prose, and the grounding check then has something to validate the
    # answer's numbers against instead of blocking every one of them.
    assessment = req.assessment
    # A registered formula the memo did not surface is one function call away.
    # Computing it here keeps the arithmetic in Python and out of the model,
    # which is the whole reason the answer can be trusted.
    if cik and as_of and assessment.get("memo"):
        # Every metric named anywhere in this conversation, not just in the
        # latest question. Turn 1 asks for the quick ratio and gets 0.74; turn
        # 3 says "compare it to the current ratio" and names nothing, so
        # recomputing from that question alone drops 0.74 out of the trusted
        # set -- and the follow-up is blocked for quoting a figure this system
        # produced two turns earlier.
        asked = " ".join(
            [req.question, *[str(t.get("question", "")) for t in (req.history or [])]]
        )
        extra_metrics = _compute_on_request(int(cik), as_of, asked, assessment)
        if extra_metrics:
            assessment = {**assessment, "memo": {
                **assessment["memo"],
                "sections": [*assessment["memo"]["sections"], extra_metrics],
            }}

    if cik and as_of and wants_trends(req.question):
        series = _trend_series(int(cik), as_of, assessment)
        if series:
            assessment = {**assessment, "trends": series}

    with tracing.span("ask", kind="chain", input={"question": req.question,
                                                   "cik": cik}) as qtrace:
        # Scoped, not trusted. History is only carried when it belongs to the
        # same filer at the same date as the assessment now loaded.
        scoped = [
            t for t in (req.history or [])
            if str(t.get("cik", cik)) == str(cik)
            and str(t.get("as_of", as_of_raw)) == str(as_of_raw)
        ]
        answer = ask(CachingClient(inner=client), assessment, req.question, history=scoped)
        total = answer.figures_verified + answer.figures_untagged
        tracing.update(qtrace, output={
            "allowed": answer.allowed, "reason": answer.reason,
            "answer": answer.text[:2000],
        })
        tracing.score("answer_allowed", answer.allowed)
        if answer.reason:
            tracing.score("guard_reason", answer.reason[:120])
        if total:
            # The number that says how much of the answer was actually
            # checkable, rather than merely unblocked.
            tracing.score("citation_coverage", answer.figures_verified / total,
                          f"{answer.figures_verified} of {total} figures bound to "
                          "a measure and period")
        tracing.flush()

    # A reader who asks to see a trend gets the trend, not a paragraph about
    # it. The UI renders whatever metric is named here.
    chart = None
    if cik and as_of:
        from agents.qa import wants_chart

        metric = wants_chart(req.question)
        if metric:
            chart = {"cik": int(cik), "as_of": as_of.isoformat(), "metric": metric}

    return JSONResponse(
        {
            "question": req.question,
            "answer": answer.text if answer.allowed else "",
            "allowed": answer.allowed,
            "reason": answer.reason,
            "mode": "company",
            # Surfaced, not hidden. A partial answer had a claim removed and a
            # repaired one needed a second pass -- a reader is entitled to know
            # which of the three they are looking at.
            "partial": answer.partial,
            "repaired": answer.repaired,
            "dropped_claims": answer.dropped_claims,
            "ungrounded_numbers": answer.ungrounded_numbers,
            # Reported, not assumed. A figure the model tagged was checked
            # against its measure AND its period; an untagged one passed the
            # weaker "appears in the assessment" test only, and the reader is
            # entitled to know which of the two they are looking at.
            "figures_verified": answer.figures_verified,
            "figures_untagged": answer.figures_untagged,
            "chart": chart,
        }
    )


if __name__ == "__main__":
    import uvicorn

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"CreditPulse IQ -> http://127.0.0.1:{port}", file=sys.stderr)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
