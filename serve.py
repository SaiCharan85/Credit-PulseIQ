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

import re
import sys
from datetime import date
from pathlib import Path

from fastapi import FastAPI, UploadFile
from fastapi import File as _File
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from agents.earnings_notes import earnings_notes
from agents.llm import load_env_file
from agents.orchestrator import Orchestrator
from agents.outcomes_lookup import load_outcomes, survived_note
from agents.schemas import SIGNAL_ORDER
from agents.screening import build as build_screen
from agents.screening import is_screening
from data.edgar import EdgarClient

app = FastAPI(title="CreditPulse IQ", docs_url="/api/docs")
INDEX = Path(__file__).parent / "web" / "index.html"

#: Measured on the 200-case L3 backtest. Shown in the UI so the number a user
#: is looking at arrives with the evidence for how much to trust it.
ARM_AUC = {"rules": 0.885, "react": 0.965}
HAZARD_AUC = 0.966

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
    cik: int
    as_of: date
    agent: str = "rules"


class BatchRequest(BaseModel):
    ciks: list[int]
    as_of: date
    agent: str = "rules"


class AskRequest(BaseModel):
    assessment: dict
    question: str
    #: Optional portfolio context for screening questions. Without it a
    #: cross-company question says what it would need rather than answering
    #: from the model's own knowledge, which nothing could check.
    ciks: list[int] = []


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX.read_text(encoding="utf8")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "arms": ARM_AUC, "hazard_baseline_auc": HAZARD_AUC}


@app.post("/api/assess")
def assess(req: AssessRequest) -> JSONResponse:
    load_env_file()
    edgar = EdgarClient()
    try:
        facts = edgar.facts(req.cik)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            {"error": f"could not load filings for CIK {req.cik}: {exc}"}, status_code=400
        )
    if not facts:
        return JSONResponse({"error": f"no XBRL facts for CIK {req.cik}"}, status_code=404)

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
            return JSONResponse(
                {"error": f"no model endpoint configured: {detail}"}, status_code=503
            )
        investigator = DistressInvestigator(CachingClient(inner=client))
        try:
            extra["filing_index"] = edgar.filing_index(req.cik)
            extra["fetch_document"] = lambda a, d: edgar.fetch_filing_document(req.cik, a, d)
        except Exception:  # noqa: BLE001
            pass
    else:
        from agents.rulebased import RuleBasedInvestigator

        investigator = RuleBasedInvestigator()

    # Earnings-quality observations ride along as context. Measured at
    # 0.51-0.61 AUC over six approaches, so they never move the graded signal.
    notes = earnings_notes(facts, req.as_of)
    result = Orchestrator(investigator, context_notes=notes).run(
        req.cik, req.as_of, facts, **extra
    )

    payload: dict = {
        "cik": req.cik,
        "as_of": req.as_of.isoformat(),
        "agent": req.agent,
        "arm_auc": ARM_AUC.get(req.agent),
        "hazard_baseline_auc": HAZARD_AUC,
        "n_facts": len(facts),
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
        return JSONResponse(payload)

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
    return JSONResponse(payload)


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
    result = Orchestrator(investigator).run(cik, as_of, facts)
    if not result.shipped:
        return {"cik": cik, "status": "blocked", "detail": result.blocked_reason[:160]}
    return {
        "cik": cik,
        "status": "ok",
        "signal": result.memo.signal,
        "risk_score": result.memo.risk_score,
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
    ciks = req.ciks[:MAX_BATCH]
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
            "requested": len(req.ciks),
            "assessed": sum(1 for r in rows if r["status"] == "ok"),
            "truncated": len(req.ciks) > MAX_BATCH,
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
    from data.signals import scan_report_text

    raw = (await file.read()).decode("utf8", errors="replace")
    if not raw.strip():
        return JSONResponse({"error": "empty file"}, status_code=400)
    found = scan_report_text(raw)
    return JSONResponse(
        {
            "filename": file.filename,
            "characters": len(raw),
            **found,
            "note": "deterministic phrase match over the filing text; no model involved",
        }
    )


@app.post("/api/ask")
def ask_question(req: AskRequest) -> JSONResponse:
    """Grounded Q&A over one finished assessment."""
    from agents.llm import CachingClient, RateLimitedClient, default_client, preflight
    from agents.qa import ask

    load_env_file()
    try:
        client = RateLimitedClient(inner=default_client())
        ok, detail = preflight(client)
    except Exception as exc:  # noqa: BLE001
        ok, detail = False, str(exc)
    if not ok:
        return JSONResponse({"error": f"no model endpoint configured: {detail}"}, status_code=503)

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

    answer = ask(CachingClient(inner=client), req.assessment, req.question)
    return JSONResponse(
        {
            "question": req.question,
            "answer": answer.text if answer.allowed else "",
            "allowed": answer.allowed,
            "reason": answer.reason,
            "ungrounded_numbers": answer.ungrounded_numbers,
        }
    )


if __name__ == "__main__":
    import uvicorn

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"CreditPulse IQ -> http://127.0.0.1:{port}", file=sys.stderr)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
