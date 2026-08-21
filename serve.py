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

import sys
from datetime import date
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from agents.llm import load_env_file
from agents.orchestrator import Orchestrator
from data.edgar import EdgarClient

app = FastAPI(title="CreditPulse IQ", docs_url="/api/docs")
INDEX = Path(__file__).parent / "web" / "index.html"

#: Measured on the 200-case L3 backtest. Shown in the UI so the number a user
#: is looking at arrives with the evidence for how much to trust it.
ARM_AUC = {"rules": 0.885, "react": 0.965}
HAZARD_AUC = 0.966


class AssessRequest(BaseModel):
    cik: int
    as_of: date
    agent: str = "rules"


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

    result = Orchestrator(investigator).run(req.cik, req.as_of, facts, **extra)

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


if __name__ == "__main__":
    import uvicorn

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"CreditPulse IQ -> http://127.0.0.1:{port}", file=sys.stderr)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
