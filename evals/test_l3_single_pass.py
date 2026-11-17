"""L3: the single-pass investigator keeps every guarantee the loop has.

Scripted model replies, real everything else: the pack is built from real
cached XBRL facts through the real ToolBox, passages come from a real
(in-memory) store, and the real critic judges the reply. Only the tokens are
fixed. What is pinned:

* a correct reply passes on the first call -- one call, not fourteen;
* a figure the pack never produced is caught by the unchanged critic;
* a misquoted passage is caught by the quote check and sent back;
* repeated failure ends in abstention, never in a shipped bad answer;
* nothing filed after the prediction date reaches the prompt.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from agents.llm import ScriptedClient
from agents.schemas import SIGNAL_INSUFFICIENT
from agents.single_pass import SinglePassInvestigator, check_quotes
from agents.tools import ToolBox
from data.edgar import EdgarClient
from rag.chunking import Chunk
from rag.embed import HashEmbedder
from rag.store import FilingStore, open_client

SLEEP_NUMBER = 827187
AS_OF = date(2026, 3, 1)  # petition was 2026-06-11
GOING = ("These conditions raise substantial doubt about the Company's ability to continue "
         "as a going concern within one year.")


@pytest.fixture(scope="module")
def facts():
    return EdgarClient(offline=True).facts(SLEEP_NUMBER)


@pytest.fixture()
def store():
    s = FilingStore(open_client("memory"), HashEmbedder())
    s.add([
        Chunk(cik=SLEEP_NUMBER, accession="past-10k", form="10-K", filing_date=date(2026, 2, 20),
              section="financials", chunk_no=0, text=GOING),
        Chunk(cik=SLEEP_NUMBER, accession="past-10q", form="10-Q", filing_date=date(2025, 11, 5),
              section="liquidity", chunk_no=0,
              text="We were not in compliance with our leverage covenant "
                   "and obtained a waiver from lenders."),
        Chunk(cik=SLEEP_NUMBER, accession="FUTURE-8K", form="8-K", filing_date=date(2026, 6, 11),
              section="full", chunk_no=0,
              text="The Company filed a voluntary petition under Chapter 11 of the Bankruptcy Code."),
    ])
    return s


def _metric(facts, name):
    return ToolBox(cik=SLEEP_NUMBER, as_of=AS_OF, facts=facts).get_metric(name)["value"]


def _reply(evidence, passages, **over) -> str:
    body = {
        "signal": "severe_risk", "risk_score": 82, "confidence": 0.5,
        "rationale": "Going-concern doubt and a covenant waiver alongside weak liquidity.",
        "residual": "", "evidence": evidence, "passages": passages,
    }
    body.update(over)
    return json.dumps(body)


def _passage_id(client: ScriptedClient, accession: str) -> str:
    prompt = client.calls[0][1]["content"]
    for line in prompt.splitlines():
        if line.startswith("[P") and accession in line:
            return line[1:line.index("]")]
    raise AssertionError(f"{accession} not in prompt")


def test_correct_reply_passes_in_one_call(facts, store):
    lta = _metric(facts, "liabilities_to_assets")
    probe = ScriptedClient(script=["{}"])
    SinglePassInvestigator(probe, store, max_retries=0).run(SLEEP_NUMBER, AS_OF, facts)
    pid = _passage_id(probe, "past-10k")

    client = ScriptedClient(script=[_reply(
        [{"metric": "liabilities_to_assets", "value": lta, "note": "high leverage"}],
        [{"id": pid,
          "quote": "substantial doubt about the Company's ability to continue as a going concern"}],
    )])
    out = SinglePassInvestigator(client, store).run(SLEEP_NUMBER, AS_OF, facts)
    assert out.signal == "severe_risk" and out.verification_passed
    assert len(client.calls) == 1
    assert out.audit_trail[-1]["tool"] == "cited_passages"


def test_future_filing_never_reaches_the_prompt(facts, store):
    client = ScriptedClient(script=["{}"])
    SinglePassInvestigator(client, store, max_retries=0).run(SLEEP_NUMBER, AS_OF, facts)
    prompt = client.calls[0][1]["content"]
    assert "FUTURE-8K" not in prompt and "Chapter 11" not in prompt
    assert "past-10k" in prompt


def test_invented_figure_is_rejected_by_the_unchanged_critic(facts, store):
    bad = _reply([{"metric": "made_up_ratio", "value": 1.23, "note": ""}], [])
    client = ScriptedClient(script=[bad, bad, bad])
    out = SinglePassInvestigator(client, store).run(SLEEP_NUMBER, AS_OF, facts)
    assert out.signal == SIGNAL_INSUFFICIENT and not out.verification_passed
    assert len(client.calls) == 3  # first try + two retries, then abstain
    assert "made_up_ratio" in out.verification_defects


def test_misquote_is_sent_back_and_corrected(facts, store):
    lta = _metric(facts, "liabilities_to_assets")
    probe = ScriptedClient(script=["{}"])
    SinglePassInvestigator(probe, store, max_retries=0).run(SLEEP_NUMBER, AS_OF, facts)
    pid = _passage_id(probe, "past-10k")
    ev = [{"metric": "liabilities_to_assets", "value": lta, "note": ""}]
    wrong = _reply(ev, [{"id": pid, "quote": "the company will certainly file for bankruptcy"}])
    right = _reply(ev, [{"id": pid, "quote": "raise substantial doubt"}])
    client = ScriptedClient(script=[wrong, right])
    out = SinglePassInvestigator(client, store).run(SLEEP_NUMBER, AS_OF, facts)
    assert out.verification_passed and out.retries == 1
    assert "[quote]" in client.calls[1][-1]["content"]


def test_unparseable_reply_ends_in_abstention(facts, store):
    client = ScriptedClient(script=["not json", "still not", "nope"])
    out = SinglePassInvestigator(client, store).run(SLEEP_NUMBER, AS_OF, facts)
    assert out.signal == SIGNAL_INSUFFICIENT


def test_quote_check_unit():
    passages = [{"id": "P1", "text": "We obtained a  waiver from our lenders."}]
    assert check_quotes({"passages": [{"id": "P1", "quote": "obtained a waiver"}]}, passages) == []
    assert check_quotes({"passages": [{"id": "P1", "quote": "lenders accelerated"}]}, passages)
    assert check_quotes({"passages": [{"id": "P9", "quote": "x"}]}, passages)
