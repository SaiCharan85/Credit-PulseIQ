"""L0: the two filing-signal tools on the ToolBox.

The behaviour these lock down is the one that is easy to get wrong: when the
data cannot be retrieved, the tool must say so. Returning "no going-concern
doubt found" after a failed fetch reads as reassurance, and reassurance drawn
from nothing is worse than an error -- it is the silent-failure-on-missing-data
mode this project exists to catch.
"""

from __future__ import annotations

from datetime import date

from agents.tools import ToolBox, tool_schemas

AS_OF = date(2024, 6, 1)

GOING_CONCERN_DOC = (
    "<p>These conditions raise substantial doubt about the Company's "
    "ability to continue as a going concern.</p>"
)


def index_row(form: str, filed: date, items: str = "") -> dict:
    return {
        "form": form,
        "items": items,
        "filing_date": filed,
        "accession": "0001-24-000001",
        "primary_document": "d.htm",
    }


def box(index=(), fetch=None) -> ToolBox:
    return ToolBox(
        cik=1, as_of=AS_OF, facts=[], filing_index=index, fetch_document=fetch
    )


class TestFilingEventsTool:
    def test_reports_events_inside_the_window(self) -> None:
        tb = box([index_row("8-K", date(2024, 3, 1), "2.04")])
        result = tb.get_filing_events()
        assert result["n_events"] == 1
        assert result["events"][0]["code"] == "2.04"

    def test_never_returns_a_filing_after_as_of(self) -> None:
        tb = box([index_row("8-K", date(2024, 9, 1), "2.04")])
        assert tb.get_filing_events()["n_events"] == 0

    def test_as_of_is_not_an_argument(self) -> None:
        """Lookahead is unrepresentable, not merely rejected (PROMPT rule 4)."""
        schema = next(
            t for t in tool_schemas() if t["function"]["name"] == "get_filing_events"
        )
        assert "as_of" not in schema["function"]["parameters"]["properties"]

    def test_absent_index_is_an_error_not_an_empty_result(self) -> None:
        """"No events" and "could not look" must not be the same answer."""
        result = box([]).get_filing_events()
        assert "error" in result
        assert "n_events" not in result

    def test_lookback_is_clamped(self) -> None:
        tb = box([index_row("8-K", date(2024, 3, 1), "2.04")])
        assert tb.get_filing_events(lookback_days=99999)["lookback_days"] <= 1826

    def test_the_call_is_recorded_in_the_audit_trail(self) -> None:
        tb = box([index_row("NT 10-K", date(2024, 5, 1))])
        tb.get_filing_events()
        assert [c["tool"] for c in tb.audit_trail()] == ["get_filing_events"]


class TestGoingConcernTool:
    def test_detects_doubt_in_the_latest_report(self) -> None:
        tb = box([index_row("10-K", date(2024, 3, 1))], lambda a, d: GOING_CONCERN_DOC)
        result = tb.check_going_concern()
        assert result["going_concern_doubt"] is True
        assert result["filing_date"] == "2024-03-01"

    def test_reads_the_latest_report_filed_before_as_of(self) -> None:
        seen: list[str] = []

        def fetch(accession: str, document: str) -> str:
            seen.append(accession)
            return "<p>All is well.</p>"

        tb = ToolBox(
            cik=1,
            as_of=AS_OF,
            facts=[],
            filing_index=[
                {**index_row("10-K", date(2024, 3, 1)), "accession": "OLD"},
                {**index_row("10-Q", date(2024, 8, 1)), "accession": "FUTURE"},
            ],
            fetch_document=fetch,
        )
        tb.check_going_concern()
        assert seen == ["OLD"]

    def test_a_failed_fetch_is_an_error_not_a_clean_bill_of_health(self) -> None:
        def boom(accession: str, document: str) -> str:
            raise RuntimeError("503 from EDGAR")

        result = box([index_row("10-K", date(2024, 3, 1))], boom).check_going_concern()
        assert "error" in result
        assert "going_concern_doubt" not in result

    def test_no_report_before_as_of_is_an_error(self) -> None:
        tb = box([index_row("10-K", date(2024, 12, 1))], lambda a, d: "")
        assert "error" in tb.check_going_concern()

    def test_missing_fetcher_is_an_error(self) -> None:
        tb = box([index_row("10-K", date(2024, 3, 1))], None)
        assert "error" in tb.check_going_concern()


class TestSchemaWiring:
    def test_both_tools_are_offered(self) -> None:
        names = [t["function"]["name"] for t in tool_schemas()]
        assert "get_filing_events" in names
        assert "check_going_concern" in names

    def test_the_investigator_can_dispatch_them(self) -> None:
        from agents.distress import DistressInvestigator

        inv = DistressInvestigator(client=None)
        tb = box([index_row("8-K", date(2024, 3, 1), "3.01")])
        assert inv._dispatch(tb, "get_filing_events", {})["n_events"] == 1
