"""L0: filing-text and filing-event signals (data/signals.py).

The as-of tests are the ones that matter. These signals are dated events, and
a covenant breach disclosed the week after the prediction date is precisely the
kind of evidence that would make a backtest look brilliant and be worthless.
"""

from __future__ import annotations

from datetime import date

from data.signals import (
    EVENT_ITEMS,
    filing_events,
    latest_report,
    scan_report_text,
    strip_html,
)

AS_OF = date(2024, 6, 1)


def row(form: str, filed: date, items: str = "", doc: str = "d.htm") -> dict:
    return {
        "form": form,
        "items": items,
        "filing_date": filed,
        "accession": "0000000000-24-000001",
        "primary_document": doc,
    }


class TestAsOfDiscipline:
    def test_events_after_as_of_are_dropped(self) -> None:
        index = [
            row("8-K", date(2024, 5, 1), "2.04"),
            row("8-K", date(2024, 6, 2), "2.04"),  # one day late
        ]
        events = filing_events(index, AS_OF)
        assert [e.filing_date for e in events] == [date(2024, 5, 1)]

    def test_an_event_on_the_as_of_date_is_visible(self) -> None:
        """Filed on the day is public on the day; excluding it would be wrong."""
        events = filing_events([row("8-K", AS_OF, "3.01")], AS_OF)
        assert len(events) == 1

    def test_events_before_the_window_are_dropped(self) -> None:
        old = row("8-K", date(2020, 1, 1), "4.01")
        assert filing_events([old], AS_OF, lookback_days=540) == []

    def test_latest_report_never_returns_a_future_filing(self) -> None:
        index = [
            row("10-K", date(2024, 3, 1)),
            row("10-Q", date(2024, 8, 1)),  # after as_of
        ]
        found = latest_report(index, AS_OF)
        assert found is not None
        assert found["filing_date"] == date(2024, 3, 1)


class TestEventExtraction:
    def test_recognised_8k_items_are_categorised(self) -> None:
        events = filing_events([row("8-K", date(2024, 5, 1), "2.04,9.01")], AS_OF)
        assert len(events) == 1
        assert events[0].code == "2.04"
        assert "covenant" in events[0].description

    def test_bankruptcy_item_is_excluded(self) -> None:
        """1.03 is the outcome; get_prior_distress_events owns it."""
        assert "1.03" not in EVENT_ITEMS
        assert filing_events([row("8-K", date(2024, 5, 1), "1.03")], AS_OF) == []

    def test_late_filing_notice_is_an_event(self) -> None:
        events = filing_events([row("NT 10-K", date(2024, 4, 1))], AS_OF)
        assert events[0].code == "late_filing"

    def test_unrelated_8k_items_are_ignored(self) -> None:
        assert filing_events([row("8-K", date(2024, 5, 1), "7.01,9.01")], AS_OF) == []

    def test_events_are_returned_in_date_order(self) -> None:
        index = [
            row("8-K", date(2024, 5, 20), "3.01"),
            row("8-K", date(2024, 2, 2), "4.01"),
        ]
        dates = [e.filing_date for e in filing_events(index, AS_OF)]
        assert dates == sorted(dates)


class TestTextScan:
    def test_going_concern_is_found_and_quoted(self) -> None:
        text = (
            "<p>These conditions raise <b>substantial doubt</b> about the "
            "Company's ability to continue as a going concern.</p>"
        )
        found = scan_report_text(text)
        assert found["going_concern_doubt"] is True
        assert "going concern" in found["going_concern_quote"].lower()

    def test_a_healthy_filing_reports_no_doubt(self) -> None:
        found = scan_report_text("<p>Liquidity remains strong.</p>")
        assert found["going_concern_doubt"] is False
        assert found["going_concern_quote"] == ""

    def test_material_weakness_is_detected(self) -> None:
        text = "We identified a material weakness in our internal control over financial reporting."
        assert scan_report_text(text)["material_weakness"] is True

    def test_script_and_style_content_is_discarded(self) -> None:
        """Boilerplate scripts must not trip the phrase match."""
        text = "<script>var s='ability to continue as a going concern';</script><p>Fine.</p>"
        assert scan_report_text(text)["going_concern_doubt"] is False

    def test_tags_between_words_do_not_break_the_match(self) -> None:
        text = "ability to <i>continue</i> as a <b>going concern</b>"
        assert scan_report_text(text)["going_concern_doubt"] is True

    def test_strip_html_collapses_entities_and_whitespace(self) -> None:
        assert strip_html("<p>a&nbsp;&amp;\n\n  b</p>").strip() == "a & b"
