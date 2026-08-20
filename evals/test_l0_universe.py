"""L0: point-in-time universe construction from EDGAR form indexes.

The survivorship tests are the reason this module exists. A universe built from
today's listings quietly excludes every company that failed, and a model
measured on it looks far better than it is.
"""

from __future__ import annotations

from datetime import date

from data.universe import (
    IndexEntry,
    annual_filers,
    parse_form_index,
    quarters,
    sample_universe,
)

IDX = """Description:           Master Index of EDGAR Dissemination Feed by Form Type
Form Type   Company Name                          CIK         Date Filed  File Name
---------------------------------------------------------------------------------
10-K        ACME CORP                             320193      2019-02-01  edgar/data/1/a.txt
10-K        DOOMED HOLDINGS INC                   789019      2019-03-15  edgar/data/2/b.txt
8-K         NOISE CO                              111111      2019-02-02  edgar/data/3/c.txt
10-K/A      ACME CORP                             320193      2019-04-01  edgar/data/1/d.txt
20-F        FOREIGN PLC                           222222      2019-03-01  edgar/data/4/e.txt
"""


class TestParsing:
    def test_only_annual_forms_are_kept(self) -> None:
        got = parse_form_index(IDX)
        assert {e.form for e in got} == {"10-K", "10-K/A", "20-F"}
        assert 111111 not in {e.cik for e in got}

    def test_company_names_with_spaces_survive(self) -> None:
        got = {e.cik: e.company for e in parse_form_index(IDX)}
        assert got[789019] == "DOOMED HOLDINGS INC"

    def test_headers_and_rules_are_ignored(self) -> None:
        assert all(e.cik > 0 for e in parse_form_index(IDX))

    def test_garbage_lines_do_not_raise(self) -> None:
        assert parse_form_index("not an index at all\n\n   \n") == []


class TestSurvivorship:
    def test_a_company_that_later_delisted_is_still_in_its_own_quarter(self) -> None:
        """The whole point: membership is earned contemporaneously."""
        calls: list[str] = []

        def fetch(url: str) -> str:
            calls.append(url)
            return IDX

        found = annual_filers(date(2019, 1, 1), date(2019, 3, 31), fetch)
        assert 789019 in found, "DOOMED HOLDINGS filed a 10-K and must be in the universe"

    def test_earliest_filing_per_cik_is_kept(self) -> None:
        found = annual_filers(date(2019, 1, 1), date(2019, 12, 31), fetch=lambda u: IDX)
        assert found[320193].filed == date(2019, 2, 1)  # not the April 10-K/A

    def test_filings_outside_the_window_are_dropped(self) -> None:
        found = annual_filers(date(2019, 3, 1), date(2019, 3, 31), fetch=lambda u: IDX)
        assert set(found) == {789019, 222222}

    def test_a_missing_quarter_does_not_void_the_rest(self) -> None:
        def flaky(url: str) -> str:
            if "QTR1" in url:
                raise RuntimeError("404")
            return IDX

        found = annual_filers(date(2019, 1, 1), date(2019, 6, 30), flaky)
        assert found, "QTR2 should still have been read"


class TestQuarters:
    def test_spans_year_boundaries(self) -> None:
        assert quarters(date(2019, 11, 1), date(2020, 2, 1)) == [(2019, 4), (2020, 1)]

    def test_single_quarter(self) -> None:
        assert quarters(date(2020, 1, 5), date(2020, 3, 1)) == [(2020, 1)]


class TestSampling:
    def _filers(self, n: int) -> dict[int, IndexEntry]:
        return {
            i: IndexEntry("10-K", f"CO {i}", i, date(2019, 1, 1)) for i in range(1, n + 1)
        }

    def test_every_kept_cik_survives_sampling(self) -> None:
        got = sample_universe(self._filers(500), keep=[1, 2, 3], n_sample=10)
        assert {1, 2, 3} <= set(got)

    def test_sample_size_is_respected(self) -> None:
        got = sample_universe(self._filers(500), keep=[1, 2], n_sample=50)
        assert len(got) == 52

    def test_deterministic_for_a_seed(self) -> None:
        a = sample_universe(self._filers(500), keep=[1], n_sample=50, seed=7)
        b = sample_universe(self._filers(500), keep=[1], n_sample=50, seed=7)
        assert a == b

    def test_zero_sample_keeps_everything(self) -> None:
        got = sample_universe(self._filers(20), keep=[1], n_sample=0)
        assert len(got) == 20

    def test_a_kept_cik_is_never_double_counted(self) -> None:
        got = sample_universe(self._filers(100), keep=[5], n_sample=99)
        assert len(got) == len(set(got)) == 100
