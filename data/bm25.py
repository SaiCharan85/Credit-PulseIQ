"""BM25 over the company directory, because a heuristic was the wrong tool.

Company resolution was a hand-rolled rule: take capitalised words, drop a
stopword list, resolve the first that matches. It failed in both directions at
once. "Can yu tll me" matched Canaan Inc. on three letters of a typo, and
"comapre this to Valaris" matched CNA Financial and never reached Valaris,
because a left-to-right scan takes whatever resolves first rather than whatever
fits best.

Both failures are the same missing thing: **a score**. A rule returns a match or
nothing, so there is no way to say one candidate fits better than another, and
no way to reject a weak match. BM25 ranks, which turns "did anything match" into
"what fits best, and is that good enough to act on".

What it buys here specifically:

* *Valaris* outranks *CNA Financial* for "comapre this to Valaris", because the
  scan is no longer positional -- the best candidate in the whole question wins.
* A weak best match can be rejected on score rather than on a length rule.
* Ambiguity is visible: two candidates within a whisker of each other is a
  question for the reader, not a coin flip.

Rare terms carry more weight than common ones, which is exactly right for
company names: "Valaris" appears in one name and "Corp" in thousands, so
matching the first means far more than matching the second. That falls out of
the IDF term rather than needing a stopword list.

Implemented directly rather than pulled in: it is thirty lines, the corpus is
ten thousand short strings, and a dependency for that is a poor trade.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

#: Standard BM25 parameters. k1 controls how fast term frequency saturates and
#: b how much document length is penalised. Company names are all short, so b
#: matters little; the defaults are kept because tuning them on ten thousand
#: two-word strings would be fitting noise.
K1 = 1.5
B = 0.75

#: Corporate furniture. Not removed -- IDF already discounts it to almost
#: nothing -- but a query made *entirely* of it should match nothing rather
#: than the longest list of Corps.
_FURNITURE = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "llc",
    "lp", "plc", "ltd", "limited", "holdings", "holding", "group", "the",
    "trust", "partners", "sa", "nv", "ag", "international", "intl", "class",
}

#: Ordinary English that happens to be somebody's company name.
#:
#: Feeding a whole question to the index found Here Group Ltd for "here",
#: Wheels Up for "up" and SOUTHERN CO for "so". BM25 was not wrong -- those are
#: genuinely the best matches for those tokens -- but a reader writing "so what
#: is the leverage here" is naming no company at all. In a corpus of ten
#: thousand names, common words have to be excluded from the query or every
#: sentence resolves to something.
_COMMON = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "cna",
    "compare", "compared", "could", "did", "do", "does", "explain", "for",
    "from", "give", "has", "have", "here", "how", "i", "if", "in", "is", "it",
    "its", "just", "me", "my", "no", "not", "now", "of", "ok", "on", "one",
    "or", "our", "out", "please", "rank", "risk", "say", "see", "should",
    "show", "so", "tell", "than", "that", "the", "their", "them", "then",
    "there", "these", "they", "this", "to", "up", "us", "was", "we", "were",
    "what", "when", "where", "which", "who", "why", "will", "with", "would",
    "you", "your", "also", "about", "against", "versus", "vs", "like", "more",
    "most", "much", "other", "others", "same", "some", "such", "very", "well",
}

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenise(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


@dataclass
class Hit:
    """One candidate, with the score that justifies it."""

    cik: int
    name: str
    ticker: str = ""
    score: float = 0.0

    @property
    def is_confident(self) -> bool:
        """Whether this is strong enough to act on without asking.

        The threshold is a floor on evidence, not on similarity: below it the
        query matched only common terms, and matching "Corp" is not knowing
        which company someone meant.
        """
        return self.score >= MIN_SCORE


#: How much of the query's total weight a candidate must account for.
#:
#: A *share*, not an absolute score. BM25 scores grow with corpus size through
#: IDF, so an absolute threshold tuned on ten thousand companies rejected
#: everything on a nine-row fixture -- and would have done the same on a
#: filtered watchlist, silently. Asking "what fraction of this query did the
#: candidate explain" has an answer that does not depend on how many companies
#: exist.
#:
#: 0.5 means a two-word query matching one word is not enough on its own, which
#: is the behaviour that stops "Chord" alone from claiming "Chord Energy Corp"
#: when the reader wrote something else entirely.
MIN_SCORE = 0.5

#: How far ahead the winner must be, on the same 0-1 share scale. Two
#: near-identical scores mean two plausible companies, and guessing between
#: them is how a reader gets confidently told about the wrong filer.
MIN_MARGIN = 0.1


@dataclass
class Index:
    """A BM25 index over company names and tickers."""

    docs: list[tuple[int, str, str, list[str]]] = field(default_factory=list)
    df: Counter = field(default_factory=Counter)
    avg_len: float = 0.0

    def add_all(self, rows) -> Index:
        for row in rows:
            name, ticker = str(row.get("name") or ""), str(row.get("ticker") or "")
            # The ticker is indexed with the name: readers use both, and a
            # ticker is the rarest possible term for its company.
            terms = tokenise(name) + ([ticker.lower()] if ticker else [])
            if not terms:
                continue
            self.docs.append((int(row["cik"]), name, ticker, terms))
            self.df.update(set(terms))
        self.avg_len = (
            sum(len(d[3]) for d in self.docs) / len(self.docs) if self.docs else 0.0
        )
        return self

    def _idf(self, term: str) -> float:
        n, df = len(self.docs), self.df.get(term, 0)
        if df == 0:
            return 0.0
        # The +0.5 smoothing keeps a term present in nearly every document from
        # going negative, which would make matching it actively harmful.
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, limit: int = 5) -> list[Hit]:
        """Best-matching companies, highest score first.

        ``Hit.score`` is normalised to the share of the query's own weight that
        the candidate accounts for, so 1.0 is a full match and the threshold
        does not move when the corpus does.
        """
        terms = [
            t for t in tokenise(query)
            if t not in _FURNITURE and t not in _COMMON and len(t) > 1
        ]
        if not terms:
            return []
        scored: list[Hit] = []
        for cik, name, ticker, doc in self.docs:
            counts = Counter(doc)
            score = 0.0
            for term in terms:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                norm = 1 - B + B * (len(doc) / (self.avg_len or 1))
                score += self._idf(term) * (tf * (K1 + 1)) / (tf + K1 * norm)
            if score > 0:
                scored.append(Hit(cik=cik, name=name, ticker=ticker, score=score))
        # Normalise by the best score any document could have achieved for this
        # query, so the number means "share of the query explained".
        ideal = sum(self._idf(t) * (K1 + 1) / (1 + K1 * (1 - B)) for t in terms)
        if ideal > 0:
            scored = [
                Hit(cik=h.cik, name=h.name, ticker=h.ticker,
                    score=min(1.0, h.score / ideal))
                for h in scored
            ]
        scored.sort(key=lambda h: (-h.score, len(h.name)))
        return scored[:limit]

    def resolve(self, query: str, exclude_cik: int | None = None) -> Hit | None:
        """The one company this phrase means, or None if that is not clear.

        Takes a *candidate phrase*, not a whole question -- see ``_COMMON`` for
        what happens otherwise. Returns nothing rather than a guess when the
        best match is weak or the runner-up is close behind: a wrong company
        confidently named is worse than admitting the question was ambiguous.
        """
        hits = [h for h in self.search(query, limit=4) if h.cik != exclude_cik]
        if not hits or not hits[0].is_confident:
            return None
        # Duplicate listings of one company (two share classes, two tickers)
        # are not ambiguity, so compare against the next *distinct* filer.
        rivals = [h for h in hits[1:] if h.name != hits[0].name]
        if rivals and hits[0].score - rivals[0].score < MIN_MARGIN:
            return None
        return hits[0]

    def best_in(self, question: str, candidates: list[str],
                exclude_cik: int | None = None) -> Hit | None:
        """The strongest company among several candidate phrases.

        Scoring every candidate and taking the maximum is what stops a typo
        early in a sentence from beating the company actually named later --
        the old scan was positional and returned whatever resolved first.
        """
        best: Hit | None = None
        for phrase in candidates:
            hit = self.resolve(phrase, exclude_cik=exclude_cik)
            if hit and (best is None or hit.score > best.score):
                best = hit
        return best


@lru_cache(maxsize=1)
def index() -> Index:
    """The directory, indexed once per process."""
    from data.company_search import load_directory

    try:
        return Index().add_all(load_directory())
    except Exception:  # noqa: BLE001 - a missing directory costs matching, not the app
        return Index()
