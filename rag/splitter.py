"""Recursive, token-budgeted text splitting with overlap.

The same algorithm as LangChain's ``RecursiveCharacterTextSplitter``: cut at
the coarsest separator that yields pieces under budget, recurse into any piece
still too large with the next finer separator, then merge neighbours back up
to the budget with an overlap carried between chunks.

Written out rather than imported because importing ``langchain_text_splitters``
also imports ``sentence_transformers`` at package load, which on this machine
drags in a torch build incompatible with NumPy 2 plus TensorFlow -- forty
seconds and a large slice of RAM to split strings.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

DEFAULT_SEPARATORS = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", "")


class RecursiveSplitter:
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        length: Callable[[str], int],
        separators: Sequence[str] = DEFAULT_SEPARATORS,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk size")
        self.size = chunk_size
        self.overlap = chunk_overlap
        self.length = length
        self.separators = tuple(separators)

    def split_text(self, text: str) -> list[str]:
        return [c for c in (s.strip() for s in self._split(text, self.separators)) if c]

    # Pieces keep their separator at the end ("keep_separator='end'"), so a
    # sentence keeps its full stop and joining pieces restores the text.
    @staticmethod
    def _pieces(text: str, sep: str) -> list[str]:
        if sep == "":
            return list(text)
        parts = text.split(sep)
        return [p + sep for p in parts[:-1]] + [parts[-1]]

    def _split(self, text: str, separators: Sequence[str]) -> list[str]:
        sep, rest = separators[-1], ()
        for i, s in enumerate(separators):
            if s == "" or s in text:
                sep, rest = s, separators[i + 1 :]
                break
        good: list[str] = []
        out: list[str] = []
        for piece in self._pieces(text, sep):
            if not piece:
                continue
            if self.length(piece) <= self.size:
                good.append(piece)
                continue
            if good:
                out.extend(self._merge(good))
                good = []
            out.extend(self._split(piece, rest) if rest else [piece])
        if good:
            out.extend(self._merge(good))
        return out

    def _merge(self, pieces: list[str]) -> list[str]:
        """Pack pieces up to the budget; start each new chunk with an overlap tail."""
        chunks: list[str] = []
        window: list[str] = []
        lengths: list[int] = []
        total = 0
        for piece in pieces:
            n = self.length(piece)
            if window and total + n > self.size:
                chunks.append("".join(window))
                # Drop from the front until what remains fits as overlap and
                # leaves room for the incoming piece.
                while window and (total > self.overlap or total + n > self.size):
                    total -= lengths.pop(0)
                    window.pop(0)
            window.append(piece)
            lengths.append(n)
            total += n
        if window:
            chunks.append("".join(window))
        return chunks
