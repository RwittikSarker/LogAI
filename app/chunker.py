"""Split a log into chunks that fit the token budget.

Design goals (from the assignment): keep original line numbers, avoid cutting
log entries, never hold extra copies of the log in memory.

* The file is streamed once. We keep only OFFSETS (a few numbers per chunk),
  never the text. `render_chunk` re-reads one chunk from disk when it is needed.
* An ENTRY is a log record that may span several lines (a stack trace). Lines
  are grouped into entries, and entries are packed into chunks whole.
* If one entry alone is bigger than the budget it is split between lines and the
  chunk is flagged `split_entry` (unavoidable, but visible).
* Each rendered line is prefixed `L<number>|` so the model can cite exact lines.
  The token counts include that prefix.
* No overlap between chunks. Overlap would make the same line count twice in the
  ledger; continuity across chunks comes from the ledger instead.

Known limit: entry grouping is a heuristic (see _START_PATTERNS). On an
unrecognised format it falls back to "indented lines continue the previous
line", and in the worst case every line is its own entry. It never breaks
correctness, only how nicely chunks are cut.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, List

from app.tokens import TokenCounter

DEFAULT_MAX_CHUNK_TOKENS = 900
DEFAULT_MAX_LINE_CHARS = 600
TRUNC_MARK = "...[truncated]"

# A line matching one of these starts a NEW entry; anything else continues it.
_START_PATTERNS = [
    re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}"),          # 2026-11-07 09:02:44 / ISO-8601
    re.compile(r"^[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}"),  # syslog: Nov  7 08:20:01
    re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}\s"),                  # nginx/apache: IP first
    re.compile(r"^\d{2}/[A-Z][a-z]{2}/\d{4}"),                  # 07/Nov/2026
    re.compile(r"^\d{2}:\d{2}:\d{2}"),                          # 09:02:44
    re.compile(r"^\["),                                         # [timestamp] ... / [error]
]
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")  # control chars (ANSI escapes, NUL...) -> space


@dataclass(frozen=True)
class Chunk:
    index: int
    start_line: int      # 1-based, inclusive
    end_line: int        # inclusive
    byte_start: int
    byte_end: int        # exclusive
    tokens: int          # estimated prompt tokens of the rendered chunk
    entries: int         # entries that BEGIN in this chunk
    split_entry: bool    # contains part of an entry too big to keep whole

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _Line:
    no: int
    start: int
    end: int
    tokens: int


def clean_line(raw: bytes, max_line_chars: int = DEFAULT_MAX_LINE_CHARS) -> str:
    """Bytes of one line -> safe text. Used by BOTH counting and rendering."""
    text = raw.rstrip(b"\r\n").decode("utf-8", errors="replace")
    text = _CTRL.sub(" ", text)
    if len(text) > max_line_chars:
        text = text[:max_line_chars] + TRUNC_MARK
    return text


def _is_start(text: str) -> bool:
    return any(p.match(text) for p in _START_PATTERNS)


def _detect_pattern_mode(path: Path, sample: int = 200) -> bool:
    """True if most lines look like entry starts (so non-matching lines are continuations)."""
    starts = total = 0
    with open(path, "rb") as f:
        for i, raw in enumerate(f):
            if i >= sample:
                break
            text = clean_line(raw)
            if not text.strip():
                continue
            total += 1
            starts += _is_start(text)
    return total > 0 and starts / total >= 0.6


def _is_continuation(text: str, pattern_mode: bool) -> bool:
    if not text.strip():
        return True  # blank lines stay with the previous entry
    if pattern_mode:
        return not _is_start(text)
    return text[0] in (" ", "\t")


def _iter_entries(path: Path, counter: TokenCounter, max_line_chars: int, pattern_mode: bool) -> Iterator[List[_Line]]:
    current: List[_Line] = []
    offset = 0
    with open(path, "rb") as f:
        for no, raw in enumerate(f, start=1):
            start, end = offset, offset + len(raw)
            offset = end
            text = clean_line(raw, max_line_chars)
            tokens = counter.count(f"L{no}|{text}") + 1  # +1 for the newline
            line = _Line(no, start, end, tokens)
            if current and _is_continuation(text, pattern_mode):
                current.append(line)
            else:
                if current:
                    yield current
                current = [line]
    if current:
        yield current


def split_log(
    path: Path | str,
    counter: TokenCounter,
    max_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
    max_line_chars: int = DEFAULT_MAX_LINE_CHARS,
) -> List[Chunk]:
    """Return the chunk plan for `path` (offsets only, no log text)."""
    path = Path(path)
    pattern_mode = _detect_pattern_mode(path)
    chunks: List[Chunk] = []
    cur: List[_Line] = []
    cur_tokens = cur_entries = 0
    cur_split = False

    def flush() -> None:
        nonlocal cur, cur_tokens, cur_entries, cur_split
        if cur:
            chunks.append(Chunk(
                index=len(chunks), start_line=cur[0].no, end_line=cur[-1].no,
                byte_start=cur[0].start, byte_end=cur[-1].end,
                tokens=cur_tokens, entries=cur_entries, split_entry=cur_split,
            ))
        cur, cur_tokens, cur_entries, cur_split = [], 0, 0, False

    for entry in _iter_entries(path, counter, max_line_chars, pattern_mode):
        entry_tokens = sum(l.tokens for l in entry)
        if entry_tokens > max_tokens:
            # Too big to keep whole: pack its lines, flag the chunks.
            flush()
            for i, line in enumerate(entry):
                if cur and cur_tokens + line.tokens > max_tokens:
                    flush()
                cur.append(line)
                cur_tokens += line.tokens
                cur_split = True
                if i == 0:
                    cur_entries += 1
            continue  # tail stays open; following entries may join it
        if cur and cur_tokens + entry_tokens > max_tokens:
            flush()
        cur.extend(entry)
        cur_tokens += entry_tokens
        cur_entries += 1
    flush()
    return chunks


def render_chunk(path: Path | str, chunk: Chunk, max_line_chars: int = DEFAULT_MAX_LINE_CHARS) -> str:
    """Read ONE chunk from disk and return it as `L<no>|<text>` lines."""
    with open(path, "rb") as f:
        f.seek(chunk.byte_start)
        data = f.read(chunk.byte_end - chunk.byte_start)
    parts = data.split(b"\n")  # not splitlines(): must match how the file was iterated
    if parts and parts[-1] == b"":
        parts.pop()
    return "\n".join(
        f"L{chunk.start_line + i}|{clean_line(part, max_line_chars)}" for i, part in enumerate(parts)
    )
