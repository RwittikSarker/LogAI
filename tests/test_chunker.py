import pytest

from app.chunker import Chunk, render_chunk, split_log, TRUNC_MARK
from app.tokens import ApproxTokenCounter

COUNTER = ApproxTokenCounter(3.0)


def write(tmp_path, text: str | bytes, name="t.log"):
    p = tmp_path / name
    p.write_bytes(text if isinstance(text, bytes) else text.encode("utf-8"))
    return p


def nginx(n):
    return f'10.0.0.{n % 250} - - [07/Nov/2026:10:00:{n % 60:02d} +0000] "GET /p?id={n} HTTP/1.1" 200 512'


TRACE = (
    "2026-11-07 09:02:44,102 ERROR [svc] Unhandled exception\n"
    "Traceback (most recent call last):\n"
    '  File "/srv/a.py", line 8, in f\n'
    "    x = y()\n"
    "pkg.errors.Boom: it broke\n"
)


def test_every_line_covered_exactly_once(tmp_path):
    lines = [nginx(i) for i in range(300)]
    p = write(tmp_path, "\n".join(lines) + "\n")
    chunks = split_log(p, COUNTER, max_tokens=200)
    assert chunks[0].start_line == 1
    assert chunks[-1].end_line == 300
    for a, b in zip(chunks, chunks[1:]):
        assert b.start_line == a.end_line + 1
        assert b.byte_start == a.byte_end
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_no_chunk_over_budget(tmp_path):
    p = write(tmp_path, "\n".join(nginx(i) for i in range(500)) + "\n")
    chunks = split_log(p, COUNTER, max_tokens=150)
    assert len(chunks) > 5
    assert all(c.tokens <= 150 for c in chunks)


def test_multiline_entry_never_cut_when_it_fits(tmp_path):
    body = []
    for i in range(40):
        body.append(nginx(i))
        if i % 10 == 5:
            body.append(TRACE.rstrip("\n"))
    p = write(tmp_path, "\n".join(body) + "\n")
    chunks = split_log(p, COUNTER, max_tokens=200)
    for c in chunks:
        text = render_chunk(p, c).split("\n")
        # a traceback header and its last line must be in the same chunk
        for i, line in enumerate(text):
            if "Traceback (most recent" in line:
                assert any("pkg.errors.Boom" in t for t in text[i:]), "traceback was cut"
        assert not text[0].split("|", 1)[1].startswith("pkg.errors"), "chunk starts mid-entry"
    assert not any(c.split_entry for c in chunks)


def test_oversized_entry_is_split_and_flagged(tmp_path):
    huge = "2026-11-07 09:00:00 ERROR boom\n" + "".join(f'  File "/x{i}.py", line {i}, in f\n' for i in range(80))
    p = write(tmp_path, nginx(1) + "\n" + huge + nginx(2) + "\n")
    chunks = split_log(p, COUNTER, max_tokens=120)
    assert any(c.split_entry for c in chunks)
    assert all(c.tokens <= 120 for c in chunks)
    assert chunks[-1].end_line == 83


def test_render_matches_file_lines_and_numbers(tmp_path):
    lines = [nginx(i) for i in range(60)]
    p = write(tmp_path, "\n".join(lines) + "\n")
    chunks = split_log(p, COUNTER, max_tokens=120)
    seen = []
    for c in chunks:
        for row in render_chunk(p, c).split("\n"):
            label, text = row.split("|", 1)
            no = int(label[1:])
            seen.append(no)
            assert text == lines[no - 1]
    assert seen == list(range(1, 61))


def test_crlf_unicode_and_no_trailing_newline(tmp_path):
    raw = "2026-11-07 09:00:00 INFO héllo wörld ✓\r\n2026-11-07 09:00:01 INFO plain\r\n2026-11-07 09:00:02 INFO last".encode("utf-8")
    p = write(tmp_path, raw)
    chunks = split_log(p, COUNTER, max_tokens=500)
    assert len(chunks) == 1 and chunks[0].end_line == 3
    rows = render_chunk(p, chunks[0]).split("\n")
    assert rows[0] == "L1|2026-11-07 09:00:00 INFO héllo wörld ✓"
    assert rows[2] == "L3|2026-11-07 09:00:02 INFO last"


def test_byte_offsets_correct_after_multibyte_lines(tmp_path):
    lines = [f"2026-11-07 09:00:{i % 60:02d} INFO ünïcödé ✓ line {i}" for i in range(80)]
    p = write(tmp_path, "\n".join(lines) + "\n")
    chunks = split_log(p, COUNTER, max_tokens=100)
    assert len(chunks) > 3
    last = render_chunk(p, chunks[-1]).split("\n")[-1]
    assert last == f"L80|{lines[-1]}"


def test_long_line_truncated_and_control_chars_removed(tmp_path):
    p = write(tmp_path, "2026-11-07 09:00:00 INFO " + "A" * 5000 + "\x1b[31m\x00\n")
    chunks = split_log(p, COUNTER, max_tokens=900, )
    row = render_chunk(p, chunks[0])
    assert row.endswith(TRUNC_MARK)
    assert "\x1b" not in row and "\x00" not in row
    assert chunks[0].tokens <= 900


def test_empty_and_blank_files(tmp_path):
    assert split_log(write(tmp_path, ""), COUNTER) == []
    chunks = split_log(write(tmp_path, "\n\n\n", "b.log"), COUNTER)
    assert chunks and chunks[-1].end_line == 3


def test_unrecognised_format_falls_back_to_indent_mode(tmp_path):
    text = "".join(f"event {i} something happened\n" + (f"   detail {i}\n" if i % 3 == 0 else "") for i in range(50))
    p = write(tmp_path, text)
    chunks = split_log(p, COUNTER, max_tokens=60)
    # a continuation (indented) line must never start a chunk
    for c in chunks:
        first = render_chunk(p, c).split("\n")[0].split("|", 1)[1]
        assert not first.startswith(" ")


def test_chunk_is_serialisable():
    c = Chunk(0, 1, 5, 0, 100, 50, 3, False)
    assert c.to_dict()["end_line"] == 5 and c.line_count == 5
