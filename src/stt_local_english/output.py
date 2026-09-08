"""Output serializers: SRT, VTT, MD, TXT, JSON."""

from __future__ import annotations

from .engine import Transcript


def _fmt_srt(seconds: float) -> str:
    ms = int(round((seconds - int(seconds)) * 1000))
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _fmt_vtt(seconds: float) -> str:
    ms = int(round((seconds - int(seconds)) * 1000))
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def render_srt(t: Transcript, max_chars: int = 80) -> str:
    lines: list[str] = []
    idx = 1
    for seg in t.segments:
        if _words_trustworthy(seg):
            for start, end, text in _word_cues(seg, max_chars):
                lines += _cue(idx, start, end, text)
                idx += 1
        else:
            for start, end, text in _proportional_cues(seg, max_chars):
                lines += _cue(idx, start, end, text)
                idx += 1
    return "\n".join(lines)


def _words_trustworthy(seg) -> bool:
    """mlx sometimes duplicates words at segment edges; detect via length ratio."""
    if not seg.words:
        return False
    joined = " ".join(w["word"] for w in seg.words).strip()
    if not joined:
        return False
    norm = lambda s: "".join(c for c in s.lower() if c.isalnum() or c.isspace())
    j, txt = norm(joined), norm(seg.text)
    return 0.55 <= len(j) / max(len(txt), 1) <= 1.45


def _word_cues(seg, max_chars: int):
    """Yield (start, end, text) cues built from word-level timestamps."""
    cue_words: list[str] = []
    cue_len = 0
    cue_start: float | None = None
    cue_end = 0.0
    for w in seg.words:
        if cue_words and cue_len + len(w["word"]) + 1 > max_chars:
            yield cue_start or seg.start, cue_end, " ".join(cue_words).strip()
            cue_words, cue_len, cue_start = [], 0, None
        if cue_start is None:
            cue_start = w["start"]
        cue_words.append(w["word"])
        cue_len += len(w["word"]) + 1
        cue_end = w["end"]
    if cue_words:
        yield cue_start or seg.start, cue_end, " ".join(cue_words).strip()


def _proportional_cues(seg, max_chars: int):
    """Split segment text into readable chunks with time proportional to length."""
    text = seg.text
    words = text.split()
    if not words:
        return
    if len(text) <= max_chars:
        yield seg.start, seg.end, text
        return
    # break on sentence boundaries where possible
    chunks: list[list[str]] = []
    cur: list[str] = []
    cur_len = 0
    for w in words:
        nxt_len = cur_len + len(w) + 1
        if cur and nxt_len > max_chars and any(tok.endswith((".", "!", "?")) for tok in cur):
            chunks.append(cur)
            cur, cur_len = [], 0
        if cur and nxt_len > max_chars:
            chunks.append(cur)
            cur, cur_len = [], 0
        cur.append(w)
        cur_len += len(w) + 1
    if cur:
        chunks.append(cur)
    span = seg.end - seg.start
    total = max(sum(len(w) + 1 for w in words), 1)
    acc = 0
    for i, ch in enumerate(chunks):
        clen = sum(len(w) + 1 for w in ch)
        s = seg.start + span * acc / total
        acc += clen
        e = seg.start + span * acc / total if i < len(chunks) - 1 else seg.end
        yield max(s, seg.start), min(e, seg.end), " ".join(ch).strip()


def _cue(idx: int, start: float, end: float, text: str) -> list[str]:
    return [str(idx), f"{_fmt_srt(start)} --> {_fmt_srt(end)}", text, ""]


def render_vtt(t: Transcript) -> str:
    lines = ["WEBVTT", ""]
    for seg in t.segments:
        lines += [f"{_fmt_vtt(seg.start)} --> {_fmt_vtt(seg.end)}", seg.text, ""]
    return "\n".join(lines)


def render_md(t: Transcript) -> str:
    paras, cur = [], []
    for seg in t.segments:
        if not cur:
            cur.append(seg)
        elif seg.start - cur[-1].end <= 1.0:
            cur.append(seg)
        else:
            paras.append(" ".join(s.text for s in cur))
            cur = [seg]
    if cur:
        paras.append(" ".join(s.text for s in cur))
    header = f"> Source: `{t.path.name}`  ·  Language: {t.language}  ·  Duration: {t.duration:.0f}s  ·  Stt-local\n\n"
    return header + "\n\n".join(paras) + "\n"


def render_txt(t: Transcript) -> str:
    return "\n".join(s.text for s in t.segments) + "\n"


def render_json(t: Transcript) -> str:
    import json

    return json.dumps(
        {
            "source": str(t.path),
            "language": t.language,
            "duration": t.duration,
            "elapsed": round(t.elapsed, 2),
            "segments": [
                {
                    "start": round(s.start, 3),
                    "end": round(s.end, 3),
                    "text": s.text,
                    "words": s.words or None,
                }
                for s in t.segments
            ],
        },
        ensure_ascii=False,
        indent=2,
    ) + "\n"


RENDERERS = {
    "srt": render_srt,
    "vtt": render_vtt,
    "md": render_md,
    "txt": render_txt,
    "json": render_json,
}
