"""Smoke-test validator for stt-local output (used by GitHub Actions CI).

Asserts that a transcription run produced well-formed srt/md/json outputs
with actual English speech, then prints a short report.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def main() -> int:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "tests/out")
    if not out_dir.is_dir():
        print(f"FAIL: output dir not found: {out_dir}")
        return 1

    stems = sorted({p.stem for p in out_dir.glob("sample.*")})
    if not stems:
        print("FAIL: no output files found")
        return 1
    stem = stems[0]

    problems: list[str] = []
    for ext in ("srt", "md", "json"):
        f = out_dir / f"{stem}.{ext}"
        if not f.is_file() or f.stat().st_size == 0:
            problems.append(f"missing/empty {f.name}")

    srt = (out_dir / f"{stem}.srt").read_text(encoding="utf-8")
    if " --> " not in srt:
        problems.append("srt has no timing lines")

    data = json.loads((out_dir / f"{stem}.json").read_text(encoding="utf-8"))
    segs = data.get("segments") or []
    if not segs:
        problems.append("json has no segments")
    else:
        text = " ".join(s["text"] for s in segs)
        if len(text.split()) < 5:
            problems.append("transcript too short for an English speech sample")
        if not re.search(r"[A-Za-z]{3,}", text):
            problems.append("transcript contains no real English words")
        if all(not s.get("words") for s in segs):
            problems.append("--word-timestamps produced no word timestamps")

    if problems:
        for p in problems:
            print(f"FAIL: {p}")
        return 1

    print("SMOKE OK")
    print(f"  segments : {len(segs)}")
    print(f"  duration : {data.get('duration')}s")
    print(f"  language : {data.get('language')}")
    print(f"  first 80 : {segs[0]['text'][:80]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
