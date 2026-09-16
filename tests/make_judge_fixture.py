"""Build a small polished/original fixture pair for the judge CI smoke test."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

OUT = Path(__file__).parent / "out"
FIXTURE = Path(__file__).parent / "judge-fixture"


def main() -> None:
    FIXTURE.mkdir(parents=True, exist_ok=True)
    for src in OUT.glob("*.json"):
        dst = FIXTURE / src.name
        shutil.copy(src, dst)
        d = json.loads(dst.read_text(encoding="utf-8"))
        d["polished"] = True
        segs = d.get("segments") or []
        if len(segs) >= 2:
            # simulate one polish change so the judge has a diff to review
            segs[1]["text"] = segs[1]["text"] + " [polished]"
        dst.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    n = len(list(FIXTURE.glob("*.json")))
    print(f"judge fixture ready: {n} file(s) in {FIXTURE}")


if __name__ == "__main__":
    main()
