"""Second-pass QA judge for polished transcripts.

Independent model reviews every correction (diff between original and
polished) and verdicts OK / SUSPECT with a reason.

Lessons baked in from the first manual judging round:
  * ids are validated by SET EQUALITY against the input ids (judges love
    renumbering) and normalized to int everywhere (JSON object keys are
    strings).
  * the judge prompt gets the SAME domain glossary as the polisher — without
    it, correct product-name fixes get flagged as "meaning change".
  * per-batch checkpoints so re-runs never re-bill completed batches.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .llm import ChatClient, LLMError
from .polish import batch_indices

log = logging.getLogger("stt-local.judge")

SYSTEM_PROMPT = (
    "You are a strict QA judge for ASR transcript corrections. You evaluate whether "
    "each correction is a legitimate transcription-error fix. You never rewrite text "
    "yourself and you never review your own edits."
)

USER_TMPL = """You are judging corrections made to an English transcript of a software-engineering
course video. For each item you get the original line, the corrected line, and the
surrounding context lines.

Glossary of REAL product/person/term names that appear in this domain
(corrections toward these spellings are legitimate, even if unfamiliar to you):
{glossary}

Verdicts:
- "OK"      : legitimate fix — typo, homophone, product/command/file name, casing,
              punctuation, grammar slip; OR justified removal of non-speech garbage
              (repeated tokens, decode duplicates, gibberish, lone punctuation).
- "SUSPECT" : meaning changed, real speech removed without garbage evidence, content
              invented, or a rewrite that goes beyond error correction.

Rules:
1. Do NOT flag a correction as SUSPECT merely because a product/person name is
   unfamiliar — check the glossary first. "Clawed Code" -> "Claude Code" is a
   classic legitimate fix.
2. Judge with the context lines in mind: a deletion between two fluently
   connecting sentences is usually a justified artifact removal.
3. Output STRICT JSON only: a single array
   [{{"id": <id exactly as given>, "verdict": "OK"|"SUSPECT", "reason": "<short; required for SUSPECT>"}}]
   — ids exactly as given (no renumbering/skipping), one entry per input id,
   no markdown, no commentary.

Items:
{items}
"""


@dataclass
class JudgeConfig:
    inputs: list[Path]
    orig_root: Path | None = None
    report_path: Path | None = None
    glossary_path: Path | None = None
    batch_segments: int = 30
    batch_chars: int = 6000
    max_retries: int = 2
    jobs: int = 1
    force: bool = False
    dry_run: bool = False
    mock: bool = False
    api_key: str | None = None
    base_url: str | None = None
    llm_model: str | None = None
    temperature: float = 0.0


# ---------------------------------------------------------------- discovery


def _default_orig_path(polished: Path) -> Path | None:
    """polished/<...>/x.json -> strip the ancestor dir literally named 'polished'."""
    for anc in [polished, *polished.parents]:
        if anc.name == "polished":
            return anc.parent / polished.relative_to(anc)
    return None


def discover_pairs(cfg: JudgeConfig) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for p in cfg.inputs:
        roots = [p] if p.is_file() else [p]
        files = [p] if p.is_file() else sorted(
            f for f in p.rglob("*.json") if not f.name.endswith(".polish-state.json")
        )
        for f in files:
            if cfg.orig_root is not None:
                try:
                    rel = f.relative_to(roots[0])
                except ValueError:
                    rel = Path(f.name)
                orig = cfg.orig_root / rel
            else:
                orig = _default_orig_path(f)
            if orig is None or not orig.is_file():
                log.warning("no original transcript found for %s — skipped", f.name)
                continue
            pairs.append((f, orig))
    # dedupe by polished path
    seen: set[Path] = set()
    out = []
    for pol, orig in pairs:
        r = pol.resolve()
        if r not in seen:
            seen.add(r)
            out.append((pol, orig))
    return out


def extract_diffs(pol_path: Path, orig_path: Path, file_id: int) -> list[dict] | None:
    try:
        pol = json.loads(pol_path.read_text(encoding="utf-8"))
        orig = json.loads(orig_path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        log.warning("bad json %s: %s", pol_path.name, e)
        return None
    ps, os_ = pol.get("segments") or [], orig.get("segments") or []
    if len(ps) != len(os_):
        log.warning(
            "segment count mismatch for %s (polished=%d original=%d) — skipped",
            pol_path.name, len(ps), len(os_),
        )
        return None
    diffs = []
    for i, (so, sp) in enumerate(zip(os_, ps)):
        ot, pt = (so.get("text") or "").strip(), (sp.get("text") or "").strip()
        if ot == pt:
            continue
        diffs.append(
            {
                "file_id": file_id,
                "idx": i,
                "start": round(float(so.get("start", 0.0)), 1),
                "original": ot,
                "polished": pt,
                "prev": (os_[i - 1].get("text") or "")[-80:] if i > 0 else "",
                "next": (os_[i + 1].get("text") or "")[:80] if i + 1 < len(os_) else "",
            }
        )
    return diffs


# ---------------------------------------------------------------- llm batch


def _parse(content: str, expected_ids: list[int]) -> dict[int, dict] | None:
    text = re.sub(r"^```(?:json)?\s*", "", content.strip())
    text = re.sub(r"\s*```$", "", text)
    s, e = text.find("["), text.rfind("]")
    if s < 0 or e <= s:
        return None
    try:
        arr = json.loads(text[s : e + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(arr, list):
        return None
    out: dict[int, dict] = {}
    for it in arr:
        if not isinstance(it, dict) or "id" not in it:
            return None
        v = it.get("verdict")
        if v not in ("OK", "SUSPECT"):
            return None
        try:
            out[int(it["id"])] = {"verdict": v, "reason": it.get("reason") or ""}
        except (TypeError, ValueError):
            return None
    if set(out.keys()) != set(expected_ids):
        return None  # judge renumbered / skipped ids -> invalid
    return out


def judge_batch(client: ChatClient, glossary: str, items: list[dict], ids: list[int]) -> tuple[dict[int, dict] | None, str | None]:
    body = json.dumps(
        [
            {"id": r["id"], "prev": r["prev"], "original": r["original"], "polished": r["polished"], "next": r["next"]}
            for r in items
        ],
        ensure_ascii=False,
    )
    user = USER_TMPL.format(glossary=glossary or "(none provided)", items=body)
    try:
        content = client.chat(SYSTEM_PROMPT, user)
    except LLMError as e:
        return None, str(e)
    verdicts = _parse(content, ids)
    if verdicts is None:
        return None, "invalid payload (structure/id-set mismatch)"
    return verdicts, None


# ---------------------------------------------------------------- entry


def run_judge(cfg: JudgeConfig) -> int:
    pairs = discover_pairs(cfg)
    if not pairs:
        print("No polished/original pairs found.", file=sys.stderr)
        return 1

    # collect diffs with global ids
    all_diffs: list[dict] = []
    file_names: dict[int, str] = {}
    skipped = 0
    for fid, (pol, orig) in enumerate(pairs):
        d = extract_diffs(pol, orig, fid)
        if d is None:
            skipped += 1
            continue
        file_names[fid] = pol.name
        all_diffs.extend(d)
    for n, d in enumerate(all_diffs):
        d["id"] = n

    report_path = cfg.report_path or (cfg.inputs[0].parent / "judge-report.md")

    if cfg.dry_run:
        for d in all_diffs[:10]:
            print(f"[{d['file_id']}:{d['file_id'] and file_names[d['file_id']][:40]} @{d['start']}s] {d['original'][:60]!r} -> {d['polished'][:60]!r}")
        print(f"\n{len(all_diffs)} diff(s) across {len(pairs) - skipped} file pair(s).")
        return 0

    if not all_diffs:
        report_path.write_text(
            f"# 二遍裁判报告\n\n- 模型: {cfg.mock and '(mock)' or (cfg.llm_model or 'auto')}\n"
            f"- diff 总数: 0（精修后无任何文本变化）\n",
            encoding="utf-8",
        )
        print("0 diffs — nothing to judge. Report:", report_path)
        return 0

    glossary = ""
    if cfg.glossary_path:
        glossary = "; ".join(
            ln.strip() for ln in cfg.glossary_path.read_text(encoding="utf-8").splitlines() if ln.strip()
        )

    client = ChatClient(api_key=cfg.api_key, base_url=cfg.base_url, model=cfg.llm_model, temperature=cfg.temperature)
    if not cfg.mock and not client.configured():
        print(
            "LLM not configured. Set STT_LLM_API_KEY + STT_LLM_MODEL (+ STT_LLM_BASE_URL), "
            "or pass --api-key/--llm-model/--api-base, or use --mock.",
            file=sys.stderr,
        )
        return 1

    texts = [d["original"] for d in all_diffs]
    batches = batch_indices(texts, cfg.batch_segments, cfg.batch_chars)

    # checkpoint (int-normalized always)
    state_path = report_path.parent / ".judge-state.json"
    state: dict[int, dict] = {}
    if report_path.parent.exists() and state_path.is_file() and not cfg.force:
        try:
            state = {int(k): v for k, v in json.loads(state_path.read_text(encoding="utf-8")).items()}
        except Exception:  # noqa: BLE001
            state = {}

    print(f"Judge model: {cfg.mock and '(mock)' or client.model}   diffs: {len(all_diffs)}   batches: {len(batches)}   jobs: {cfg.jobs}")
    print("-" * 72)

    def work(bi: int, ids: list[int]) -> dict[int, dict]:
        key = str(bi)
        if not cfg.force and all(i in state for i in ids):
            return {i: state[i] for i in ids}
        items = [all_diffs[i] for i in ids]
        if cfg.mock:
            return {i: {"verdict": "OK", "reason": ""} for i in ids}
        last = None
        for a in range(cfg.max_retries + 1):
            verdicts, err = judge_batch(client, glossary, items, ids)
            if verdicts is not None:
                return verdicts
            last = err
            log.warning("batch %d attempt %d failed: %s", bi, a + 1, err)
            time.sleep(2 * (a + 1))
        raise RuntimeError(f"batch {bi} failed after retries: {last}")

    fails: list[int] = []
    if cfg.jobs <= 1:
        for bi, ids in enumerate(batches):
            try:
                state.update(work(bi, ids))
            except Exception as e:  # noqa: BLE001
                fails.append(bi)
                print(f"batch {bi} FAIL: {e}", file=sys.stderr)
            _save_state(state_path, state)
    else:
        with ThreadPoolExecutor(max_workers=cfg.jobs) as ex:
            futs = {ex.submit(work, bi, ids): bi for bi, ids in enumerate(batches)}
            for fut in as_completed(futs):
                bi = futs[fut]
                try:
                    state.update(fut.result())
                except Exception as e:  # noqa: BLE001
                    fails.append(bi)
                    print(f"batch {bi} FAIL: {e}", file=sys.stderr)
                _save_state(state_path, state)

    if fails:
        print(f"{len(fails)} batch(es) failed: {fails} — re-run same command to resume.", file=sys.stderr)
        return 1

    # report
    sus = [(d, state[d["id"]]["reason"]) for d in all_diffs if state[d["id"]]["verdict"] == "SUSPECT"]
    ok = len(all_diffs) - len(sus)
    u = client.usage
    lines = [
        "# 二遍裁判报告（stt-local judge）",
        "",
        f"- 模型: {cfg.mock and '(mock)' or client.model}",
        f"- 评审 diff 总数: {len(all_diffs)}",
        f"- OK: {ok} ({ok / len(all_diffs) * 100:.1f}%)",
        f"- SUSPECT: {len(sus)} ({len(sus) / len(all_diffs) * 100:.2f}%)",
        f"- LLM 用量: {u['requests']} requests / {u['prompt_tokens']} + {u['completion_tokens']} tokens",
        "",
        "> 注意：裁判对领域产品名/术语可能保守误报（SUSPECT 需带上下文人工复核，",
        "> 优先看非术语类的改写与删除）。",
        "",
    ]
    if sus:
        lines.append("## SUSPECT 明细")
        for d, reason in sus:
            lines.append(f"- [{file_names[d['file_id']][:46]} @{d['start']}s] {reason}")
            lines.append(f"  - 原: {d['original'][:110]!r}")
            lines.append(f"  - 改: {d['polished'][:110]!r}")
    else:
        lines.append("全部通过，无需人工复核。")
    tmp = report_path.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(report_path)

    print("-" * 72)
    print(f"Done: {ok} OK / {len(sus)} SUSPECT of {len(all_diffs)} diffs.")
    print(f"Report: {report_path}")
    if state_path.is_file():
        state_path.unlink()  # clean checkpoint on full success
    return 0


def _save_state(state_path: Path, state: dict[int, dict]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps({str(k): v for k, v in state.items()}, ensure_ascii=False), encoding="utf-8")
    tmp.replace(state_path)
