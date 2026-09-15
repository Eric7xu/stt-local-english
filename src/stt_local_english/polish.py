"""Alignment-safe transcript polishing via an LLM.

Design invariants (the whole point of this module):
  * The LLM NEVER owns timing data. It only returns corrected text keyed by
    segment id, one-to-one with the input. All timestamps/word data stay in
    this module's hands, so subtitles remain aligned regardless.
  * Strict validation: id/order equality + per-segment length-ratio bounds.
    Invalid batches are retried, then fall back to original text.
  * Per-file checkpoint state so interrupted runs never re-bill completed
    batches.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from .engine import Segment, Transcript
from .llm import ChatClient, LLMError
from .output import _fmt_srt, render_md

log = logging.getLogger("stt-local.polish")

SYSTEM_PROMPT = (
    "You are a meticulous transcription corrector for English speech-to-text output. "
    "You fix transcription errors ONLY. You never paraphrase, never reorder, never "
    "add or remove sentences, and never invent content."
)

USER_TMPL = """You will get numbered segments from an automatic transcript of a technical
course video (software engineering / AI tooling).

Glossary of terms that are often mis-transcribed (use these exact spellings):
{glossary}

Rules:
1. Fix obvious ASR errors only: wrong homophones, mangled technical terms,
   product/command names, missing apostrophes, broken words.
2. You MUST NOT add, remove, merge, split or reorder segments. Output exactly
   one entry per input id, in the same order.
3. Keep the meaning and wording as close to the original as possible.
4. If a segment is non-speech garbage / hallucination (e.g. "A-A-A-A-...",
   random repeated tokens, ads read by TTS), output an EMPTY string "" for it.
5. If you are unsure, keep the original text unchanged.
6. Output STRICT JSON only: a single array of objects
   [{{"id": <same id>, "text": "<corrected text>"}}, ...] — no markdown, no
   commentary.

Segments:
{segments}
"""


@dataclass
class PolishConfig:
    inputs: list[Path]
    output_root: Path | None = None
    glossary_path: Path | None = None
    batch_segments: int = 40
    batch_chars: int = 6000
    max_retries: int = 2
    jobs: int = 1
    force: bool = False
    dry_run: bool = False
    mock: bool = False
    # llm client kwargs
    api_key: str | None = None
    base_url: str | None = None
    llm_model: str | None = None
    temperature: float = 0.0


@dataclass
class FileResult:
    path: Path
    ok: bool = False
    skipped: bool = False
    batches: int = 0
    fallback_batches: int = 0
    empty_segments: int = 0
    error: str | None = None
    usage: dict = field(default_factory=dict)


# ---------------------------------------------------------------- discovery


def discover_json(cfg: PolishConfig) -> list[Path]:
    files: list[Path] = []
    for p in cfg.inputs:
        if p.is_file() and p.suffix == ".json":
            files.append(p)
        elif p.is_dir():
            for f in sorted(p.rglob("*.json")):
                if f.name.endswith(".polish-state.json"):
                    continue
                files.append(f)
    seen: set[Path] = set()
    out: list[Path] = []
    for f in files:
        r = f.resolve()
        if r not in seen:
            seen.add(r)
            out.append(f)
    return out


def output_dir_for(json_path: Path, cfg: PolishConfig) -> Path:
    if cfg.output_root is not None:
        return cfg.output_root
    # default: <sibling of the json's folder> with -polished suffix
    parent = json_path.parent
    return parent.parent / (parent.name + "-polished")


# ---------------------------------------------------------------- batching


def plan_batches(segments: list[dict], cfg: PolishConfig) -> list[list[int]]:
    """Group segment indices into LLM-sized batches."""
    batches: list[list[int]] = []
    cur: list[int] = []
    cur_chars = 0
    for i, seg in enumerate(segments):
        text = seg.get("text") or ""
        if cur and (len(cur) >= cfg.batch_segments or cur_chars + len(text) > cfg.batch_chars):
            batches.append(cur)
            cur, cur_chars = [], 0
        cur.append(i)
        cur_chars += len(text) + 1
    if cur:
        batches.append(cur)
    return batches


# ---------------------------------------------------------------- state


def _state_path(json_path: Path, cfg: PolishConfig) -> Path:
    return output_dir_for(json_path, cfg) / (json_path.stem + ".polish-state.json")


def _load_state(json_path: Path, cfg: PolishConfig) -> dict:
    sp = _state_path(json_path, cfg)
    if cfg.force:
        return {}
    if sp.is_file():
        try:
            return json.loads(sp.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _save_state(json_path: Path, cfg: PolishConfig, state: dict) -> None:
    sp = _state_path(json_path, cfg)
    sp.parent.mkdir(parents=True, exist_ok=True)
    tmp = sp.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    tmp.replace(sp)


# ---------------------------------------------------------------- llm batch


def _parse_ids_payload(content: str) -> list[dict] | None:
    """Extract strict JSON array from model output (tolerate code fences)."""
    text = content.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        arr = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(arr, list):
        return None
    out = []
    for item in arr:
        if not isinstance(item, dict) or "id" not in item:
            return None
        out.append({"id": item["id"], "text": item.get("text")})
    return out


def _validate(payload: list[dict] | None, ids: list[int], segments: list[dict]) -> list[str] | None:
    """Return corrected texts aligned to `ids`, or None if invalid."""
    if payload is None:
        return None
    by_id = {}
    for item in payload:
        try:
            by_id[int(item["id"])] = item.get("text")
        except (TypeError, ValueError):
            return None
    if set(by_id.keys()) != set(ids):
        return None
    texts: list[str] = []
    for i, seg_id in enumerate(ids):
        new = by_id[seg_id]
        if new is None:
            new = ""
        if not isinstance(new, str):
            return None
        old = segments[seg_id].get("text") or ""
        ratio = len(new) / max(len(old), 1)
        # empty string allowed (hallucination removal); otherwise length must be sane
        if new and not (0.25 <= ratio <= 2.6):
            return None
        texts.append(new.strip())
    return texts


def polish_batch(client: ChatClient, glossary: str, segments: list[dict], ids: list[int]) -> tuple[list[str] | None, str | None]:
    seg_lines = "\n".join(
        json.dumps({"id": i, "text": (segments[i].get("text") or "")}, ensure_ascii=False) for i in ids
    )
    user = USER_TMPL.format(glossary=glossary or "(none provided)", segments=seg_lines)
    try:
        content = client.chat(SYSTEM_PROMPT, user)
    except LLMError as e:
        return None, str(e)
    payload = _parse_ids_payload(content)
    texts = _validate(payload, ids, segments)
    if texts is None:
        return None, "invalid payload (structure/ids/length-ratio)"
    return texts, None


# ---------------------------------------------------------------- outputs


def _render_srt_proportional(segments: list[Segment]) -> str:
    """SRT from corrected text; cue times split proportionally inside each
    (accurate) segment span. Word-level data is intentionally NOT reused here
    because the corrected text no longer matches the original words."""
    from .output import _proportional_cues

    lines: list[str] = []
    idx = 1
    for seg in segments:
        if not seg.text:
            continue
        for start, end, text in _proportional_cues(seg, 80):
            lines += [str(idx), f"{_fmt_srt(start)} --> {_fmt_srt(end)}", text, ""]
            idx += 1
    return "\n".join(lines)


def finalize_outputs(json_path: Path, cfg: PolishConfig, segments: list[dict], corrected: list[str], meta: dict) -> dict:
    out_dir = output_dir_for(json_path, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    segs: list[Segment] = []
    empty_count = 0
    for i, seg in enumerate(segments):
        text = corrected[i]
        if not text:
            empty_count += 1
        segs.append(Segment(start=float(seg.get("start", 0.0)), end=float(seg.get("end", 0.0)), text=text))

    duration = float(segs[-1].end) if segs else 0.0
    tr = Transcript(path=json_path, language=str(meta.get("language") or "en"), duration=duration, segments=segs, elapsed=0.0)

    # json: original structure + corrected text + polish meta
    doc = json.loads(json_path.read_text(encoding="utf-8"))
    for i, seg in enumerate(doc.get("segments") or []):
        seg["text"] = corrected[i]
        seg["original_text"] = None  # drop stale; original preserved in git/history
    doc["polished"] = True
    doc["polish_meta"] = {**meta, "empty_segments": empty_count}
    (out_dir / json_path.name).write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    (out_dir / (json_path.stem + ".srt")).write_text(_render_srt_proportional(segs), encoding="utf-8")
    (out_dir / (json_path.stem + ".md")).write_text(render_md(tr), encoding="utf-8")
    return {"empty_segments": empty_count}


# ---------------------------------------------------------------- per-file


def polish_file(json_path: Path, cfg: PolishConfig, client: ChatClient, glossary: str) -> FileResult:
    res = FileResult(path=json_path)
    out_dir = output_dir_for(json_path, cfg)
    marker = out_dir / json_path.name
    if not cfg.force and marker.is_file() and marker.stat().st_size > 0:
        res.ok = True
        res.skipped = True
        return res

    try:
        doc = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        res.error = f"bad json: {e}"
        return res
    segments = doc.get("segments") or []
    if not segments:
        res.error = "no segments"
        return res

    batches = plan_batches(segments, cfg)
    res.batches = len(batches)
    state = _load_state(json_path, cfg)
    done_batches: dict[str, list[str]] = state.get("batches", {})
    fallback = 0

    for bi, ids in enumerate(batches):
        key = str(bi)
        if key in done_batches:
            continue
        texts = done_batches.get(key)
        if texts is None:
            if cfg.mock:
                texts = [segments[i].get("text") or "" for i in ids]
            else:
                texts = None
                err = None
                for attempt in range(cfg.max_retries + 1):
                    texts, err = polish_batch(client, glossary, segments, ids)
                    if texts is not None:
                        break
                    log.warning("batch %d/%s attempt %d failed: %s", bi, json_path.name, attempt + 1, err)
                    time.sleep(2 * (attempt + 1))
                if texts is None:
                    log.warning("batch %d/%s fell back to ORIGINAL text (%s)", bi, json_path.name, err)
                    texts = [segments[i].get("text") or "" for i in ids]
                    fallback += 1
        done_batches[key] = texts
        state["batches"] = done_batches
        state["fallbacks"] = fallback
        _save_state(json_path, cfg, state)

    # flatten in batch order (batches are ordered, contiguous)
    corrected: list[str] = [""] * len(segments)
    for bi, ids in enumerate(batches):
        for j, i in enumerate(ids):
            corrected[i] = done_batches[str(bi)][j]

    meta = {
        "model": cfg.mock and "mock" or (cfg.llm_model or client.model),
        "batches": len(batches),
        "fallback_batches": fallback,
        "usage": {} if cfg.mock else dict(client.usage),
        "language": doc.get("language"),
    }
    out = finalize_outputs(json_path, cfg, segments, corrected, meta)
    res.fallback_batches = fallback
    res.empty_segments = out["empty_segments"]
    res.usage = meta["usage"]
    res.ok = True

    # clean state file on success
    sp = _state_path(json_path, cfg)
    if sp.is_file():
        sp.unlink()
    return res


# ---------------------------------------------------------------- entry


def run_polish(cfg: PolishConfig) -> int:
    files = discover_json(cfg)
    if not files:
        print("No transcript .json files found.", file=sys.stderr)
        return 1

    glossary = ""
    if cfg.glossary_path:
        glossary = "; ".join(
            ln.strip() for ln in cfg.glossary_path.read_text(encoding="utf-8").splitlines() if ln.strip()
        )

    client = ChatClient(
        api_key=cfg.api_key, base_url=cfg.base_url, model=cfg.llm_model, temperature=cfg.temperature
    )
    if not cfg.mock and not client.configured():
        print(
            "LLM not configured. Set STT_LLM_API_KEY + STT_LLM_MODEL "
            "(+ STT_LLM_BASE_URL for non-OpenAI providers), or use --mock.",
            file=sys.stderr,
        )
        return 1

    print(f"Files: {len(files)}   model: {cfg.mock and '(mock)' or client.model}   jobs: {cfg.jobs}")
    print("-" * 72)

    def work(fp: Path) -> FileResult:
        return polish_file(fp, cfg, client, glossary)

    results: list[FileResult] = []
    if cfg.jobs <= 1:
        for i, fp in enumerate(files, 1):
            r = work(fp)
            results.append(r)
            tag = "SKIP" if r.skipped else ("OK" if r.ok else "FAIL")
            print(f"[{i}/{len(files)}] {tag:4s} {fp.name}" + (f"  ({r.error})" if r.error else ""))
    else:
        with ThreadPoolExecutor(max_workers=cfg.jobs) as ex:
            futs = {ex.submit(work, fp): fp for fp in files}
            for i, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                results.append(r)
                fp = futs[fut]
                tag = "SKIP" if r.skipped else ("OK" if r.ok else "FAIL")
                print(f"[{i}/{len(files)}] {tag:4s} {fp.name}" + (f"  ({r.error})" if r.error else ""))

    print("-" * 72)
    ok = sum(1 for r in results if r.ok and not r.skipped)
    skipped = sum(1 for r in results if r.skipped)
    failed = sum(1 for r in results if not r.ok)
    fb = sum(r.fallback_batches for r in results)
    u = client.usage
    print(f"Done: {ok} polished, {skipped} skipped, {failed} failed; fallback batches: {fb}")
    if not cfg.mock:
        print(f"LLM usage: {u['requests']} requests, {u['prompt_tokens']} prompt + {u['completion_tokens']} completion tokens")
    return 1 if failed else 0
