"""Media discovery + batch pipeline with resume support."""

from __future__ import annotations

import logging
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from . import SUPPORTED_MEDIA_EXTS
from .engine import Transcript, transcribe_media
from .output import RENDERERS

log = logging.getLogger("stt-local")


@dataclass
class BatchConfig:
    inputs: list[Path]
    output_root: Path | None  # None => write next to source media
    formats: tuple[str, ...] = ("srt", "md")
    model: str = "mlx-community/whisper-large-v3-turbo"
    language: str = "en"
    word_timestamps: bool = False
    force: bool = False
    recursive: bool = True
    jobs: int = 1
    condition_on_previous_text: bool = True


def _fmt_ts(seconds: float) -> str:
    s = int(seconds % 60)
    m = int((seconds // 60) % 60)
    h = int(seconds // 3600)
    return f"{h:d}:{m:02d}:{s:02d}"


def discover_media(cfg: BatchConfig) -> list[Path]:
    files: list[Path] = []
    for p in cfg.inputs:
        if p.is_file():
            if p.suffix.lower() in SUPPORTED_MEDIA_EXTS:
                files.append(p)
            else:
                log.warning("Skip unsupported file: %s", p)
        elif p.is_dir():
            it = p.rglob("*") if cfg.recursive else p.glob("*")
            for f in sorted(it):
                if f.is_file() and f.suffix.lower() in SUPPORTED_MEDIA_EXTS:
                    files.append(f)
        else:
            log.warning("Input not found: %s", p)
    # dedupe, keep first occurrence order
    seen: set[Path] = set()
    out: list[Path] = []
    for f in files:
        r = f.resolve()
        if r not in seen:
            seen.add(r)
            out.append(f)
    return out


def _output_dir_for(media: Path, cfg: BatchConfig) -> Path:
    """Where the outputs for `media` should live."""
    if cfg.output_root is not None:
        return cfg.output_root
    # Default: sibling hidden folder `<parent>/.transcripts-<dirname>` is confusing;
    # we write next to the media file by default (same basename, different ext).
    return media.parent


def _is_done(media: Path, cfg: BatchConfig, single_format: str | None = None) -> bool:
    fmts = (single_format,) if single_format else cfg.formats
    for fmt in fmts:
        out = (_output_dir_for(media, cfg) / media.stem).with_suffix(f".{fmt}")
        if not out.exists() or out.stat().st_size == 0:
            return False
    return True


def _done_missing(media: Path, cfg: BatchConfig) -> tuple[list[str], bool]:
    """Return (done_formats, all_done)."""
    done = []
    for fmt in cfg.formats:
        out = (_output_dir_for(media, cfg) / media.stem).with_suffix(f".{fmt}")
        if out.exists() and out.stat().st_size > 0:
            done.append(fmt)
    return done, len(done) == len(cfg.formats)


def transcribe_one(media: Path, cfg: BatchConfig) -> dict:
    """Transcribe a single media file, writing all requested formats.

    Returns {"ok": bool, "reason": str|None, "elapsed": float, "out": [paths], "skipped": bool}
    """
    out_dir = _output_dir_for(media, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.force:
        done, all_done = _done_missing(media, cfg)
        if all_done:
            return {"ok": True, "reason": None, "elapsed": 0.0, "out": [], "skipped": True}
        if done:
            log.info("  partial: missing %s for %s", ",".join(f for f in cfg.formats if f not in done), media.name)

    t_start = time.monotonic()
    tr: Transcript = transcribe_media(
        media,
        model_ref=cfg.model,
        language=cfg.language,
        word_timestamps=cfg.word_timestamps,
        condition_on_previous_text=cfg.condition_on_previous_text,
    )
    written: list[Path] = []
    for fmt in cfg.formats:
        render = RENDERERS[fmt]
        out = (out_dir / media.stem).with_suffix(f".{fmt}")
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text(render(tr), encoding="utf-8")
        tmp.replace(out)
        written.append(out)
    return {"ok": True, "reason": None, "elapsed": time.monotonic() - t_start, "out": written, "skipped": False}


def run_batch(cfg: BatchConfig, media_files: list[Path], log_every: int = 1) -> dict:
    n = len(media_files)
    ok = skipped = failed = 0
    total_media_s = 0.0
    total_wall = 0.0
    failures: list[tuple[Path, str]] = []

    for i, media in enumerate(media_files, 1):
        prefix = f"[{i}/{n}]"
        try:
            res = transcribe_one(media, cfg)
            if res["skipped"]:
                skipped += 1
                print(f"{prefix} SKIP   {media.name} (already done)", flush=True)
                continue
            ok += 1
            total_wall += res["elapsed"]
            print(
                f"{prefix} OK     {media.name}  (transcribed in {res['elapsed']:.1f}s "
                f"→ {', '.join(str(p) for p in res['out'])})",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001 - batch must keep going
            failed += 1
            failures.append((media, str(e)))
            print(f"{prefix} FAIL   {media.name}: {e}", file=sys.stderr, flush=True)

    summary = {
        "total": n,
        "ok": ok,
        "skipped": skipped,
        "failed": failed,
        "wall_s": total_wall,
        "failures": failures,
    }
    return summary
