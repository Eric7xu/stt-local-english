"""stt-local: local English speech-to-text CLI (mlx-whisper / Apple Silicon)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import DEFAULT_MODEL, __version__
from .batch import BatchConfig, discover_media, run_batch
from .engine import check_mlx
from .output import RENDERERS

log = logging.getLogger("stt-local")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="stt-local",
        description=(
            "Transcribe English video/audio locally on Apple Silicon into timestamped "
            "SRT / VTT / Markdown / JSON. Powered by mlx-whisper (Whisper large-v3-turbo)."
        ),
    )
    p.add_argument("inputs", nargs="+", type=Path, help="media files or directories to transcribe")
    p.add_argument("-o", "--output-dir", type=Path, default=None,
                   help="write transcripts here, mirroring input structure (default: next to each media file)")
    p.add_argument("-m", "--model", default=DEFAULT_MODEL,
                   help=f"mlx-whisper model repo/path (default: {DEFAULT_MODEL})")
    p.add_argument("-f", "--format", dest="formats", action="append", choices=sorted(RENDERERS),
                   default=None, help="output format; repeatable (default: srt md)")
    p.add_argument("-l", "--language", default="en", help="language hint, 'auto' to detect (default: en)")
    p.add_argument("--word-timestamps", action="store_true", help="include per-word timestamps (in JSON/SRT)")
    p.add_argument("--no-md", action="store_true", help="shortcut: only write SRT")
    p.add_argument("--force", action="store_true", help="re-transcribe even if outputs exist")
    p.add_argument("--no-recursive", action="store_true", help="do not descend into subdirectories")
    p.add_argument("--dry-run", action="store_true", help="list media that would be transcribed, then exit")
    p.add_argument("--verbose", action="store_true", help="verbose mlx-whisper logging")
    p.add_argument("--version", action="version", version=f"stt-local {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.no_md:
        fmts = ("srt",)
    else:
        fmts = tuple(args.formats) if args.formats else ("srt", "md")

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    check_mlx()

    cfg = BatchConfig(
        inputs=args.inputs,
        output_root=args.output_dir,
        formats=fmts,
        model=args.model,
        language=args.language,
        word_timestamps=args.word_timestamps,
        force=args.force,
        recursive=not args.no_recursive,
    )

    media = discover_media(cfg)
    if not media:
        print("No supported media files found.", file=sys.stderr)
        return 1

    if args.dry_run:
        for m in media:
            print(m)
        print(f"\n{len(media)} file(s).")
        return 0

    print(f"Model: {cfg.model}")
    print(f"Language: {cfg.language}   Formats: {', '.join(fmts)}   Files: {len(media)}")
    print("-" * 72)

    summary = run_batch(cfg, media)

    print("-" * 72)
    print(f"Done: {summary['ok']} ok, {summary['skipped']} skipped, {summary['failed']} failed "
          f"of {summary['total']} ({summary['wall_s']:.0f}s transcription time).")
    if summary["failures"]:
        for media, err in summary["failures"]:
            print(f"  FAILED {media}: {err}", file=sys.stderr)
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
