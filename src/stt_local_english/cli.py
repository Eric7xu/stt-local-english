"""stt-local: local English speech-to-text CLI (mlx-whisper / Apple Silicon)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import DEFAULT_MODEL, __version__
from .batch import BatchConfig, discover_media, run_batch
from .engine import resolve_engine
from .judge import JudgeConfig, run_judge
from .output import RENDERERS
from .polish import PolishConfig, run_polish

log = logging.getLogger("stt-local")


def build_polish_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="stt-local polish",
        description=(
            "Alignment-safe LLM polishing of stt-local transcripts: fixes ASR errors "
            "while keeping every timestamp. The LLM only returns corrected text per "
            "segment id; timing data never leaves this tool."
        ),
    )
    p.add_argument("inputs", nargs="+", type=Path, help="transcript .json files or directories")
    p.add_argument("-o", "--output-dir", type=Path, default=None,
                   help="output root (default: <sibling> -polished/ next to each input folder)")
    p.add_argument("--glossary", type=Path, default=None,
                   help="text file of domain terms (one per line) to protect from mis-correction")
    p.add_argument("--api-base", default=None, help="OpenAI-compatible base URL (env STT_LLM_BASE_URL)")
    p.add_argument("--api-key", default=None, help="API key (env STT_LLM_API_KEY or OPENAI_API_KEY)")
    p.add_argument("--llm-model", default=None, help="model name (env STT_LLM_MODEL)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--batch-segments", type=int, default=40, help="max segments per LLM request (default 40)")
    p.add_argument("--batch-chars", type=int, default=6000, help="max chars per LLM request (default 6000)")
    p.add_argument("--max-retries", type=int, default=2, help="retries per failed batch (default 2)")
    p.add_argument("--jobs", type=int, default=1, help="parallel files (default 1)")
    p.add_argument("--force", action="store_true", help="re-polish even if outputs exist")
    p.add_argument("--dry-run", action="store_true", help="list transcript files and exit")
    p.add_argument("--mock", action="store_true", help="no network: pass-through text, exercise the full pipeline")
    return p


def build_judge_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="stt-local judge",
        description=(
            "Second-pass QA judge: an independent model reviews every polish diff "
            "(original vs polished) and verdicts OK / SUSPECT with reasons. "
            "Use a DIFFERENT model family from the polisher to avoid self-review bias."
        ),
    )
    p.add_argument("inputs", nargs="+", type=Path,
                   help="polished .json files or directories (default orig-root auto-detects the sibling 'polished' component)")
    p.add_argument("--orig-root", type=Path, default=None,
                   help="root of the ORIGINAL transcripts (default: auto: replace the 'polished' path component)")
    p.add_argument("-o", "--report", type=Path, default=None,
                   help="report markdown path (default: judge-report.md next to the input folder)")
    p.add_argument("--glossary", type=Path, default=None,
                   help="same glossary file used for polishing — injects domain terms to avoid false positives")
    p.add_argument("--api-base", default=None, help="OpenAI-compatible base URL (env STT_LLM_BASE_URL)")
    p.add_argument("--api-key", default=None, help="API key (env STT_LLM_API_KEY or OPENAI_API_KEY)")
    p.add_argument("--llm-model", default=None, help="judge model (env STT_LLM_MODEL; pick a DIFFERENT family from the polisher)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--batch-segments", type=int, default=30)
    p.add_argument("--batch-chars", type=int, default=6000)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--force", action="store_true", help="re-judge even if checkpoints exist")
    p.add_argument("--dry-run", action="store_true", help="list diffs that would be judged, then exit")
    p.add_argument("--mock", action="store_true", help="no network: all-OK verdicts, exercise the pipeline")
    return p


def judge_main(argv: list[str]) -> int:
    args = build_judge_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    cfg = JudgeConfig(
        inputs=args.inputs,
        orig_root=args.orig_root,
        report_path=args.report,
        glossary_path=args.glossary,
        batch_segments=args.batch_segments,
        batch_chars=args.batch_chars,
        max_retries=args.max_retries,
        jobs=args.jobs,
        force=args.force,
        dry_run=args.dry_run,
        mock=args.mock,
        api_key=args.api_key,
        base_url=args.api_base,
        llm_model=args.llm_model,
        temperature=args.temperature,
    )
    return run_judge(cfg)


def polish_main(argv: list[str]) -> int:
    args = build_polish_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    cfg = PolishConfig(
        inputs=args.inputs,
        output_root=args.output_dir,
        glossary_path=args.glossary,
        batch_segments=args.batch_segments,
        batch_chars=args.batch_chars,
        max_retries=args.max_retries,
        jobs=args.jobs,
        force=args.force,
        dry_run=args.dry_run,
        mock=args.mock,
        api_key=args.api_key,
        base_url=args.api_base,
        llm_model=args.llm_model,
        temperature=args.temperature,
    )
    if args.dry_run:
        files = __import__("stt_local_english.polish", fromlist=["discover_json"]).discover_json(cfg)
        for f in files:
            print(f)
        print(f"\n{len(files)} transcript file(s).")
        return 0
    return run_polish(cfg)


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
    p.add_argument("--no-condition-previous", action="store_true",
                  help="disable conditioning on previous text (less repetition-hallucination on non-speech/music; may break sentence consistency)")
    p.add_argument("--engine", choices=["auto", "mlx", "faster"], default="auto",
                  help="transcription backend: mlx (Apple Silicon GPU) or faster (faster-whisper, cross-platform); auto = mlx if available")
    p.add_argument("--fw-device", default=None, help="faster backend device: auto/cpu/cuda (default auto)")
    p.add_argument("--vad", action="store_true", help="faster backend only: VAD filter to skip non-speech (fewer music hallucinations)")
    p.add_argument("--no-md", action="store_true", help="shortcut: only write SRT")
    p.add_argument("--force", action="store_true", help="re-transcribe even if outputs exist")
    p.add_argument("--no-recursive", action="store_true", help="do not descend into subdirectories")
    p.add_argument("--dry-run", action="store_true", help="list media that would be transcribed, then exit")
    p.add_argument("--verbose", action="store_true", help="verbose mlx-whisper logging")
    p.add_argument("--version", action="version", version=f"stt-local {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "polish":
        return polish_main(argv[1:])
    if argv and argv[0] == "judge":
        return judge_main(argv[1:])
    args = build_parser().parse_args(argv)

    if args.no_md:
        fmts = ("srt",)
    else:
        fmts = tuple(args.formats) if args.formats else ("srt", "md")

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    engine = resolve_engine(args.engine)

    cfg = BatchConfig(
        inputs=args.inputs,
        output_root=args.output_dir,
        formats=fmts,
        model=args.model,
        language=args.language,
        word_timestamps=args.word_timestamps,
        force=args.force,
        recursive=not args.no_recursive,
        condition_on_previous_text=not args.no_condition_previous,
        engine=engine,
        fw_device=args.fw_device,
        vad=args.vad,
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

    print(f"Engine: {engine}   Model: {args.model}")
    print(f"Language: {args.language}   Formats: {', '.join(fmts)}   Files: {len(media)}")
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
