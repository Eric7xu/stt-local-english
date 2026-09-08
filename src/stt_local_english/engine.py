"""Transcription engine: wraps mlx_whisper with model loading + error resilience."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("stt-local")

try:  # mlx is macOS-only; import lazily so --help works everywhere
    import mlx_whisper
    MLX_OK = True
except Exception:  # pragma: no cover - non-macOS or missing mlx
    mlx_whisper = None
    MLX_OK = False


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[dict] = field(default_factory=list)


@dataclass
class Transcript:
    path: Path
    language: str
    duration: float
    segments: list[Segment]
    elapsed: float


def check_mlx() -> None:
    if not MLX_OK:
        raise SystemExit(
            "mlx-whisper is not available. This tool is built for Apple Silicon macOS.\n"
            "Run: uv sync   (or reinstall with `uv add mlx-whisper` inside the project)."
        )


class ModelCache:
    """Placeholder kept for API stability; mlx-whisper 0.4+ caches models itself.

    Loading happens lazily inside the first ``transcribe`` call (ModelHolder), so
    re-passing the same model string costs nothing on later files.
    """

    def get(self, model_ref: str):
        return model_ref


_CACHE = ModelCache()


def _clean_segments(segs, word_timestamps: bool) -> list[Segment]:
    out: list[Segment] = []
    for s in segs:
        text = (s.get("text") or "").strip()
        if not text:
            continue
        words = []
        if word_timestamps:
            for w in s.get("words", []) or []:
                wt = (w.get("word") or w.get("text") or "").strip()
                if wt:
                    words.append({"word": wt, "start": float(w.get("start", 0.0)), "end": float(w.get("end", 0.0))})
        out.append(Segment(start=float(s.get("start", 0.0)), end=float(s.get("end", 0.0)), text=text, words=words))
    return out


def transcribe_media(
    media_path: Path,
    model_ref: str,
    language: str = "en",
    word_timestamps: bool = False,
    condition_on_previous_text: bool = True,
    verbose: bool = False,
) -> Transcript:
    """Transcribe one media file to a Transcript with segments."""
    check_mlx()
    t0 = time.monotonic()
    model = _CACHE.get(model_ref)

    result = mlx_whisper.transcribe(
        str(media_path),
        path_or_hf_repo=model,
        language=language if language != "auto" else None,
        verbose=verbose,
        word_timestamps=word_timestamps,
        condition_on_previous_text=condition_on_previous_text,
    )
    elapsed = time.monotonic() - t0

    raw_segs = result.get("segments") or []
    info = result.get("info") or {}
    segs = _clean_segments(raw_segs, word_timestamps)
    duration = float(info.get("duration") or 0.0)
    if not duration and segs:
        duration = segs[-1].end
    detected_lang = result.get("language") or language

    if not segs:
        log.warning("No speech segments detected in %s", media_path.name)

    return Transcript(
        path=media_path,
        language=detected_lang,
        duration=duration,
        segments=segs,
        elapsed=elapsed,
    )
