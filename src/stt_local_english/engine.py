"""Transcription engine backends.

Two interchangeable engines, identical output contract (Transcript/Segment):

  * mlx   — mlx-whisper, macOS Apple Silicon only, GPU-native (default on Mac)
  * faster — faster-whisper (CTranslate2), Linux/Windows/macOS, CPU or CUDA

Engine selection (--engine): "auto" picks mlx when available, else faster.
Model names are shared: mlx-community/whisper-large-v3-turbo maps to
"large-v3-turbo" on the faster backend automatically.
"""

from __future__ import annotations

import logging
import re
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

try:  # faster-whisper: cross-platform backend (Linux/Windows/macOS)
    import faster_whisper
    FW_OK = True
except Exception:  # pragma: no cover
    faster_whisper = None
    FW_OK = False


class LLMError(RuntimeError):  # re-exported convenience  # pragma: no cover
    pass


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
            "mlx-whisper is not available. The mlx engine requires Apple Silicon macOS.\n"
            "On Linux/Windows use --engine faster (faster-whisper installs automatically).\n"
            "Run: uv sync   (inside the project)."
        )


def resolve_engine(engine: str = "auto") -> str:
    """Return the concrete backend name ('mlx' | 'faster'), or exit friendly."""
    if engine == "mlx":
        check_mlx()
        return "mlx"
    if engine == "faster":
        if not FW_OK:
            raise SystemExit(
                "faster-whisper is not installed. Run `uv sync` (it is included "
                "automatically on non-macOS platforms), or `uv sync --extra fw`."
            )
        return "faster"
    # auto
    if MLX_OK:
        return "mlx"
    if FW_OK:
        return "faster"
    raise SystemExit(
        "No transcription backend available. On Apple Silicon run `uv sync` "
        "(installs mlx-whisper); on Linux/Windows `uv sync` installs "
        "faster-whisper automatically."
    )


def _fw_model_name(model_ref: str) -> str:
    """Map mlx-community repo ids to faster-whisper size names; pass through ct2 repos."""
    if not model_ref.startswith("mlx-community/"):
        return model_ref
    name = model_ref.split("/", 1)[1]
    if name.startswith("whisper-"):
        name = name[len("whisper-") :]
    name = re.sub(r"-(int4|int8|4bit|8bit|q4|q8)$", "", name)
    log.info("faster backend: using model %r", name)
    return name


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


class ModelCache:
    """Placeholder for API stability; mlx-whisper caches models itself."""

    def get(self, model_ref: str):
        return model_ref


_CACHE = ModelCache()

# faster-whisper models are heavy; cache loaded instances per (model, device, compute)
_FW_CACHE: dict[tuple, object] = {}


def _get_fw_model(model_ref: str, fw_device: str | None):
    from faster_whisper import WhisperModel

    key = (model_ref, fw_device or "auto")
    if key not in _FW_CACHE:
        log.info("Loading faster-whisper model %s ...", model_ref)
        t0 = time.monotonic()
        _FW_CACHE[key] = WhisperModel(model_ref, device=fw_device or "auto", compute_type="auto")
        log.info("faster-whisper model loaded in %.1fs", time.monotonic() - t0)
    return _FW_CACHE[key]


def _finalize(path: Path, language: str, duration: float, segs: list[Segment], elapsed: float) -> Transcript:
    if not duration and segs:
        duration = segs[-1].end
    if not segs:
        log.warning("No speech segments detected in %s", path.name)
    return Transcript(path=path, language=language, duration=duration, segments=segs, elapsed=elapsed)


def _transcribe_mlx(
    media_path: Path, model_ref: str, language: str, word_timestamps: bool,
    condition_on_previous_text: bool, verbose: bool,
) -> Transcript:
    t0 = time.monotonic()
    result = mlx_whisper.transcribe(
        str(media_path),
        path_or_hf_repo=_CACHE.get(model_ref),
        language=language if language != "auto" else None,
        verbose=verbose,
        word_timestamps=word_timestamps,
        condition_on_previous_text=condition_on_previous_text,
    )
    elapsed = time.monotonic() - t0
    info = result.get("info") or {}
    segs = _clean_segments(result.get("segments") or [], word_timestamps)
    return _finalize(
        media_path,
        result.get("language") or language,
        float(info.get("duration") or 0.0),
        segs,
        elapsed,
    )


def _transcribe_fw(
    media_path: Path, model_ref: str, language: str, word_timestamps: bool,
    condition_on_previous_text: bool, verbose: bool,
    fw_device: str | None, vad: bool,
) -> Transcript:
    t0 = time.monotonic()
    model = _get_fw_model(_fw_model_name(model_ref), fw_device)
    segments_iter, info = model.transcribe(
        str(media_path),
        language=None if language == "auto" else language,
        word_timestamps=word_timestamps,
        condition_on_previous_text=condition_on_previous_text,
        vad_filter=vad,
    )
    segs: list[Segment] = []
    for s in segments_iter:  # generator — iteration performs the decoding
        text = (s.text or "").strip()
        if not text:
            continue
        words = []
        if word_timestamps:
            for w in s.words or []:
                wt = (w.word or "").strip()
                if wt:
                    words.append({"word": wt, "start": float(w.start), "end": float(w.end)})
        segs.append(Segment(start=float(s.start), end=float(s.end), text=text, words=words))
    elapsed = time.monotonic() - t0
    return _finalize(media_path, info.language or language, float(info.duration or 0.0), segs, elapsed)


def transcribe_media(
    media_path: Path,
    model_ref: str,
    language: str = "en",
    word_timestamps: bool = False,
    condition_on_previous_text: bool = True,
    verbose: bool = False,
    engine: str = "auto",
    fw_device: str | None = None,
    vad: bool = False,
) -> Transcript:
    """Transcribe one media file with the selected backend."""
    backend = resolve_engine(engine)
    t0 = time.monotonic()
    if backend == "mlx":
        return _transcribe_mlx(media_path, model_ref, language, word_timestamps, condition_on_previous_text, verbose)
    return _transcribe_fw(
        media_path, model_ref, language, word_timestamps, condition_on_previous_text, verbose, fw_device, vad
    )
