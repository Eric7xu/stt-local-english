# stt-local-english

<div align="center">

## stt-local-english

**Local English speech-to-text CLI: word-level timestamped SRT/MD/JSON from any video/audio,
with alignment-safe LLM post-editing built in.**

[![CI](https://github.com/Eric7xu/stt-local-english/actions/workflows/smoke.yml/badge.svg)](https://github.com/Eric7xu/stt-local-english/actions/workflows/smoke.yml)
[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)]()
[![Platform](https://img.shields.io/badge/transcribe-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

English | [中文](README_ZH.md)

</div>

---

Local English speech-to-text CLI built on `mlx-whisper` + Whisper `large-v3-turbo`
(macOS Apple Silicon) and `faster-whisper` (Linux/Windows).

Turns any video/audio file into timestamped **SRT / VTT / Markdown / JSON / TXT**.
Designed for the "batch subtitle → LLM refinement" workflow: word-level timestamps,
checkpoint/resume, structured JSON output.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🍎 GPU-native | Fully utilizes Apple Silicon unified memory (MLX) — measured **11–13× real-time** on an M5 |
| 📄 5 output formats | `srt` / `vtt` / `md` / `txt` / `json`, multi-select |
| ⏱ Word-level timing | Cue boundaries = first word onset → last word end (`--word-timestamps`) |
| 🛡 Artifact fallback | Segments with unreliable word alignment automatically fall back to proportional sentence splitting — no duplicated subtitle text |
| ♻️ Resume | Files with all target formats already generated are skipped; re-run the same command after any interruption |
| 📦 Model caching | Weights cached on disk + loaded once per process |
| 🌲 Directory batch | Recursive scanning, mirrored output structure |

---

## Installation

### ⚠️ Platform requirements (read first)

| Item | Requirement | Notes |
|---|---|---|
| OS | **`transcribe`: macOS uses the mlx engine; Linux/Windows automatically use the faster-whisper engine** | CPU works on Linux/Win (NVIDIA is faster); `--vad` helps with music-section hallucinations |
| `polish` / `judge` | ✅ Cross-platform | Pure Python standard library |
| ffmpeg | needed by `transcribe` only | `brew install ffmpeg` / `apt install ffmpeg` |
| Package manager | uv | `brew install uv` or see the [uv docs](https://docs.astral.sh/uv/) |
| Network | first run only | Downloads dependencies + model weights (offline afterwards) |

### From zero in 5 steps (fresh machine)

```bash
# 1. Prerequisites (skip if installed)
brew install uv ffmpeg          # Linux: install uv + ffmpeg via your package manager

# 2. Clone (replace with your own repo URL)
git clone https://github.com/<you>/stt-local-english.git
cd stt-local-english

# 3. Create the env and install deps (uv installs Python 3.12 automatically)
uv sync

# 4. Verify
uv run stt-local --help

# 5. Transcribe the first file (model downloads automatically on first run, see below)
uv run stt-local your_video.mp4 -o ./out -f srt -f md
```

**Global install** (use `stt-local` from any directory):

```bash
uv tool install .
# upgrade / uninstall
uv tool upgrade stt-local-english
uv tool uninstall stt-local-english
```

---

## How models work (not committed; downloaded on demand)

Model weights are **not stored in the git repo** (~1.5 GB for the default) — instead:

1. On the first transcription run, `mlx-whisper` pulls `mlx-community/whisper-large-v3-turbo`
   from HuggingFace automatically
2. It is cached at `~/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo/`
3. **Downloaded exactly once** — fully offline afterwards
4. Deleting the cache re-downloads; a new machine downloads its own copy

```bash
# inspect cache size / clean up manually
ls ~/.cache/huggingface/hub/ | grep whisper
rm -rf ~/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo
```

**Too big / slow network?** Use the int4 quantized build (~500 MB, slightly lower quality):

```bash
stt-local clip.mp4 -m mlx-community/whisper-large-v3-turbo-int4
```

**Can't reach HuggingFace (mainland China)?** Use the mirror (one env var, no proxy):

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1   # optional: faster downloads
stt-local clip.mp4
```

---

## Quick start

```bash
# ① Single file → writes .srt + .md next to the video
stt-local 001_video.mp4

# ② Subtitles only
stt-local 001_video.mp4 --no-md

# ③ Whole directory (recursive), output mirrored to transcripts/
stt-local ./videos/ -o ./transcripts/

# ④ Recommended for real runs (word-level timestamps + three formats, for later LLM refinement)
stt-local ./videos/ -o ./transcripts/videos/ \
          -f srt -f md -f json --word-timestamps

# ⑤ See what would be processed
stt-local ./videos/ --dry-run
```

---

## CLI reference

```
usage: stt-local [-h] [-o OUTPUT_DIR] [-m MODEL] [-f {json,md,srt,txt,vtt}]
                 [-l LANGUAGE] [--word-timestamps] [--no-md] [--force]
                 [--no-recursive] [--dry-run] [--verbose] [--version]
                 inputs [inputs ...]
```

| Flag | Default | Description |
|---|---|---|
| `inputs` | — | media files or directories (multiple allowed; directories are recursive by default) |
| `-o, --output-dir` | none | output root mirroring the input structure; **omit → written next to each media file** |
| `-m, --model` | `mlx-community/whisper-large-v3-turbo` | HF repo or local path — see the model table (the faster backend auto-maps names) |
| `-f, --format` | `srt md` | output format; repeatable (e.g. `-f srt -f json`) |
| `-l, --language` | `en` | language hint; `auto` to detect |
| `--engine` | `auto` | `mlx` (Apple Silicon GPU) / `faster` (cross-platform, the Linux/Win default) / `auto` |
| `--fw-device` | `auto` | faster backend device: `cpu` / `cuda` / `auto` |
| `--vad` | off | faster backend: VAD filter skips non-speech (fixes music/outro hallucinations) |
| `--word-timestamps` | off | word-level timestamps: word-timed SRT cues, `words` in JSON |
| `--no-condition-previous` | off | disable conditioning on previous text; cures the `A-A-A-...` repetition hallucination on music/outro segments |
| `--no-md` | off | shortcut: SRT only |
| `--force` | off | re-transcribe even if outputs exist |
| `--no-recursive` | off | don't descend into subdirectories |
| `--dry-run` | off | list files that would be processed, then exit |
| `--verbose` | off | verbose logging (progress bars) |
| `--version` | — | print version |

### Choosing a model

The default `large-v3-turbo` is the best accuracy/speed balance. Other HF repos (MLX builds):

| Model | Params | Speed | Accuracy | Notes |
|---|---|---|---|---|
| `mlx-community/whisper-turbo` | ~809M | ★★★★★ | ★★★★ | fast, good enough |
| `mlx-community/whisper-large-v3-turbo` | 809M | ★★★★ | ★★★★★ | **default**, recommended |
| `mlx-community/whisper-large-v3-turbo-int4` | 809M quant. | ★★★★★ | ★★★★☆ | memory-saver, slightly lower quality |
| `mlx-community/whisper-large-v3` | 1.54B | ★★ | ★★★★★ | top accuracy, ~4× slower |
| `mlx-community/whisper-small` | 244M | ★★★★★★ | ★★★ | quick pipeline validation |

Small-model sanity check: `stt-local clip.mp4 -m mlx-community/whisper-small -f txt`

---

## Output formats

### SRT (subtitles)
- One cue per segment by default; with `--word-timestamps` each cue's times come from the
  **actual aligned first/last word**
- Long segments wrap at word boundaries (≤80 chars), each wrapped line keeps its own precise timing
- If a segment's word alignment is deemed unreliable (duplicates/misalignment artifacts), that
  segment automatically falls back to **proportional sentence splitting** — duplicated text is never emitted

### MD (reading version)
Paragraphs grouped by speech pauses (>1 s gaps) — for direct reading or feeding to an LLM.

### JSON (structured; preferred for LLM refinement)
```jsonc
{
  "source": "/path/001_video.mp4",   // source file
  "language": "en",                  // detected/specified language
  "duration": 180.0,                 // audio duration (seconds)
  "elapsed": 12.3,                   // this run's transcription time (seconds)
  "segments": [
    {
      "start": 0.0,                  // segment start (seconds)
      "end": 5.9,                    // segment end (seconds)
      "text": "Welcome back to the series.",
      "words": [                     // present only with --word-timestamps
        {"word": "Welcome", "start": 0.0, "end": 0.5},
        {"word": " back",   "start": 0.5, "end": 0.9}
        // ...one entry per word
      ]
    }
  ]
}
```

---

## Resume semantics

Rule: **a media file is skipped when all target-format outputs already exist and are non-empty.**
- Interrupted batch → re-run the same command; finished files `SKIP` instantly, the rest continue
- `--force` re-processes everything
- Atomic writes: output is written to `.tmp` then renamed — no half files

---

## Real-world benchmark (dev machine: Apple M5 / 24 GB)

- 57-second clip: **4.4 s** (≈13× real-time)
- Full run of 176 videos (~19.4 h of audio, with word timestamps + 3 output formats): **~1h41m**
  (≈11.5× real-time), 0 failures
- Output size ~160 KB per episode (srt+md+json) — a 176-episode set is ~28 MB

---

## Where the LLM key lives (no zshrc, no keychain)

Credentials live in a **dedicated lightweight config file** (a mini zshrc that does exactly one
job — but only stt-local reads it; it never enters any shell or repo):

```
~/.config/stt-local/keys.env        # mode 600, readable by you only
```

```bash
# fill in (uncomment to activate)
# STT_LLM_API_KEY=sk-xxx
# STT_LLM_BASE_URL=https://api.deepseek.com/v1
# STT_LLM_MODEL=deepseek-chat

chmod 600 ~/.config/stt-local/keys.env   # verify permissions
```

- Precedence: CLI flags > environment variables > keys.env
- Edits take effect immediately — no sourcing or restarts
- To use the same vars in a shell: `set -a; source ~/.config/stt-local/keys.env; set +a`
- **Migrating machines**: copy this one file along with the project (same pattern as gh's
  `~/.config/gh`)

## LLM polishing (`stt-local polish`)

**Alignment-safe** typo repair for transcribed `.json` files: the LLM only returns corrected
text, one-to-one per segment id — timestamps are always handled by this tool, so subtitles can
never drift. Works with any OpenAI-compatible endpoint.

```bash
# configure (write ~/.config/stt-local/keys.env once, see previous section; or override via env)
# STT_LLM_API_KEY / STT_LLM_BASE_URL / STT_LLM_MODEL

# pipeline self-test first (mock, no network, no cost)
stt-local polish transcripts/videos/ --mock -o /tmp/pol-test

# real run (a glossary is recommended to protect proper nouns)
stt-local polish transcripts/videos/ -o transcripts/videos-polished \
    --glossary terms.txt --jobs 4
```

Key points:
- Writes three files into `-o`: `.json` (with `polished: true` and usage stats), `.srt`, `.md`
- Resume: a checkpoint is written after every batch — interrupted runs never re-bill
- Validation: batches with wrong count/order/length-ratio are retried, then fall back to the
  original text (visible as `fallback` in logs)
- Hallucination handling: segments the LLM deems non-speech garbage become empty strings and
  are skipped in srt/md
- Glossary `terms.txt`: one term per line (e.g. `CLAUDE.md`, `agent skills`, `OpenRouter`)
- Output shape is pinned by a few-shot example + explicit id rules (no renumbering/skipping),
  which cuts invalid-payload retries

## Second-pass QA judge (`stt-local judge`)

An **independent model family** reviews every polish diff (avoiding self-review bias) and
produces an OK/SUSPECT report:

```bash
stt-local judge transcripts/polished/ \
    --orig-root transcripts/ \
    --glossary terms.txt \
    --llm-model openai/gpt-4o-mini \
    -o judge-report.md
```

Key points:
- The original-transcript root is auto-detected by default (a `polished` ancestor path component
  is stripped); override with `--orig-root`
- The judge prompt injects the same glossary — **without it, judges false-positive correct
  product-name fixes as SUSPECT** (hard-earned lesson)
- Strict validation: returned ids must exactly equal the input ids (guards against renumbering);
  int keys normalized end to end
- Resume + `--mock` for offline self-testing
- Calibrated: term fixes / garbage removal → OK; real speech replaced or destroyed → SUSPECT
  (verified in all three directions with a fixture)

---

## Development / CI

The repo ships a GitHub Actions smoke test (`.github/workflows/smoke.yml`):
on a macOS arm64 runner: `uv sync` → transcribe `tests/sample.mp3` with whisper-tiny →
validate srt/md/json outputs. A second job proves `polish`/`judge` on **Ubuntu** (Linux
transcription included).

- Runs automatically on every push to `main` and on new PRs
- **`main` is protected: `transcribe-smoke` must pass before a PR can merge**
- Local CI rehearsal:
  ```bash
  uv sync --frozen
  uv run stt-local tests/sample.mp3 -o tests/out -m mlx-community/whisper-tiny \
    -f srt -f md -f json --word-timestamps
  uv run python tests/smoke_check.py tests/out
  ```

---

## FAQ

**Q: `mlx-whisper is not available`?**
Not on Apple Silicon, or deps are missing. Confirm the chip with
`sysctl -n machdep.cpu.brand_string` (must contain "Apple"), then re-run `uv sync`.

**Q: Windows / Linux / Intel Mac?**
**Transcription works everywhere now**: on non-macOS platforms `uv sync` automatically installs
the faster-whisper engine (CPU alone works; NVIDIA is faster) — identical commands. Macs default
to the mlx engine (switch with `--engine faster`). `polish` / `judge` were always cross-platform.

**Q: ffmpeg-related errors?**
mlx-whisper shells out to the system ffmpeg. macOS: `brew install ffmpeg`.

**Q: Chinese/other-language videos turn into garbled English?**
This tool targets English. For Chinese use FunASR/SenseVoice-family models; or `-l zh`
(Whisper can transcribe Chinese but a dedicated model is more accurate).

**Q: One file keeps FAILing?**
Read the error. For OOM switch to the `-int4` quantized model; a repeatedly failing file is
skipped so the batch continues — check the `FAIL` summary in the log afterwards.

**Q: Weird duplicated words in subtitles?**
Word-alignment artifacts. Detection+fallback is built in; if one still appears, that segment's
word timing was hopeless — re-transcribing that single file with `--force` usually resolves it.

**Q: Can I watch progress?**
Each finished file prints one `[i/n] OK ...` line; add `--verbose` for per-segment progress bars.
For batches, `2>&1 | tee transcribe.log` is recommended.

**Q: Where do outputs go by default?**
Without `-o`, **next to each media file** (same stem, different extension). Use `-o` to keep
source directories clean.

---

## Project structure

```
stt-local-english/
├── pyproject.toml            # dependencies & CLI entry point (stt-local)
├── README.md
└── src/stt_local_english/
    ├── __init__.py           # version & constants (default model, supported formats)
    ├── cli.py                # argparse entry point
    ├── engine.py             # mlx-whisper / faster-whisper backends + normalization
    ├── batch.py              # media discovery, batch loop, resume checks
    ├── polish.py             # alignment-safe LLM polishing
    ├── judge.py              # second-pass QA judge
    ├── llm.py                # OpenAI-compatible client (stdlib only)
    └── output.py             # srt/vtt/md/txt/json renderers + word-level/fallback logic
```

---

## Suggested LLM refinement flow

1. Read each episode's `.json` `segments[].text` and have the LLM fix mis-hearings, punctuation
   and term spelling
2. Output a corrected-text array that maps 1:1 onto `segments` (order must not change)
3. Refill the SRT: keep the timeline (`start/end` or `words`) untouched, replace text only
4. Verify: same cue count as the old SRT, timestamps strictly monotonic

> The `words` array enables finer subtitle refill; for paragraph-level polishing, ignore `words`
> and align on `segments` only.
