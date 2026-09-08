"""stt-local-english: local English speech-to-text CLI on Apple Silicon."""

__version__ = "0.1.0"

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"
SUPPORTED_AUDIO_EXTS = {
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma",
}
SUPPORTED_VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".ts", ".m4v"}
SUPPORTED_MEDIA_EXTS = SUPPORTED_AUDIO_EXTS | SUPPORTED_VIDEO_EXTS
