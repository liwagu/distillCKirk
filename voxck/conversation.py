"""Private local audio and text archives, independent of live reply cancellation.

The JSONL file is an append-only event log. For assistant playback updates,
the latest event for a given ``turn`` replaces that turn's earlier text.
This module never sends archives to a model or any other network service.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import uuid
import wave

import numpy as np


SAMPLE_RATE = 16_000
MAX_RESUME_BYTES = 4 * 1024 * 1024
MAX_RESUME_MESSAGES = 256
MAX_RESUME_CONTENT = 100_000
MAX_RESUME_TOTAL = 1_000_000


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _unique_name(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex}"


def _private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)


class ResumeError(ValueError):
    """The resume seed was rejected and has not been consumed."""


class ConversationJournal:
    """One connection's durable local archive beneath ``root``.

    Calls are synchronous so ``save_audio`` completes before work is queued.
    I/O errors are deliberately propagated: callers must report a failed
    backup rather than claiming that audio was saved.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        _private_dir(self.root)
        self.directory = self.root / _unique_name("session")
        _private_dir(self.directory)
        self.audio_directory = self.directory / "audio"
        _private_dir(self.audio_directory)
        self.events_path = self.directory / "events.jsonl"
        fd = os.open(self.events_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self._events = os.fdopen(fd, "w", encoding="utf-8")
        self._closed = False
        self._append("session_started", sample_rate=SAMPLE_RATE)

    def _append(self, kind: str, **fields) -> None:
        if self._closed:
            raise RuntimeError("Conversation journal is closed")
        event = {"type": kind, "timestamp": _timestamp(), **fields}
        self._events.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._events.flush()
        os.fsync(self._events.fileno())

    def save_audio(self, wav: np.ndarray) -> str:
        """Atomically save mono float audio as a private PCM16 WAV, then log it."""
        if self._closed:
            raise RuntimeError("Conversation journal is closed")
        samples = np.asarray(wav, dtype=np.float32)
        if samples.ndim != 1 or not samples.size or not np.isfinite(samples).all():
            raise ValueError("Audio must be a nonempty, finite, one-dimensional array")
        clip_id = uuid.uuid4().hex
        destination = self.audio_directory / f"{clip_id}.wav"
        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
        fd, temporary = tempfile.mkstemp(prefix=f".{clip_id}-", suffix=".tmp", dir=self.audio_directory)
        try:
            with os.fdopen(fd, "wb") as output:
                with wave.open(output, "wb") as audio:
                    audio.setnchannels(1)
                    audio.setsampwidth(2)
                    audio.setframerate(SAMPLE_RATE)
                    audio.writeframes(pcm.tobytes())
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, destination)
            self._append("audio_saved", clip_id=clip_id,
                         path=str(destination.relative_to(self.directory)),
                         sample_rate=SAMPLE_RATE, samples=int(samples.size),
                         duration_s=float(samples.size / SAMPLE_RATE))
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return clip_id

    def record_user(self, clip_id: str, text: str) -> None:
        self._append("user", clip_id=clip_id, role="user", content=text)

    def record_assistant(self, turn: int, text: str, interrupted: bool = False,
                         after_clip_id: str = "") -> None:
        self._append("assistant", turn=turn, role="assistant", content=text,
                     interrupted=bool(interrupted), after_clip_id=after_clip_id)

    def record_error(self, clip_id: str, error: str | Exception) -> None:
        self._append("error", clip_id=clip_id, error=str(error))

    def close(self) -> None:
        """Finalize the journal; repeated closes are harmless."""
        if self._closed:
            return
        try:
            self._append("session_closed")
        finally:
            self._closed = True
            self._events.close()


def load_resume(root: Path) -> list[dict]:
    """Load a one-use ``resume-next.json`` seed and archive it after validation.

    The seed is ``{"messages": [{"role": "user"|"assistant", "content": "..."}]}``.
    Invalid data raises ``ResumeError`` and remains at its original path.
    Missing data returns an empty list. No message is generated automatically.
    """
    root = Path(root)
    seed = root / "resume-next.json"
    try:
        with seed.open("rb") as source:
            raw = source.read(MAX_RESUME_BYTES + 1)
    except FileNotFoundError:
        return []
    if len(raw) > MAX_RESUME_BYTES:
        raise ResumeError("Resume seed is too large")
    try:
        document = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as error:
        raise ResumeError("Resume seed is not valid JSON") from error
    if not isinstance(document, dict) or not isinstance(document.get("messages"), list):
        raise ResumeError("Resume seed must contain a messages list")
    source_messages = document["messages"]
    if not source_messages or len(source_messages) > MAX_RESUME_MESSAGES:
        raise ResumeError(f"Resume seed must contain 1 to {MAX_RESUME_MESSAGES} messages")
    messages = []
    total = 0
    for index, message in enumerate(source_messages):
        if not isinstance(message, dict) or message.get("role") not in ("user", "assistant"):
            raise ResumeError(f"Resume message {index + 1} has an invalid role")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip() or len(content) > MAX_RESUME_CONTENT:
            raise ResumeError(f"Resume message {index + 1} has invalid or oversized content")
        total += len(content)
        if total > MAX_RESUME_TOTAL:
            raise ResumeError("Resume conversation is too large")
        messages.append({"role": message["role"], "content": content})
    _private_dir(root)
    seed.chmod(0o600)
    # A unique destination preserves every consumed seed instead of overwriting it.
    seed.rename(root / f"{_unique_name('resume-consumed')}.json")
    return messages
