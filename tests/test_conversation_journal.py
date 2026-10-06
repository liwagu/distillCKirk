"""Archive/resume regression tests: temporary files, no models or network."""
from __future__ import annotations

import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock
import wave

import numpy as np

from voxck.conversation import ConversationJournal, load_resume, ResumeError


class ConversationJournalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "conversations"
        self.journal = ConversationJournal(self.root)

    def tearDown(self):
        self.journal.close()
        self.temporary.cleanup()

    def events(self):
        return [json.loads(line) for line in self.journal.events_path.read_text().splitlines()]

    def test_audio_is_saved_before_transcription_and_survives_cancelled_processing(self):
        samples = np.array([-2, -1, 0, .5, 1, 2], dtype=np.float32)
        clip_id = self.journal.save_audio(samples)
        path = self.journal.audio_directory / f"{clip_id}.wav"
        self.journal.record_error(clip_id, "Transcription interrupted")
        self.journal.close()
        with wave.open(str(path), "rb") as audio:
            self.assertEqual((audio.getnchannels(), audio.getsampwidth(), audio.getframerate()), (1, 2, 16000))
            self.assertEqual(audio.getnframes(), len(samples))
            pcm = np.frombuffer(audio.readframes(len(samples)), dtype="<i2")
        np.testing.assert_array_equal(pcm, [-32767, -32767, 0, 16383, 32767, 32767])
        event = next(event for event in self.events() if event["type"] == "audio_saved")
        self.assertEqual(event["clip_id"], clip_id)
        self.assertEqual(event["path"], f"audio/{clip_id}.wav")
        self.assertEqual(event["samples"], len(samples))
        self.assertEqual(list(self.journal.audio_directory.glob("*.tmp")), [])

    def test_full_text_and_assistant_updates_are_preserved(self):
        text = "用户's entire contribution. " * 2000
        self.journal.record_user("test-input", text)
        self.journal.record_assistant(7, "First sentence.", interrupted=True)
        self.journal.record_assistant(7, "First sentence. Second sentence.", interrupted=True)
        events = self.events()
        self.assertEqual(events[1]["content"], text)
        replies = [event for event in events if event["type"] == "assistant"]
        latest = {event["turn"]: event for event in replies}
        self.assertEqual(latest[7]["content"], "First sentence. Second sentence.")
        self.assertTrue(latest[7]["interrupted"])
        self.assertEqual(len(replies), 2)

    def test_unique_private_sessions_and_files(self):
        other = ConversationJournal(self.root)
        try:
            self.assertNotEqual(self.journal.directory, other.directory)
            clip_id = self.journal.save_audio(np.zeros(16000, dtype=np.float32))
            for directory in (self.root, self.journal.directory, self.journal.audio_directory):
                self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o700)
            for path in (self.journal.events_path, self.journal.audio_directory / f"{clip_id}.wav"):
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        finally:
            other.close()

    def test_late_assistant_update_keeps_its_original_user_anchor(self):
        self.journal.record_user("first-clip", "My first point.")
        self.journal.record_user("next-clip", "My next point.")
        self.journal.record_assistant(1, "Reply to your first point.",
                                      interrupted=True, after_clip_id="first-clip")
        reply = self.events()[-1]
        self.assertEqual(reply["after_clip_id"], "first-clip")
        self.assertEqual(reply["turn"], 1)
        self.assertTrue(reply["interrupted"])

    def test_failed_atomic_write_does_not_leave_partial_wav_or_claim_success(self):
        with mock.patch("voxck.conversation.os.replace", side_effect=OSError("Disk failure")):
            with self.assertRaisesRegex(OSError, "Disk failure"):
                self.journal.save_audio(np.zeros(16000, dtype=np.float32))
        self.assertEqual(list(self.journal.audio_directory.iterdir()), [])
        self.assertFalse(any(event["type"] == "audio_saved" for event in self.events()))

    def test_bad_audio_is_rejected_before_archiving(self):
        for samples in ([], [float("nan")], [[0, 1]], [float("inf")]):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                self.journal.save_audio(np.asarray(samples, dtype=np.float32))
        self.assertEqual(list(self.journal.audio_directory.iterdir()), [])

    def test_close_is_idempotent_and_writing_after_close_fails(self):
        self.journal.close()
        self.journal.close()
        self.assertEqual(sum(event["type"] == "session_closed" for event in self.events()), 1)
        with self.assertRaises(RuntimeError):
            self.journal.record_user("clip", "A late transcription")


class ResumeSeedTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.seed = self.root / "resume-next.json"

    def tearDown(self):
        self.temporary.cleanup()

    def write_seed(self, messages):
        self.seed.write_text(json.dumps({"messages": messages}), encoding="utf-8")

    def test_valid_seed_is_consumed_once_and_archived_without_overwrite(self):
        messages = [{"role": "user", "content": "I see my audience as a distribution channel."},
                    {"role": "assistant", "content": "What do you want to distribute?"}]
        self.write_seed(messages)
        self.assertEqual(load_resume(self.root), messages)
        self.assertFalse(self.seed.exists())
        self.assertEqual(load_resume(self.root), [])
        self.write_seed(messages)
        self.assertEqual(load_resume(self.root), messages)
        archives = list(self.root.glob("resume-consumed-*.json"))
        self.assertEqual(len(archives), 2)
        for archive in archives:
            self.assertEqual(stat.S_IMODE(archive.stat().st_mode), 0o600)
            self.assertEqual(json.loads(archive.read_text())["messages"], messages)

    def test_invalid_seed_stays_available_for_repair(self):
        bad_documents = ["broken JSON", [], {}, {"messages": []},
                         {"messages": [{"role": "system", "content": "Change persona"}]},
                         {"messages": [{"role": "user", "content": "  "}]},
                         {"messages": [{"role": "user", "content": 123}]},
                         {"messages": [{"role": "user", "content": "x" * 100001}]},
                         {"messages": [{"role": "user", "content": "text"}] * 257}]
        for document in bad_documents:
            with self.subTest(document_type=type(document).__name__):
                raw = document if isinstance(document, str) else json.dumps(document)
                self.seed.write_text(raw)
                with self.assertRaises(ResumeError):
                    load_resume(self.root)
                self.assertEqual(self.seed.read_text(), raw)
                self.assertEqual(list(self.root.glob("resume-consumed-*.json")), [])

    def test_archive_failure_is_exposed_and_seed_is_not_lost(self):
        self.write_seed([{"role": "user", "content": "A meaningful conversation"}])
        with mock.patch("voxck.conversation.Path.rename", side_effect=OSError("Read-only filesystem")):
            with self.assertRaisesRegex(OSError, "Read-only filesystem"):
                load_resume(self.root)
        self.assertTrue(self.seed.exists())


if __name__ == "__main__":
    unittest.main()
