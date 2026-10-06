"""Completed microphone input must survive reply interruption and shutdown.

These regressions use fake models and VAD events only; no weights or network.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import threading
import unittest
import wave

import numpy as np

from voxck.orchestrator import Session


FRAGMENTS = {
    1: "I build AI tools and share what I learn.",
    2: "My audience becomes a distribution channel for my products.",
    3: "The product still needs to solve a real customer problem.",
}


class FakeWs:
    def __init__(self):
        self.closed = False
        self.fail_text = False
        self.failed_sends = 0
        self.messages = []
        self.binary = []

    async def send_str(self, payload):
        if self.fail_text:
            self.failed_sends += 1
            raise ConnectionResetError("fake transport reset before closed flag updated")
        self.messages.append(json.loads(payload))

    async def send_bytes(self, payload):
        self.binary.append(payload)


class FakeVad:
    def __init__(self):
        self.events = []

    def feed(self, _pcm):
        events, self.events = self.events, []
        return events


class MarkerAsr:
    def __init__(self):
        self.calls = []
        self.failures = {}
        self.block_marker = None
        self.entered = threading.Event()
        self.release = threading.Event()

    def transcribe(self, pcm):
        marker = int(pcm[0])
        self.calls.append(marker)
        if marker == self.block_marker:
            self.entered.set()
            if not self.release.wait(timeout=5):
                raise RuntimeError("fake ASR was not released")
        if marker in self.failures:
            failure = self.failures[marker]
            if isinstance(failure, Exception):
                raise failure
            return failure
        return FRAGMENTS[marker]


class RecordingLlm:
    cached_tokens = prompt_tokens = 0

    def __init__(self):
        self.requests = []

    async def stream_sentences(self, _http, messages):
        self.requests.append([dict(message) for message in messages])
        yield "A useful product gives the audience a reason to return.", True


class FakeTts:
    def __init__(self):
        self.interrupts = 0

    def stream(self, _text):
        yield b"\x00\x00" * 160

    def interrupt(self):
        self.interrupts += 1


class FakeSemantic:
    enabled = False


def clip(marker):
    return np.full(1600, marker, dtype=np.float32)


class AsrIngestionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.ws = FakeWs()
        self.vad = FakeVad()
        self.asr = MarkerAsr()
        self.llm = RecordingLlm()
        self.tts = FakeTts()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.app = {
            "cfg": {"chat_turns": 12, "tts_min_words": 1},
            "make_vad": lambda: self.vad,
            "asr": self.asr,
            "tts": self.tts,
            "llm": self.llm,
            "gpu_lock": asyncio.Lock(),
            "mlx_exec": self.executor,
            "semantic": FakeSemantic(),
            "http": None,
            "persona": {"prompt": "Debate opponent."},
        }
        self.session = Session(self.ws, self.app)

    async def asyncTearDown(self):
        self.asr.release.set()
        await self.session.close()
        self.executor.shutdown(wait=True)

    async def wait_until(self, condition):
        async with asyncio.timeout(2):
            while not condition():
                await asyncio.sleep(0.005)

    async def feed(self, *events):
        self.vad.events = list(events)
        await self.session.on_audio(b"\x00\x00" * 128)

    def user_history(self):
        return [message["content"] for message in self.session.history
                if message["role"] == "user"]

    def shown_users(self):
        return [message["text"] for message in self.ws.messages
                if message["type"] == "user"]

    def assert_complete_argument(self, request, expected):
        # Request preparation may collapse adjacent user fragments into one
        # message. What matters is that every fragment arrives once and in order.
        user_text = "\n".join(message["content"] for message in request
                              if message["role"] == "user")
        offsets = []
        for fragment in expected:
            self.assertEqual(user_text.count(fragment), 1)
            offsets.append(user_text.index(fragment))
        self.assertEqual(offsets, sorted(offsets))

    async def test_same_vad_batch_preserves_complete_clip_before_next_speech_start(self):
        # The old code created respond(), then canceled it before it even ran.
        await self.feed(("speech_end", None), ("turn_complete", clip(1)),
                        ("speech_start", None))
        await self.wait_until(lambda: len(self.user_history()) == 1)
        self.assertEqual(self.user_history(), [FRAGMENTS[1]])
        self.assertEqual(self.shown_users(), [FRAGMENTS[1]])
        self.assertEqual(self.llm.requests, [])  # The user is still speaking.

        await self.feed(("speech_end", None), ("turn_complete", clip(2)))
        await self.wait_until(lambda: bool(self.llm.requests))
        self.assert_complete_argument(self.llm.requests[-1],
                                      [FRAGMENTS[1], FRAGMENTS[2]])

    async def test_next_speech_start_does_not_cancel_in_flight_asr(self):
        self.asr.block_marker = 1
        await self.feed(("speech_start", None), ("speech_end", None),
                        ("turn_complete", clip(1)))
        await self.wait_until(self.asr.entered.is_set)
        await asyncio.wait_for(self.feed(("speech_start", None)), timeout=0.5)
        self.asr.release.set()
        await self.wait_until(lambda: self.shown_users() == [FRAGMENTS[1]])
        self.assertEqual(self.user_history(), [FRAGMENTS[1]])
        self.assertEqual(self.llm.requests, [])

        await self.feed(("speech_end", None), ("turn_complete", clip(2)))
        await self.wait_until(lambda: bool(self.llm.requests))
        self.assert_complete_argument(self.llm.requests[-1],
                                      [FRAGMENTS[1], FRAGMENTS[2]])

    async def test_multiple_completed_clips_are_transcribed_in_order_before_answer(self):
        await self.feed(
            ("speech_start", None), ("speech_end", None), ("turn_complete", clip(1)),
            ("speech_start", None), ("speech_end", None), ("turn_complete", clip(2)),
            ("speech_start", None), ("speech_end", None), ("turn_complete", clip(3)),
        )
        await self.wait_until(lambda: bool(self.llm.requests))
        expected = [FRAGMENTS[1], FRAGMENTS[2], FRAGMENTS[3]]
        self.assertEqual(self.asr.calls, [1, 2, 3])
        self.assertEqual(self.shown_users(), expected)
        self.assertEqual(self.user_history(), expected)
        self.assertEqual(len(self.llm.requests), 1)
        self.assert_complete_argument(self.llm.requests[0], expected)

    async def test_close_drains_accepted_completed_clips_without_starting_reply(self):
        self.asr.block_marker = 1
        await self.feed(("speech_start", None), ("speech_end", None),
                        ("turn_complete", clip(1)))
        await self.wait_until(self.asr.entered.is_set)
        await self.feed(("speech_start", None), ("speech_end", None),
                        ("turn_complete", clip(2)))
        self.ws.closed = True
        closing = asyncio.create_task(self.session.close())
        # Let close mark the session as closing before releasing executor work.
        await asyncio.sleep(0)
        self.asr.release.set()
        await asyncio.wait_for(closing, timeout=2)
        self.assertEqual(self.asr.calls, [1, 2])
        self.assertEqual(self.user_history(), [FRAGMENTS[1], FRAGMENTS[2]])
        self.assertEqual(self.llm.requests, [])
        self.assertEqual(self.session.input_queue.qsize(), 0)
        self.assertTrue(self.session.input_task is None or self.session.input_task.done())

    async def assert_failed_fragment_is_reported_without_reanswering_previous(self, failure):
        self.asr.failures[2] = failure
        # A valid first fragment has arrived, but the user is still speaking.
        await self.feed(("speech_end", None), ("turn_complete", clip(1)),
                        ("speech_start", None))
        await self.wait_until(lambda: self.user_history() == [FRAGMENTS[1]])
        await self.feed(("speech_end", None), ("turn_complete", clip(2)))
        await self.session.input_queue.join()
        await asyncio.sleep(0)  # Expose an incorrectly scheduled stale reply.
        self.assertTrue(any(message["type"] == "sys" and "repeat that part" in message["text"]
                            for message in self.ws.messages))
        self.assertEqual(self.user_history(), [FRAGMENTS[1]])
        self.assertEqual(self.shown_users(), [FRAGMENTS[1]])
        self.assertEqual(self.llm.requests, [])
        # An ASR failure must not kill ingestion of the next good segment.
        await self.feed(("speech_start", None), ("speech_end", None),
                        ("turn_complete", clip(3)))
        await self.wait_until(lambda: bool(self.llm.requests))
        self.assertEqual(self.asr.calls, [1, 2, 3])
        self.assert_complete_argument(self.llm.requests[-1], [FRAGMENTS[1], FRAGMENTS[3]])

    async def test_empty_asr_reports_missing_segment_and_does_not_answer_previous_text(self):
        await self.assert_failed_fragment_is_reported_without_reanswering_previous("")

    async def test_failed_asr_reports_missing_segment_and_next_good_clip_still_works(self):
        with self.assertLogs("voxck", level="ERROR"):
            await self.assert_failed_fragment_is_reported_without_reanswering_previous(
                RuntimeError("fake ASR decode failed"))

    async def test_transport_reset_still_drains_two_clips_into_history_and_local_archive(self):
        await self.session.close()
        with tempfile.TemporaryDirectory() as directory:
            self.app["conversation_root"] = Path(directory)
            self.session = Session(self.ws, self.app)
            self.asr.block_marker = 1
            await self.feed(("speech_start", None), ("speech_end", None),
                            ("turn_complete", clip(1)))
            await self.wait_until(self.asr.entered.is_set)
            # The socket can fail before aiohttp changes its closed property.
            self.ws.fail_text = True
            self.assertFalse(self.ws.closed)
            await self.feed(("speech_start", None), ("speech_end", None),
                            ("turn_complete", clip(2)))
            closing = asyncio.create_task(self.session.close())
            await asyncio.sleep(0)
            self.asr.release.set()
            await asyncio.wait_for(closing, timeout=2)
            self.assertGreater(self.ws.failed_sends, 0)
            self.assertFalse(self.ws.closed)
            self.assertEqual(self.asr.calls, [1, 2])
            self.assertEqual(self.user_history(), [FRAGMENTS[1], FRAGMENTS[2]])
            self.assertEqual(self.llm.requests, [])
            self.assertEqual(self.session.input_queue.qsize(), 0)
            events = [json.loads(line) for line in
                      self.session.journal.events_path.read_text().splitlines()]
            self.assertEqual([event["content"] for event in events if event["type"] == "user"],
                             [FRAGMENTS[1], FRAGMENTS[2]])
            self.assertEqual(events[-1]["type"], "session_closed")
            saved_audio = [event for event in events if event["type"] == "audio_saved"]
            self.assertEqual(len(saved_audio), 2)
            for event in saved_audio:
                with wave.open(str(self.session.journal.directory / event["path"]), "rb") as audio:
                    self.assertEqual(audio.getnframes(), 1600)
                    self.assertEqual(audio.getframerate(), 16000)

    async def test_cancel_before_reply_body_starts_then_answer_next_fragment_only_once(self):
        await self.feed(("speech_end", None), ("turn_complete", clip(1)))
        # task_done wakes this waiter before the newly scheduled answer runs.
        await self.session.input_queue.join()
        first_reply = self.session.reply_task
        self.assertIsNotNone(first_reply)
        self.assertFalse(first_reply.done())
        await self.feed(("speech_start", None))
        self.assertTrue(first_reply.cancelled())
        self.assertEqual(self.llm.requests, [])

        await self.feed(("speech_end", None), ("turn_complete", clip(2)))
        await self.session.input_queue.join()
        await self.wait_until(lambda: bool(self.llm.requests))
        await self.session.reply_task
        # Ordinary subsequent microphone packets must not rerun this reply.
        await self.feed()
        await asyncio.sleep(0)
        self.assertEqual(len(self.llm.requests), 1)
        self.assert_complete_argument(self.llm.requests[0], [FRAGMENTS[1], FRAGMENTS[2]])


if __name__ == "__main__":
    unittest.main()
