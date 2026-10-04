"""回合状态回归：只用 fake ASR/LLM/TTS，不加载模型、不访问网络。"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import threading
import unittest

import numpy as np

from voxck.orchestrator import Session


FIRST = "The claim needs evidence."
SECOND = "Let's discuss the actual argument."
OLD_LONG = "Feminism has become much more about hating men than empowering women."
FRESH = "Equal opportunity does not require identical outcomes."


class FakeWs:
    closed = False

    def __init__(self):
        self.messages = []
        self.binary = []

    async def send_str(self, payload):
        self.messages.append(json.loads(payload))

    async def send_bytes(self, payload):
        self.binary.append(payload)


class FakeVad:
    def __init__(self):
        self.events = []

    def feed(self, _pcm):
        events, self.events = self.events, []
        return events


class FakeAsr:
    def transcribe(self, _pcm):
        return "What supports this claim?"


class FakeLlm:
    cached_tokens = prompt_tokens = 0

    def __init__(self):
        self.requests = []
        self.pause_after_first = None

    async def stream_sentences(self, _http, messages):
        self.requests.append([dict(message) for message in messages])
        yield FIRST, True
        if self.pause_after_first is not None:
            await self.pause_after_first.wait()
        yield SECOND, False


class FakeTts:
    def __init__(self):
        self.interrupts = 0
        self.active = 0
        self.max_active = 0
        self.closed = 0
        self.in_next = threading.Event()
        self.release_next = None
        self.samples_per_segment = 16000
        self.texts = []

    def stream(self, _text):
        self.texts.append(_text)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            yield b"\x00\x00" * self.samples_per_segment
            if self.release_next is not None:
                self.in_next.set()
                if not self.release_next.wait(timeout=2):
                    raise RuntimeError("fake next was not released")
        finally:
            self.active -= 1
            self.closed += 1

    def interrupt(self):
        self.interrupts += 1


class FakeSemantic:
    enabled = False


class ScriptedLlm(FakeLlm):
    """每次请求一份固定输出，第二次请求可挂起以验证 retry 的取消。"""

    def __init__(self, replies, pause_request=None):
        super().__init__()
        self.replies = replies
        self.pause_request = pause_request
        self.waiting = asyncio.Event()
        self.closed = 0

    async def stream_sentences(self, _http, messages):
        self.requests.append([dict(message) for message in messages])
        index = len(self.requests) - 1
        try:
            if index == self.pause_request:
                self.waiting.set()
                await asyncio.Event().wait()
            for i, sentence in enumerate(self.replies[index]):
                yield sentence, i == 0
        finally:
            self.closed += 1


class SessionLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.ws = FakeWs()
        self.vad = FakeVad()
        self.llm = FakeLlm()
        self.tts = FakeTts()
        self.executor = ThreadPoolExecutor(max_workers=1)
        app = {
            "cfg": {"chat_turns": 12, "tts_min_words": 1},
            "make_vad": lambda: self.vad,
            "asr": FakeAsr(), "tts": self.tts, "llm": self.llm,
            "gpu_lock": asyncio.Lock(), "mlx_exec": self.executor,
            "semantic": FakeSemantic(), "http": None,
            "persona": {"prompt": "Debate opponent."},
        }
        self.session = Session(self.ws, app)

    async def asyncTearDown(self):
        if self.tts.release_next is not None:
            self.tts.release_next.set()
        if self.session.reply_task and not self.session.reply_task.done():
            self.session.reply_task.cancel()
            try:
                await self.session.reply_task
            except asyncio.CancelledError:
                pass
        self.executor.shutdown(wait=True)

    async def reply(self):
        task = asyncio.create_task(self.session.respond(np.zeros(16000, dtype=np.float32)))
        self.session.reply_task = task
        await task
        return self.session.turn_id

    async def wait_until(self, condition):
        async with asyncio.timeout(2):
            while not condition():
                await asyncio.sleep(0.005)

    def assistant_history(self):
        return [entry["content"] for entry in self.session.history if entry["role"] == "assistant"]

    async def test_generation_done_waits_for_actual_playback_done_without_face(self):
        turn = await self.reply()
        self.assertFalse(self.session.generating)
        self.assertTrue(self.session.speaking)
        self.assertEqual(self.assistant_history(), [])
        begin = next(m for m in self.ws.messages if m["type"] == "turn")
        self.assertEqual(begin, {"type": "turn", "id": turn, "fps": 0, "face": False})
        done = next(m for m in self.ws.messages if m["type"] == "generation_done")
        self.assertEqual(done["audio_ms"], 2000)
        states = [m["state"] for m in self.ws.messages if m["type"] == "state"]
        self.assertEqual(states, ["speaking"])

        await self.session.on_control({"type": "playback_done", "id": turn, "played_ms": 2000})
        self.assertFalse(self.session.speaking)
        self.assertIsNone(self.session.active_turn)
        self.assertEqual(self.assistant_history(), [FIRST + " " + SECOND])
        self.assertEqual(self.ws.messages[-1], {"type": "state", "state": "listening"})

    async def test_speech_start_interrupts_generated_but_buffered_audio_and_trims_history(self):
        old = await self.reply()
        self.vad.events = [("speech_start", None)]
        await self.session.on_audio(b"\x00\x00" * 128)
        self.assertFalse(self.session.speaking)
        self.assertTrue(self.session.face_stop.is_set())
        self.assertIn({"type": "interrupt", "id": old}, self.ws.messages)
        self.assertEqual(self.tts.interrupts, 1)

        # 清队列后的迟到帧必须丢弃；只有用户实际听完整的第一段进历史。
        n = len(self.ws.binary)
        await self.session.send_frame(old, 0, b"late jpeg")
        self.assertEqual(len(self.ws.binary), n)
        await self.session.on_control({"type": "playback_progress", "id": old, "played_ms": 1500})
        self.assertEqual(self.assistant_history(), [FIRST])
        new = await self.reply()
        self.assertEqual(new, old + 1)
        self.assertTrue(self.session.speaking)
        previous = [m["content"] for m in self.llm.requests[-1] if m["role"] == "assistant"]
        self.assertEqual(previous, [FIRST])

        # 旧回合 ACK 不能结束新回合。
        await self.session.on_control({"type": "playback_done", "id": old, "played_ms": 1500})
        self.assertTrue(self.session.speaking)
        self.assertEqual(self.session.active_turn.id, new)
        await self.session.on_control({"type": "playback_done", "id": new})
        self.assertFalse(self.session.speaking)
        self.assertEqual(self.tts.max_active, 1)

    async def test_interrupt_during_generation_closes_old_stream_before_new_turn(self):
        self.llm.pause_after_first = asyncio.Event()
        task = asyncio.create_task(self.session.respond(np.zeros(16000, dtype=np.float32)))
        self.session.reply_task = task
        await self.wait_until(lambda: self.tts.closed == 1)
        old = self.session.turn_id
        self.vad.events = [("speech_start", None)]
        await self.session.on_audio(b"\x00\x00" * 128)
        self.assertTrue(task.cancelled())
        self.assertFalse(any(m["type"] == "generation_done" for m in self.ws.messages))
        self.assertEqual(self.tts.active, 0)
        await self.session.on_control({"type": "playback_progress", "id": old, "played_ms": 1000})
        self.assertEqual(self.assistant_history(), [FIRST])
        self.llm.pause_after_first = None
        await self.reply()
        self.assertEqual(self.tts.max_active, 1)

    async def test_cancel_waits_for_in_flight_executor_next_before_closing_generator(self):
        self.tts.release_next = threading.Event()
        task = asyncio.create_task(self.session.respond(np.zeros(16000, dtype=np.float32)))
        self.session.reply_task = task
        await self.wait_until(self.tts.in_next.is_set)
        old = self.session.turn_id
        self.vad.events = [("speech_start", None)]
        asyncio.get_running_loop().call_later(0.02, self.tts.release_next.set)
        await self.session.on_audio(b"\x00\x00" * 128)
        self.assertTrue(task.cancelled())
        self.assertEqual(self.tts.closed, 1)
        self.assertEqual(self.tts.active, 0)
        self.assertIn({"type": "interrupt", "id": old}, self.ws.messages)
        self.tts.release_next = None
        await self.reply()
        self.assertEqual(self.tts.max_active, 1)

    async def test_unknown_and_invalid_playback_reports_do_not_change_active_turn(self):
        turn = await self.reply()
        for payload in (
            {"type": "playback_done", "id": turn + 99, "played_ms": 2000},
            {"type": "playback_done", "id": "invalid", "played_ms": 2000},
            {"type": "playback_progress", "id": turn, "played_ms": -1},
            {"type": "playback_progress", "id": turn, "played_ms": float("nan")},
        ):
            await self.session.on_control(payload)
        self.assertTrue(self.session.speaking)
        self.assertEqual(self.assistant_history(), [])

    async def test_natural_done_normalizes_sample_clock_float_boundary(self):
        class SingleSentenceLlm(FakeLlm):
            async def stream_sentences(self, _http, _messages):
                yield FIRST, True

        self.session.llm = SingleSentenceLlm()
        self.tts.samples_per_segment = 1001
        turn = await self.reply()
        reported_ms = 1001 / 16000 * 1000
        self.assertLess(reported_ms, self.session.active_turn.audio_ms)

        # 中途 progress 不能凭接近段尾就冒充整段已听完。
        await self.session.on_control({"type": "playback_progress", "id": turn, "played_ms": reported_ms})
        self.assertEqual(self.assistant_history(), [])
        await self.session.on_control({"type": "playback_done", "id": turn, "played_ms": reported_ms})
        self.assertEqual(self.assistant_history(), [FIRST])
        self.assertFalse(self.session.speaking)

    async def test_repeated_long_opener_is_skipped_but_fresh_followup_is_spoken(self):
        self.session.repeats.add(OLD_LONG)
        llm = self.session.llm = ScriptedLlm([(OLD_LONG, FRESH)])
        await self.reply()
        self.assertEqual(self.tts.texts, [FRESH])
        self.assertEqual(len(llm.requests), 1)
        self.assertEqual(llm.requests[0][-1], {"role": "user", "content": "What supports this claim?"})
        self.assertEqual([m["text"] for m in self.ws.messages if m["type"] == "assistant"], [FRESH])

    async def test_repeat_only_retries_once_and_keeps_latest_user_last(self):
        self.session.repeats.add(OLD_LONG)
        llm = self.session.llm = ScriptedLlm([(OLD_LONG,), (OLD_LONG, FRESH)])
        await self.reply()
        self.assertEqual(self.tts.texts, [FRESH])
        self.assertEqual(len(llm.requests), 2)
        retry = llm.requests[1]
        self.assertEqual(retry[-1], {"role": "user", "content": "What supports this claim?"})
        self.assertEqual(retry[-2]["role"], "system")
        self.assertIn("latest user's specific question", retry[-2]["content"])
        self.assertIn(OLD_LONG, retry[-2]["content"])

    async def test_two_repeat_only_attempts_do_not_fallback_to_old_audio(self):
        self.session.repeats.add(OLD_LONG)
        llm = self.session.llm = ScriptedLlm([(OLD_LONG,), (OLD_LONG,)])
        turn = await self.reply()
        self.assertEqual(len(llm.requests), 2)
        self.assertEqual(self.tts.texts, [])
        self.assertEqual(self.ws.binary, [])
        self.assertFalse(any(m["type"] == "assistant" for m in self.ws.messages))
        self.assertTrue(any(m["type"] == "sys" and "fresh reply" in m["text"] for m in self.ws.messages))
        await self.session.on_control({"type": "playback_done", "id": turn, "played_ms": 0})
        self.assertFalse(self.session.speaking)

    async def test_explicit_explanation_allows_restatement_and_short_opener_stays_natural(self):
        class ExplainAsr:
            def transcribe(self, _pcm):
                return "Can you explain that again?"

        self.session.asr = ExplainAsr()
        self.session.repeats.add(OLD_LONG)
        self.session.llm = ScriptedLlm([(OLD_LONG,)])
        turn = await self.reply()
        self.assertEqual(self.tts.texts, [OLD_LONG])
        await self.session.on_control({"type": "playback_done", "id": turn})
        self.session.asr = FakeAsr()
        self.session.repeats.add(FIRST)
        self.session.llm = ScriptedLlm([(FIRST,)])
        await self.reply()
        self.assertEqual(self.tts.texts, [OLD_LONG, FIRST])

    async def test_retry_can_be_cancelled_and_closes_both_streams(self):
        self.session.repeats.add(OLD_LONG)
        llm = self.session.llm = ScriptedLlm([(OLD_LONG,), (FRESH,)], pause_request=1)
        task = asyncio.create_task(self.session.respond(np.zeros(16000, dtype=np.float32)))
        self.session.reply_task = task
        await asyncio.wait_for(llm.waiting.wait(), timeout=2)
        self.vad.events = [("speech_start", None)]
        await self.session.on_audio(b"\x00\x00" * 128)
        self.assertTrue(task.cancelled())
        self.assertEqual(len(llm.requests), 2)
        self.assertEqual(llm.closed, 2)
        self.assertEqual(self.tts.texts, [])
        self.assertFalse(any(m["type"] == "generation_done" for m in self.ws.messages))

    async def test_content_violation_does_not_trigger_repeat_retry(self):
        self.session.repeats.add(OLD_LONG)
        llm = self.session.llm = ScriptedLlm([(OLD_LONG, "According to G01, this is settled.")])
        await self.reply()
        self.assertEqual(len(llm.requests), 1)
        self.assertEqual(self.tts.texts, [])
        self.assertTrue(self.session.app["llm_ready"])  # 正文已到，审核拒绝不等于 API 断网。
        self.assertTrue(any(m["type"] == "sys" and "fresh reply" in m["text"]
                            for m in self.ws.messages))

    async def test_empty_eof_clears_stale_brain_readiness_and_reports_error(self):
        self.session.app["llm_ready"] = True
        self.session.llm = ScriptedLlm([()])
        await self.reply()
        self.assertFalse(self.session.app["llm_ready"])
        self.assertEqual(self.tts.texts, [])
        self.assertTrue(any(m["type"] == "sys" and "without answer text" in m["text"]
                            for m in self.ws.messages))


if __name__ == "__main__":
    unittest.main()
