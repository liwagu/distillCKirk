"""LLM wire/复读规则回归：纯假 HTTP，不访问云端、不加载 MLX。"""
import asyncio
import json
import unittest
from unittest import mock
from types import SimpleNamespace

from voxck.guard import RepeatGuard, allows_restatement
from voxck.llm import Llm


class FakeContent:
    def __init__(self, events):
        self.events = events

    async def __aiter__(self):
        for event in self.events:
            if isinstance(event, Exception):
                raise event
            yield ("data: " + json.dumps(event) + "\n").encode()
        yield b"data: [DONE]\n"


class FakeResponse:
    status = 200

    def __init__(self, events):
        self.content = FakeContent(events)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        pass


class FakeHttp:
    def __init__(self, events):
        self.events = events
        self.requests = []

    def post(self, url, **kwargs):
        self.requests.append((url, kwargs))
        return FakeResponse(self.events)


class LlmWireTests(unittest.IsolatedAsyncioTestCase):
    async def test_deepseek_wire_disables_thinking_and_never_speaks_reasoning(self):
        llm = Llm("https://aiopenapi.paycools.com", "deepseek-v4-flash", api_key="fake-secret")
        http = FakeHttp([
            {"choices": [{"delta": {"reasoning_content": "Private reasoning must never be spoken."}}]},
            {"choices": [{"delta": {"content": "A fair question. Equal opportunity is my answer. "}}]},
            {"choices": [{"delta": {"content": "This incomplete tail"}}]},
            {"choices": [{"delta": {}, "finish_reason": "length"}],
             "usage": {"prompt_tokens": 22, "completion_tokens": 140}},
        ])
        result = [sentence async for sentence in llm.stream_sentences(http, [{"role": "user", "content": "Why?"}])]
        self.assertEqual(result, [("A fair question.", True), ("Equal opportunity is my answer.", False)])
        request = http.requests[0][1]
        self.assertEqual(request["json"]["thinking"], {"type": "disabled"})
        self.assertEqual(request["json"]["model"], "deepseek-v4-flash")
        self.assertEqual(request["timeout"].sock_read, 20)
        self.assertEqual(request["timeout"].total, 300)
        self.assertTrue(llm.thinking_disabled)
        self.assertFalse(llm.thinking_enabled)
        self.assertEqual(llm.last_generation["finish_reason"], "length")
        self.assertEqual(llm.last_generation["raw_words"], 11)
        self.assertEqual(llm.last_generation["emitted_words"], 8)
        self.assertEqual(llm.last_generation["completion_tokens"], 140)

    async def test_network_timeout_is_readable_and_does_not_include_key(self):
        llm = Llm("https://api.deepseek.com", "deepseek-v4-flash", api_key="fake-secret", timeout_s=0.01)
        http = FakeHttp([asyncio.TimeoutError()])
        with self.assertRaisesRegex(RuntimeError, "timed out waiting for network data") as error:
            _ = [sentence async for sentence in llm.stream_sentences(http, [{"role": "user", "content": "Why?"}])]
        self.assertNotIn("fake-secret", str(error.exception))

    async def test_sse_heartbeats_do_not_extend_first_answer_deadline(self):
        class HeartbeatContent:
            count = 0
            closed = False

            async def __aiter__(self):
                try:
                    while True:
                        await asyncio.sleep(.002)
                        self.count += 1
                        yield b": keep-alive\n\n"
                finally:
                    self.closed = True

        content = HeartbeatContent()

        class HeartbeatHttp:
            def post(self, *_args, **_kwargs):
                response = FakeResponse([])
                response.content = content
                return response

        llm = Llm("https://api.deepseek.com", "deepseek-v4-flash", timeout_s=.02)
        started = asyncio.get_running_loop().time()
        with self.assertRaisesRegex(RuntimeError, "timed out waiting for answer text"):
            _ = [sentence async for sentence in llm.stream_sentences(HeartbeatHttp(), [])]
        self.assertLess(asyncio.get_running_loop().time() - started, .2)
        self.assertGreater(content.count, 2)
        self.assertTrue(content.closed)
        self.assertEqual(llm.last_generation["raw_words"], 0)

    async def test_consumer_tts_wait_after_first_content_is_outside_first_answer_deadline(self):
        llm = Llm("https://api.deepseek.com", "deepseek-v4-flash", timeout_s=.01)
        http = FakeHttp([
            {"choices": [{"delta": {"content": "A clear first answer. "}}]},
            {"choices": [{"delta": {"content": "The next point is different. "}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ])
        result = []
        async for sentence in llm.stream_sentences(http, []):
            result.append(sentence)
            await asyncio.sleep(.025)  # 模拟消费者 TTS > 首正文等待预算。
        self.assertEqual(len(result), 2)
        self.assertTrue(llm.last_generation["complete"])

    async def test_reasoning_only_eof_is_reported_as_no_answer(self):
        llm = Llm("https://api.deepseek.com", "deepseek-v4-flash")
        http = FakeHttp([
            {"choices": [{"delta": {"reasoning_content": "Hidden internal text."}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ])
        with self.assertRaisesRegex(RuntimeError, "ended without answer text"):
            _ = [sentence async for sentence in llm.stream_sentences(http, [])]
        self.assertEqual(llm.last_generation["raw_words"], 0)

    async def test_warm_timeout_keeps_local_service_ready_but_brain_degraded(self):
        from voxck.orchestrator import build_app

        class FakeComponent:
            def __init__(self, *_args, **_kwargs):
                pass

            def warm(self):
                pass

        async def never_answer(_llm, _http, _messages):
            await asyncio.Event().wait()
            yield "unreachable", True

        semantic = SimpleNamespace(warm=mock.AsyncMock(), close=mock.AsyncMock(), report=lambda: "")
        cfg = {"persona": "unused", "asr_model": "fake-asr", "tts_model": "fake-tts",
               "llm_url": "https://api.deepseek.com", "llm_model": "deepseek-v4-flash",
               "max_tokens": 300, "temperature": .7, "host": "127.0.0.1", "port": 8891,
               "llm_warm_timeout_s": .01}
        fake_modules = {"voxck.asr": SimpleNamespace(Asr=FakeComponent),
                        "voxck.tts": SimpleNamespace(Tts=FakeComponent),
                        "voxck.vad": SimpleNamespace(VadTurnDetector=FakeComponent)}
        with mock.patch("voxck.orchestrator.persona_mod.load", return_value={"prompt": "test", "meta": {}}), \
                mock.patch("voxck.orchestrator.SemanticGuard", return_value=semantic), \
                mock.patch.dict("sys.modules", fake_modules), \
                mock.patch.object(Llm, "stream_sentences", never_answer):
            app = build_app(cfg)
            try:
                startup = next(fn for fn in app.on_startup if fn.__name__ == "_startup")
                await asyncio.wait_for(startup(app), timeout=1)
                health = next(route.handler for route in app.router.routes()
                              if route.method == "GET" and route.resource.canonical == "/health")
                status = json.loads((await health(SimpleNamespace(app=app))).text)
                self.assertTrue(status["ready"])
                self.assertFalse(status["llm_ready"])
                self.assertTrue(status["thinking_disabled"])
                self.assertFalse(status["thinking_enabled"])
                self.assertFalse(status["face_ready"])
            finally:
                cleanup = next(fn for fn in app.on_cleanup if fn.__name__ == "_cleanup")
                await cleanup(app)


class RepeatRuleTests(unittest.TestCase):
    def test_exact_long_opener_normalizes_punctuation_and_whitespace(self):
        guard = RepeatGuard()
        guard.add("Women should have equal access to education.")
        self.assertTrue(guard.is_exact_repeat("WOMEN should have equal access  to education!"))
        self.assertFalse(guard.is_exact_repeat("Women deserve equal access to education."))
        guard.add("Right, I agree.")
        self.assertFalse(guard.is_exact_repeat("Right, I agree."))

    def test_explicit_restatement_is_allowed_but_stop_repeating_is_not(self):
        for user in ("Repeat that please.", "Explain your answer.", "What do you mean?", "没听懂",
                     "Please explain that again.", "I didn't understand your point.",
                     "Please restate your position.", "Could you quote his exact words?"):
            self.assertTrue(allows_restatement(user))
        for user in ("Stop repeating that sentence and answer my question.",
                     "Don't repeat yourself. Explain why.", "Explain why feminism harms women.",
                     "I cannot explain why that policy is wrong.", "You repeat yourself every time.",
                     "I do not want you to repeat yourself."):
            self.assertFalse(allows_restatement(user), user)


if __name__ == "__main__":
    unittest.main()
