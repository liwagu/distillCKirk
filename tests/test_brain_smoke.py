"""Text-brain acceptance checks with in-memory SSE; no sockets or models."""
import asyncio
import json
import unittest

from scripts.brain_smoke import DEFAULT_TURNS, converse, repetition_checks, validate_client
from voxck.llm import Llm


SECRET = "dummy-private-key-not-for-report"


class Content:
    def __init__(self, reply, mode="normal"):
        self.reply, self.mode = reply, mode

    async def __aiter__(self):
        if self.mode == "timeout":
            await asyncio.sleep(60)
        if self.mode == "keepalive_timeout":
            yield b": keep-alive\n"
            await asyncio.sleep(60)
        payload = {"model": "deepseek-v4-flash", "choices": [{"delta": {"content": self.reply}}]}
        yield ("data: " + json.dumps(payload) + "\n").encode()
        yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n'
        yield b"data: [DONE]\n"


class Response:
    def __init__(self, reply, mode):
        self.status = 500 if mode == "http_error" else 200
        self.content = Content(reply, mode)

    async def text(self):
        return "Rejected secret " + SECRET


class Request:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, *args):
        return False


class Session:
    def __init__(self, replies, mode="normal"):
        self.replies, self.mode, self.requests = replies, mode, []

    def post(self, _url, **kwargs):
        self.requests.append({**kwargs, "json": json.loads(json.dumps(kwargs["json"]))})
        return Request(Response(self.replies[len(self.requests) - 1], self.mode))


class BrainSmokeTests(unittest.IsolatedAsyncioTestCase):
    def client(self):
        return Llm("https://api.deepseek.com", "deepseek-v4-flash", api_key=SECRET, timeout_s=60)

    async def run_fixture(self, replies, mode="normal", timeout=1):
        report = {"turns": []}
        output = []
        session = Session(replies, mode)
        ok = await converse(self.client(), session, "Only sourced persona.", DEFAULT_TURNS,
                            timeout, report, emit=output.append)
        return ok, report, session, output

    async def test_three_turns_use_real_llm_parser_and_history(self):
        replies = ["I'm doing well. What would you like to discuss?",
                   "My strongest concern is whether the claimed policy actually helps families.",
                   "A controlled comparison showing a persistent difference would challenge the choices explanation."]
        ok, report, session, _output = await self.run_fixture(replies)
        self.assertTrue(ok)
        self.assertEqual(len(session.requests), 3)
        self.assertEqual(report["turns"][2]["full_reply"], replies[2])
        self.assertEqual(report["turns"][2]["finish_reason"], "stop")
        self.assertEqual(report["turns"][2]["production_raw_words"], len(replies[2].split()))
        self.assertEqual([x["role"] for x in session.requests[2]["json"]["messages"]],
                         ["system", "user", "assistant", "user", "assistant", "user"])
        self.assertEqual(session.requests[2]["json"]["messages"][-1]["content"], DEFAULT_TURNS[2])
        self.assertEqual(session.requests[0]["json"]["thinking"], {"type": "disabled"})
        self.assertNotIn(SECRET, json.dumps(report))

    async def test_first_timeout_aborts_without_two_more_requests(self):
        ok, report, session, _ = await self.run_fixture([""], "timeout", timeout=0.01)
        self.assertFalse(ok)
        self.assertEqual(len(session.requests), 1)
        self.assertEqual(report["aborted_after_turn"], 1)
        self.assertEqual(report["unattempted_turns"], 2)

    async def test_first_empty_body_aborts(self):
        ok, report, session, _ = await self.run_fixture([""])
        self.assertFalse(ok)
        self.assertEqual(len(session.requests), 1)
        self.assertEqual(report["turns"][0]["raw_words"], 0)

    async def test_http_200_keepalive_without_content_still_aborts(self):
        ok, report, session, _ = await self.run_fixture([""], "keepalive_timeout", timeout=0.01)
        self.assertFalse(ok)
        self.assertEqual(len(session.requests), 1)
        self.assertEqual(report["turns"][0]["http_status"], 200)
        self.assertEqual(report["turns"][0]["sse_comment_events"], 1)
        self.assertIsNone(report["turns"][0]["first_content_ms"])

    async def test_full_reply_repetition_is_a_failed_diagnostic(self):
        reply = "Women make choices that explain the comparison."
        ok, report, session, _ = await self.run_fixture([reply, reply, reply])
        self.assertFalse(ok)
        self.assertEqual(len(session.requests), 3)
        self.assertEqual(report["turns"][1]["repetition"]["exact_prior_reply_turns"], [1])

    async def test_api_key_reflected_in_error_is_redacted(self):
        ok, report, session, output = await self.run_fixture([""], "http_error")
        self.assertFalse(ok)
        self.assertEqual(len(session.requests), 1)
        self.assertNotIn(SECRET, json.dumps(report))
        self.assertNotIn(SECRET, "\n".join(output))

    def test_duplicate_sentence_is_detected(self):
        self.assertTrue(repetition_checks("This claim needs stronger empirical evidence. This claim needs stronger empirical evidence.", [])["repeated"])

    def test_default_accepts_configured_supported_models_without_switching(self):
        for model in ("deepseek-v4-flash", "deepseek-flash", "deepseek-v4-pro"):
            client = Llm("https://api.deepseek.com", model)
            validate_client(client)
            self.assertEqual(client.model, model)

    def test_explicit_expected_model_is_strict(self):
        client = Llm("https://api.deepseek.com", "deepseek-v4-pro")
        with self.assertRaisesRegex(ValueError, "differs from expected"):
            validate_client(client, "deepseek-v4-flash")
        validate_client(client, "deepseek-v4-pro")

    def test_local_or_unsupported_models_are_rejected_without_requests(self):
        for url, model in (("http://127.0.0.1:8080", "deepseek-v4-pro"),
                           ("https://api.deepseek.com", "Qwen3-30B-A3B")):
            with self.assertRaisesRegex(ValueError, "supported remote"):
                validate_client(Llm(url, model))

    def test_thinking_enabled_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "thinking.type=disabled"):
            validate_client(Llm("https://api.deepseek.com", "deepseek-v4-pro",
                                extra_body={"thinking": {"type": "enabled"}}))


if __name__ == "__main__":
    unittest.main()
