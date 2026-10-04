#!/usr/bin/env python
"""Exercise the configured brain/persona for three text turns, without audio/video.

Requires a configured supported remote DeepSeek with thinking disabled. No model fallback,
server startup, ASR, TTS or face loading. Reports contain no API key/headers.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from difflib import SequenceMatcher
import json
import os
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlsplit, urlunsplit

import aiohttp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from voxck.llm import Llm
from voxck.persona import load as load_persona

DEFAULT_TURNS = [
    "Hi, how's it going?",
    "Why do you oppose feminism? Give me your strongest reason, not a slogan.",
    "What evidence would change your mind? Suppose women in the same role, with the same hours "
    "and experience, still earn less. What would that evidence mean for your claim?",
]
SUPPORTED_MODELS = {"deepseek-v4-flash", "deepseek-flash", "deepseek-v4-pro"}


def validate_client(client: Llm, expected_model: str | None = None) -> None:
    """Validate the selected configuration; never select or switch a model."""
    if not client.remote or client.model.casefold() not in SUPPORTED_MODELS:
        raise ValueError(f"Configured model {client.model!r} is not a supported remote DeepSeek; no request/fallback attempted")
    if expected_model is not None and client.model.casefold() != expected_model.casefold():
        raise ValueError(f"Configured model {client.model!r} differs from expected {expected_model!r}; no request/fallback attempted")
    if client.extra_body.get("thinking") != {"type": "disabled"}:
        raise ValueError("Configured client is not sending thinking.type=disabled; no request attempted")


def load_config(config_path: Path, env_path: Path) -> dict:
    """Match run.py's environment precedence, while never logging env values."""
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if os.environ.get("LLM_URL"):
        cfg["llm_url"] = os.environ["LLM_URL"]
        cfg["llm_model"] = os.environ.get("LLM_MODEL", cfg["llm_model"])
    cfg["llm_api_key"] = os.environ.get("LLM_API_KEY", "")
    # Same direct/proxy selection as the production runner.
    host = urlsplit(cfg["llm_url"]).hostname or ""
    bypass = ["127.0.0.1", "localhost", "::1"]
    if not os.environ.get("LLM_VIA_PROXY"):
        bypass.append(host)
    for key in ("NO_PROXY", "no_proxy"):
        os.environ[key] = ",".join(x for x in bypass + [os.environ.get(key, "")] if x)
    return cfg


def safe_url(url: str) -> str:
    parts = urlsplit(url)
    netloc = parts.hostname or ""
    if parts.port:
        netloc += f":{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def redact(text: str, api_key: str) -> str:
    if api_key:
        text = text.replace(api_key, "[REDACTED]")
    return re.sub(r"(?i)Bearer\s+[^\s\"']+", "Bearer [REDACTED]", text)


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def repetition_checks(reply: str, earlier: list[str]) -> dict:
    sentences = [normalize(x) for x in re.split(r"(?<=[.!?])\s+", reply) if normalize(x)]
    duplicates = sorted({s for s in sentences if len(s.split()) >= 4 and sentences.count(s) > 1})
    current = normalize(reply)
    similarities = [round(SequenceMatcher(None, normalize(x), current).ratio(), 4) for x in earlier]
    exact = [i + 1 for i, x in enumerate(earlier) if normalize(x) == current and current]
    almost = [i + 1 for i, score in enumerate(similarities)
              if score >= 0.88 and min(len(normalize(earlier[i]).split()), len(current.split())) >= 6]
    return {"duplicate_sentences": duplicates, "similarity_to_prior_replies": similarities,
            "exact_prior_reply_turns": exact, "near_duplicate_prior_reply_turns": almost,
            "repeated": bool(duplicates or exact or almost)}


class Capture:
    """Observe public SSE content without changing production Llm parsing."""
    def __init__(self):
        self.parts: list[str] = []
        self.response_models: set[str] = set()
        self.finish_reason = None
        self.done = False
        self.reasoning_words = 0
        self.status = None
        self.response_at = None
        self.first_content_at = None
        self.sse_comments = 0
        self.request_model = None
        self.request_thinking = None

    def consume(self, raw: bytes):
        for line in raw.decode("utf-8", "ignore").splitlines():
            if line.startswith(":"):
                self.sse_comments += 1
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                self.done = True
                continue
            try:
                value = json.loads(data)
            except json.JSONDecodeError:
                continue
            if value.get("model"):
                self.response_models.add(str(value["model"]))
            choice = (value.get("choices") or [{}])[0]
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                self.parts.append(delta["content"])
                if delta["content"] and self.first_content_at is None:
                    self.first_content_at = time.perf_counter()
            # Count unexpected reasoning, but never retain its text.
            reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
            self.reasoning_words += len(reasoning.split())


class CapturedContent:
    def __init__(self, content, capture: Capture):
        self.content, self.capture = content, capture

    async def __aiter__(self):
        async for raw in self.content:
            self.capture.consume(raw)
            yield raw


class CapturedResponse:
    def __init__(self, response, capture: Capture):
        self.response = response
        self.status = response.status
        self.content = CapturedContent(response.content, capture)
        capture.status = response.status
        capture.response_at = time.perf_counter()

    def __getattr__(self, name):
        return getattr(self.response, name)


class CapturedRequest:
    def __init__(self, request, capture: Capture):
        self.request, self.capture = request, capture

    async def __aenter__(self):
        return CapturedResponse(await self.request.__aenter__(), self.capture)

    async def __aexit__(self, *args):
        return await self.request.__aexit__(*args)


class CapturedSession:
    def __init__(self, session, capture: Capture):
        self.session, self.capture = session, capture

    def post(self, url, **kwargs):
        body = kwargs.get("json") or {}
        self.capture.request_model = body.get("model")
        self.capture.request_thinking = body.get("thinking")
        return CapturedRequest(self.session.post(url, **kwargs), self.capture)


async def converse(client: Llm, session, persona: str, prompts: list[str], timeout: float,
                   report: dict, emit=print) -> bool:
    messages = [{"role": "system", "content": persona}]
    earlier: list[str] = []
    for index, user in enumerate(prompts, 1):
        messages.append({"role": "user", "content": user})
        capture, emitted = Capture(), []
        start = time.perf_counter()
        error = None
        try:
            async with asyncio.timeout(timeout):
                async for sentence, _first in client.stream_sentences(CapturedSession(session, capture), messages):
                    emitted.append(sentence)
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": redact(str(exc), client.api_key)}
        raw_reply = redact("".join(capture.parts).strip(), client.api_key)
        audible_reply = redact(" ".join(emitted).strip(), client.api_key)
        stats = dict(client.last_generation)
        repeated = repetition_checks(raw_reply, earlier)
        row = {"turn": index, "user": user, "full_reply": raw_reply,
               "production_emitted_reply": audible_reply, "finish_reason": capture.finish_reason or stats.get("finish_reason"),
               "raw_words": len(raw_reply.split()), "production_raw_words": stats.get("raw_words"),
               "emitted_words": stats.get("emitted_words"), "completion_tokens": stats.get("completion_tokens"),
               "request_model": capture.request_model, "response_models": sorted(capture.response_models),
               "thinking_request": capture.request_thinking, "unexpected_reasoning_words": capture.reasoning_words,
               "http_status": capture.status, "stream_done": capture.done,
               "http_response_ms": round((capture.response_at - start) * 1000) if capture.response_at else None,
               "first_content_ms": round((capture.first_content_at - start) * 1000) if capture.first_content_at else None,
               "sse_comment_events": capture.sse_comments,
               "elapsed_ms": round((time.perf_counter() - start) * 1000), "error": error,
               "repetition": repeated}
        row["checks"] = {"request_model_matches_config": capture.request_model == client.model,
                         "thinking_disabled": capture.request_thinking == {"type": "disabled"},
                         "body_received": bool(raw_reply), "request_succeeded": error is None,
                         "finished_normally": capture.done and row["finish_reason"] == "stop",
                         "reasoning_not_returned": capture.reasoning_words == 0,
                         "not_repeated": not repeated["repeated"]}
        row["passed"] = all(row["checks"].values())
        report["turns"].append(row)
        emit(f"\n[{index} user] {user}\n[{index} reply] {raw_reply or '(no body)'}")
        emit(f"model={capture.request_model or client.model} thinking={capture.request_thinking} "
             f"finish_reason={row['finish_reason']} raw_words={row['raw_words']} "
             f"elapsed_ms={row['elapsed_ms']} repeated={repeated['repeated']} passed={row['passed']}")
        if error:
            emit(f"error={error['type']}: {error['message']}")
        # A dead/empty gateway must not receive two more diagnostic requests.
        if error is not None or not raw_reply:
            report["aborted_after_turn"] = index
            report["unattempted_turns"] = len(prompts) - index
            return False
        messages.append({"role": "assistant", "content": raw_reply})
        earlier.append(raw_reply)
    return len(report["turns"]) == len(prompts) and all(t["passed"] for t in report["turns"])


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/assistant.json")
    parser.add_argument("--env", type=Path, default=ROOT / ".env.local")
    parser.add_argument("--report", type=Path, default=ROOT / "logs/brain-smoke.json")
    parser.add_argument("--timeout", type=float, default=60, help="Hard per-turn wall timeout, seconds (default 60)")
    parser.add_argument("--expect-model", default=None, help="Optional: fail before HTTP if actual configured model differs")
    parser.add_argument("--prompt", action="append", help="Override turns; repeat for each prompt")
    args = parser.parse_args()
    if not 0 < args.timeout < float("inf"):
        parser.error("timeout must be finite and positive")
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "mode": "configured_brain_text_only",
              "config": str(args.config), "persona": None, "model": None,
              "expected_model": args.expect_model, "fallback_used": False,
              "timeout_seconds_per_turn": args.timeout, "turns": [], "status": "failed",
              "assessment_limit": "Repetition checks are deterministic diagnostics; inspect full replies to judge whether they answer the latest question."}
    try:
        cfg = load_config(args.config, args.env)
        client = Llm(cfg["llm_url"], cfg["llm_model"], max_tokens=cfg.get("max_tokens", 600),
                     temperature=cfg.get("temperature", 0.7), api_key=cfg.get("llm_api_key", ""),
                     extra_body=cfg.get("llm_extra_body"), timeout_s=args.timeout)
        report.update(model=client.model, endpoint=safe_url(client.url), thinking_request=client.extra_body.get("thinking"),
                      max_tokens=client.max_tokens, temperature=client.temperature)
        validate_client(client, args.expect_model)
        persona_path = Path(cfg["persona"])
        if not persona_path.is_absolute():
            persona_path = ROOT / persona_path
        report["persona"] = str(persona_path)
        persona = load_persona(persona_path)["prompt"]
        print(f"model={client.model} endpoint={safe_url(client.url)} thinking={report['thinking_request']} fallback=False")
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=args.timeout, sock_connect=min(10, args.timeout), sock_read=args.timeout), trust_env=True) as session:
            ok = await converse(client, session, persona, args.prompt or DEFAULT_TURNS, args.timeout, report)
        report["status"] = "passed" if ok else "failed"
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": redact(str(exc), os.environ.get("LLM_API_KEY", ""))}
        print(f"FAILED: {report['error']['type']}: {report['error']['message']}")
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status'].upper()}: {args.report}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
