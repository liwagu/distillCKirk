"""本地 LLM 客户端：mlx_lm.server 的 OpenAI 兼容接口，按句流式产出。

关键点（均有本机实测支撑，见 docs/prefill-measured.md）：
- 必须关思考模式：Qwen3 默认吐 reasoning 字段，思考 token 全排在可朗读内容之前。
  请求体字段名是 chat_template_kwargs（不是 CLI 的 chat_template_args）。
- server 自带 LRUPromptCache + fetch_nearest_cache，跨轮自动复用最长公共前缀。
  实测：追加式续轮命中 96-98%，打断重写历史仍命中 99%。无需改成 append-only 历史。
- 按句切分后逐句喂 TTS，首句决定体感延迟。
"""
from __future__ import annotations

import json
import asyncio
import logging
import re
from contextvars import ContextVar
from typing import AsyncIterator

import aiohttp

log = logging.getLogger("voxck.llm")

# 句子边界：句号/问号/感叹号 + 空白。避免在缩写和小数点处误切。
_ABBREV = r"(?<!\bMr)(?<!\bMrs)(?<!\bDr)(?<!\bSt)(?<!\bvs)(?<!\bU\.S)(?<!\d)"
_SENT_END = re.compile(_ABBREV + r"([.!?])[\"')\]]*\s+")


class Llm:
    def __init__(self, base_url: str, model: str, max_tokens: int = 200,
                 temperature: float = 0.7, api_key: str = "",
                 extra_body: dict | None = None, timeout_s: float | None = None):
        base = base_url.rstrip("/")
        # 本地 mlx_lm.server 是 http://127.0.0.1:port，云端通常已带 /v1
        self.url = base + ("" if base.endswith("/v1") else "/v1") + "/chat/completions"
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.api_key = api_key
        self.remote = not ("127.0.0.1" in base or "localhost" in base)
        self.timeout_s = float(timeout_s or (20 if self.remote else 300))
        # sock_read 只限制等待网络输入；TTS 消费流的时间不吃掉这个预算。
        self.request_timeout = (aiohttp.ClientTimeout(total=300, connect=min(10, self.timeout_s),
                                                      sock_read=self.timeout_s) if self.remote
                                else aiohttp.ClientTimeout(total=300))
        # 关思考模式，各家字段不同（思考 token 全排在可朗读内容之前，辩论等不起）：
        #   本地 mlx_lm + Qwen3 → chat_template_kwargs.enable_thinking=false
        #   DeepSeek（api-docs.deepseek.com）→ thinking.type="disabled"
        # 其他云端由配置里的 llm_extra_body 给。
        if extra_body is not None:
            self.extra_body = extra_body
        elif not self.remote:
            self.extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        elif "deepseek" in base or "deepseek" in model.lower():
            # 中转站域名里没有 deepseek，但模型名有；不关的话 140 个 token 全花在 reasoning_content，正文为空
            self.extra_body = {"thinking": {"type": "disabled"}}
        else:
            self.extra_body = {}
        self.last_usage: dict = {}
        # 请求诊断随当前 asyncio task 隔离；多个浏览器不会互相覆盖这份记录。
        self._generation = ContextVar("llm_generation", default=None)

    @property
    def last_generation(self) -> dict:
        return self._generation.get() or {}

    @property
    def thinking_enabled(self) -> bool | None:
        mode = (self.extra_body.get("thinking") or {}).get("type")
        if mode in ("disabled", "enabled"):
            return mode == "enabled"
        value = (self.extra_body.get("chat_template_kwargs") or {}).get("enable_thinking")
        return value if isinstance(value, bool) else None

    @property
    def thinking_disabled(self) -> bool:
        return self.thinking_enabled is False

    @property
    def headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    async def stream_sentences(self, session: aiohttp.ClientSession,
                               messages: list[dict]) -> AsyncIterator[tuple[str, bool]]:
        """逐句产出 (sentence, is_first)。调用方可在首句到达时立刻起 TTS。"""
        body = {
            "model": self.model, "messages": messages,
            "max_tokens": self.max_tokens, "temperature": self.temperature,
            "stream": True, "stream_options": {"include_usage": True},
            **self.extra_body,
        }
        latest_user = next((m.get("content", "") for m in reversed(messages)
                            if m.get("role") == "user"), "")
        stats = {"model": self.model, "finish_reason": None, "raw_words": 0,
                 "emitted_words": 0, "completion_tokens": 0, "complete": False}
        self._generation.set(stats)
        raw_parts: list[str] = []
        buf, first = "", True
        first_content = False
        # SSE 心跳是网络数据，却不是模型正文。仅首个正文前启用此 deadline，
        # 在 yield 给 TTS 消费者之前解除，避免把合成耗时误算成模型无响应。
        content_deadline = asyncio.timeout(self.timeout_s if self.remote else None)
        try:
            async with content_deadline:
                async with session.post(self.url, json=body, headers=self.headers,
                                        timeout=self.request_timeout) as resp:
                    if resp.status >= 400:
                        detail = (await resp.text())[:400]
                        if self.api_key:
                            detail = detail.replace(self.api_key, "[redacted]")
                        raise RuntimeError(f"LLM {resp.status} from {self.url}: {detail}")
                    async for raw in resp.content:
                        line = raw.decode("utf-8", "ignore").strip()
                        if not line.startswith("data: "):
                            continue
                        payload = line[6:]
                        if payload == "[DONE]":
                            break
                        try:
                            d = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        if d.get("usage"):
                            self.last_usage = d["usage"]
                            stats["completion_tokens"] = d["usage"].get("completion_tokens", 0)
                        ch = d.get("choices") or [{}]
                        choice = ch[0]
                        if choice.get("finish_reason"):
                            stats["finish_reason"] = choice["finish_reason"]
                        delta = choice.get("delta") or {}
                        piece = delta.get("content")
                        if not piece:
                            continue
                        if not first_content and piece.strip():
                            first_content = True
                            content_deadline.reschedule(None)
                        raw_parts.append(piece)
                        buf += piece
                        while True:
                            m = _SENT_END.search(buf)
                            if not m:
                                break
                            cut = m.end()
                            sent = buf[:cut].strip()
                            buf = buf[cut:]
                            if sent:
                                stats["emitted_words"] += len(sent.split())
                                yield sent, first
                                first = False
            if not first_content:
                raise RuntimeError("LLM response ended without answer text. Check the configured brain service.")
            tail = buf.strip()
            if tail:
                # max_tokens 截断会留下半句话，仅首句可在没有完整句时保底。
                if tail[-1] in ".!?\"')]" or first:
                    stats["emitted_words"] += len(tail.split())
                    yield tail, first
                else:
                    log.debug("丢弃被截断的尾句: %r", tail[-60:])
            stats["complete"] = True
        except asyncio.TimeoutError as e:
            if content_deadline.expired():
                raise RuntimeError(f"LLM timed out waiting for answer text ({self.timeout_s:g}s). "
                                   "The service sent no answer content; SSE keep-alives do not count.") from e
            raise RuntimeError(f"LLM timed out waiting for network data ({self.timeout_s:g}s). "
                               "Check the configured brain service.") from e
        except aiohttp.ClientError as e:
            raise RuntimeError(f"LLM connection failed ({type(e).__name__}). "
                               "Check the configured brain service.") from e
        finally:
            stats["raw_words"] = len("".join(raw_parts).split())
            log.info("llm_generation model=%s user=%r finish_reason=%s raw_words=%d "
                     "emitted_words=%d completion_tokens=%d complete=%s",
                     self.model, " ".join(str(latest_user).split())[:160],
                     stats["finish_reason"], stats["raw_words"], stats["emitted_words"],
                     stats["completion_tokens"], stats["complete"])

    @property
    def cached_tokens(self) -> int:
        u = self.last_usage
        # OpenAI/mlx_lm 风格与 DeepSeek 风格两种字段
        return ((u.get("prompt_tokens_details") or {}).get("cached_tokens")
                or u.get("prompt_cache_hit_tokens") or 0)

    @property
    def prompt_tokens(self) -> int:
        return self.last_usage.get("prompt_tokens", 0)
