"""语义守卫：用 TypeSafe 的 Noul 原语做正则做不到的判断。

为什么需要：guard.py 的正则只能匹配见过的句式。同一条禁令我在提示词里写了三遍、
在正则里堵了两轮，模型每次换个说法就绕过去：
  "You're not as attractive…"        → 堵住
  "if you're single by 30, you've…"  → 绕过（改成假设句）
  "You're not as happy as you were…" → 绕过（换了形容词）
这本质上是语义判断，正则永远在追着打。

还有一类正则**根本做不到**：分辨引用的数字是不是编的。
模型曾把账本里关于女性的统计安到男性头上（"Men who stay single past 30 have a 50%
chance of never marrying"）——句式完全正常，只有对照账本才知道是伪造。

延迟与分层（实测 TypeSafe 中位 463ms）：
- 首句最敏感（决定首音延迟）也最高危（模型爱拿违规句开场）。默认不拦，只审计。
- 第二句起有余量：TTS 的 RTF 0.26，生成比播放快约 4 倍，463ms 藏得进去。
- strict 模式对所有句子同步检查，代价是首音 +~460ms。由配置决定。

阈值 0.50 的依据（10 句验证集实测）：违规句落在 0.73-0.98，干净句落在 0.05-0.14，
中间是空白带。按文档指引「漏判代价更大时取低阈值」——漏一句违规会直接进用户耳朵，
误杀一句只是少说一句（回复本来多句）——所以取偏低的 0.50，离最高干净句仍有 3.6 倍距离。

密钥从环境变量读（.env.local，已 gitignore）。离线时自动降级为只用正则。
可选 TYPESAFE_PROXY 只作用于语义守卫的 HTTP 客户端，不改变其他服务的代理。
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

log = logging.getLogger("voxck.semantic")

# 账本里真实存在的数字。除此之外的统计断言都视为编造。
LEDGER_FIGURES = [
    "single at 30 → 50% chance of never having kids (about WOMEN)",
    "one in five women / 1-in-5 sexual assault figure (which he disputes as inflated)",
]

_Q_PERSONAL = dict(
    instructions=(
        "Does `sentence` tell the LISTENER something about their own life — their body, age, "
        "attractiveness, happiness, fertility, marriage prospects, or how they will feel later? "
        "This includes hypotheticals aimed at the listener, e.g. 'if you're in your thirties "
        "and you've built a career…'."
    ),
    criteria={
        "true": ("It addresses the listener directly about themselves, including as a "
                 "conditional or a rhetorical question about their own situation."),
        "false": ("It is about a population, a third party, an abstract claim, or about the "
                  "speaker. Saying 'you' while quoting an argument or referring to the "
                  "listener's stated POSITION rather than their life is false."),
    },
)

_Q_INVENTED = dict(
    instructions=(
        "Does `sentence` assert a specific statistic, study, survey, poll, institution, or "
        "named authority that is NOT among `allowed_figures`? Vague appeals such as 'the data "
        "shows' with no number and no named source do not count."
    ),
    criteria={
        "true": ("It states a number, percentage, named study, named institution, or named "
                 "researcher that is not in the allowed list — or applies an allowed figure "
                 "to the wrong group."),
        "false": ("It cites nothing specific, or only uses figures from the allowed list, "
                  "applied to the group they actually describe."),
    },
)


class SemanticGuard:
    """每句两个独立判断，合并为一次请求（文档：独立问题批在一起并行跑）。"""

    def __init__(self, enabled: bool = True, threshold: float = 0.50,
                 strict: bool = False, timeout: float = 10.0):
        self.threshold = threshold
        self.strict = strict
        self.timeout = timeout
        self.audit: list[dict[str, Any]] = []
        self.failures = 0
        # 必须串行。实测：串行 406-464ms 极稳；并发 4 个请求合计 6780ms
        # （每个 ~1695ms），比串行还慢 —— 请求在服务端排队或触发退避，不是真并行。
        # 不加锁的话每轮飞出的多个请求会一起撞破超时，全部 fail-open。
        # 注意：文档建议的「多个问题批进一次请求」我们已经做了（每句 2 问 1 请求），
        # 这里限制的是并发的**请求数**，两者不冲突。
        self._lock = asyncio.Lock()
        self._client = None
        self.enabled = False
        if not enabled:
            return
        if not os.environ.get("TYPESAFE_API_KEY"):
            log.info("未设置 TYPESAFE_API_KEY，语义守卫关闭（正则仍生效）")
            return
        try:
            from typesafe_sdk import AsyncTypeSafeClient
            proxy = os.environ.get("TYPESAFE_PROXY", "").strip()
            if proxy:
                import httpx2
                from typesafe_sdk.constants import DEFAULT_TIMEOUT
                self._client = AsyncTypeSafeClient(http_client=httpx2.AsyncClient(
                    proxy=proxy, trust_env=False, timeout=DEFAULT_TIMEOUT))
            else:
                self._client = AsyncTypeSafeClient()
            self.enabled = True
            log.info("语义守卫已启用（阈值 %.2f，%s）", threshold,
                     "strict：全部句子同步检查" if strict else "首句只审计、其后同步检查")
        except Exception:
            log.warning("TypeSafe 初始化失败，语义守卫关闭", exc_info=True)

    async def warm(self) -> None:
        """预热。首次调用实测 6.7s（冷启动），之后中位 463ms。
        不预热的话前几句必定超时，而超时是 fail-open —— 违规句会直接放行。"""
        if not self.enabled:
            return
        t0 = time.perf_counter()
        old, self.timeout = self.timeout, 30.0
        await self.judge("This is a warmup sentence about nothing in particular.")
        self.timeout = old
        self.audit.clear()
        log.info("语义守卫预热 %.1fs", time.perf_counter() - t0)

    async def judge(self, sentence: str) -> tuple[str | None, float]:
        """返回 (违规原因, 概率)。

        注意区分两种「没拦」：判定干净 → (None, 概率)；调用失败 → (None, -1.0)。
        失败是 fail-open，必须能在审计里认出来，否则 0.00 看起来像模型自信地说没问题。
        """
        if not self.enabled:
            return None, -1.0
        from typesafe_sdk import Noul
        t0 = time.perf_counter()
        try:
            async with self._lock:          # 串行：并发会让每个请求都变慢并撞破超时
                resp = await asyncio.wait_for(
                    self._client.system_one(
                        state={"sentence": sentence,
                               "allowed_figures": LEDGER_FIGURES},
                        questions={"personal": Noul(**_Q_PERSONAL),
                                   "invented": Noul(**_Q_INVENTED)}),
                    timeout=self.timeout)
        except asyncio.TimeoutError:
            self.failures += 1
            self.audit.append({"sentence": sentence, "error": "timeout",
                               "ms": round((time.perf_counter()-t0)*1000)})
            log.warning("语义守卫超时（fail-open，本句未经语义检查）")
            return None, -1.0
        except Exception as e:
            self.failures += 1
            self.audit.append({"sentence": sentence, "error": type(e).__name__, "ms": 0})
            log.warning("语义守卫调用失败（fail-open）: %s", type(e).__name__)
            return None, -1.0
        ms = (time.perf_counter() - t0) * 1000
        p_personal = resp.nouls["personal"].noul
        p_invented = resp.nouls["invented"].noul
        self.audit.append({"sentence": sentence, "personal": p_personal,
                           "invented": p_invented, "ms": round(ms)})
        if p_personal >= self.threshold:
            return "语义：对用户本人下判词", p_personal
        if p_invented >= self.threshold:
            return "语义：引用了账本外的数字/来源", p_invented
        return None, max(p_personal, p_invented)

    def should_block_inline(self, is_first: bool) -> bool:
        """首句默认只审计不拦：它决定首音延迟，而 463ms 藏不进去。"""
        return self.enabled and (self.strict or not is_first)

    async def audit_only(self, sentence: str) -> None:
        """不阻塞地判一句，只为积累审计数据，用来反哺正则。"""
        if self.enabled:
            with __import__("contextlib").suppress(Exception):
                await self.judge(sentence)

    def report(self) -> str:
        if not self.audit:
            return "（无审计数据）"
        n = len(self.audit)
        judged = [a for a in self.audit if "error" not in a]
        errs = [a for a in self.audit if "error" in a]
        hi = [a for a in judged
              if max(a["personal"], a["invented"]) >= self.threshold]
        ms = sorted(a["ms"] for a in judged) or [0]
        out = [f"语义守卫审计：{n} 句（成功 {len(judged)}，失败 {len(errs)}），"
               f"{len(hi)} 句触线，中位延迟 {ms[len(ms)//2]}ms"]
        if errs:
            out.append(f"  ⚠ {len(errs)} 句因调用失败未经检查（fail-open）")
        for a in sorted(hi, key=lambda x: -max(x["personal"], x["invented"]))[:8]:
            out.append(f"  p_personal={a['personal']:.2f} p_invented={a['invented']:.2f}  "
                       f"{' '.join(a['sentence'].split())[:88]}")
        return "\n".join(out)

    async def close(self) -> None:
        if self._client is not None:
            with __import__("contextlib").suppress(Exception):
                await self._client.aclose()
