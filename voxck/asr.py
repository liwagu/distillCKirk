"""Qwen3-ASR（纯 MLX，无 torch/transformers 依赖）。

源码级验证过的契约：
- 用官方非 "-hf" 仓库 Qwen/Qwen3-ASR-1.7B。transformers 5.x 那个「静默丢权重」的坑
  **不适用**于本包 —— 它的转写路径完全不依赖 transformers，自带 BPE 分词器。
- transcribe(audio, *, context=..., language=...)，audio 收 float32 [-1,1] @16k。
- **热词机制就是 `context` 这一个字符串参数**，被原样注入 system 消息。
  参考实现作者实测：10 个热词会让模型整段幻觉出词表内容。上限 2-3 个，
  其余靠转写后确定性替换兜底。
- 最短约 0.2 秒（3200 样本）。低于 200 样本 stft 直接抛异常。

调用方应通过 asyncio.to_thread 调用 —— 它是阻塞的 GPU 工作。
"""
from __future__ import annotations

import logging
import re

import numpy as np

log = logging.getLogger("voxck.asr")

SR = 16000
MIN_SAMPLES = 3200          # 0.2s，低于此不值得转写


class Asr:
    def __init__(self, model_id: str = "Qwen/Qwen3-ASR-1.7B",
                 hotwords: list[str] | None = None,
                 corrections: dict[str, str] | None = None,
                 language: str = "en",
                 dtype: str = "float16"):
        import mlx.core as mx
        from mlx_qwen3_asr import Session

        dt = {"float16": mx.float16, "bfloat16": mx.bfloat16, "float32": mx.float32}[dtype]
        self.session = Session(model_id, dtype=dt)
        self.language = language

        # 热词务必克制：概率性偏置，词越多越容易把没说的话脑补成词表内容
        hw = [w.strip() for w in (hotwords or []) if w.strip()][:3]
        self.context = ", ".join(hw) if hw else None
        if hw:
            log.info("ASR 热词: %s", hw)

        # 确定性后校正：同音误写的兜底，不走模型，没有幻觉风险
        self._corr = [(re.compile(rf"\b{re.escape(k)}\b", re.I), v)
                      for k, v in (corrections or {}).items()]

    def transcribe(self, wav: np.ndarray) -> str:
        """wav: float32 [-1,1] 单声道 16kHz（VAD 产出的整段发言）。"""
        a = np.asarray(wav)
        if a.ndim > 1:
            # 先转 float 再降混：int16 数组直接 .mean() 会被 numpy 提升成 float64，
            # 之后既不匹配 int16 分支也不匹配 float32 分支，走进错误的缩放路径
            a = a.astype(np.float32).reshape(a.shape[0], -1).mean(axis=-1)
        if a.dtype == np.int16:
            a = a.astype(np.float32) / 32768.0
        else:
            a = a.astype(np.float32)
        a = np.ascontiguousarray(a.reshape(-1))

        if a.size < MIN_SAMPLES:
            return ""

        kw = {"language": self.language}
        if self.context:
            kw["context"] = self.context
        try:
            res = self.session.transcribe(a, **kw)
        except Exception:
            log.exception("ASR 失败")
            return ""

        text = res if isinstance(res, str) else getattr(res, "text", "") or ""
        text = text.strip()
        for pat, rep in self._corr:
            text = pat.sub(rep, text)
        return text

    def warm(self) -> None:
        """预热：首次调用要付 Metal kernel 编译。0.5 秒静音足够触发。"""
        try:
            self.transcribe(np.zeros(SR // 2, dtype=np.float32))
        except Exception:
            log.debug("ASR 预热失败", exc_info=True)
