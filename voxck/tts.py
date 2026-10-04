"""Qwen3-TTS 流式合成 + 零样本音色克隆（mlx-audio 0.5.3）。

源码级验证过的契约：
- 参考音必须是 **24kHz** mx.array。extract_speaker_embedding 假定 24k 且从不校验，
  喂 16k 进去会得到错误声纹、且不报任何错。用 load_audio(path, sample_rate=24000)。
- 参考音要**预加载一次、每次传同一个 array**。传路径会导致每句重读重采样；
  且 ICL 记忆键是 (ref_text, (size, sum))，同一个 array 才能命中。
- 输出是 24kHz、dtype 可能是 bfloat16（声码器不被量化）→ 必须先 astype(float32)，
  否则 np.asarray 直接报错。管线全程 16k，需重采样。
- 打断后声码器状态是脏的：_generate_icl 的清理在循环之后且无 try/finally，
  放弃生成器会跳过它。必须自己调 reset_streaming_state()。
- 最小有效块 ~1 秒语音 / 12-15 词。0.3 秒的短块净倍率 1.12x，反而落后实时。
"""
from __future__ import annotations

import logging
from pathlib import Path

import mlx.core as mx
import numpy as np
from scipy.signal import resample_poly

log = logging.getLogger("voxck.tts")

PIPE_SR = 16000
TTS_SR = 24000
FADE_MS = 8          # 每次调用首块淡入，磨掉声码器状态重置造成的起步爆音


class Tts:
    def __init__(self, model_id: str, ref_wav: str | None = None,
                 ref_text_path: str | None = None, streaming_interval: float = 0.32,
                 root: Path | None = None):
        from mlx_audio.tts.utils import load as load_tts
        from mlx_audio.utils import load_audio

        self.model = load_tts(model_id)
        # post_load_hook 把整个 speech-tokenizer 加载包在 try/except 里，失败只打警告；
        # 之后 use_icl 静默变 False，报一个驴唇不对马嘴的 "Voice not supported"。
        assert getattr(self.model, "speech_tokenizer", None) is not None, \
            "speech_tokenizer 未加载，音色克隆不可用（检查模型是否为 -Base 变体且下载完整）"

        self.streaming_interval = streaming_interval
        self.ref_audio = None
        self.ref_text = None
        root = root or Path(__file__).resolve().parent.parent
        if ref_wav and ref_text_path:
            wp, tp = root / ref_wav, root / ref_text_path
            if wp.exists() and tp.exists():
                self.ref_audio = load_audio(str(wp), sample_rate=TTS_SR)  # 必须 24k
                self.ref_text = tp.read_text(encoding="utf-8").strip()
                log.info("音色克隆已启用: %s (%.1fs)", wp.name, self.ref_audio.size / TTS_SR)
            else:
                log.warning("参考音缺失 (%s / %s)，使用默认音色", wp, tp)

        # 世代计数器而非布尔标志。stream() 是生成器函数，函数体要到第一次取值
        # 才执行——若在体内重置布尔标志，VAD 线程在「创建生成器」与「首次取值」
        # 之间发来的打断会被静默清掉。改为在调用时立即捕获世代号。
        self._epoch = 0

    def interrupt(self) -> None:
        """打断：作废当前世代，并清理声码器跨调用状态。可从任意线程调用。"""
        self._epoch += 1
        self._reset_vocoder()

    def _reset_vocoder(self) -> None:
        try:
            self.model.speech_tokenizer.decoder.reset_streaming_state()
        except Exception:
            log.debug("reset_streaming_state 失败（可能尚未流式过）", exc_info=True)

    def stream(self, text: str):
        """产出 16kHz int16 单声道 PCM bytes。非生成器：立即捕获世代号后再返回生成器。"""
        return self._stream(text.strip(), self._epoch)

    def _stream(self, text: str, epoch: int):
        if not text:
            return
        kw = dict(text=text, stream=True, streaming_interval=self.streaming_interval,
                  verbose=False)
        if self.ref_audio is not None:
            kw.update(ref_audio=self.ref_audio, ref_text=self.ref_text)

        first = True
        gen = self.model.generate(**kw)
        try:
            for res in gen:
                if self._epoch != epoch:      # 被打断（世代已推进）
                    break
                a = res.audio
                if a is None or a.size == 0:
                    continue
                # 声码器不被量化，bf16 下 numpy 没有对应 dtype，必须先转 float32
                f = np.asarray(a.astype(mx.float32), dtype=np.float32).reshape(-1)
                if first:
                    n = min(int(TTS_SR * FADE_MS / 1000), f.size)
                    if n:
                        f[:n] = f[:n] * np.linspace(0.0, 1.0, n, dtype=np.float32)
                    first = False
                # 24k → 16k，2:3 有理重采样
                f16 = resample_poly(f, PIPE_SR, TTS_SR).astype(np.float32)
                np.clip(f16, -1.0, 1.0, out=f16)
                yield (f16 * 32767.0).astype(np.int16).tobytes()
        finally:
            gen.close()
            if self._epoch != epoch:
                # _generate_icl 的清理写在循环之后且无 try/finally，
                # 放弃生成器时 GeneratorExit 会直接跳过它，留下脏的声码器状态。
                # 注意：不要在这里调 mx.clear_cache() —— 它会把下一句马上要用的
                # MLX 缓冲还回去，反而给刚被打断的那一轮增加延迟。
                self._reset_vocoder()

    def warm(self) -> None:
        """GPU 空闲会掉低功耗态（mlx-lm#432 记录可慢 7 倍）。回合间跑一次废合成保温。"""
        for _ in self.stream("Okay."):
            break
