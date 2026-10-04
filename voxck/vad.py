"""两级判停：Silero VAD v5（有没有在说话）+ SmartTurn v3（说完了没有）。

纯 CPU ONNX，不碰 GPU —— 这是刻意的：GPU 被 ASR/TTS/LLM 争用，而判停必须
在助手说话期间也保持灵敏（打断）。实测 Silero 0.139ms/32ms 帧、SmartTurn 18ms。

源码级验证过的契约：
- Silero v5 输入是 (1, 576) = 64 样本上下文 + 512 样本当前帧。
  **两个维度都是 dynamic**，所以漏掉上下文不会报错，只会永远返回错误的概率。
- state 是单个融合张量 (2,1,128) float32。v4 的两个 (2,1,64) h/c 不兼容。
- sr 是 **rank-0 标量** int64。np.array([16000]) 是形状错误。
- 不要定时 reset_states()。v5 状态定长，中途重置会丢上下文、压垮随后几帧的概率。
- SmartTurn 输出张量名叫 "logits" 但**已经过 Sigmoid**，是概率。再 sigmoid 一次
  会把一切压到 0.5-0.73，判停实际失效。
- SmartTurn 要恰好 8 秒 @16k，不足则**在前面补零**（右对齐），且归一化覆盖整个
  含前导零的缓冲。用 transformers 的 WhisperFeatureExtractor(chunk_length=8)。
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("voxck.vad")

SR = 16000
FRAME = 512          # Silero v5 @16k 的帧长
CONTEXT = 64         # 必须前置的上下文样本数
ST_SECONDS = 8       # SmartTurn 固定窗口


_PATH_CACHE: dict[tuple[str, str], str] = {}


def _hf_file(repo: str, filename: str) -> str:
    """解析一次就缓存。否则每建一个 VadTurnDetector（= 每条 WS 连接）都要联网
    去 HF 校验，网络慢时连接阶段直接卡住。"""
    key = (repo, filename)
    if key not in _PATH_CACHE:
        from huggingface_hub import hf_hub_download
        try:
            _PATH_CACHE[key] = hf_hub_download(repo_id=repo, filename=filename,
                                               local_files_only=True)
        except Exception:
            _PATH_CACHE[key] = hf_hub_download(repo_id=repo, filename=filename)
    return _PATH_CACHE[key]


class VadTurnDetector:
    """喂 16kHz int16 PCM（bytes 或 ndarray），产出事件。

    事件: ("speech_start", None) / ("speech_end", None) / ("turn_complete", np.ndarray)
    turn_complete 带的是整段发言的 float32 [-1,1] 波形，直接给 ASR。
    """

    def __init__(self,
                 vad_repo: str = "runanywhere/silero-vad-v5",
                 vad_file: str = "silero_vad.onnx",
                 turn_repo: str = "pipecat-ai/smart-turn-v3",
                 turn_file: str = "smart-turn-v3.2-cpu.onnx",
                 threshold: float = 0.5,
                 start_frames: int = 2,          # 连续 N 帧超阈值才算开口（抗咳嗽）
                 hangover_ms: int = 200,         # 静音多久后去问 SmartTurn
                 turn_threshold: float = 0.5,
                 preroll_ms: int = 320,          # 开口前保留多少音频（否则吃掉第一个词）
                 max_silence_ms: int = 900,      # SmartTurn 一直说"没说完"时的兜底
                 max_utterance_s: float = 30.0,
                 threads: int = 1):
        import onnxruntime as ort

        so = ort.SessionOptions()
        # 线程数必须压住：实测 CPU 打满时 CoreML/ANE 的 p99 会劣化 7 倍。
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.vad = ort.InferenceSession(_hf_file(vad_repo, vad_file), so,
                                        providers=["CPUExecutionProvider"])
        names = {i.name for i in self.vad.get_inputs()}
        if not {"input", "state", "sr"} <= names:
            raise RuntimeError(f"不是 Silero v5 图（输入为 {names}）——v4 的 h/c 布局不兼容")

        self.turn = ort.InferenceSession(_hf_file(turn_repo, turn_file), so,
                                         providers=["CPUExecutionProvider"])
        self.turn_in = self.turn.get_inputs()[0].name

        from transformers import WhisperFeatureExtractor
        self.fx = WhisperFeatureExtractor(feature_size=80, sampling_rate=SR,
                                          hop_length=160, chunk_length=ST_SECONDS,
                                          n_fft=400)

        self.threshold = threshold
        self.start_frames = start_frames
        self.hangover_frames = max(1, int(hangover_ms * SR / 1000 / FRAME))
        # 兜底：SmartTurn 是韵律模型，在不熟悉的口音/语速上可能一直判"没说完"。
        # 没有这个上限，回合会一直不提交直到 max_utterance_s（用户干等 30 秒）。
        self.max_silence_frames = max(self.hangover_frames,
                                      int(max_silence_ms * SR / 1000 / FRAME))
        self.turn_threshold = turn_threshold
        self.max_samples = int(max_utterance_s * SR)

        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._ctx = np.zeros(CONTEXT, dtype=np.float32)
        self._sr = np.array(SR, dtype=np.int64)        # rank-0 标量，不是 [16000]
        self._pending = np.zeros(0, dtype=np.float32)  # 不足一帧的余数
        self._utter: list[np.ndarray] = []             # 当前发言累积
        self._speaking = False
        self._hits = 0
        self._silence = 0
        # 前置缓冲：判定开口需要连续 start_frames 帧，那几帧之前的音频照样含语音
        # （起音、爆破音）。不回补就会吃掉用户的第一个词。
        from collections import deque
        self._preroll_n = max(start_frames, int(preroll_ms * SR / 1000 / FRAME))
        self._preroll: deque[np.ndarray] = deque(maxlen=self._preroll_n)

    # ── Silero 单帧 ────────────────────────────────────────────────────────
    def _vad_prob(self, frame: np.ndarray) -> float:
        x = np.concatenate([self._ctx, frame])[None, :]      # (1, 576)
        out, self._state = self.vad.run(
            None, {"input": x.astype(np.float32), "state": self._state, "sr": self._sr})
        self._ctx = x[0, -CONTEXT:]                          # 取拼接后的最后 64
        return float(out[0][0])

    # ── SmartTurn：这段音频听起来说完了吗 ───────────────────────────────────
    def _turn_complete(self, wav: np.ndarray) -> float:
        need = ST_SECONDS * SR
        if wav.size > need:
            wav = wav[-need:]
        elif wav.size < need:
            wav = np.pad(wav, (need - wav.size, 0))          # 前面补零，右对齐
        feats = self.fx(wav, sampling_rate=SR, return_tensors="np",
                        padding="max_length", max_length=need,
                        truncation=True, do_normalize=True)["input_features"]
        prob = self.turn.run(None, {self.turn_in: feats.astype(np.float32)})[0]
        return float(np.asarray(prob).reshape(-1)[0])        # 已过 Sigmoid，不要再压

    # ── 入口 ──────────────────────────────────────────────────────────────
    def feed(self, pcm) -> list[tuple[str, np.ndarray | None]]:
        if isinstance(pcm, (bytes, bytearray, memoryview)):
            samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        else:
            a = np.asarray(pcm)
            # int16 数组要先转 float 再缩放；直接 astype(int16) 会把 float32 的
            # [-1,1] 全部截断成 0（验证 agent 在别处实测到的静默归零 bug）
            samples = (a.astype(np.float32) / 32768.0) if a.dtype == np.int16 \
                else a.astype(np.float32).reshape(-1)

        buf = np.concatenate([self._pending, samples]) if self._pending.size else samples
        n = (buf.size // FRAME) * FRAME
        self._pending = buf[n:].copy()                       # copy：别持有调用方内存
        events: list[tuple[str, np.ndarray | None]] = []

        for i in range(0, n, FRAME):
            frame = buf[i:i + FRAME]
            p = self._vad_prob(frame)
            voiced = p >= self.threshold

            if not self._speaking:
                self._preroll.append(frame.copy())
                self._hits = self._hits + 1 if voiced else 0
                if self._hits >= self.start_frames:
                    self._speaking = True
                    self._silence = 0
                    self._utter = list(self._preroll)   # 回补起音，含当前帧
                    self._preroll.clear()
                    events.append(("speech_start", None))
                continue

            self._utter.append(frame.copy())
            if voiced:
                self._silence = 0
            else:
                self._silence += 1
                # 静音超过硬上限：不再问 SmartTurn，直接提交。
                timeout = self._silence >= self.max_silence_frames
                # 只在悬停点、以及之后每隔 hangover_frames 帧问一次 SmartTurn。
                # 每帧都问的话 700ms 内要跑 22 次 × 50ms，CPU 直接吃满。
                due = (self._silence >= self.hangover_frames
                       and (self._silence - self.hangover_frames) % self.hangover_frames == 0)
                if timeout or due:
                    done = timeout or self._turn_complete(
                        np.concatenate(self._utter)) >= self.turn_threshold
                    if done:
                        wav = np.concatenate(self._utter)
                        events.append(("speech_end", None))
                        events.append(("turn_complete", wav))
                        self._reset_utterance()
                    else:
                        # 只是句中停顿，继续听。注意 _silence 不清零——否则
                        # SmartTurn 若持续判"没说完"，兜底上限永远追不上。
                        pass

            if sum(a.size for a in self._utter) >= self.max_samples:
                wav = np.concatenate(self._utter)
                events.append(("speech_end", None))
                events.append(("turn_complete", wav))
                self._reset_utterance()

        return events

    def _reset_utterance(self) -> None:
        self._speaking = False
        self._hits = 0
        self._silence = 0
        self._utter = []
        self._preroll.clear()

    def reset_session(self) -> None:
        """仅在真正的会话边界调用。不要定时调——会丢 LSTM 上下文。"""
        self._state[:] = 0.0
        self._ctx[:] = 0.0
        self._pending = np.zeros(0, dtype=np.float32)
        self._reset_utterance()
