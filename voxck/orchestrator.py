"""编排层：浏览器 WS ⇄ VAD/ASR/LLM/TTS 回合状态机。

设计要点：
- 下行不做服务端节拍。TTS 生成快于实时（RTF~0.35），全速推给浏览器，
  由前端绝对时间轴排程负责无缝播放。打断时前端 flush 已排程源即可。
  （参考实现的 paced 喂入是为了给扩散渲染器按实时供料，我们没有渲染器。）
- GPU 是串行资源：ASR / TTS / LLM 共用一块 Metal。用一把锁避免三者并发，
  实测并发会让 TTS 的 RTF 从 0.35 劣化到 0.92（逼近断流阈值）。
- 打断：VAD 检测到用户开口 → 取消 LLM 与 TTS → 通知前端清队列。
"""
from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
import math
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import time
from pathlib import Path

import aiohttp
import numpy as np
from aiohttp import web

from . import persona as persona_mod
from . import guard
from .semantic_guard import SemanticGuard
from .llm import Llm
from .conversation import ConversationJournal, ResumeError, load_resume

log = logging.getLogger("voxck")
ROOT = Path(__file__).resolve().parent.parent


def sentence_key(sent: str) -> str:
    return " ".join(sent.split())[:160]


@dataclass
class PlaybackTurn:
    """生成完成不等于用户听完；历史只提交已完整播放的合成段。"""

    id: int
    user: dict
    segments: list[dict] = field(default_factory=list)
    audio_ms: float = 0.0
    played_ms: float = 0.0
    generation_done: bool = False
    interrupted: bool = False
    assistant: dict | None = None


@dataclass
class InputUtterance:
    pcm: np.ndarray
    ended_at: float
    clip_id: str = ""


class Session:
    """一条浏览器连接的回合状态机。"""

    def __init__(self, ws: web.WebSocketResponse, app: web.Application):
        self.ws = ws
        self.app = app
        self.cfg = app["cfg"]
        self.vad = app["make_vad"]()
        self.asr = app["asr"]
        self.tts = app["tts"]
        self.llm: Llm = app["llm"]
        self.gpu: asyncio.Lock = app["gpu_lock"]
        self.mlx = app["mlx_exec"]
        self.sem: SemanticGuard = app["semantic"]
        self.http: aiohttp.ClientSession = app["http"]

        self.history: list[dict] = [{"role": "system", "content": app["persona"]["prompt"]}]
        self.speaking = False
        self.generating = False
        self.reply_task: asyncio.Task | None = None
        # 用户输入与可取消的回复分开：新 speech_start 绝不能取消 ASR。
        self.input_queue: asyncio.Queue[InputUtterance] = asyncio.Queue()
        self.input_task: asyncio.Task | None = None
        self.user_speaking = False
        self._input_busy = False
        self._processing_audio = False
        self._input_revision = 0
        self._answered_revision = -1
        self._latest_input: tuple[dict, float, float] | None = None
        self._closing = False
        self._closed = False
        self.journal: ConversationJournal | None = None
        self.resume_count = 0
        self._user_clip_ids: dict[int, str] = {}
        archive_root = app.get("conversation_root")
        if archive_root is not None:
            self.journal = ConversationJournal(Path(archive_root))
            try:
                restored = load_resume(Path(archive_root))
                self.history.extend(restored)
                self.resume_count = len(restored)
            except ResumeError:
                log.exception("Conversation restore rejected; seed preserved")
        self.active_turn: PlaybackTurn | None = None
        self.playback_turns: dict[int, PlaybackTurn] = {}
        self._respond_lock = asyncio.Lock()
        self.t_speech_end = 0.0
        self._rx_bytes = 0
        self._rx_logged = 0
        self._rx_peak = 0.0
        # 复读检测按会话保持：跨回合才能发现"上一轮说过同一句"
        self.repeats = guard.RepeatGuard()
        self.recent_openers: list[str] = []
        # 数字人：每回合一个 id，浏览器据此丢弃打断后迟到的帧；stop 事件让渲染线程尽快退出
        self.face = app.get("face")
        self.face_exec = app.get("face_exec")
        self.turn_id = 0
        self.face_stop = threading.Event()
        self._face_speed = 9.9          # 上一段渲染的实时倍率（自适应降帧用）
        self._face_jobs = 0             # 在途渲染段数；>0 时待机帧暂停，避免游标打架
        self.idle_task: asyncio.Task | None = None

    async def _mlx(self, fn, *args):
        """把 MLX 工作派到那一个专用线程。

        MLX 的 Stream 是**线程局部**的：在 A 线程创建的生成器，到 B 线程再取值会抛
        'There is no Stream(gpu, N) in current thread'。asyncio.to_thread 用的是默认
        线程池、每次线程不定，所以必须固定到单线程执行器。
        单线程同时也满足 TTS 的单飞要求（声码器是共享的单份流式状态）。
        """
        return await asyncio.get_running_loop().run_in_executor(self.mlx, fn, *args)

    # ── 出站 ──────────────────────────────────────────────────────────────
    async def send(self, **kw):
        if not self.ws.closed:
            try:
                await self.ws.send_str(json.dumps(kw))
            except (ConnectionError, aiohttp.ClientConnectionError):
                # 连接关闭与 closed 标志更新可以有先后差；输入仍必须落盘、排空。
                log.debug("WebSocket closed before text delivery")

    # 下行二进制帧格式：首字节 'A' = 16k int16 PCM；'V' = 视频帧，后接 <H 回合id><I pts毫秒> + JPEG
    async def send_audio(self, pcm: bytes):
        if not self.ws.closed:
            await self.ws.send_bytes(b"A" + pcm)

    async def send_frame(self, turn: int, pts_ms: int, jpg: bytes):
        if pts_ms != self.IDLE_PTS and (self.active_turn is None or turn != self.active_turn.id):
            return                      # 已打断的渲染任务可能仍有排在事件循环里的发送
        if jpg and not self.ws.closed:
            await self.ws.send_bytes(b"V" + struct.pack("<HI", turn & 0xFFFF, pts_ms) + jpg)

    def _commit_played_history(self, turn: PlaybackTurn) -> None:
        text = " ".join(seg["text"] for seg in turn.segments
                        if seg["end_ms"] is not None and seg["end_ms"] <= turn.played_ms).strip()
        if not text:
            return
        if turn.assistant is not None:
            if turn.assistant["content"] == text:
                return
            turn.assistant["content"] = text
            self._archive_assistant(turn, text)
            return
        # 迟到的旧回合进度必须插回其原始 user 后面，不能污染新回合的顺序。
        for i, message in enumerate(self.history):
            if message is turn.user:
                turn.assistant = {"role": "assistant", "content": text}
                self.history.insert(i + 1, turn.assistant)
                self._archive_assistant(turn, text)
                return

    def _archive_assistant(self, turn: PlaybackTurn, text: str) -> None:
        if self.journal is not None:
            try:
                self.journal.record_assistant(turn.id, text, turn.interrupted,
                    after_clip_id=self._user_clip_ids.get(id(turn.user), ""))
            except OSError:
                log.exception("Could not archive assistant playback")

    async def on_control(self, message: dict) -> None:
        """浏览器报告实际播放进度。旧回合 ACK 可补历史，不能结束新回合。"""
        kind = message.get("type")
        if kind not in ("playback_progress", "playback_done"):
            return
        try:
            turn = self.playback_turns.get(int(message.get("id", message.get("turn"))))
        except (TypeError, ValueError, OverflowError):
            return
        if turn is None:
            return
        value = message.get("played_ms")
        if value is None and kind == "playback_done" and turn.generation_done and not turn.interrupted:
            value = turn.audio_ms     # 兼容只报告 playback_done(id) 的客户端
        try:
            played_ms = float(value)
        except (TypeError, ValueError, OverflowError):
            return
        if not math.isfinite(played_ms) or played_ms < 0:
            return
        if (kind == "playback_done" and turn.generation_done and not turn.interrupted
                and self.active_turn is turn):
            # 自然 drain 已证明所有音频播完。秒→毫秒的浮点误差不能漏掉最后整段历史。
            played_ms = turn.audio_ms
        turn.played_ms = max(turn.played_ms, min(played_ms, turn.audio_ms))
        self._commit_played_history(turn)
        log.info("playback_ack type=%s turn=%d played_ms=%.3f audio_ms=%.3f "
                 "generation_done=%s interrupted=%s committed_words=%d",
                 kind, turn.id, turn.played_ms, turn.audio_ms, turn.generation_done,
                 turn.interrupted, len((turn.assistant or {}).get("content", "").split()))
        if kind == "playback_done" and turn.generation_done and self.active_turn is turn:
            self.active_turn = None
            self.speaking = False
            self.face_stop.set()
            await self.send(type="state", state="listening")

    FACE_CTX_S = 0.4          # 分段渲染时带上前一段的 0.4s 音频做 Whisper 左侧上下文
    FACE_MIN_S = 1.0          # 攒够 1s 音频就开一段渲染，不等整句 TTS 合成完

    IDLE_PTS = 0xFFFFFFFF     # 待机帧的 pts 标记：浏览器立即画，不排时间轴

    async def start_face(self):
        """连接建立：告诉浏览器画面参数，并开始推待机帧。
        待机是已确认闭嘴的中性画面；说话帧按实际回复音频另行生成。"""
        if self.face is None:
            await self.send(type="face", ready=False, reason="Live face renderer is unavailable.")
            return
        await self.send(type="face", ready=True, fps=self.face.fps, w=self.face.out_w, h=self.face.out_h)
        self.idle_task = asyncio.create_task(self._idle_loop())

    async def _idle_loop(self):
        loop = asyncio.get_running_loop()
        interval = 1.0 / self.face.fps
        nxt = loop.time()
        try:
            while not self.ws.closed:
                nxt += interval                                   # 虚拟时刻表，不按帧 sleep 漂移
                await asyncio.sleep(max(0.0, nxt - loop.time()))
                if self.speaking or self._face_jobs > 0:
                    nxt = loop.time()
                    continue
                await self.send_frame(0, self.IDLE_PTS, self.face.next_idle_jpeg())
        except asyncio.CancelledError:
            return
        except Exception:
            log.debug("idle loop ended", exc_info=True)

    def _render_job(self, turn: int, pts0: float, wav: np.ndarray, skip: int, stop: threading.Event,
                    loop, t_audio0: float):
        try:
            self._render_job_inner(turn, pts0, wav, skip, stop, loop, t_audio0)
        finally:
            self._face_jobs -= 1

    def _render_job_inner(self, turn: int, pts0: float, wav: np.ndarray, skip: int, stop: threading.Event,
                          loop, t_audio0: float):
        """渲染线程：一段音频 → 逐帧 JPEG → 交给事件循环发出。pts 相对本回合音频起点。

        自适应降帧：渲染吞吐不足时（上一段 < 1.1 倍实时，或这段已经落后于播放进度），
        本段每两帧算一帧。浏览器按 pts 显示、缺帧时保持上一帧，所以只是嘴动得粗一点。
        """
        if stop.is_set():
            return
        fps = self.face.fps
        # 已落后播放多少秒（近似）：浏览器会把首块音频扣到第一帧到达（~0.8s），这段不算落后
        late = (time.perf_counter() - t_audio0) - 0.8 - pts0 if t_audio0 else 0.0
        stride = 2 if (self._face_speed < 1.1 or late > 0.4) else 1
        t = time.perf_counter()
        n = 0
        try:
            for fr in self.face.render(wav, stop, skip_frames=skip, stride=stride):
                if stop.is_set():
                    break
                pts_ms = int(round((pts0 + n * stride / fps) * 1000))
                n += 1
                asyncio.run_coroutine_threadsafe(self.send_frame(turn, pts_ms, self.face.jpeg(fr)), loop)
        except Exception:
            log.exception("face render failed")
            return
        dt = time.perf_counter() - t
        audio_s = n * stride / fps
        if not stop.is_set() and dt > 0:
            self._face_speed = audio_s / dt                 # 实时倍率：>1 才跟得上
        log.info("脸 %d 帧 %.2fs（音频 %.1fs @ %.1fs，×%.2f 实时，步长 %d，落后 %.1fs%s）",
                 n, dt, audio_s, pts0, audio_s / dt if dt else 0, stride, max(0.0, late),
                 "，打断" if stop.is_set() else "")

    # ── 入站音频 ──────────────────────────────────────────────────────────
    async def on_audio(self, pcm: bytes):
        if self._closing:
            return
        # 诊断：确认上行音频真的到了，以及它有没有声音（全静音 = 麦克风没通）
        self._rx_bytes += len(pcm)
        a = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        if a.size:
            self._rx_peak = max(self._rx_peak, float(np.sqrt(np.mean(a * a))))
        if self._rx_bytes - self._rx_logged >= 16000 * 2 * 10:     # 每 10 秒报一次
            self._rx_logged = self._rx_bytes
            # 用区间**峰值**判断麦克风通不通。之前用单块瞬时 RMS，采样点常落在
            # 词间静音上，于是一边正常转写一边报「麦克风可能没通」，纯属误导。
            log.info("上行音频 %.0fs · 区间峰值 %.4f%s",
                     self._rx_bytes / 2 / 16000, self._rx_peak,
                     "  ← 一直是 0，麦克风没通" if self._rx_peak < 0.002 else "")
            self._rx_peak = 0.0
        self._processing_audio = True
        try:
            for ev, buf in self.vad.feed(pcm):
                log.info("VAD 事件: %s%s", ev,
                         f" ({buf.size/16000:.2f}s)" if buf is not None else "")
                if ev == "speech_start":
                    self.user_speaking = True
                    self._input_revision += 1
                    if self.speaking or (self.reply_task and not self.reply_task.done()):
                        await self.interrupt()
                    await self.send(type="state", state="listening")
                elif ev == "speech_end":
                    self.user_speaking = False
                    self.t_speech_end = time.perf_counter()
                    await self.send(type="state", state="transcribing")
                elif ev == "turn_complete":
                    # 在任何 await 之前保存并入队；同一包里的下一次开口也不会丢这段。
                    self._enqueue_input(buf)
                    if self.speaking or (self.reply_task and not self.reply_task.done()):
                        await self.interrupt()
        finally:
            self._processing_audio = False
            self._maybe_answer()

    def _enqueue_input(self, pcm: np.ndarray) -> None:
        job = InputUtterance(np.asarray(pcm, dtype=np.float32).copy(),
                             self.t_speech_end or time.perf_counter())
        if self.journal is not None:
            try:
                job.clip_id = self.journal.save_audio(job.pcm)
            except (OSError, ValueError):
                log.exception("Audio backup failed; input remains queued in memory")
                asyncio.create_task(self.send(type="sys", text=
                    "Audio backup failed. This segment is still being transcribed."))
        self._input_revision += 1
        self.input_queue.put_nowait(job)
        if self.input_task is None or self.input_task.done():
            self.input_task = asyncio.create_task(self._drain_inputs())

    async def _transcribe_input(self, job: InputUtterance) -> tuple[dict, float, float] | None:
        async with self.gpu:
            started = time.perf_counter()
            text = await self._mlx(self.asr.transcribe, job.pcm)
        asr_ms = (time.perf_counter() - started) * 1000
        text = (text or "").strip()
        if not text:
            if self.journal is not None:
                self.journal.record_error(job.clip_id, "ASR returned no transcript")
            await self.send(type="sys", text="Could not transcribe a speech segment. "
                            + ("Its audio was saved locally. " if job.clip_id else "")
                            + "Please repeat that part.")
            return None
        user = {"role": "user", "content": text}
        self._user_clip_ids[id(user)] = job.clip_id
        # 真实历史保留对象与顺序；模型请求里再合并连续的用户片段。
        self.history.append(user)
        if self.journal is not None:
            try:
                self.journal.record_user(job.clip_id, text)
            except OSError:
                log.exception("Transcript backup failed")
                await self.send(type="sys", text="Transcript backup failed; the text is still in this conversation.")
        await self.send(type="user", text=text)
        log.info("input_transcribed clip=%s duration_s=%.3f words=%d", job.clip_id,
                 job.pcm.size / 16000, len(text.split()))
        return user, asr_ms, job.ended_at

    async def _drain_inputs(self) -> None:
        self._input_busy = True
        try:
            while not self.input_queue.empty():
                job = self.input_queue.get_nowait()
                try:
                    result = await self._transcribe_input(job)
                    if result is not None:
                        self._latest_input = result
                    else:
                        self._latest_input = None
                except Exception as error:
                    self._latest_input = None
                    log.exception("Input transcription failed")
                    if self.journal is not None:
                        with contextlib.suppress(OSError):
                            self.journal.record_error(job.clip_id, error)
                    with contextlib.suppress(ConnectionError, aiohttp.ClientConnectionError):
                        await self.send(type="sys", text="Speech transcription failed. "
                                        + ("Its audio was saved locally. " if job.clip_id else "")
                                        + "Please repeat that part.")
                finally:
                    self.input_queue.task_done()
        finally:
            self._input_busy = False
            self._maybe_answer()

    def _maybe_answer(self) -> None:
        if (self._closing or self.ws.closed or self.user_speaking or self._input_busy
                or self._processing_audio or not self.input_queue.empty()
                or self._latest_input is None or self._answered_revision == self._input_revision):
            return
        if self.reply_task is not None and not self.reply_task.done():
            return
        revision = self._input_revision
        self._answered_revision = revision
        self.reply_task = asyncio.create_task(self._answer_latest(revision))

    async def _answer_latest(self, revision: int) -> None:
        # create_task 后的新语音可能先到；再次验证，避免开始回答半截发言。
        if (self._closing or self.ws.closed or self.user_speaking or self._input_busy
                or not self.input_queue.empty() or revision != self._input_revision):
            self._answered_revision = -1
            return
        user, asr_ms, ended_at = self._latest_input
        async with self._respond_lock:
            await self._respond_user(user, asr_ms, ended_at)

    def _request_history(self) -> list[dict]:
        messages: list[dict] = []
        for message in self.history:
            if (messages and message["role"] == "user" and messages[-1]["role"] == "user"):
                messages[-1]["content"] += "\n" + message["content"]
            else:
                messages.append(dict(message))
        keep = self.cfg["chat_turns"] * 2
        # 本地完整历史不裁剪。请求窗口按合并后的轮次计算，整段用户论述不会被切掉。
        if len(messages) - 1 > keep + 8:
            messages = [messages[0]] + messages[-keep:]
        return messages

    async def close(self) -> None:
        if self._closed:
            return
        self._closing = True
        try:
            await self.interrupt()
        finally:
            try:
                if self.input_task is not None:
                    await asyncio.shield(self.input_task)
            finally:
                if self.idle_task is not None:
                    self.idle_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await self.idle_task
                if self.journal is not None:
                    self.journal.close()
                self._closed = True

    async def interrupt(self):
        """打断：停生成、停合成、清前端队列。"""
        turn = self.active_turn
        if turn is not None:
            turn.interrupted = True
            self._commit_played_history(turn)
        self.active_turn = None
        self.speaking = False
        self.face_stop.set()
        # 先清浏览器排程，再等在途模型工作退出；不能让清声音卡在 GPU 的 next() 后面。
        await self.send(type="interrupt", id=turn.id if turn is not None else self.turn_id)
        if self.reply_task and not self.reply_task.done():
            self.reply_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.reply_task
        # next()/close()/decoder reset 必须在同一条 MLX 线程按队列顺序执行。
        await self._mlx(self.tts.interrupt)

    # ── 一个回合 ──────────────────────────────────────────────────────────
    async def respond(self, pcm: np.ndarray):
        """Compatibility entry point; the live microphone uses the independent input queue."""
        async with self._respond_lock:
            await self._respond(pcm)

    async def _respond(self, pcm: np.ndarray):
        result = await self._transcribe_input(InputUtterance(pcm,
            self.t_speech_end or time.perf_counter()))
        if result is not None:
            await self._respond_user(*result)

    async def _respond_user(self, user: dict, asr_ms: float, t0: float):
        text = user["content"]
        turn: PlaybackTurn | None = None
        checks: list[asyncio.Task] = []
        self.generating = True
        try:
            self.speaking = True
            self.turn_id = self.turn_id % 0xFFFF + 1
            turn_id = self.turn_id
            turn = PlaybackTurn(turn_id, user)
            self.active_turn = turn
            self.playback_turns[turn_id] = turn
            # 旧回合进度在下一轮 ASR 期间仍能回填；只保留有限数量的回合记录。
            while len(self.playback_turns) > 4:
                self.playback_turns.pop(next(iter(self.playback_turns)))
            face_stop = self.face_stop = threading.Event()
            turn_pcm = bytearray()              # 本回合全部音频（16k int16）
            rendered = 0                        # 已交给渲染的样本数 → 下一段的 pts 起点
            t_audio0 = 0.0                      # 首块音频发出的时刻（估算浏览器播放进度）
            await self.send(type="turn", id=turn_id, fps=self.face.fps if self.face is not None else 0,
                            face=self.face is not None)
            await self.send(type="state", state="speaking")

            def submit_render(final: bool) -> None:
                """把 turn_pcm 里还没渲染的部分交给渲染线程（攒够 FACE_MIN_S 或收尾时）。"""
                nonlocal rendered
                if self.face is None or self.active_turn is not turn:
                    return
                total = len(turn_pcm) // 2
                if total - rendered < (1 if final else int(self.FACE_MIN_S * 16000)):
                    return
                ctx = min(rendered, int(self.FACE_CTX_S * 16000))
                seg = np.frombuffer(bytes(turn_pcm[(rendered - ctx) * 2: total * 2]), dtype="<i2").astype(np.float32) / 32768.0
                skip = int(round(ctx / 16000 * self.face.fps))
                pts0 = rendered / 16000.0
                rendered = total
                loop = asyncio.get_running_loop()
                self._face_jobs += 1
                loop.run_in_executor(self.face_exec, self._render_job, turn_id, pts0, seg, skip, face_stop, loop, t_audio0)
            reply_parts: list[str] = []
            t_llm = time.perf_counter()
            ttft_ms = tta_ms = 0.0

            # TTS 攒块策略：最小有效块 ~12 词。逐句喂会让短句的净倍率变成 1.12x，
            # 反而落后于实时（源码级实测，见 docs/voice-apis.md）。
            # 唯一例外是**首块**——它可以短，因为它买的是首音延迟。
            MIN_WORDS = self.cfg.get("tts_min_words", 12)
            pending: list[str] = []
            spoken_first = False

            async def flush_tts(force: bool = False) -> None:
                nonlocal pending, spoken_first, tta_ms, t_audio0
                if not pending:
                    return
                text = " ".join(pending)
                if not force and not spoken_first and len(text.split()) < MIN_WORDS:
                    # 首块允许短：抢首音
                    pass
                elif not force and len(text.split()) < MIN_WORDS:
                    return
                pending = []
                # 字幕：这段文字将从当前累计音频位置开始念，浏览器按音频时钟到点再显示
                segment = {"text": text, "start_ms": len(turn_pcm) / 32, "end_ms": None}
                turn.segments.append(segment)
                await self.send(type="caption", turn=turn_id, pts_ms=int(segment["start_ms"]), text=text)
                text = guard.for_speech(text)      # markdown 会被 TTS 念出来
                async with self.gpu:                        # TTS 独占 GPU（单飞：
                    # speech_tokenizer.decoder 是单一共享对象、单份流式状态，
                    # 两个并发 generate(stream=True) 会互相污染声码器缓冲）
                    gen = self.tts.stream(text)             # 立即捕获世代号
                    try:
                        while True:
                            # 每块都过线程取，合成期间事件循环继续接收麦克风和播放 ACK。
                            chunk = await self._mlx(next, gen, None)
                            if chunk is None:
                                if turn.audio_ms > segment["start_ms"]:
                                    segment["end_ms"] = turn.audio_ms
                                self._commit_played_history(turn)
                                break
                            if self.active_turn is not turn:
                                break
                            if not tta_ms:
                                tta_ms = (time.perf_counter() - t0) * 1000
                            await self.send_audio(chunk)
                            if not t_audio0:
                                t_audio0 = time.perf_counter()
                            turn_pcm.extend(chunk)
                            turn.audio_ms = len(turn_pcm) / 32
                            # 攒够 1s 就画嘴，不等整句合成完。
                            submit_render(final=False)
                    finally:
                        # cancel 不会停止 executor 中的 next()，排队 close 才能安全收束。
                        await self._mlx(gen.close)
                spoken_first = True
                submit_render(final=True)

            budget = guard.WordBudget(self.cfg.get("word_ceiling", 90))
            self.repeats.new_turn()
            dropped = 0
            repeat_drops: list[str] = []
            other_drops = 0
            restatement = guard.allows_restatement(text)
            over = False
            held: tuple | None = None       # 提前一句待判的 (句子, 检查任务)

            async def accept(sent: str) -> None:
                nonlocal over
                if not budget.allow(sent):
                    log.info("到达词数上限，本轮在句子边界收尾")
                    over = True
                    return
                reply_parts.append(sent)
                pending.append(sent)
                await flush_tts(force=not spoken_first)

            async def resolve(sent: str, task) -> None:
                """等该句的语义结论——此时它已经跑了整整一句的时间，通常早已就绪。

                但守卫服务偶发 6.5s 的慢响应（实测），会让整条语音链停住 5 秒：音频断流、
                嘴定格。所以只等一个预算（默认 1.5s），超时就先说、审计里记一笔——守卫是
                安全网，不是闸门。"""
                nonlocal dropped, other_drops
                budget = float(self.cfg.get("semantic_guard", {}).get("wait_budget_s", 1.5))
                try:
                    swhy, sp = await asyncio.wait_for(asyncio.shield(task), timeout=budget)
                except asyncio.TimeoutError:
                    log.warning("语义守卫 %.1fs 未答，先说：%s", budget, " ".join(sent.split())[:80])
                    self.sem.audit.append({"sentence": sentence_key(sent), "error": "wait_budget"})
                    await accept(sent)
                    return
                if swhy:
                    log.warning("丢弃违规句 [%s p=%.2f]: %s", swhy, sp,
                                " ".join(sent.split())[:100])
                    dropped += 1
                    other_drops += 1
                    return
                await accept(sent)

            # 尾部提醒保持最后一条是当前 user；明确要求重说时不要求换开场。
            msgs = self._request_history()
            if self.recent_openers and not restatement:
                msgs.insert(-1, {"role": "system", "content":
                    "You have already opened recent turns with these lines. Open differently "
                    "this time: " + " / ".join(f'"{o}"' for o in self.recent_openers)})

            async def process_sentence(sent: str, is_first: bool) -> None:
                nonlocal ttft_ms, dropped, other_drops, held
                if is_first:
                    if not ttft_ms:
                        ttft_ms = (time.perf_counter() - t_llm) * 1000
                    self.recent_openers.append(sent.strip())
                    del self.recent_openers[:-3]        # 只留最近三次
                # 第一层：正则，零成本，只认见过的句式
                ok, why = guard.clean(sent)
                if ok is None:
                    dropped += 1
                    other_drops += 1
                    return
                # 首句只新增完整长句重复检查；短回应和近似开场不额外等待。
                # 若用户明确要求重说/解释，可复述，但仍经过内容守卫。
                repeated = (self.repeats.is_exact_repeat(sent) if is_first
                            else self.repeats.is_repeat(sent)) if not restatement else False
                if repeated:
                    log.info("丢弃复读句: %s", " ".join(sent.split())[:90])
                    repeat_drops.append(sent)
                    dropped += 1
                    return
                self.repeats.add(sent)
                # 第二层：语义（TypeSafe Noul），补正则做不到的——换了说法的人生判词、
                # 以及账本外编造的数字。
                if (not reply_parts and held is None) or not self.sem.enabled:
                    # 首句决定首音延迟，一秒都不能等：直接出声，只在后台审计。
                    if self.sem.enabled:
                        checks.append(asyncio.create_task(self.sem.audit_only(sent)))
                    await accept(sent)
                    return
                # 其余句子提前一句判：合成第 N 句时第 N+1 句的检查同时在跑，
                # 轮到它时结论已就绪。直接同步 await 会真卡住 TTS
                # （实测中位 ~460ms，尾部会超 3s 触发 fail-open）。
                task = asyncio.create_task(self.sem.judge(sent))
                checks.append(task)
                if held is not None:
                    await resolve(*held)
                held = (sent, task) if not over else None
                if over and task:
                    task.cancel()

            async def generate(attempt: int, request_messages: list[dict]) -> None:
                nonlocal held
                generated_words = 0
                accepted_before = len(" ".join(reply_parts).split())
                source = self.llm.stream_sentences(self.http, request_messages)
                try:
                    async with contextlib.aclosing(source):
                        async for sent, is_first in source:
                            if self.active_turn is not turn or over:
                                break
                            self.app["llm_ready"] = True
                            generated_words += len(sent.split())
                            await process_sentence(sent, is_first)
                    if not generated_words:
                        self.app["llm_ready"] = False
                        raise RuntimeError("LLM response ended without answer text. Check the configured brain service.")
                    if held is not None:
                        if self.speaking and not over:
                            await resolve(*held)
                        else:
                            held[1].cancel()
                        held = None
                    if self.speaking:
                        await flush_tts(force=True)
                except Exception:
                    self.app["llm_ready"] = False
                    raise
                finally:
                    stats = getattr(self.llm, "last_generation", {})
                    log.info("reply_generation turn=%d attempt=%d model=%s user=%r "
                             "finish_reason=%s raw_words=%d emitted_words=%d accepted_words=%d "
                             "repeat_drops=%d other_drops=%d",
                             turn.id, attempt, getattr(self.llm, "model", self.cfg.get("llm_model", "fake")),
                             sentence_key(text), stats.get("finish_reason"),
                             stats.get("raw_words", generated_words), generated_words,
                             len(" ".join(reply_parts).split()) - accepted_before,
                             len(repeat_drops), other_drops)

            await generate(1, msgs)
            if (not reply_parts and repeat_drops and not other_drops
                    and self.active_turn is turn and not over):
                # 纯复读造成整轮空才重生成一次；绝不把刚丢弃的旧句回放。
                log.warning("本轮只有复读，最多重生成一次 turn=%d", turn.id)
                retry_msgs = list(msgs)
                rejected = list(dict.fromkeys(repeat_drops))[:8]
                retry_msgs.insert(-1, {"role": "system", "content":
                    "Answer the latest user's specific question directly in fresh, brief sentences. "
                    "Do not repeat or quote any of these already-spoken lines: "
                    + " / ".join(f'"{s}"' for s in rejected)})
                await generate(2, retry_msgs)
            if not reply_parts and self.active_turn is turn:
                await self.send(type="sys", text="Could not generate a fresh reply. Please try another question.")
            if dropped:
                log.info("本轮丢弃 %d 句（复读 %d，内容 %d）", dropped, len(repeat_drops), other_drops)

            reply = " ".join(reply_parts).strip()
            if reply:
                await self.send(
                    type="assistant", id=turn_id, text=reply,
                    asr_ms=round(asr_ms), ttft_ms=round(ttft_ms), tta_ms=round(tta_ms),
                    e2e_ms=round((time.perf_counter() - t0) * 1000),
                    words=len(reply.split()),
                    cached=self.llm.cached_tokens, prompt_tokens=self.llm.prompt_tokens)
            turn.generation_done = True
            await self.send(type="generation_done", id=turn_id, audio_ms=round(turn.audio_ms), text=reply)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception("respond failed")
            if turn is not None and self.active_turn is turn:
                turn.interrupted = True
                self.active_turn = None
                self.speaking = False
                self.face_stop.set()
                await self.send(type="interrupt", id=turn.id)
            await self.send(type="sys", text=f"error: {e}")
        finally:
            self.generating = False
            if turn is None or turn.interrupted:
                for task in checks:
                    if not task.done():
                        task.cancel()
            # 生成结束之后仍在说话，只有 playback_done 或 interrupt 能回到 listening。
            if self.active_turn is None:
                self.speaking = False
                await self.send(type="state", state="listening")


async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(max_msg_size=4 << 20)
    await ws.prepare(request)
    sess = Session(ws, request.app)
    log.info("client connected")
    if sess.resume_count:
        await sess.send(type="sys", text=f"Restored {sess.resume_count} saved messages. Continue when ready.")
        for message in sess.history[1:]:
            await sess.send(type="restored", role=message["role"], text=message["content"])
    await sess.start_face()
    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                await sess.on_audio(msg.data)
            elif msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    message = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                if isinstance(message, dict):
                    await sess.on_control(message)
    finally:
        await sess.close()
        log.info("client gone")
    return ws


def build_app(cfg: dict) -> web.Application:
    app = web.Application()
    app["cfg"] = cfg
    app["conversation_root"] = ROOT / cfg.get("conversation_dir", "conversations")
    app["ready"] = False
    app["llm_ready"] = False
    app["persona"] = persona_mod.load(ROOT / cfg["persona"])
    app["gpu_lock"] = asyncio.Lock()
    sem_cfg = cfg.get("semantic_guard", {})
    app["semantic"] = SemanticGuard(
        enabled=sem_cfg.get("enabled", True),
        threshold=sem_cfg.get("threshold", 0.50),
        strict=sem_cfg.get("strict", False),
        timeout=sem_cfg.get("timeout", 3.0))
    # 所有 MLX 调用都必须跑在同一个线程（Stream 是线程局部的），
    # 且 TTS 声码器是共享单例、必须单飞。一个线程同时满足两条。
    app["mlx_exec"] = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
    app["llm"] = Llm(cfg["llm_url"], cfg["llm_model"],
                     max_tokens=cfg["max_tokens"], temperature=cfg["temperature"],
                     api_key=cfg.get("llm_api_key", ""),
                     extra_body=cfg.get("llm_extra_body"),
                     timeout_s=cfg.get("llm_timeout_s"))

    async def _startup(a):
        # trust_env：云端大脑要走系统代理（这台机器直连是 SSL EOF）；
        # 127.0.0.1 已在 NO_PROXY 里，本地 server 不受影响
        a["http"] = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300), trust_env=True,
            connector=aiohttp.TCPConnector(force_close=False))
        # 组件在 startup 里加载，避免 import 期就吃显存
        from .asr import Asr
        from .tts import Tts
        from .vad import VadTurnDetector
        log.info("loading ASR…");  a["asr"] = Asr(cfg["asr_model"],
                                                  hotwords=cfg.get("hotwords", []),
                                                  corrections=cfg.get("corrections"))
        log.info("loading TTS…");  a["tts"] = Tts(cfg["tts_model"],
                                                  ref_wav=a["persona"]["meta"].get("ref_wav"),
                                                  ref_text_path=a["persona"]["meta"].get("ref_text"))
        a["make_vad"] = lambda: VadTurnDetector(**cfg.get("vad", {}))

        # 预热三件事，否则第一轮要付全部代价（实测首轮 TTFT 9.0s → 预热后 0.7s）：
        #  1. Metal kernel 编译（ASR / TTS 首次调用）
        #  2. GPU 低功耗态（mlx-lm#432：空闲后首次推理可慢 7 倍）
        #  3. LLM 的人设前缀 prefill —— 4100 token 冷启动约 1.1s，预热后走缓存命中 99.8%
        log.info("warming up…")
        t0 = time.perf_counter()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(a["mlx_exec"], a["asr"].warm)
        await loop.run_in_executor(a["mlx_exec"], a["tts"].warm)
        fc = cfg.get("face") or {}
        a["face"] = None
        if fc.get("enabled"):
            try:
                from .face import FaceRenderer
                log.info("loading face renderer…")
                t_f = time.perf_counter()
                mlx_submit = lambda fn, *args: a["mlx_exec"].submit(fn, *args)
                a["face"] = await loop.run_in_executor(
                    None, lambda: FaceRenderer(fc.get("base_video", "web/media/idle.mp4"), mlx_submit=mlx_submit,
                                               fps=fc.get("fps", 20), r0=fc.get("r0", 9),
                                               jpeg_quality=fc.get("jpeg_quality", 80),
                                               mask_mode=fc.get("mask_mode", "mouth"), expand=fc.get("expand", 0.1),
                                               pre_crop=fc.get("pre_crop"), view_scale=fc.get("view_scale", 3.0),
                                               idle_frame=fc.get("idle_frame", 0),
                                               background=fc.get("background")))
                a["face_exec"] = ThreadPoolExecutor(max_workers=1, thread_name_prefix="face")
                log.info("face renderer ready in %.1fs（%dfps，底子 %s）", time.perf_counter() - t_f,
                         a["face"].fps, fc.get("base_video", "web/media/idle.mp4"))
            except Exception:
                log.exception("face renderer failed to load — 继续无脸运行")
                a["face"] = None
        try:
            msgs = [{"role": "system", "content": a["persona"]["prompt"]},
                    {"role": "user", "content": "Hello."}]
            async with asyncio.timeout(float(cfg.get("llm_warm_timeout_s", 12))):
                async with contextlib.aclosing(a["llm"].stream_sentences(a["http"], msgs)) as source:
                    async for _ in source:
                        a["llm_ready"] = True
                        break               # 首句证明 LLM 可用，也把人设前缀灌进缓存
        except Exception:
            a["llm_ready"] = False
            log.warning("LLM 预热失败；大脑当前不可用，本地语音/脸继续启动", exc_info=True)
        await a["semantic"].warm()
        a["ready"] = True
        log.info("warm in %.1fs", time.perf_counter() - t0)
        log.info("ready on http://%s:%s", cfg["host"], cfg["port"])

    async def _heartbeat(a):
        """保温心跳：GPU 空闲会掉低功耗态（mlx-lm#432），复发后下一次推理慢数倍。
        实测对话间隔稍长时首音从 1.9s 涨到 6.2s、10.5s —— 用户只是停顿想一下，
        回复就慢十秒。启动预热只管第一次，之后每次空闲都会复发，所以要持续心跳。
        代价极小：一次小矩阵乘，不碰任何模型。"""
        import mlx.core as mx

        def tick():
            a_ = mx.random.normal((512, 512), dtype=mx.float16)
            mx.eval(a_ @ a_)

        loop = asyncio.get_running_loop()
        while True:
            try:
                await asyncio.sleep(a["cfg"].get("keepwarm_s", 15))
                if a["gpu_lock"].locked():
                    continue            # 真有活在跑，不插队
                await loop.run_in_executor(a["mlx_exec"], tick)
            except asyncio.CancelledError:
                return
            except Exception:
                log.debug("心跳失败", exc_info=True)

    async def _start_heartbeat(a):
        a["hb"] = asyncio.create_task(_heartbeat(a))

    async def _cleanup(a):
        a["ready"] = False
        a["llm_ready"] = False
        if a.get("hb"):
            a["hb"].cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await a["hb"]
        await a["http"].close()
        rep = a["semantic"].report()
        if rep and "无审计" not in rep:
            log.info("\n%s", rep)
        await a["semantic"].close()
        if a.get("face") is not None:
            a["face"].close()
        if a.get("face_exec"):
            a["face_exec"].shutdown(wait=False, cancel_futures=True)
        a["mlx_exec"].shutdown(wait=False, cancel_futures=True)

    app.on_startup.append(_startup)
    app.on_startup.append(_start_heartbeat)
    app.on_cleanup.append(_cleanup)
    async def index(_request: web.Request) -> web.FileResponse:
        # aiohttp 的 add_static 不会把目录请求映射到 index.html，
        # 访问 "/" 会直接 403。必须显式给根路由。
        return web.FileResponse(ROOT / "web" / "index.html")

    async def health(request: web.Request) -> web.Response:
        a = request.app
        return web.json_response({
            "ready": bool(a.get("ready", False)),
            "llm_ready": bool(a.get("llm_ready", False)),
            "face_ready": a.get("face") is not None,
            "models": {"asr": cfg["asr_model"], "tts": cfg["tts_model"], "llm": cfg["llm_model"]},
            "remote_llm": a["llm"].remote,
            "thinking_enabled": a["llm"].thinking_enabled,
            "thinking_disabled": a["llm"].thinking_disabled,
        })

    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/health", health)
    app.router.add_get("/", index)
    app.router.add_static("/", ROOT / "web", show_index=False)
    return app
