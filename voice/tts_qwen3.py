"""Qwen3-TTS streaming synthesis for the local voice-debate pipeline.

Component contract
------------------
  in : str  (one sentence / clause group of English text)
  out: bytes, 16 kHz mono PCM signed 16-bit little-endian, 20 ms frames (640 B)

Everything MLX-touching runs on ONE dedicated worker thread. Callers use
`stream()` (sync generator) or `astream()` (async generator). `interrupt()`
kills the in-flight utterance for barge-in.

Measured on M5 Max / mlx-audio 0.5.3 / Qwen3-TTS-12Hz-1.7B-Base-8bit:
  TTFA 0.157-0.256 s idle, 0.31-0.42 s under concurrent GPU load
  RTF  0.41-0.53 idle, 0.76-0.92 under load
  peak 3.94 GB, 12-29 chunks/utterance at streaming_interval=0.32
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import AsyncIterator, Iterator, Optional

import mlx.core as mx
import numpy as np

log = logging.getLogger(__name__)

# ---- immutable facts read out of the installed source -----------------------
# qwen3_tts/config.py:29,221  sample_rate = 24000
# qwen3_tts/config.py:143     frame_rate  = 12.5  (1 code frame = 1920 samples = 80 ms)
TTS_SR = 24_000
PIPE_SR = 16_000
CODE_HZ = 12.5

FRAME_MS = 20
FRAME_SAMPLES = PIPE_SR * FRAME_MS // 1000          # 320
FRAME_BYTES = FRAME_SAMPLES * 2                     # 640
FADE_IN_SAMPLES = 128                               # 8 ms @16k, kills the start-of-utterance click
FADE_OUT_SAMPLES = 160                              # 10 ms @16k, used on interrupt

# 24k -> 16k is up=2, down=3. mlx_audio's kaiser_best FIR is 2*64*max(up,down)+1
# = 385 taps at the 48 kHz upsampled rate => ~96 input samples of half-support on
# EACH side. So a streaming resampler needs both a left context (discarded after
# filtering) and a right lookahead (held back until the next chunk arrives),
# otherwise resample_poly's edge padding corrupts ~1 sample at every chunk seam
# (measured -45 dBFS ticks every 320 ms without the lookahead).
_RESAMP_CTX = 384       # 16 ms of left context, multiple of `down`
_RESAMP_LOOK = 192      # 8 ms of held-back lookahead, multiple of `down`
                        # -> the resampler adds exactly 8 ms of latency


class _Resampler24to16:
    """Stateful, bit-accurate-at-the-seams 24 kHz -> 16 kHz float32 resampler.

    Wraps mlx_audio.resample.resample_audio_array (scipy resample_poly with the
    kaiser_best FIR). Cost: ~0.2 ms per 320 ms chunk (5120 output samples x ~128
    effective taps) -- under 1% of the ~145 ms the model takes to produce that
    chunk, so resampling to 16 kHz is free in the latency budget. The alternative
    is to run the browser AudioContext at 24 kHz and skip this entirely; we keep
    16 kHz because the whole pipeline (VAD, ASR, echo reference) is 16 kHz.
    """

    def __init__(self) -> None:
        from mlx_audio.resample import resample_audio_array  # scipy-backed

        self._fn = resample_audio_array
        self.reset()

    def reset(self) -> None:
        self._ctx = np.zeros(0, dtype=np.float32)    # left context, already emitted
        self._hold = np.zeros(0, dtype=np.float32)   # right lookahead, not yet emitted

    def push(self, x24: np.ndarray) -> np.ndarray:
        buf = np.concatenate([self._hold, np.asarray(x24, dtype=np.float32)])
        n = ((buf.size - _RESAMP_LOOK) // 3) * 3     # whole `down` periods, safe to emit
        if n <= 0:
            self._hold = buf
            return np.zeros(0, dtype=np.float32)
        proc, self._hold = buf[:n], buf[n:]
        x = np.concatenate([self._ctx, proc, self._hold])   # ctx | emit | lookahead
        y = self._fn(x, TTS_SR, PIPE_SR)                    # -> float32
        lo = self._ctx.size * 2 // 3                        # exact: ctx is a multiple of 3
        out = y[lo:lo + n * 2 // 3]
        emitted = np.concatenate([self._ctx, proc])
        keep = min(_RESAMP_CTX, (emitted.size // 3) * 3)
        self._ctx = emitted[emitted.size - keep:] if keep else np.zeros(0, dtype=np.float32)
        return out

    def drain(self) -> np.ndarray:
        """Flush the held lookahead at end of utterance (edge-padded tail)."""
        if self._hold.size == 0:
            return np.zeros(0, dtype=np.float32)
        pad = (-self._hold.size) % 3
        tail = np.concatenate([self._hold, np.zeros(pad, dtype=np.float32)])
        x = np.concatenate([self._ctx, tail])
        y = self._fn(x, TTS_SR, PIPE_SR)
        out = y[self._ctx.size * 2 // 3:]
        self.reset()
        return out


def _f32_to_pcm16(x: np.ndarray) -> bytes:
    return np.clip(x * 32767.0, -32768.0, 32767.0).astype(np.int16).tobytes()


def fade_out_frame(last_frame: bytes, n: int = FADE_OUT_SAMPLES) -> bytes:
    """Ramp the tail of the last frame to zero. Send this on interrupt so the
    client's buffer ends on a zero crossing instead of a click."""
    s = np.frombuffer(last_frame, dtype=np.int16).astype(np.float32).copy()
    k = min(n, s.size)
    s[-k:] *= np.linspace(1.0, 0.0, k, dtype=np.float32)
    return s.astype(np.int16).tobytes()


@dataclass
class TtsConfig:
    model_path: str = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
    ref_wav: str = ""                 # persona frontmatter: ref_wav
    ref_text: str = ""                # persona frontmatter: ref_text (verbatim)
    streaming_interval: float = 0.32  # -> int(0.32*12.5) = 4 code frames = 320 ms audio
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    max_tokens: int = 4096            # 12.5 Hz -> 327 s ceiling
    lang_code: str = "english"
    keep_warm_interval: float = 0.35  # s between idle GPU pings
    keep_warm_dim: int = 1024         # 1024^3 fp16 matmul ~2 GFLOP, ~0.2 ms
    wired_limit_gb: float = 8.0       # 0 disables; peak measured 3.94 GB
    seed: Optional[int] = None        # pin for reproducible prosody A/B


_STOP = object()


class Tts:
    """Load once, stream many. Thread-confined MLX, cancellable, self-warming."""

    def __init__(self, cfg: TtsConfig) -> None:
        self.cfg = cfg
        self._model = None
        self._ref_audio_24k: Optional[mx.array] = None
        self._jobs: "queue.Queue" = queue.Queue()
        self._live: set = set()          # cancel events of submitted-or-running jobs
        self._gen_id = 0
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._closed = False
        self._thread = threading.Thread(target=self._worker, name="tts-mlx", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ load
    def wait_ready(self, timeout: Optional[float] = None) -> bool:
        return self._ready.wait(timeout)

    def _load(self) -> None:
        from mlx_audio.tts.utils import load as load_tts        # tts/utils.py:135
        from mlx_audio.utils import load_audio                  # utils.py:621

        if self.cfg.wired_limit_gb:
            try:
                mx.set_wired_limit(int(self.cfg.wired_limit_gb * (1 << 30)))
            except Exception as e:  # not fatal, just a paging hint
                log.warning("set_wired_limit failed: %s", e)

        t0 = time.perf_counter()
        # load() -> load_model() -> base_load_model() -> Model.post_load_hook(),
        # which loads the HF text tokenizer, the speech_tokenizer/ subdir (encoder
        # codebooks initialised in place) and wraps the vocoder in mx.compile.
        self._model = load_tts(self.cfg.model_path)
        if self._model.speech_tokenizer is None or not self._model.speech_tokenizer.has_encoder:
            raise RuntimeError(
                "speech tokenizer encoder missing - ICL voice cloning needs a "
                "*-Base-* repo with a speech_tokenizer/ subdirectory"
            )

        # TRAP: load_audio() returns an mx.array unchanged. extract_speaker_embedding()
        # hard-asserts 24 kHz. Resolve the wav ONCE here, at 24 kHz mono float32, and
        # reuse the same array so Model._icl_cache (keyed on (ref_text, (size, sum)))
        # hits and the speech-tokenizer encode is never repeated.
        self._ref_audio_24k = load_audio(self.cfg.ref_wav, sample_rate=TTS_SR)
        if self._ref_audio_24k.ndim != 1:
            self._ref_audio_24k = self._ref_audio_24k.reshape(-1)
        mx.eval(self._ref_audio_24k)
        ref_s = self._ref_audio_24k.size / TTS_SR
        if not (2.0 <= ref_s <= 15.0):
            log.warning("ref_wav is %.1fs; 5-8 s is the sweet spot "
                        "(prefill grows 12.5 KV positions per second of reference)", ref_s)
        if not self.cfg.ref_text.strip():
            raise ValueError("ref_text must be the verbatim transcript of ref_wav")

        if self.cfg.seed is not None:
            mx.random.seed(self.cfg.seed)

        self._warm_a = mx.random.normal((self.cfg.keep_warm_dim,) * 2).astype(mx.float16)
        self._warm_b = mx.random.normal((self.cfg.keep_warm_dim,) * 2).astype(mx.float16)
        mx.eval(self._warm_a, self._warm_b)

        log.info("Qwen3-TTS loaded in %.1fs (ref %.1fs, sr=%d)",
                 time.perf_counter() - t0, ref_s, self._model.sample_rate)
        self._prewarm()
        self._ready.set()

    def _prewarm(self) -> None:
        """One throwaway utterance. Pays the mx.compile of the vocoder, the first
        Metal kernel JIT, and populates Model._icl_cache with (ref_codes, ref_text_ids)
        so the first real turn does not pay the speech-tokenizer encode."""
        t0 = time.perf_counter()
        for _ in self._raw_chunks("Ready.", cancel=threading.Event()):
            pass
        log.info("TTS prewarm %.2fs", time.perf_counter() - t0)

    # ------------------------------------------------------- the actual call
    def _raw_chunks(self, text: str, cancel: threading.Event):
        """THE streaming call. model.generate(...) is a plain Python generator.

        Routing (qwen3_tts.py:1126 generate -> :2204 _generate_icl): tts_model_type
        == "base" AND ref_audio AND ref_text AND speech_tokenizer.has_encoder
        => ICL zero-shot cloning. generate() silently raises the repetition_penalty
        to max(rp, 1.5) on this path; passing 1.05 has no effect.

        Yields GenerationResult (tts/models/base.py:72). In streaming mode:
          .audio                -> mx.array [samples], 1-D, 24 kHz,
                                   dtype follows the un-quantised vocoder weights
                                   (bf16 or f32 -> ALWAYS .astype(mx.float32) before
                                   numpy; numpy has no bfloat16)
          .sample_rate          -> 24000
          .is_streaming_chunk   -> True
          .is_final_chunk       -> True on the last one only
          .samples/.token_count/.real_time_factor -> per-chunk telemetry
        Chunk size: max(1, int(streaming_interval*12.5)) code frames; 0.32 -> 4
        frames -> 4*1920 = 7680 samples = 320 ms at 24 kHz.
        In streaming mode the reference audio is NOT prepended to the vocoder, so
        there is no ref-prefix to trim (unlike the non-streaming path, which
        decodes ref+gen and cuts proportionally).
        """
        gen = self._model.generate(
            text=text,
            ref_audio=self._ref_audio_24k,   # mx.array, 24 kHz mono float32
            ref_text=self.cfg.ref_text,
            lang_code=self.cfg.lang_code,
            temperature=self.cfg.temperature,
            top_k=self.cfg.top_k,
            top_p=self.cfg.top_p,
            max_tokens=self.cfg.max_tokens,
            stream=True,
            streaming_interval=self.cfg.streaming_interval,
            verbose=False,                    # keeps tqdm inert; it is never .close()d on interrupt
        )
        try:
            for result in gen:
                if cancel.is_set():
                    break
                yield result
        finally:
            # GeneratorExit unwinds _generate_icl WITHOUT running its trailing
            # reset_streaming_state()/mx.clear_cache(), so do it by hand.
            gen.close()
            self._reset_decoder()

    def _reset_decoder(self) -> None:
        """Tear-down after an interrupt.

        What is actually left dirty:
          * Qwen3TTSSpeechTokenizerDecoder._transformer_cache (growing KV cache)
          * every CausalConv1d._buffer and DecoderBlockUpsample._overflow
            (speech_tokenizer.py:82 / :658 reset_state())
        What is NOT left dirty: the talker KV cache and the code-predictor cache
        are locals of _generate_icl, freed when the frame unwinds; Model._icl_cache
        is a pure memo. So the model object stays usable with no reset at all --
        _generate_icl calls reset_streaming_state() itself at the top of the next
        streaming call. We reset eagerly anyway so an interrupt cannot leak KV
        growth into a non-streaming decode() and so memory settles between turns.
        No MLX work stays queued: the loop does mx.eval(input_embeds, is_eos) and
        is_eos.item() every step, so the graph is fully materialised at each yield.
        """
        try:
            st = getattr(self._model, "speech_tokenizer", None)
            if st is not None:
                st.decoder.reset_streaming_state()   # forwards through mlx.gc_func
        except Exception as e:
            log.warning("decoder reset failed: %s", e)

    # ------------------------------------------------------------- keep warm
    def _warm_ping(self) -> None:
        """mlx-lm#432: the Metal GPU drops to a low-power state when idle and the
        next inference runs up to ~7x slower, recurring after EVERY idle gap.

        Cheapest fix: a small matmul on the same (default) stream at a few Hz.
        1024^3 fp16 = 2 GFLOP, ~0.2 ms on an M5 Max -> <0.1% duty cycle at 3 Hz.
        It runs on the TTS worker thread, in the queue-get timeout, so it can
        never contend with a real generation. Do not warm from another thread.
        """
        mx.eval(self._warm_a @ self._warm_b)

    # ----------------------------------------------------------- worker loop
    def _worker(self) -> None:
        try:
            self._load()
        except Exception:
            log.exception("TTS load failed")
            self._ready.set()
            return
        while not self._closed:
            try:
                job = self._jobs.get(timeout=self.cfg.keep_warm_interval)
            except queue.Empty:
                self._warm_ping()
                continue
            if job is _STOP:
                break
            self._run_job(job)

    def _run_job(self, job) -> None:
        text, sink, cancel, gen_id = job
        rs = _Resampler24to16()
        carry = b""
        first = True
        t0 = time.perf_counter()
        n_out = 0
        try:
            for result in self._raw_chunks(text, cancel):
                # bf16 -> f32 BEFORE numpy; np.array(bfloat16 mx.array) raises.
                x24 = np.asarray(result.audio.astype(mx.float32), dtype=np.float32)
                y16 = rs.push(x24)
                if y16.size == 0:
                    continue
                if first:
                    k = min(FADE_IN_SAMPLES, y16.size)
                    y16 = y16.copy()
                    y16[:k] *= np.linspace(0.0, 1.0, k, dtype=np.float32)
                    sink(("ttfa", time.perf_counter() - t0))
                    first = False
                buf = carry + _f32_to_pcm16(y16)
                cut = (len(buf) // FRAME_BYTES) * FRAME_BYTES
                for i in range(0, cut, FRAME_BYTES):
                    sink(("pcm", buf[i:i + FRAME_BYTES]))
                    n_out += FRAME_SAMPLES
                carry = buf[cut:]
                if cancel.is_set():
                    break
            if not cancel.is_set():
                tail = rs.drain()
                buf = carry + (_f32_to_pcm16(tail) if tail.size else b"")
                if buf:
                    buf = buf + b"\x00" * ((-len(buf)) % FRAME_BYTES)
                    for i in range(0, len(buf), FRAME_BYTES):
                        sink(("pcm", buf[i:i + FRAME_BYTES]))
                        n_out += FRAME_SAMPLES
        except Exception:
            log.exception("TTS job failed")
            sink(("error", None))
        finally:
            dt = time.perf_counter() - t0
            dur = n_out / PIPE_SR
            log.info("tts gen=%d %.2fs audio in %.2fs (RTF %.2f)%s",
                     gen_id, dur, dt, dt / max(dur, 1e-6),
                     " [INTERRUPTED]" if cancel.is_set() else "")
            with self._lock:
                self._live.discard(cancel)
            sink(("end", bool(cancel.is_set())))

    # --------------------------------------------------------- public API
    def _submit(self, text: str, sink):
        cancel = threading.Event()
        with self._lock:
            self._gen_id += 1
            gen_id = self._gen_id
            self._live.add(cancel)
        self._jobs.put((text, sink, cancel, gen_id))
        return gen_id, cancel

    def interrupt(self) -> None:
        """Barge-in. Idempotent, safe from any thread, returns immediately.

        Latency budget: this only stops PRODUCTION. The in-flight loop iteration
        finishes first (one talker step + 16 code-predictor steps + possibly one
        vocoder chunk, ~40-190 ms). What the user actually still hears is whatever
        the browser has already buffered, so the orchestrator MUST also send a
        control frame ({"type":"flush"}) and the client MUST drop its queue with a
        ~10 ms ramp. Keep client-side buffering under ~200 ms or the 300 ms target
        is unreachable no matter how fast this returns.
        """
        with self._lock:
            live = list(self._live)      # in-flight AND queued, of every turn
        for ev in live:
            ev.set()
        while True:                      # drop queued-but-unstarted utterances
            try:
                job = self._jobs.get_nowait()
            except queue.Empty:
                break
            if job is _STOP:
                self._jobs.put(_STOP)
                break
            job[2].set()
            with self._lock:
                self._live.discard(job[2])
            job[1](("end", True))

    def stream(self, text: str) -> Iterator[bytes]:
        """Synchronous generator: yields 16 kHz mono int16-LE PCM, 640-byte frames."""
        out: "queue.Queue" = queue.Queue()
        self._submit(text, out.put)
        while True:
            kind, payload = out.get()
            if kind == "pcm":
                yield payload
            elif kind == "ttfa":
                log.info("TTFA %.3fs", payload)
            elif kind in ("end", "error"):
                return

    async def astream(self, text: str) -> AsyncIterator[bytes]:
        """Asyncio facade for the websocket gateway. Same frames."""
        import asyncio

        loop = asyncio.get_running_loop()
        aq: "asyncio.Queue" = asyncio.Queue()
        self._submit(text, lambda item: loop.call_soon_threadsafe(aq.put_nowait, item))
        while True:
            kind, payload = await aq.get()
            if kind == "pcm":
                yield payload
            elif kind == "ttfa":
                log.info("TTFA %.3fs", payload)
            elif kind in ("end", "error"):
                return

    def close(self) -> None:
        self._closed = True
        self.interrupt()
        self._jobs.put(_STOP)
        self._thread.join(timeout=5)


# --------------------------------------------------------------------------
# Sentence batching for the orchestrator.
#
# There IS a fixed per-call cost and it cannot be cached away: the ICL prefill
# is [role | codec_prefix | ref_text_tokens + TARGET_text_tokens + eos | codec_bos
# + ref_codes]  (qwen3_tts.py:606 _prepare_icl_generation_inputs, step 6). The
# target text sits BEFORE the reference codec block, so no prefix KV cache can
# ever be shared across sentences -- unlike the LLM, TTS re-prefills every call.
# For a 6 s reference that is ~75 code positions + ~20 ref-text tokens + target
# tokens, re-run per sentence, plus an ECAPA speaker-encoder forward that is NOT
# memoised (only ref_codes and ref_text_ids are).
#
# Amortisation, with the measured numbers: wall = TTFA + RTF * audio_seconds.
#   0.3 s of speech -> 0.20 + 0.14 = 0.34 s  -> net 1.12x, FALLS BEHIND realtime
#   0.5 s           -> 0.20 + 0.23 = 0.43 s  -> 0.85x, no headroom under load
#   1.0 s           -> 0.20 + 0.45 = 0.65 s  -> 0.65x, safe
#   2.0 s           -> 0.20 + 0.90 = 1.10 s  -> 0.55x, comfortable
# Under concurrent load (TTFA 0.42 / RTF 0.92) the 1 s case is 1.34x -- so the
# floor is ~1 s of speech per call, i.e. roughly 12-15 words.
# --------------------------------------------------------------------------
_SENT_END = ".!?;:"


def sentence_batches(token_stream, min_words: int = 12, first_min_words: int = 4):
    """Consume an LLM token stream, emit text chunks sized for TTS.

    First chunk is deliberately tiny (first clause) to minimise mouth-open
    latency; every later chunk is >= min_words (~1 s of speech) so the fixed
    per-call prefill stays amortised.
    """
    buf = ""
    first = True
    for tok in token_stream:
        buf += tok
        if buf and buf[-1] in _SENT_END and len(buf.split()) >= (
            first_min_words if first else min_words
        ):
            yield buf.strip()
            buf = ""
            first = False
    if buf.strip():
        yield buf.strip()
