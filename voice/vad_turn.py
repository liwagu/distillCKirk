"""Two-stage endpointing: Silero VAD v5 + SmartTurn v3, both ONNX on CPU.

Feed 16 kHz mono int16 PCM; get three events: speech_start / speech_end / turn_complete.

Division of labour
------------------
Stage 1  Silero v5 (~0.14 ms per 32 ms frame) answers only "is someone talking right now".
Stage 2  SmartTurn v3 (~18 ms, 8 s acoustic window) answers "has this person finished the
         sentence". It is an ACOUSTIC model, not a text model: prosody, trailing
         intonation and filled pauses ("uh", "um") are the signal, so the trailing
         silence must be inside the window - the model needs to hear the pause.

Silence is not the same as "done". A naive pipeline has only stage 1, so it must wait
600-800 ms of silence before it dares commit. This one asks stage 2 at 160 ms of silence;
if the answer is "finished", it commits immediately:

    end-of-turn latency  ~=  160 ms (hangover) + ~20 ms (mel + inference)  ~=  180-200 ms

If the answer is "not finished" it keeps waiting and re-asks at 400 / 700 / 1100 ms with a
progressively lower bar, with a 2000 ms hard commit as backstop. Cost is 1-4 extra ~20 ms
CPU inferences per turn, which buys 400-600 ms of conversational latency.

Thread budget (hard constraint: ANE/CoreML p99 degrades ~7x when every core is saturated,
so this component must not hog the CPU):
  - Silero runs inline on the calling thread, intra_op=1. Duty cycle ~0.4 %.
  - SmartTurn runs on a dedicated single-worker executor, intra_op=2 by default.
  - The state machine is only ever mutated on the event-loop thread; probe results are
    posted back with loop.call_soon_threadsafe. No locks.

VERIFIED CONTRACTS (2026-09-21). Everything below was read out of source or decoded out of
the actual ONNX protobuf, not recalled:

  Silero v5 - snakers4/silero-vad src/silero_vad/data/silero_vad.onnx, 2,327,524 bytes.
    runanywhere/silero-vad-v5 ships a BYTE-IDENTICAL file (same size; sha256 of the last
    16 KiB matches: ab840758d343869c2f14d069c533326da32555969975acdc6f9d27975d3543fb).
    Decoded graph IO (opset 16, graph name "spox_graph"):
      input   float32  [dynamic, dynamic]     <- 64-sample context ++ 512-sample frame = 576
      state   float32  [2, dynamic, 128]      <- LSTM h/c, threaded between calls
      sr      int64    scalar (rank 0!)       <- not [1]
      output  float32  [dynamic, 1]           <- P(speech) for this frame
      stateN  float32  [dynamic, dynamic, dynamic]  -> feed back as `state`
    NOTE both `input` dims are DYNAMIC. Feeding a bare 512-sample frame without the 64
    sample context does NOT raise - it silently returns a plausible but wrong probability.
    That is the trap; see SileroVadOnnx.__call__.

  SmartTurn v3 - pipecat-ai/smart-turn-v3, smart-turn-v3.2-cpu.onnx, 8,679,182 bytes (int8).
    Decoded graph IO (opset 18):
      input_features float32 [batch("s6"), 80, 800]
      logits         float32 [batch, 1]
    The output is NAMED "logits" but the graph contains exactly one Sigmoid node, applied at
    the classifier head, so the value is ALREADY a probability in [0, 1]. Do not apply a
    second sigmoid. Threshold 0.5 => 1 = complete. (pipecat inference.py reads
    outputs[0][0].item() directly and compares to 0.5.)
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# --- Silero v5 contract (do not change; changing these means feeding the wrong model) ---
SR = 16000
VAD_FRAME = 512           # the only legal frame length at 16 kHz (256 at 8 kHz)
VAD_CONTEXT = 64          # v5 addition: the previous frame's last 64 samples are prepended
VAD_WINDOW = VAD_FRAME + VAD_CONTEXT      # 576 - the real length of ONNX input 'input'
FRAME_MS = VAD_FRAME * 1000.0 / SR        # 32.0
VAD_STATE_SHAPE = (2, 1, 128)

# --- SmartTurn v3 contract ---
TURN_SECONDS = 8
TURN_SAMPLES = TURN_SECONDS * SR          # 128000
N_FFT, HOP, N_MELS = 400, 160, 80
MEL_FRAMES = TURN_SAMPLES // HOP          # 800 (rfft yields 801 frames; the last is dropped)
NORM_EPS = 1e-7
MEL_FLOOR = 1e-10


# =============================================================================
# Whisper-style log-mel, numpy only.
#
# Bit-compatible with
#   transformers.WhisperFeatureExtractor(chunk_length=8)(
#       audio, sampling_rate=16000, padding="max_length", max_length=128000,
#       truncation=True, do_normalize=True, return_tensors="np"
#   ).input_features.squeeze(0)
# which is exactly what pipecat feeds smart-turn. Mirrors pipecat's vendored
# _whisper_features.compute_whisper_log_mel_features.
# =============================================================================

def _hz_to_mel_slaney(freq: np.ndarray) -> np.ndarray:
    freq = np.atleast_1d(np.asarray(freq, dtype=np.float64))
    mels = 3.0 * freq / 200.0
    log_region = freq >= 1000.0
    mels[log_region] = 15.0 + np.log(freq[log_region] / 1000.0) * (27.0 / np.log(6.4))
    return mels


def _mel_to_hz_slaney(mels: np.ndarray) -> np.ndarray:
    mels = np.atleast_1d(np.asarray(mels, dtype=np.float64))
    freq = 200.0 * mels / 3.0
    log_region = mels >= 15.0
    freq[log_region] = 1000.0 * np.exp((np.log(6.4) / 27.0) * (mels[log_region] - 15.0))
    return freq


def _mel_filterbank() -> np.ndarray:
    """(201, 80) triangular filterbank, Slaney frequency axis + Slaney area normalization.

    Equivalent to transformers.audio_utils.mel_filter_bank(
        num_frequency_bins=201, num_mel_filters=80, min_frequency=0.0,
        max_frequency=8000.0, sampling_rate=16000, norm="slaney", mel_scale="slaney").
    The HTK variant (mel_scale="htk") is numerically quite different and will shift the
    output probability across the board. Do not mix them.
    """
    n_bins = N_FFT // 2 + 1                                    # 201
    mel_min = float(_hz_to_mel_slaney(np.array([0.0]))[0])
    mel_max = float(_hz_to_mel_slaney(np.array([SR / 2.0]))[0])
    filter_freqs = _mel_to_hz_slaney(np.linspace(mel_min, mel_max, N_MELS + 2))
    fft_freqs = np.linspace(0, SR // 2, n_bins)

    diff = np.diff(filter_freqs)
    slopes = filter_freqs[None, :] - fft_freqs[:, None]        # (201, 82)
    down = -slopes[:, :-2] / diff[:-1]
    up = slopes[:, 2:] / diff[1:]
    fb = np.maximum(0.0, np.minimum(down, up))
    enorm = 2.0 / (filter_freqs[2:N_MELS + 2] - filter_freqs[:N_MELS])
    return fb * enorm[None, :]


_MEL_FB = _mel_filterbank()                    # (201, 80)
_HANN = np.hanning(N_FFT + 1)[:-1]             # periodic Hann == torch.hann_window(400)


def whisper_log_mel(audio: np.ndarray) -> np.ndarray:
    """8 s of float32 waveform -> (80, 800) float32 log-mel.

    `audio` must already be exactly TURN_SAMPLES samples with the real speech RIGHT-aligned
    (zeros padded at the FRONT). That is what pipecat's truncate_audio_to_last_n_seconds
    does - np.pad(audio, (padding, 0)) - and it is the distribution the model was trained
    and benchmarked on.

    Normalization happens BEFORE the spectrogram and over the WHOLE 8 s buffer including the
    leading zeros. This looks wrong and is right: because the caller pre-pads to exactly
    128000 samples, HF's pad() adds nothing, so the attention mask is all ones and
    zero_mean_unit_var_norm reduces to (x - x.mean()) / sqrt(x.var() + 1e-7) over everything.
    Normalizing only the voiced part shifts every probability.
    """
    x = np.asarray(audio, dtype=np.float32)
    if x.shape != (TURN_SAMPLES,):
        raise ValueError(f"expected ({TURN_SAMPLES},) float32, got {x.shape}")
    # float32 here on purpose: the reference normalizes the float32 buffer, and only the
    # spectrogram promotes to float64.
    x = (x - x.mean()) / np.sqrt(x.var() + NORM_EPS)

    pad = N_FFT // 2                                            # center=True reflect padding
    padded = np.pad(x.astype(np.float64), (pad, pad), mode="reflect")
    frames = sliding_window_view(padded, N_FFT)[::HOP]          # (801, 400)
    spec = np.fft.rfft(frames * _HANN, axis=-1)                 # (801, 201)
    power = (np.abs(spec) ** 2).T                               # (201, 801)

    mel = np.maximum(MEL_FLOOR, _MEL_FB.T @ power)              # (80, 801)
    log_spec = np.log10(mel)[:, :-1]                            # drop last frame -> (80, 800)
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).astype(np.float32)


# =============================================================================
# Events / configuration
# =============================================================================

@dataclass
class TurnEvent:
    """type in {speech_start, speech_end, turn_complete}.

    speech_start   provisional=True means "suspected barge-in while the assistant is
                   speaking" - duck the volume, do not stop yet. A second speech_start with
                   provisional=False is the real interrupt.
                   resumed=True means the user spoke again during hangover (same turn, not
                   a new one).
    speech_end     spurious=True means the segment was shorter than min_speech_ms or the
                   barge-in was never confirmed (cough / keyboard / "mm-hmm"). The caller
                   should resume playback and discard the turn.
    turn_complete  pcm is the whole segment for the ASR (preroll + speech + trailing
                   silence). reason in {semantic, timeout, max_duration}.
    """
    type: str
    ts: float                     # time.monotonic()
    stream_ms: float              # position in the fed audio stream
    probability: float = 0.0      # speech_*: VAD probability. turn_complete: SmartTurn prob.
    provisional: bool = False
    resumed: bool = False
    spurious: bool = False
    reason: str = ""
    speech_ms: float = 0.0
    silence_ms: float = 0.0
    pcm: Optional[bytes] = None


@dataclass
class VadTurnConfig:
    # --- stage 1: Silero hysteresis ---
    speech_threshold: float = 0.55        # onset threshold
    release_threshold: float = 0.35       # sustain threshold (onset - 0.15, the usual rule)
    start_frames: int = 2                 # 2 consecutive frames = 64 ms before we believe it
    # --- energy gate (Silero occasionally fires on keyboard clacks / desk knocks) ---
    gate_db_over_floor: float = 8.0
    abs_floor_dbfs: float = -55.0
    # --- stage 2: endpointing ---
    speech_end_ms: float = 160.0          # silence before speech_end == first probe point
    probe_schedule: Sequence[tuple] = ((160.0, 0.60), (400.0, 0.50),
                                       (700.0, 0.40), (1100.0, 0.30))
    hard_commit_ms: float = 2000.0        # backstop when SmartTurn keeps saying "not done"
    min_speech_ms: float = 160.0          # shorter than this is noise, not a turn
    preroll_ms: float = 500.0             # extra audio before onset (pipecat PRE_SPEECH_MS)
    max_utterance_ms: float = 30000.0     # hard cut so buffers stay bounded
    # --- barge-in ---
    bargein_threshold: float = 0.70       # stricter onset while the assistant talks
    bargein_frames: int = 2               # 2 frames = 64 ms to raise suspicion
    bargein_confirm_ms: float = 250.0     # sustained human voice before a real interrupt
    aec_settle_ms: float = 120.0          # AEC reconverge window right after playback starts
    # --- threads ---
    vad_threads: int = 1
    turn_threads: int = 2                 # 4 threads measured 18.0 ms; 2 is kinder to cores
    # --- buffering ---
    history_seconds: float = 34.0


# =============================================================================
# Ring buffer, addressed by absolute sample index
# =============================================================================

class _Ring:
    """Fixed-capacity int16 ring addressed on an absolute sample timeline.

    Invariant: the sample with absolute index `i` lives at `_buf[i % cap]`, for every
    i in [total - cap, total). Every read/write preserves that, including the
    oversized-write path - which is why that path advances `total` by the dropped
    count BEFORE writing instead of blitting to _buf[0] (that would desynchronise
    the modulo mapping unless total happened to be a multiple of cap).
    """

    def __init__(self, capacity: int):
        self._buf = np.zeros(capacity, dtype=np.int16)
        self._cap = capacity
        self.total = 0                       # total samples ever written

    def write(self, x: np.ndarray) -> None:
        n = int(x.size)
        if n <= 0:
            return
        if n > self._cap:                    # keep only the tail, but keep the clock honest
            drop = n - self._cap
            self.total += drop
            x = x[drop:]
            n = self._cap
        w = self.total % self._cap
        end = w + n
        if end <= self._cap:
            self._buf[w:end] = x
        else:
            k = self._cap - w
            self._buf[w:] = x[:k]
            self._buf[:n - k] = x[k:]
        self.total += n

    def read(self, start: int, end: int) -> np.ndarray:
        """Absolute range [start, end); anything already overwritten is clamped away."""
        start = max(int(start), self.total - self._cap, 0)
        end = min(int(end), self.total)
        if end <= start:
            return np.zeros(0, dtype=np.int16)
        s = start % self._cap
        n = end - start
        if s + n <= self._cap:
            return self._buf[s:s + n].copy()
        k = self._cap - s
        return np.concatenate([self._buf[s:], self._buf[:n - k]])


# =============================================================================
# The two ONNX wrappers
# =============================================================================

def _session(model_path: str, intra_threads: int):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.inter_op_num_threads = 1
    so.intra_op_num_threads = max(1, int(intra_threads))
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_path, sess_options=so,
                                providers=["CPUExecutionProvider"])


class SileroVadOnnx:
    """Streaming v5 wrapper. Transcribed from snakers4/silero-vad utils_vad.py OnnxWrapper.

        input  'input'  float32 [1, 576]  <- 64 samples of context ++ 512 new samples
               'state'  float32 [2, 1, 128]  (LSTM h/c, you must thread it yourself)
               'sr'     int64   rank-0 scalar (NOT [1])
        output 'output' float32 [1, 1] = P(this frame is speech)
               'stateN' float32 -> next call's 'state'

    v4 used 1536-sample frames and two separate (2,1,64) h/c tensors. Completely
    incompatible; a v4 file will fail the input-name check below.
    """

    def __init__(self, model_path: str, intra_threads: int = 1):
        self.session = _session(model_path, intra_threads)

        names = {i.name for i in self.session.get_inputs()}
        if names != {"input", "state", "sr"}:
            raise RuntimeError(
                f"{model_path} is not a Silero v5 graph (inputs {sorted(names)}, expected "
                "input/state/sr). v4 and other exports fail here - do not continue.")
        if len(self.session.get_outputs()) != 2:
            raise RuntimeError(
                f"{model_path}: expected 2 outputs (output, stateN), got "
                f"{[o.name for o in self.session.get_outputs()]}")
        self._sr = np.array(SR, dtype=np.int64)      # rank-0
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros(VAD_STATE_SHAPE, dtype=np.float32)
        self._ctx = np.zeros(VAD_CONTEXT, dtype=np.float32)

    def __call__(self, frame_f32: np.ndarray) -> float:
        """frame_f32: (512,) float32 in [-1, 1]. Returns P(speech).

        The 64-sample context prepend is mandatory. Both `input` dims are dynamic in the
        graph, so passing a bare 512-sample frame does not raise - it just returns a
        subtly wrong number forever. This is the single easiest way to get a v5
        integration that "works" and endpoints badly.
        """
        if frame_f32.shape != (VAD_FRAME,):
            raise ValueError(f"expected ({VAD_FRAME},), got {frame_f32.shape}")
        window = np.concatenate([self._ctx, frame_f32])[None, :]      # (1, 576)
        out, state = self.session.run(
            None, {"input": window, "state": self._state, "sr": self._sr})
        self._state = state
        self._ctx = window[0, -VAD_CONTEXT:]                          # tail of THIS frame
        return float(np.asarray(out).reshape(-1)[0])


class SmartTurnOnnx:
    """v3 wrapper.

        input  'input_features' float32 [1, 80, 800]  = Whisper log-mel of 8 s @ 16 kHz
        output 'logits'         float32 [1, 1]

    Despite the name, the graph ends in a Sigmoid, so the value is already
    P(the speaker has FINISHED). Do not sigmoid it again.
    """

    def __init__(self, model_path: str, intra_threads: int = 2):
        self.session = _session(model_path, intra_threads)
        names = [i.name for i in self.session.get_inputs()]
        if names != ["input_features"]:
            raise RuntimeError(
                f"{model_path}: input names {names}, expected ['input_features'] "
                "(smart-turn-v3.x).")

    def __call__(self, audio_8s_f32: np.ndarray) -> float:
        feats = whisper_log_mel(audio_8s_f32)[None, :, :]             # (1, 80, 800)
        out = self.session.run(None, {"input_features": feats})[0]
        return float(np.asarray(out).reshape(-1)[0])


# =============================================================================
# State machine
# =============================================================================

_IDLE, _SPEECH, _HANGOVER = 0, 1, 2


class VadTurnDetector:
    """Two-stage endpoint detector.

    Usage (asyncio, recommended):
        det = VadTurnDetector(vad_path, turn_path, loop=loop, on_event=handler)
        ...                                  # per websocket frame:
        det.feed(pcm_bytes)
        det.set_assistant_speaking(True)     # when TTS actually starts playing
        det.close()

    on_event is called synchronously on the event-loop thread. Do not await heavy work
    inside it - enqueue and return.

    speech_start / speech_end are produced inline inside feed() (Silero is sub-millisecond).
    turn_complete usually arrives ~20 ms later, posted back from the SmartTurn worker.

    States:
        IDLE     --start_frames consecutive voiced frames-->  SPEECH    [speech_start]
        SPEECH   --silence >= speech_end_ms-->                HANGOVER  [speech_end]
        HANGOVER --voice again-->                             SPEECH    [speech_start resumed]
        HANGOVER --probe says "finished" / backstop-->        IDLE      [turn_complete]
    """

    def __init__(
        self,
        vad_model_path: str,
        turn_model_path: str,
        *,
        config: Optional[VadTurnConfig] = None,
        on_event: Optional[Callable[[TurnEvent], None]] = None,
        loop=None,
        inline_probe: bool = False,
    ):
        self.cfg = config or VadTurnConfig()
        self._on_event = on_event
        self._loop = loop
        self._inline = inline_probe or loop is None

        self.vad = SileroVadOnnx(vad_model_path, self.cfg.vad_threads)
        self.turn = SmartTurnOnnx(turn_model_path, self.cfg.turn_threads)

        self._ring = _Ring(int(self.cfg.history_seconds * SR))
        self._pending = np.zeros(0, dtype=np.int16)   # remainder shorter than one frame
        self._tail = b""                              # remainder shorter than one int16

        # Noise floor, learned separately for playing / not playing. While the assistant
        # talks, echo residue raises the floor - which is exactly what we want, since the
        # barge-in gate rises with it. Switching back on stop means we are not deaf for
        # seconds after a loud reply.
        self._floor_idle = -70.0
        self._floor_play = -70.0

        self._state = _IDLE
        self._voiced_run = 0
        self._voiced_ms = 0.0
        self._silence_ms = 0.0
        self._utt_start = 0               # absolute sample index
        self._provisional = False
        self._confirmed = False
        self._probes_done: set = set()
        self._gen = 0                     # probe generation; bump to invalidate in-flight
        self._probe_inflight = False

        self._assistant_speaking = False
        self._playback_started = 0.0

        self._exec: Optional[ThreadPoolExecutor] = None
        if not self._inline:
            self._exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix="smartturn")

        self._inline_events: list = []

    # --- public -------------------------------------------------------------

    @property
    def stream_ms(self) -> float:
        return self._ring.total * 1000.0 / SR

    def set_assistant_speaking(self, speaking: bool) -> None:
        """True when the first TTS packet actually starts playing; False when it finishes
        or is interrupted.

        The rising edge opens an aec_settle_ms window: the browser's AEC adaptive filter
        has to reconverge when the far-end signal changes abruptly, and it always leaks
        during roughly the first 100 ms. No barge-in decisions inside that window.
        """
        if speaking and not self._assistant_speaking:
            self._playback_started = time.monotonic()
        self._assistant_speaking = speaking

    def feed(self, pcm) -> list:
        """Feed 16 kHz mono int16 PCM (bytes or np.int16). Returns events raised inline."""
        self._inline_events = []

        if isinstance(pcm, (bytes, bytearray, memoryview)):
            raw = self._tail + bytes(pcm)
            n = len(raw) - (len(raw) % 2)
            self._tail = raw[n:]
            samples = np.frombuffer(raw[:n], dtype=np.int16)
        else:
            samples = np.asarray(pcm, dtype=np.int16).reshape(-1)

        if samples.size:
            self._ring.write(samples)
            self._pending = (np.concatenate([self._pending, samples])
                             if self._pending.size else samples)

        nframes = self._pending.size // VAD_FRAME
        for i in range(nframes):
            self._process_frame(self._pending[i * VAD_FRAME:(i + 1) * VAD_FRAME])
        if nframes:
            self._pending = self._pending[nframes * VAD_FRAME:].copy()

        return self._inline_events

    def reset(self) -> None:
        """Back to IDLE, invalidating in-flight probes, and clear Silero's LSTM state.

        Call this only on real session boundaries (new speaker / reconnect). Do NOT copy
        the "reset_states() every 5 seconds" habit: v5's state is fixed-size (2,1,128) so
        it cannot grow, and clearing it mid-sentence throws away context and makes the
        next couple of frames' probabilities collapse.
        """
        self._gen += 1
        self._state = _IDLE
        self._voiced_run = 0
        self._voiced_ms = 0.0
        self._silence_ms = 0.0
        self._provisional = self._confirmed = False
        self._probes_done.clear()
        self._probe_inflight = False
        self.vad.reset()

    def close(self) -> None:
        if self._exec is not None:
            self._exec.shutdown(wait=False)
            self._exec = None

    # --- internals ----------------------------------------------------------

    def _emit(self, ev: TurnEvent) -> None:
        self._inline_events.append(ev)
        if self._on_event is not None:
            self._on_event(ev)

    def _process_frame(self, frame_i16: np.ndarray) -> None:
        cfg = self.cfg
        f32 = frame_i16.astype(np.float32) / 32768.0
        prob = self.vad(f32)

        rms = float(np.sqrt(np.mean(f32 * f32)) + 1e-12)
        db = 20.0 * math.log10(rms)
        floor = self._floor_play if self._assistant_speaking else self._floor_idle
        energy_ok = db >= max(cfg.abs_floor_dbfs, floor + cfg.gate_db_over_floor)

        in_utt = self._state in (_SPEECH, _HANGOVER)
        if in_utt:
            thr = cfg.release_threshold
        elif self._assistant_speaking:
            thr = cfg.bargein_threshold
        else:
            thr = cfg.speech_threshold
        voiced = prob >= thr and (energy_ok or in_utt)

        if not voiced:      # learn the floor from non-speech frames only, slowly
            if self._assistant_speaking:
                self._floor_play += 0.02 * (db - self._floor_play)
            else:
                self._floor_idle += 0.02 * (db - self._floor_idle)

        if self._state == _IDLE:
            self._idle_frame(voiced, prob)
        else:
            self._utt_frame(voiced, prob)

    def _idle_frame(self, voiced: bool, prob: float) -> None:
        cfg = self.cfg
        if not voiced:
            self._voiced_run = 0
            return

        if self._assistant_speaking:
            # Inside the AEC reconvergence window everything that leaks through is our
            # own voice, by assumption. Do not even start suspecting.
            if (time.monotonic() - self._playback_started) * 1000.0 < cfg.aec_settle_ms:
                self._voiced_run = 0
                return
            need = cfg.bargein_frames
        else:
            need = cfg.start_frames

        self._voiced_run += 1
        if self._voiced_run < need:
            return

        self._state = _SPEECH
        self._voiced_ms = self._voiced_run * FRAME_MS
        self._silence_ms = 0.0
        self._voiced_run = 0
        self._probes_done.clear()
        self._gen += 1
        # Rewind the onset by `need` frames: that is how far the decision itself lagged.
        self._utt_start = self._ring.total - int(need * VAD_FRAME)
        self._provisional = self._assistant_speaking     # assistant talking -> duck first
        self._confirmed = not self._provisional
        self._emit(TurnEvent("speech_start", time.monotonic(), self.stream_ms,
                             probability=prob, provisional=self._provisional))

    def _utt_frame(self, voiced: bool, prob: float) -> None:
        cfg = self.cfg
        if voiced:
            self._voiced_ms += FRAME_MS
            self._silence_ms = 0.0
            if self._state == _HANGOVER:
                # Spoke again during hangover: same turn continuing. Kill pending probes.
                self._state = _SPEECH
                self._probes_done.clear()
                self._gen += 1
                self._emit(TurnEvent("speech_start", time.monotonic(), self.stream_ms,
                                     probability=prob, provisional=self._provisional,
                                     resumed=True))
            # Barge-in confirmation: only a sustained bargein_confirm_ms of real voice
            # earns a true interrupt.
            if (self._provisional and not self._confirmed
                    and self._voiced_ms >= cfg.bargein_confirm_ms):
                self._confirmed = True
                self._emit(TurnEvent("speech_start", time.monotonic(), self.stream_ms,
                                     probability=prob, provisional=False))
            if (self._ring.total - self._utt_start) * 1000.0 / SR >= cfg.max_utterance_ms:
                self._commit(1.0, "max_duration")
            return

        self._silence_ms += FRAME_MS

        if self._state == _SPEECH and self._silence_ms >= cfg.speech_end_ms:
            self._state = _HANGOVER
            # Noise rejection: too short, or a barge-in that never got confirmed.
            spurious = ((self._voiced_ms < cfg.min_speech_ms)
                        or (self._provisional and not self._confirmed))
            self._emit(TurnEvent("speech_end", time.monotonic(), self.stream_ms,
                                 probability=prob, spurious=spurious,
                                 speech_ms=self._voiced_ms, silence_ms=self._silence_ms))
            if spurious:
                self._abandon()
                return

        if self._state != _HANGOVER:
            return

        if self._silence_ms >= cfg.hard_commit_ms:
            self._commit(0.0, "timeout")
            return

        for at_ms, thr in cfg.probe_schedule:
            if self._silence_ms >= at_ms and at_ms not in self._probes_done:
                if self._probe_inflight:
                    break          # retry on a later frame; do NOT burn the probe slot
                self._probes_done.add(at_ms)
                self._launch_probe(thr)
                break

    # --- SmartTurn probe ----------------------------------------------------

    def _segment_for_turn(self) -> np.ndarray:
        """The 8 s SmartTurn window: real audio right-aligned, zeros in front.

        This mirrors pipecat's truncate_audio_to_last_n_seconds + front zero-pad, which is
        the training and benchmark distribution. Taking "the last 8 s of the microphone"
        instead would drag the previous assistant reply or the user's previous sentence
        into the window and the probability drifts.
        """
        start = self._utt_start - int(self.cfg.preroll_ms * SR / 1000.0)
        seg = self._ring.read(start, self._ring.total).astype(np.float32) / 32768.0
        if seg.size >= TURN_SAMPLES:
            return np.ascontiguousarray(seg[-TURN_SAMPLES:])
        out = np.zeros(TURN_SAMPLES, dtype=np.float32)
        if seg.size:
            out[-seg.size:] = seg
        return out

    def _launch_probe(self, threshold: float) -> None:
        audio = self._segment_for_turn()
        gen = self._gen
        if self._inline:
            self._on_probe_done(gen, threshold, self.turn(audio))
            return
        self._probe_inflight = True

        def work():
            try:
                p = self.turn(audio)
            except Exception:
                p = -1.0
            self._loop.call_soon_threadsafe(self._on_probe_done, gen, threshold, p)

        self._exec.submit(work)

    def _on_probe_done(self, gen: int, threshold: float, prob: float) -> None:
        self._probe_inflight = False
        if gen != self._gen or self._state != _HANGOVER:
            return                        # stale (user resumed / already committed)
        if prob >= threshold:
            self._commit(prob, "semantic")

    # --- commit -------------------------------------------------------------

    def _commit(self, prob: float, reason: str) -> None:
        start = self._utt_start - int(self.cfg.preroll_ms * SR / 1000.0)
        pcm = self._ring.read(start, self._ring.total).tobytes()
        ev = TurnEvent("turn_complete", time.monotonic(), self.stream_ms,
                       probability=prob, reason=reason,
                       speech_ms=self._voiced_ms, silence_ms=self._silence_ms, pcm=pcm)
        self._abandon()
        self._emit(ev)

    def _abandon(self) -> None:
        self._gen += 1
        self._state = _IDLE
        self._voiced_run = 0
        self._voiced_ms = 0.0
        self._silence_ms = 0.0
        self._provisional = self._confirmed = False
        self._probes_done.clear()
