"""MuseTalk-MLX 端到端实测 —— 测官方基准漏掉的部分，并在 GPU 争抢下复测。

官方 scripts/bench_realtime.py 只测 run_batched（UNet + VAE decode）。
本脚本把运行时真实路径拆开分别计时，并区分：
  - 可预计算（只依赖底子视频，离线做一次）：VAE encode
  - 运行时每帧必付（依赖实时音频）：Whisper enc / UNet / VAE decode / 贴回

因为本项目的人物身份与底子视频循环是固定的，可预计算部分不计入实时预算。

用法: .venv/bin/python scripts/bench_musetalk.py
"""
import sys, time, threading
import numpy as np
import mlx.core as mx

mx.set_default_device(mx.gpu)
from huggingface_hub import snapshot_download
from musetalk_mlx.pipeline_mlx import MuseTalkPipeline

FPS = 25.0
TARGET_MS = 1000.0 / FPS          # 40 ms/frame budget


def timeit(fn, iters, warmup=3):
    for _ in range(warmup): fn()
    mx.synchronize()
    t = time.perf_counter()
    for _ in range(iters): fn()
    mx.synchronize()
    return (time.perf_counter() - t) / iters * 1000.0    # ms per call


class Contender:
    """模拟本地 LLM 解码占用 GPU（依赖链，防惰性求值消除）。"""
    def __init__(self, n=4096, layers=28):
        self.stop = threading.Event(); self.n=n; self.layers=layers; self.steps=0
    def _run(self):
        a = (mx.random.normal((self.n, self.n)) * 0.01).astype(mx.float16)
        x = (mx.random.normal((self.n, self.n)) * 0.01).astype(mx.float16)
        while not self.stop.is_set():
            for _ in range(self.layers):
                x = (a @ x) * 0.01
            mx.eval(x); self.steps += 1
    def __enter__(self):
        self.t = threading.Thread(target=self._run, daemon=True); self.t.start()
        time.sleep(3.0)   # 让争抢负载稳定
        return self
    def __exit__(self, *a):
        self.stop.set(); self.t.join(timeout=10)


def bench(pipe, label):
    B = 8
    face = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)      # 底子视频的一帧人脸裁剪
    lat  = mx.random.normal((B, 8, 32, 32)).astype(mx.float16)
    aud  = mx.random.normal((B, 50, 384)).astype(mx.float16)
    wav  = mx.random.normal((16000 * 30,)).astype(mx.float32)        # 30s 音频段
    from musetalk_mlx.whisper.log_mel import log_mel_spectrogram

    print(f"\n{'='*78}\n {label}\n{'='*78}")

    # --- 可预计算（不计入实时预算）---
    ms = timeit(lambda: mx.eval(pipe.get_latents_for_unet(face)), 20)
    print(f"  [预计算] VAE encode (masked+ref)   {ms:8.2f} ms/帧   ← 离线做一次，运行时 0")

    # --- 运行时每帧必付 ---
    def whisper_pass():
        mel = log_mel_spectrogram(wav[:16000*30])
        mx.eval(pipe.whisper_encoder(mel))
    ms_w = timeit(whisper_pass, 5, warmup=2)
    per_frame_w = ms_w / (30 * FPS)
    print(f"  [运行时] Whisper enc 30s 音频       {ms_w:8.2f} ms  = {per_frame_w:6.3f} ms/帧（摊薄）")

    for bs in (1, 2, 4, 8, 16):
        lb = mx.repeat(lat[:1], bs, axis=0) if bs > B else lat[:bs]
        ab = mx.repeat(aud[:1], bs, axis=0) if bs > B else aud[:bs]
        ms_core = timeit(lambda: pipe.run_batched(lb, ab, batch_size=bs), 10)
        per = ms_core / bs
        total = per + per_frame_w
        fps = 1000.0 / total
        flag = "OK " if total < TARGET_MS else "MISS"
        print(f"  [运行时] UNet+VAEdec bs={bs:<2}          {ms_core:8.2f} ms/批 = {per:6.2f} ms/帧"
              f" | +whisper = {total:6.2f} ms → {fps:6.1f} fps  [{flag}]")

    print(f"  峰值显存 {mx.get_peak_memory()/1e9:.2f} GB   (25fps 预算 = {TARGET_MS:.0f} ms/帧)")


def main():
    snap = snapshot_download("mlx-community/MuseTalk-1.5-fp16")
    t0 = time.perf_counter()
    pipe = MuseTalkPipeline.from_pretrained_mlx(snap)
    print(f"模型加载 {time.perf_counter()-t0:.1f}s")

    bench(pipe, "条件 A：GPU 空闲")

    mx.reset_peak_memory()
    with Contender() as c:
        bench(pipe, "条件 B：有 LLM 级负载争抢 GPU（真实工作条件）")
        print(f"  争抢线程完成 {c.steps} 步")


if __name__ == "__main__":
    main()
