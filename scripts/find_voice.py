#!/usr/bin/env python
"""按声纹在长音频里找出目标说话人的片段。

为什么需要：播客常有嘉宾、共同主持、广告口播。按能量/语音占比挑段会挑到别人身上，
克隆出来就是错的人 —— 而且你听之前完全不知道。

做法：用一段**确定是目标本人**的音频当锚点（例如自述身世的片段），
算 ECAPA-TDNN 声纹嵌入，再在目标音频上滑窗比对余弦相似度。
用的是 TTS 模型自带的 speaker encoder，不额外引入模型。

用法:
    .venv/bin/python scripts/find_voice.py --anchor a.mp3 --anchor-at 200 \\
        --target b.mp3 --top 5
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
SR = 24000            # extract_speaker_embedding 假定 24k 且不校验


def decode(path: str, at: float, dur: float) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{at}", "-i", path, "-t", f"{dur}",
         "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def duration(path: str) -> float:
    return float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True, check=True).stdout.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", required=True, help="确定是目标本人的音频")
    ap.add_argument("--anchor-at", type=float, required=True, help="锚点起始秒")
    ap.add_argument("--anchor-len", type=float, default=12.0)
    ap.add_argument("--target", required=True, help="要扫描的音频")
    ap.add_argument("--win", type=float, default=8.0, help="候选片段长度")
    ap.add_argument("--hop", type=float, default=6.0, help="滑窗步长")
    ap.add_argument("--skip", type=float, default=45.0, help="跳过开头多少秒（广告/片头）")
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args()

    from mlx_audio.tts.utils import load as load_tts
    print("加载 speaker encoder…")
    model = load_tts("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit")

    def embed(x: np.ndarray) -> np.ndarray:
        e = model.extract_speaker_embedding(mx.array(x))
        v = np.asarray(e.astype(mx.float32)).reshape(-1)
        return v / (np.linalg.norm(v) + 1e-9)

    anchor = embed(decode(a.anchor, a.anchor_at, a.anchor_len))
    print(f"锚点: {Path(a.anchor).name} @ {a.anchor_at:.0f}s ({a.anchor_len:.0f}s)")

    total = duration(a.target)
    print(f"扫描: {Path(a.target).name}  {total/60:.0f}分钟  "
          f"步长{a.hop:.0f}s → 约 {int((total-a.skip)/a.hop)} 个候选\n")

    rows = []
    at = a.skip
    while at + a.win < total:
        x = decode(a.target, at, a.win)
        if x.size < int(SR * a.win * 0.9):
            break
        peak = float(np.abs(x).max())
        rms = float(np.sqrt((x ** 2).mean()))
        # 先用便宜的门槛筛掉静音段和削波段，省下声纹计算
        if rms > 0.01 and peak < 0.99:
            sim = float(anchor @ embed(x))
            rows.append((sim, at, rms, peak))
        at += a.hop
        if len(rows) % 40 == 0 and rows:
            print(f"  …{at/60:.0f}分钟", end="\r", flush=True)

    rows.sort(reverse=True)
    print(" " * 30)
    print(f"{'相似度':>7}  {'位置':>9}  {'RMS':>6}  {'峰值':>5}")
    for sim, at, rms, peak in rows[:a.top]:
        print(f"{sim:7.3f}  {int(at)//60:02d}:{int(at)%60:02d}  "
              f"{rms:6.3f}  {peak:5.2f}")
    if rows:
        lo = np.percentile([r[0] for r in rows], 10)
        hi = rows[0][0]
        print(f"\n相似度范围 {lo:.3f} – {hi:.3f}"
              f"（跨度 {hi-lo:.3f}；跨度小说明整集只有一个说话人）")
        print(f"\n取最佳片段做参考音:")
        print(f"  .venv/bin/python scripts/make_ref.py {a.target} "
              f"--start {rows[0][1]:.0f} --window {a.win + 1:.0f} --len {a.win:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
