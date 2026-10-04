#!/usr/bin/env python
"""从一段长音频里切出音色克隆参考片段，并自动生成匹配的文字稿。

用法:
    .venv/bin/python scripts/make_ref.py <音频文件> [--start 秒] [--len 8]

输出 assets/ck/ref.wav（24kHz 单声道）+ assets/ck/ref.txt（逐字稿）。

为什么每一步都必要（全部源码级验证过，见 voxck/tts.py 注释）：
- **必须 24kHz**：extract_speaker_embedding 假定 24k 但从不校验。喂 16k 进去
  得到的是错误声纹，克隆出来不像人，而且不报任何错。
- **必须在静音处切**：切在词中间，模型会学到一个残缺音素开头的发音习惯。
- **文字稿必须逐字对应**：ICL 靠 (音频, 文字) 配对学映射，对不上就学歪。
  所以这里直接用管线自己的 ASR 转写，保证一致。
- **响度归一但不压限**：削波会毁掉高频细节，而辅音的清晰度就在那里。

选材比参数重要得多：坐着录的播客 > 集会喊话。零样本克隆连韵律和音区一起复制，
拿对着人群喊的素材克隆出来，说什么都在吼。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "assets" / "ck"
TTS_SR = 24000


def ffprobe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration:stream=channels,sample_rate,codec_name",
         "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def decode(path: Path, sr: int, start: float | None = None,
           dur: float | None = None) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{start}"]
    cmd += ["-i", str(path)]
    if dur is not None:
        cmd += ["-t", f"{dur}"]
    cmd += ["-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def find_pauses(x: np.ndarray, sr: int, win_ms: int = 25) -> np.ndarray:
    """返回每个窗口的 RMS，用来找静音边界。"""
    w = int(sr * win_ms / 1000)
    n = len(x) // w
    return np.sqrt((x[:n * w].reshape(n, w) ** 2).mean(axis=1) + 1e-12)


def pick_segment(x: np.ndarray, sr: int, want: float) -> tuple[int, int]:
    """挑一段 want 秒、两端都落在静音上、且中间语音连续的片段。"""
    rms = find_pauses(x, sr)
    win = int(sr * 0.025)
    speech = rms > max(np.percentile(rms, 60) * 0.35, 1e-4)
    need = int(want * sr / win)

    best, best_score = None, -1.0
    for s in range(0, len(speech) - need - 1, 4):
        e = s + need
        if speech[s] or speech[e]:          # 两端必须在静音上
            continue
        seg = speech[s:e]
        ratio = seg.mean()                  # 语音占比：太低是空录音，太高是没换气
        if not 0.60 <= ratio <= 0.93:
            continue
        chunk = x[s * win:e * win]
        peak = float(np.abs(chunk).max())
        if peak > 0.99:                     # 削波
            continue
        # 偏好：语音占比适中 + 动态范围大（信息多）+ 不贴顶
        score = (1 - abs(ratio - 0.78)) + float(chunk.std()) * 2 + (0.99 - peak) * 0.3
        if score > best_score:
            best_score, best = score, (s * win, e * win)
    return best or (0, min(len(x), int(want * sr)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("--start", type=float, default=None, help="从第几秒开始找（跳过片头音乐）")
    ap.add_argument("--window", type=float, default=180.0, help="在多长的范围内找（秒）")
    ap.add_argument("--len", dest="length", type=float, default=8.0, help="参考片段秒数")
    a = ap.parse_args()

    if not a.audio.exists():
        print(f"找不到 {a.audio}", file=sys.stderr)
        return 1

    info = ffprobe(a.audio)
    total = float(info["format"]["duration"])
    st = info["streams"][0]
    print(f"源: {a.audio.name}  {total/60:.1f}分钟  {st.get('sample_rate')}Hz "
          f"{st.get('channels')}ch  {st.get('codec_name')}")

    # 默认跳过前 60 秒（片头音乐/口播广告），这是最常见的污染源
    start = a.start if a.start is not None else min(60.0, total * 0.05)
    window = min(a.window, max(0.0, total - start))
    if window < a.length + 2:
        print("音频太短", file=sys.stderr)
        return 1

    print(f"在 {start:.0f}s – {start+window:.0f}s 内挑选 {a.length:.0f}s 片段…")
    x = decode(a.audio, TTS_SR, start, window)
    i0, i1 = pick_segment(x, TTS_SR, a.length)
    seg = x[i0:i1].copy()
    at = start + i0 / TTS_SR
    print(f"选中 {at:.1f}s – {at + len(seg)/TTS_SR:.1f}s")

    # 响度归一到 -23 LUFS 量级的近似（RMS 法），不压限、不削波
    rms = float(np.sqrt((seg ** 2).mean()))
    if rms > 1e-6:
        seg = seg * min(0.1 / rms, 0.99 / max(float(np.abs(seg).max()), 1e-6))
    peak = float(np.abs(seg).max())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import soundfile as sf
    wav_path = OUT_DIR / "ref.wav"
    sf.write(wav_path, seg, TTS_SR, subtype="PCM_16")   # 24kHz，绝不能是 16k

    # 用管线自己的 ASR 转写，保证 ref_text 与 ref_wav 逐字对应
    print("转写…")
    from voxck.asr import Asr
    seg16 = decode(a.audio, 16000, at, len(seg) / TTS_SR)
    text = Asr().transcribe(seg16).strip()
    (OUT_DIR / "ref.txt").write_text(text + "\n", encoding="utf-8")

    print()
    print(f"  ref.wav  {wav_path}  ({len(seg)/TTS_SR:.1f}s, {TTS_SR}Hz, peak {peak:.2f})")
    print(f"  ref.txt  {text}")
    print()
    warn = []
    if len(text.split()) < 12:
        warn.append("词数偏少——片段可能含大段静音")
    if peak > 0.97:
        warn.append("接近削波")
    if len(seg) / TTS_SR < 4:
        warn.append("短于 4 秒，克隆质量会下降")
    for w in warn:
        print(f"  ⚠ {w}")
    print("\n下一步：听一遍 ref.wav 确认是单人、正常音量、无背景音乐，然后重启 run.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
