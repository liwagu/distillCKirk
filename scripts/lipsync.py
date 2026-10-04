#!/usr/bin/env python
"""MuseTalk 端到端：底子视频 + 音频 → 真·口型同步视频。

这不是播放录像。每一帧的嘴部都是按当时那句话现算的：
    音频 → Whisper 编码 ─┐
                         ├→ UNet 单步 → VAE 解码 → 羽化贴回
    人脸裁剪 → VAE 编码 ─┘
头部动作、眨眼、肩膀来自底子视频（所以自然），嘴是生成的。

与上游 build_video.py 的区别：它依赖上游 MuseTalk 仓库的 DWPose + bisenet
人脸分割做融合，需要再拉一整套依赖。这里用已有的 SCRFD 检测 + 羽化矩形遮罩，
边缘略糙但不引入新依赖，足以验证口型是否真的对得上。

用法:
    .venv/bin/python scripts/lipsync.py --video base.mp4 --audio reply.wav --out out.mp4
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ref-feathertalk" / "data_utils"))

CROP = 256
FPS = 25


def face_boxes(video: str, det, expand: float = 0.55):
    """逐帧取人脸框。缺检时沿用上一帧，避免裁剪跳动。"""
    cap = cv2.VideoCapture(video)
    frames, boxes, last = [], [], None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
        H, W = f.shape[:2]
        b = None
        try:
            bb, idx, _ = det.detect(f)
            idx = np.asarray(idx).reshape(-1)
            if idx.size:
                c = np.asarray(bb, float)[idx]
                i = int(np.argmax(c[:, 3]))
                x, y, w, h = c[i, :4]
                cx, cy = x + w / 2, y + h / 2
                s = max(w, h) * (1 + expand)
                b = (cx - s / 2, cy - s / 2, s)
        except Exception:
            pass
        if b is None:
            b = last
        else:
            # 平滑：人脸框逐帧抖动会让贴回区域跟着抖，看起来像脸在呼吸
            if last is not None:
                b = tuple(0.75 * np.array(last) + 0.25 * np.array(b))
            last = b
        boxes.append(b)
    cap.release()
    return frames, boxes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()

    from detect_face import SCRFD
    from huggingface_hub import snapshot_download
    from musetalk_mlx.pipeline_mlx import MuseTalkPipeline

    mx.set_default_device(mx.gpu)
    det = SCRFD(str(ROOT / "ref-feathertalk" / "data_utils" / "scrfd_2.5g_kps.onnx"),
                confThreshold=0.5)

    print("读帧 + 检测人脸…", flush=True)
    frames, boxes = face_boxes(a.video, det)
    if not frames or boxes[0] is None:
        print("底子视频里没检测到人脸", file=sys.stderr)
        return 1
    print(f"  {len(frames)} 帧")

    print("加载 MuseTalk…", flush=True)
    pipe = MuseTalkPipeline.from_pretrained_mlx(
        snapshot_download("mlx-community/MuseTalk-1.5-fp16"))
    pipe.astype(mx.float16)

    print("编码音频…", flush=True)
    chunks = pipe.encode_audio_from_wav(a.audio, fps=FPS)
    n = int(chunks.shape[0])
    print(f"  {n} 帧音频特征（{n/FPS:.1f}s）")

    # 底子视频循环使用以覆盖音频长度；用往返循环避免接缝处的跳变
    order = []
    L = len(frames)
    while len(order) < n:
        order += list(range(L)) + list(range(L - 2, 0, -1))
    order = order[:n]

    print("裁剪 + VAE 编码…", flush=True)
    lat, metas = [], []
    for k in order:
        f, b = frames[k], boxes[k]
        x0, y0, s = b
        H, W = f.shape[:2]
        xi, yi, si = int(x0), int(y0), int(s)
        pad = max(0, -xi, -yi, xi + si - W, yi + si - H)
        if pad:
            f = cv2.copyMakeBorder(f, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
            xi, yi = xi + pad, yi + pad
        crop = cv2.resize(f[yi:yi + si, xi:xi + si], (CROP, CROP))
        lat.append(pipe.get_latents_for_unet(crop))
        metas.append((k, xi, yi, si, pad))
    lat = mx.concatenate(lat, axis=0)

    print(f"生成口型（{n} 帧，batch={a.batch}）…", flush=True)
    import time
    t0 = time.perf_counter()
    recon = pipe.run_batched(lat, chunks, batch_size=a.batch)
    dt = time.perf_counter() - t0
    print(f"  {dt:.1f}s → {n/dt:.1f} fps")

    # 羽化遮罩：只把下半张脸贴回去，边缘渐变避免硬边
    m = np.zeros((CROP, CROP), np.float32)
    m[CROP // 2:, :] = 1.0
    m = cv2.GaussianBlur(m, (0, 0), CROP * 0.06)[..., None]

    H0, W0 = frames[0].shape[:2]
    tmp = str(Path(a.out).with_suffix(".silent.mp4"))
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W0, H0))
    print("贴回 + 写出…", flush=True)
    for i, (k, xi, yi, si, pad) in enumerate(metas):
        f = frames[k].copy()
        if pad:
            f = cv2.copyMakeBorder(f, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        patch = cv2.resize(recon[i], (si, si)).astype(np.float32)
        dst = f[yi:yi + si, xi:xi + si].astype(np.float32)
        mm = cv2.resize(m, (si, si))[..., None]
        f[yi:yi + si, xi:xi + si] = (patch * mm + dst * (1 - mm)).astype(np.uint8)
        if pad:
            f = f[pad:pad + H0, pad:pad + W0]
        vw.write(f)
    vw.release()

    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-i", a.audio,
                    "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", "-movflags", "+faststart", a.out],
                   check=True)
    Path(tmp).unlink(missing_ok=True)
    print(f"\n完成 → {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
