#!/usr/bin/env python
"""扫描视频，找出人脸稳定、够大、位置连续的片段，供剪循环用。

针对这个素材的具体情况：画面是分屏的（他在左半、提问者在右半），
而且左半会换人。所以默认只看左半边，并输出每段的人脸尺寸与位置，
由人眼最终确认是不是本人 —— 自动区分身份需要人脸识别模型，
而上一轮的教训是：没有对照实验就不要相信一个嵌入的区分能力。

用法:
    .venv/bin/python scripts/scan_face.py <视频> [--half left|right|full] [--step 0.5]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ref-feathertalk" / "data_utils"))


def duration(p: str) -> float:
    return float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", p],
        capture_output=True, text=True, check=True).stdout.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--half", default="left", choices=["left", "right", "full"])
    ap.add_argument("--step", type=float, default=0.5, help="采样间隔秒")
    ap.add_argument("--min-face", type=float, default=0.12,
                    help="人脸高度占画面高度的最小比例")
    a = ap.parse_args()

    from detect_face import SCRFD
    det = SCRFD(str(ROOT / "ref-feathertalk" / "data_utils" / "scrfd_2.5g_kps.onnx"),
                confThreshold=0.5)

    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = duration(a.video)
    print(f"{Path(a.video).name}  {total:.0f}s @ {fps:.0f}fps  扫描 {a.half} 半边\n")

    rows = []
    t = 0.0
    while t < total:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if a.half == "left":
            frame = frame[:, : w // 2]
        elif a.half == "right":
            frame = frame[:, w // 2:]
        best = None
        try:
            # SCRFD.detect 返回 (bboxes, indices, kpss)：bboxes 是 [x,y,w,h]，
            # indices 是 NMS 之后存活的下标
            bboxes, indices, _ = det.detect(frame)
            idx = np.asarray(indices).reshape(-1)
            if idx.size:
                cand = np.asarray(bboxes, dtype=float)[idx]
                i = int(np.argmax(cand[:, 3]))          # 取最大的那张脸
                x, y, bw, bh = cand[i, :4]
                if bh / h >= a.min_face:
                    best = (x + bw / 2, y + bh / 2, bh / h)
        except Exception:
            pass
        rows.append((t, best))
        t += a.step
    cap.release()

    # 连续片段：连续多个采样点都有脸，且中心位置不大幅跳动（跳动 = 换人/换镜头）
    segs, cur = [], None
    for t, b in rows:
        if b is None:
            if cur and cur["n"] >= 6:
                segs.append(cur)
            cur = None
            continue
        if cur and abs(b[0] - cur["cx"]) < 60 and abs(b[1] - cur["cy"]) < 60:
            cur["end"] = t
            cur["n"] += 1
            cur["cx"] = 0.7 * cur["cx"] + 0.3 * b[0]
            cur["cy"] = 0.7 * cur["cy"] + 0.3 * b[1]
            cur["sz"] = max(cur["sz"], b[2])
        else:
            if cur and cur["n"] >= 6:
                segs.append(cur)
            cur = {"start": t, "end": t, "n": 1, "cx": b[0], "cy": b[1], "sz": b[2]}
    if cur and cur["n"] >= 6:
        segs.append(cur)

    segs.sort(key=lambda s: -(s["end"] - s["start"]))
    have = sum(1 for _, b in rows if b)
    print(f"有脸的采样点 {have}/{len(rows)}（{have/max(len(rows),1)*100:.0f}%）")
    print(f"连续片段 {len(segs)} 段\n")
    print(f"{'起':>7} {'止':>7} {'时长':>6} {'脸高占比':>8}")
    for s in segs[:12]:
        d = s["end"] - s["start"]
        print(f"{s['start']:7.1f} {s['end']:7.1f} {d:6.1f}s {s['sz']*100:7.0f}%")
    if segs:
        print(f"\n最长一段：{segs[0]['start']:.1f}s – {segs[0]['end']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
