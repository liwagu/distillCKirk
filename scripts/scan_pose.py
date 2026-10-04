#!/usr/bin/env python
"""扫描长视频，找出「正对镜头 + 脸够大 + 位置稳定」的片段。

朝向怎么算：SCRFD 给 5 个关键点（左眼、右眼、鼻、左嘴角、右嘴角）。
鼻子相对两眼中点的水平偏移，除以两眼间距 —— 这个比值就是偏航的代理量。
正对镜头时鼻子居中（≈0），侧坐时鼻子偏向一侧（越大越侧）。
垂直方向同理可估俯仰，但对我们不重要，只用来排除极端低头。

不做身份判断：检测器只知道「有脸」，不知道是谁。合辑类素材必然混入他人，
最后仍需人眼确认 —— 上一轮的教训是没有对照实验就别信一个嵌入的区分力。

用法:
    .venv/bin/python scripts/scan_pose.py <视频> [--step 2] [--max-yaw 0.18]
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
    ap.add_argument("--step", type=float, default=2.0)
    ap.add_argument("--max-yaw", type=float, default=0.18, help="越小越严格（正脸）")
    ap.add_argument("--min-face", type=float, default=0.14, help="脸高占画面高的最小比例")
    ap.add_argument("--min-run", type=float, default=6.0, help="片段最短秒数")
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()

    from detect_face import SCRFD
    det = SCRFD(str(ROOT / "ref-feathertalk" / "data_utils" / "scrfd_2.5g_kps.onnx"),
                confThreshold=0.5)
    cap = cv2.VideoCapture(a.video)
    total = duration(a.video)
    print(f"{Path(a.video).name[:60]}  {total/60:.1f}分钟  步长{a.step}s "
          f"→ 约 {int(total/a.step)} 个采样点", flush=True)

    rows, t, n = [], 0.0, 0
    while t < total:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        rec = None
        try:
            bb, idx, kps = det.detect(frame)
            idx = np.asarray(idx).reshape(-1)
            if idx.size:
                cand = np.asarray(bb, float)[idx]
                i = int(np.argmax(cand[:, 3]))
                x, y, bw, bh = cand[i, :4]
                k = np.asarray(kps, float)[idx][i]
                if bh / h >= a.min_face:
                    eye_mid = (k[0] + k[1]) / 2.0
                    eye_d = float(np.linalg.norm(k[0] - k[1])) + 1e-6
                    yaw = float(k[2][0] - eye_mid[0]) / eye_d      # 鼻子水平偏移
                    pitch = float(k[2][1] - eye_mid[1]) / eye_d
                    rec = (abs(yaw), pitch, bh / h,
                           x + bw / 2, y + bh / 2)
        except Exception:
            pass
        rows.append((t, rec))
        t += a.step
        n += 1
        if n % 200 == 0:
            print(f"  …{t/60:.0f}分钟", end="\r", flush=True)
    cap.release()

    # 连续片段：连续正脸 + 位置不跳（跳 = 换镜头/换人）
    segs, cur = [], None
    for t, r in rows:
        good = r is not None and r[0] <= a.max_yaw and abs(r[1]) < 2.2
        if not good:
            if cur and cur["end"] - cur["start"] >= a.min_run:
                segs.append(cur)
            cur = None
            continue
        _, _, sz, cx, cy = r
        if cur and abs(cx - cur["cx"]) < w * .08 and abs(cy - cur["cy"]) < h * .08:
            cur["end"] = t
            cur["yaw"] = min(cur["yaw"], r[0])
            cur["sz"] = max(cur["sz"], sz)
            cur["cx"] = .7 * cur["cx"] + .3 * cx
            cur["cy"] = .7 * cur["cy"] + .3 * cy
        else:
            if cur and cur["end"] - cur["start"] >= a.min_run:
                segs.append(cur)
            cur = {"start": t, "end": t, "yaw": r[0], "sz": sz, "cx": cx, "cy": cy}
    if cur and cur["end"] - cur["start"] >= a.min_run:
        segs.append(cur)

    segs.sort(key=lambda s: -(s["end"] - s["start"]))
    have = sum(1 for _, r in rows if r)
    front = sum(1 for _, r in rows if r and r[0] <= a.max_yaw)
    print(" " * 30)
    print(f"有脸 {have}/{len(rows)} · 其中正脸 {front}（阈值 yaw≤{a.max_yaw}）")
    print(f"连续正脸片段 {len(segs)} 段\n")
    print(f"{'起':>8} {'止':>8} {'时长':>6} {'最正yaw':>8} {'脸高':>6}")
    for s in segs[:a.top]:
        d = s["end"] - s["start"]
        print(f"{s['start']:8.1f} {s['end']:8.1f} {d:5.0f}s {s['yaw']:8.3f} {s['sz']*100:5.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
