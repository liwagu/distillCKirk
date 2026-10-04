#!/usr/bin/env python
"""从一段影像里切出 idle / talk 两条循环轨。

判断说话还是倾听：不再引入模型，直接量**嘴部区域的帧间运动**。
说话时下半张脸持续变化，倾听时基本不动。用 SCRFD 的关键点定位嘴，
再对该区域做逐帧差分。

选循环点的原则：在**眨眼附近**剪接。眨眼时眼睛闭着，头部姿态的不连续
最不容易被察觉（Video Textures 的老技巧），比任何淡化算法都有效。
眨眼用眼部区域的短促运动尖峰来找。

用法:
    .venv/bin/python scripts/make_loops.py <视频> --ranges 37:55 58:79 302:356 \\
        [--half left] [--secs 20]
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
OUT = ROOT / "web" / "media"


def analyse(video: str, ranges: list[tuple[float, float]], half: str):
    """逐帧量嘴部与眼部运动。返回 [(t, mouth_motion, eye_motion, box)]。"""
    from detect_face import SCRFD
    det = SCRFD(str(ROOT / "ref-feathertalk" / "data_utils" / "scrfd_2.5g_kps.onnx"),
                confThreshold=0.5)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rows = []
    for (a, b) in ranges:
        prev_m = prev_e = None
        t = a
        while t < b:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if half == "left":
                frame = frame[:, : w // 2]
            elif half == "right":
                frame = frame[:, w // 2:]
            try:
                bb, idx, kps = det.detect(frame)
                idx = np.asarray(idx).reshape(-1)
                if not idx.size:
                    t += 1 / fps * 2
                    continue
                cand = np.asarray(bb, float)[idx]
                i = int(np.argmax(cand[:, 3]))
                x, y, bw, bh = cand[i, :4].astype(int)
                k = np.asarray(kps, float)[idx][i]      # 5 点：双眼、鼻、双嘴角
            except Exception:
                t += 1 / fps * 2
                continue
            # 嘴部：两嘴角之间，向上下各扩一点
            mx0 = int(min(k[3][0], k[4][0]) - bw * .12); mx1 = int(max(k[3][0], k[4][0]) + bw * .12)
            my0 = int(min(k[3][1], k[4][1]) - bh * .12); my1 = int(max(k[3][1], k[4][1]) + bh * .18)
            # 眼部：两眼之间
            ex0 = int(min(k[0][0], k[1][0]) - bw * .12); ex1 = int(max(k[0][0], k[1][0]) + bw * .12)
            ey0 = int(min(k[0][1], k[1][1]) - bh * .10); ey1 = int(max(k[0][1], k[1][1]) + bh * .10)
            H, W = frame.shape[:2]
            cl = lambda v, hi: max(0, min(int(v), hi))
            m = frame[cl(my0, H):cl(my1, H), cl(mx0, W):cl(mx1, W)]
            e = frame[cl(ey0, H):cl(ey1, H), cl(ex0, W):cl(ex1, W)]
            if m.size and e.size:
                m = cv2.cvtColor(cv2.resize(m, (48, 32)), cv2.COLOR_BGR2GRAY).astype(np.float32)
                e = cv2.cvtColor(cv2.resize(e, (64, 20)), cv2.COLOR_BGR2GRAY).astype(np.float32)
                if prev_m is not None:
                    rows.append((t,
                                 float(np.abs(m - prev_m).mean()),
                                 float(np.abs(e - prev_e).mean()),
                                 (x, y, bw, bh)))
                prev_m, prev_e = m, e
            t += 1 / fps * 2                 # 隔帧采样，够用且快一倍
    cap.release()
    return rows, fps


def pick(rows, secs: float, want_talk: bool, idle_pct: float = 30.0):
    """挑一段 secs 秒、嘴部运动符合要求、且两端都落在眨眼附近的窗口。"""
    if not rows:
        return None
    ts = np.array([r[0] for r in rows])
    mm = np.array([r[1] for r in rows])
    ee = np.array([r[2] for r in rows])
    eye_hi = np.percentile(ee, 82)            # 眨眼 = 眼部运动尖峰
    # 待机轨的绝对门槛：只挑"相对最低"是不够的 —— 整段素材都在说话时，
    # 最低的那段仍然是在说话（实测选到运动量 4.06 的片段，画面里他嘴张着）。
    idle_max = np.percentile(mm, idle_pct)
    best, best_score = None, -1e9
    for i in range(len(rows)):
        j = np.searchsorted(ts, ts[i] + secs)
        if j >= len(rows) or ts[j] - ts[i] < secs * .9:
            continue
        if ts[j] - ts[i] > secs * 1.15:       # 跨越了片段间断
            continue
        seg = mm[i:j]
        motion = float(seg.mean())
        if not want_talk and motion > idle_max:
            continue                          # 达不到"安静"的绝对门槛，跳过
        # 说话轨要运动大且持续；待机轨要运动小且平稳
        score = motion if want_talk else -motion
        score -= float(seg.std()) * (0.3 if want_talk else 1.0)
        # 两端落在眨眼上加分：眨眼时眼睛闭着，剪接点最不易察觉
        score += 0.6 * ((ee[i] > eye_hi) + (ee[j] > eye_hi))
        if score > best_score:
            best_score, best = score, (ts[i], ts[j], motion)
    return best


def cut(video: str, start: float, dur: float, half: str, out: Path, fps: float):
    vf = []
    if half in ("left", "right"):
        vf.append(f"crop=iw/2:ih:{0 if half=='left' else 'iw/2'}:0")
    vf += ["scale=640:-2", f"fps={min(30, round(fps))}"]
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start}",
                    "-i", video, "-t", f"{dur}", "-an",
                    "-vf", ",".join(vf), "-pix_fmt", "yuv420p",
                    "-c:v", "libx264", "-crf", "20", "-preset", "slow",
                    "-movflags", "+faststart", str(out)], check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--ranges", nargs="+", required=True, help="如 37:55 302:356")
    ap.add_argument("--half", default="left", choices=["left", "right", "full"])
    ap.add_argument("--secs", type=float, default=20.0)
    a = ap.parse_args()

    ranges = [(float(r.split(":")[0]), float(r.split(":")[1])) for r in a.ranges]
    print(f"分析 {sum(b-x for x,b in ranges):.0f}s 素材…")
    rows, fps = analyse(a.video, ranges, a.half)
    if not rows:
        print("没检测到可用人脸", file=sys.stderr)
        return 1
    mm = np.array([r[1] for r in rows])
    print(f"采样 {len(rows)} 点 · 嘴部运动 中位 {np.median(mm):.2f} "
          f"(10% {np.percentile(mm,10):.2f} / 90% {np.percentile(mm,90):.2f})\n")

    OUT.mkdir(parents=True, exist_ok=True)
    for name, want_talk in [("talk", True), ("idle", False)]:
        p = pick(rows, a.secs, want_talk)
        if not p:
            print(f"  {name}: 找不到符合要求的 {a.secs:.0f}s 窗口"
                  + ("（素材里没有足够安静的片段 —— 需要一段他在倾听的影像）"
                     if not want_talk else ""))
            continue
        s, e, motion = p
        cut(a.video, s, e - s, a.half, OUT / f"{name}.mp4", fps)
        print(f"  {name}.mp4  {s:6.1f}s – {e:6.1f}s  ({e-s:.1f}s) "
              f"嘴部运动 {motion:.2f}")
    print(f"\n输出到 {OUT}/ —— 刷新页面即可看到")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
