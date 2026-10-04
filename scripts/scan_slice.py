#!/usr/bin/env python
"""扫描视频的一个时间片，找「脸大、正对镜头、位置稳定、时长够」的候选底子片段，并为每个候选出接触表。

输出到 --out 目录：
  samples.json   每个采样点：t, face_pct(脸高/画面高), yaw(鼻偏移/眼距，0=正脸), pitch, cx, cy
  segments.json  候选片段：start, end, dur, min_yaw, mean_yaw, face_pct, motion（中心位移均值）
  seg_<k>.jpg    每个候选 6 帧的接触表（左上角标时间），用来人眼确认是他本人、没话筒挡嘴

用法: .venv/bin/python scripts/scan_slice.py <视频> --t0 0 --t1 600 --step 0.4 --out DIR
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ref-feathertalk" / "data_utils"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("--t0", type=float, default=0); ap.add_argument("--t1", type=float, required=True)
    ap.add_argument("--step", type=float, default=0.4); ap.add_argument("--out", required=True)
    ap.add_argument("--max-yaw", type=float, default=0.16); ap.add_argument("--min-face", type=float, default=0.16)
    ap.add_argument("--min-run", type=float, default=2.5)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    import onnxruntime as ort
    from detect_face import SCRFD
    det = SCRFD(str(ROOT / "ref-feathertalk" / "data_utils" / "scrfd_2.5g_kps.onnx"), confThreshold=0.5)
    cap = cv2.VideoCapture(a.video)
    rows, t = [], a.t0
    while t < a.t1:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok: break
        h, w = frame.shape[:2]
        rec = None
        try:
            bb, idx, kps = det.detect(frame)
            idx = np.asarray(idx).reshape(-1)
            if idx.size:
                cand = np.asarray(bb, float)[idx]; i = int(np.argmax(cand[:, 3]))
                x, y, bw, bh = cand[i, :4]; k = np.asarray(kps, float)[idx][i]
                eye_mid = (k[0] + k[1]) / 2; eye_d = float(np.linalg.norm(k[0] - k[1])) + 1e-6
                rec = dict(t=round(t, 2), face_pct=round(bh / h, 3), yaw=round(float(k[2][0] - eye_mid[0]) / eye_d, 3),
                           pitch=round(float(k[2][1] - eye_mid[1]) / eye_d, 3), cx=round(float(x + bw / 2) / w, 3), cy=round(float(y + bh / 2) / h, 3),
                           n_faces=int(idx.size))
        except Exception:
            pass
        rows.append(rec or dict(t=round(t, 2)))
        t += a.step
    json.dump(rows, open(out / "samples.json", "w"))
    # 连续片段：正脸 + 脸够大 + 位置不跳
    segs, cur = [], None
    def close():
        nonlocal cur
        if cur and cur["end"] - cur["start"] >= a.min_run:
            cur["dur"] = round(cur["end"] - cur["start"], 1); cur["mean_yaw"] = round(float(np.mean(cur["yaws"])), 3)
            cur["min_yaw"] = round(float(np.min(cur["yaws"])), 3); cur["motion"] = round(float(np.mean(cur["moves"])) if cur["moves"] else 0, 4)
            cur["face_pct"] = round(float(np.mean(cur["sizes"])), 3); cur["multi_face_frac"] = round(float(np.mean(cur["multi"])), 2)
            for k_ in ("yaws", "moves", "sizes", "multi", "cx", "cy"): cur.pop(k_, None)
            segs.append(cur)
        cur = None
    for r in rows:
        good = "yaw" in r and abs(r["yaw"]) <= a.max_yaw and r["face_pct"] >= a.min_face and abs(r["pitch"]) < 2.2
        if not good: close(); continue
        if cur and abs(r["cx"] - cur["cx"]) < 0.08 and abs(r["cy"] - cur["cy"]) < 0.08:
            cur["moves"].append(abs(r["cx"] - cur["cx"]) + abs(r["cy"] - cur["cy"]))
            cur["end"] = r["t"]; cur["yaws"].append(abs(r["yaw"])); cur["sizes"].append(r["face_pct"]); cur["multi"].append(r["n_faces"] > 1)
            cur["cx"] = .7 * cur["cx"] + .3 * r["cx"]; cur["cy"] = .7 * cur["cy"] + .3 * r["cy"]
        else:
            close(); cur = dict(start=r["t"], end=r["t"], cx=r["cx"], cy=r["cy"], yaws=[abs(r["yaw"])], sizes=[r["face_pct"]], moves=[], multi=[r["n_faces"] > 1])
    close()
    segs.sort(key=lambda s: -s["dur"])
    for k, s in enumerate(segs):
        s["id"] = f"{int(a.t0)}_{k}"; s["sheet"] = str(out / f"seg_{k}.jpg")
        tiles = []
        for j in range(6):
            tt = s["start"] + (s["end"] - s["start"]) * j / 5
            cap.set(cv2.CAP_PROP_POS_MSEC, tt * 1000); ok, fr = cap.read()
            if not ok: continue
            fr = cv2.resize(fr, (480, 270)); cv2.putText(fr, f"{tt:.1f}s", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2); tiles.append(fr)
        if tiles: cv2.imwrite(s["sheet"], np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])
    json.dump(segs, open(out / "segments.json", "w"), indent=1)
    print(f"采样 {len(rows)} 点，有脸 {sum(1 for r in rows if 'yaw' in r)}，候选片段 {len(segs)}")
    for s in segs[:12]: print(f"  {s['id']:>7}  {s['start']:7.1f}–{s['end']:7.1f}  {s['dur']:5.1f}s  yaw {s['mean_yaw']:.3f}  脸 {s['face_pct']*100:.0f}%  多人 {s['multi_face_frac']}  {s['sheet']}")
    cap.release(); return 0

if __name__ == "__main__":
    raise SystemExit(main())
