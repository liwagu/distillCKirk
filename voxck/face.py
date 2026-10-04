"""MuseTalk 实时口型渲染器 —— 不用训练，用他本人的影像当底子，按音频现算嘴部。

三块芯片分工，流水线并行（每帧 40ms 预算）：
  ANE  (神经引擎, Core ML)：UNet 单步去噪            实测 34 ms/帧，与 GPU 物理隔离
  GPU  (Metal, MLX)：      Whisper 音频编码 + VAE 解码   只解码下半张脸的潜变量（16–29 ms/帧）
  CPU  (numpy)：           羽化贴回底子帧              2–8 ms/帧

为什么这样切（全部本机实测，见 docs/musetalk-measured.md）：
  - GPU 在持续负载 ~1s 后降到爆发态的 1/3（矩阵乘 42→13 TFLOPS）。九月基准里的 30.7 fps
    是爆发态数字，持续态 UNet+VAE 全放 GPU 只有 14.5 fps。
  - UNet 放 ANE 后 GPU 只剩 VAE 解码，且只解码贴回会用到的下半张脸。
  - 底子帧的人脸框、VAE 编码都不依赖音频，离线算一次缓存到磁盘。

线程约束：MLX 的 Stream 是线程局部的，所有 MLX 调用必须在同一条线程上。
构造时传入 mlx_submit（orchestrator 的单线程执行器），不传则自建。
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from typing import Callable, Iterator

import cv2
import numpy as np

log = logging.getLogger("voxck.face")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                                   # 直接运行本文件时也能 import voxck
sys.path.insert(0, str(ROOT / "ref-feathertalk" / "data_utils"))

CROP = 256
FPS = 25
UNET_PKG = ROOT / "assets" / "coreml" / "musetalk_unet_ane_b1.mlpackage"   # Apple ANE 结构版（scripts/export_unet_ane.py）
CACHE_DIR = ROOT / "assets" / "cache"
DEC_BS = 4                     # GPU 解码批大小；ANE 逐帧


class FaceRenderer:
    def __init__(self, base_video: str, mlx_submit: Callable | None = None,
                 r0: int = 9, expand: float = 0.1, fps: int = FPS, jpeg_quality: int = 80,
                 mask_mode: str = "mouth", background: dict | None = None,
                 pre_crop: list | None = None, view_scale: float = 3.0,
                 idle_frame: int = 0):
        """expand：人脸框外扩比例。**必须小**（0.1）：MuseTalk 的训练裁剪是紧贴人脸的框，脸填满 256；
        之前用 0.55 时嘴在裁剪里只有二十几像素，模型输出是一团糊、几乎不张嘴（实测对比 expand_ab.jpg）。
        r0：潜变量从第几行开始解码（32 行=256px）。贴回遮罩从 128px 起渐入，
        r0=9 → 从 72px 起解码，留 56px 给卷积感受野吃掉边界误差。
        fps：输出帧率。ANE UNet 流水线里约 40ms/帧，25fps 撑不住、20fps 有 20% 余量；
        Whisper 特征按该帧率切块，浏览器按 pts 显示，所以任意帧率都对得上音频。"""
        self.r0 = r0
        self.fps = fps
        self.jpeg_quality = jpeg_quality
        self.expand = expand
        self.mask_mode = mask_mode                 # "mouth"：只贴嘴周椭圆；"lower"：贴整个下半张脸
        self.background = background or {}         # {"enabled": bool, "image": 路径 或 None, "top": "#0b1a33", "bottom": "#1c2f57"}
        self.pre_crop = pre_crop                   # 先裁掉底子的一部分（比例 [x0,y0,x1,y1]）：电视分屏只留他那一半
        self.view_scale = view_scale               # 取景窗高 = 脸高 × view_scale（3.0 = 头肩构图）
        self.idle_frame = idle_frame               # 已人工确认闭嘴的帧，按底子重采样后的25fps索引
        self.out_w, self.out_h = 640, 480          # 输出取景：以人脸为中心的固定 4:3 头肩构图
        if not Path(base_video).is_absolute():
            base_video = str(ROOT / base_video)
        self._ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx") if mlx_submit is None else None
        self._mlx = mlx_submit or (lambda fn, *a: self._ex.submit(fn, *a))
        t0 = time.perf_counter()
        self._mlx(self._init_mlx).result()
        from voxck.ane_worker import AneUnet
        self.unet = AneUnet(str(UNET_PKG), batch=DEC_BS, layout="bc1s")   # 独立进程，见 ane_worker.py
        log.info("模型加载 %.1fs（MuseTalk MLX + UNet Core ML 子进程）", time.perf_counter() - t0)
        self._mlx(self._precompute, base_video, expand).result()
        self._cursor = self.order.index(self.idle_frame)
        self.stats: dict = {}

    # ───────────── MLX 线程内 ─────────────
    def _init_mlx(self):
        import mlx.core as mx
        from huggingface_hub import snapshot_download
        from musetalk_mlx.pipeline_mlx import MuseTalkPipeline
        mx.set_default_device(mx.gpu)
        self.mx = mx
        self.pipe = MuseTalkPipeline.from_pretrained_mlx(snapshot_download("mlx-community/MuseTalk-1.5-fp16"))
        self.pipe.astype(mx.float16)
        sf, r0, vae = self.pipe.scaling_factor, self.r0, self.pipe.vae

        def dec(z):                                     # (B,4,32-r0,32) → (B,H,256,3) 0..255 RGB
            img = vae.decode(z / sf)
            img = mx.clip(img / 2 + 0.5, 0, 1) * 255
            return img.transpose(0, 2, 3, 1)
        self._dec = mx.compile(dec)
        z = mx.zeros((DEC_BS, 4, 32 - r0, 32), dtype=mx.float16)
        mx.eval(self._dec(z))                           # 编译预热

    def _precompute(self, video: str, expand: float):
        """读底子帧（重采样到 25fps）→ 人脸框 → 裁剪 → VAE 编码。潜变量与框缓存到磁盘。"""
        mx = self.mx
        t0 = time.perf_counter()
        cap = cv2.VideoCapture(video)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or FPS
        raw = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            raw.append(f)
        cap.release()
        if not raw:
            raise RuntimeError(f"底子视频读不到帧: {video}")
        n25 = int(len(raw) / src_fps * FPS)
        self.frames = [raw[min(len(raw) - 1, int(round(i / FPS * src_fps)))] for i in range(n25)]
        if self.pre_crop:
            H0, W0 = self.frames[0].shape[:2]
            x0, y0, x1, y1 = (int(round(self.pre_crop[0] * W0)), int(round(self.pre_crop[1] * H0)),
                              int(round(self.pre_crop[2] * W0)), int(round(self.pre_crop[3] * H0)))
            self.frames = [np.ascontiguousarray(f[y0:y1, x0:x1]) for f in self.frames]
        self.H, self.W = self.frames[0].shape[:2]

        st = Path(video).stat()
        bg = self.background if self.background.get("enabled") else {}
        bg_key = json.dumps(bg, sort_keys=True)
        key = hashlib.md5(f"v2|{Path(video).resolve()}|{st.st_mtime_ns}|{st.st_size}|{expand}|{CROP}|{bg_key}|{self.pre_crop}".encode()).hexdigest()[:12]
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if bg:
            # 换背景：先抠像合成，再检测/裁剪/编码——这样 VAE 参考潜变量里也是新背景，贴回不留原背景光晕
            self._composite_background(bg, CACHE_DIR / f"alpha-{Path(video).stem}-{key}.npz")
        cache = CACHE_DIR / f"base-{Path(video).stem}-{key}.npz"
        if cache.exists():
            d = np.load(cache)
            self.lat = mx.array(d["lat"]).astype(mx.float16)
            self.metas = [tuple(int(v) for v in m) for m in d["metas"]]
            self.kps = d["kps"]                                   # (N,5,2) 裁剪坐标 0..1
            log.info("底子缓存命中 %s（%d 帧）", cache.name, len(self.metas))
        else:
            from detect_face import SCRFD
            det = SCRFD(str(ROOT / "ref-feathertalk" / "data_utils" / "scrfd_2.5g_kps.onnx"), confThreshold=0.5)
            lat, metas, kps_all, last, last_k = [], [], [], None, None
            for f in self.frames:
                b = k5 = None
                try:
                    bb, idx, kps = det.detect(f)
                    idx = np.asarray(idx).reshape(-1)
                    if idx.size:
                        c = np.asarray(bb, float)[idx]
                        i = int(np.argmax(c[:, 3]))
                        x, y, w, h = c[i, :4]
                        s = max(w, h) * (1 + expand)
                        b = np.array([x + w / 2 - s / 2, y + h / 2 - s / 2, s])
                        k5 = np.asarray(kps, float)[idx][i]        # 双眼、鼻、双嘴角（帧坐标）
                except Exception:
                    pass
                if b is None:
                    b, k5 = last, last_k
                elif last is not None:
                    # 中心轻平滑、尺寸重平滑：转头快时贴回区域跟得上，尺寸又不呼吸
                    b = np.array([0.5 * last[0] + 0.5 * b[0], 0.5 * last[1] + 0.5 * b[1], 0.85 * last[2] + 0.15 * b[2]])
                if b is None:
                    raise RuntimeError("底子视频开头检测不到人脸")
                last, last_k = b, k5
                xi, yi, si = int(b[0]), int(b[1]), int(b[2])
                pad = max(0, -xi, -yi, xi + si - self.W, yi + si - self.H)
                fp = f
                if pad:
                    fp = cv2.copyMakeBorder(f, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
                    xi, yi = xi + pad, yi + pad
                crop = cv2.resize(fp[yi:yi + si, xi:xi + si], (CROP, CROP))
                lat.append(self.pipe.get_latents_for_unet(crop))
                metas.append((xi, yi, si, pad))
                kn = (k5 + pad - np.array([xi, yi])) / si if k5 is not None else np.full((5, 2), np.nan)
                kps_all.append(kn)
            self.lat = mx.concatenate(lat, axis=0).astype(mx.float16)
            mx.eval(self.lat)
            self.metas = metas
            self.kps = np.array(kps_all, dtype=np.float32)
            np.savez(cache, lat=np.array(self.lat.astype(mx.float32)), metas=np.array(metas), kps=self.kps)
            log.info("底子预计算 %d 帧 %.1fs → 缓存 %s", len(metas), time.perf_counter() - t0, cache.name)
        L = len(self.frames)
        self.order = list(range(L)) + list(range(L - 2, 0, -1))      # 往返循环，接缝不跳
        # 取景窗：整段底子用同一个窗（人脸中心/尺寸取中位数），头动是真的、镜头不抖。
        # 竖幅底子直接 object-fit:cover 会把脸裁掉（实测只剩下巴和话筒），所以在服务端定框。
        boxes = np.array([(xi - pad + si / 2, yi - pad + si / 2, si) for (xi, yi, si, pad) in self.metas], float)
        fcx, fcy, si_m = np.median(boxes, axis=0)
        fh = si_m / (1 + self.expand)
        vh = int(min(self.H, fh * self.view_scale)); vw = int(min(self.W, vh * 4 / 3)); vh = int(vw * 3 / 4)
        vx = int(np.clip(fcx - vw / 2, 0, self.W - vw)); vy = int(np.clip(fcy - vh * 0.42, 0, self.H - vh))
        self.view = (vx, vy, vw, vh)
        self._prepare_idle()
        log.info("取景窗 x=%d y=%d %dx%d（脸高≈%dpx）→ 输出 %dx%d", vx, vy, vw, vh, int(fh), self.out_w, self.out_h)
        # 羽化遮罩（裁剪坐标，裁掉 r0 以上那部分不解码的行）
        #  lower：整个下半张脸——MuseTalk 原版做法，脸颊下巴一起重绘，糊得明显、话筒会被抹掉
        #  mouth：按关键点只贴嘴周椭圆（含下巴活动范围）——脸颊/下颌线/话筒保留原片清晰度
        m = np.zeros((CROP, CROP), np.float32)
        m[CROP // 2:, :] = 1.0
        self.mask = cv2.GaussianBlur(m, (0, 0), CROP * 0.06)[8 * self.r0:, :]
        self.masks = None
        if self.mask_mode == "mouth":
            masks = []
            for k in self.kps:
                if np.isnan(k).any():
                    masks.append(self.mask); continue
                ml, mr, nose = k[3] * CROP, k[4] * CROP, k[2] * CROP
                cx, cy = (ml[0] + mr[0]) / 2, (ml[1] + mr[1]) / 2
                mw = max(24.0, float(np.hypot(*(mr - ml))))
                mouth_nose = max(12.0, cy - nose[1])
                e = np.zeros((CROP, CROP), np.float32)
                # 椭圆：宽 2.1 倍嘴宽，中心略偏下（张嘴时下巴往下），高覆盖到鼻下与下巴
                cv2.ellipse(e, (int(cx), int(cy + 0.35 * mouth_nose)), (int(1.05 * mw), int(0.85 * mouth_nose + 0.55 * mw)), 0, 0, 360, 1.0, -1)
                masks.append(cv2.GaussianBlur(e, (0, 0), CROP * 0.045)[8 * self.r0:, :])
            self.masks = masks

    def _backdrop(self, bg: dict) -> np.ndarray:
        """演播室背景：给了图片就用图片（拉伸到底子尺寸），否则深蓝渐变 + 柔光。"""
        img = bg.get("image")
        if img:
            pth = Path(img) if Path(img).is_absolute() else ROOT / img
            im = cv2.imread(str(pth))
            if im is not None:
                return cv2.resize(im, (self.W, self.H), interpolation=cv2.INTER_AREA)
            log.warning("背景图读不到 %s，改用渐变", pth)
        def hexc(h, d):
            h = (bg.get(h) or d).lstrip("#"); return np.array([int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)], np.float32)  # BGR
        top, bot = hexc("top", "#0b1a33"), hexc("bottom", "#1c2f57")
        t = np.linspace(0, 1, self.H, dtype=np.float32)[:, None, None]
        grad = top * (1 - t) + bot * t
        yy, xx = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        spot = np.exp(-(((xx - self.W * 0.5) / (self.W * 0.55)) ** 2 + ((yy - self.H * 0.35) / (self.H * 0.6)) ** 2))
        out = grad * (0.85 + 0.35 * spot[..., None])
        return np.clip(out, 0, 255).astype(np.uint8)

    def _composite_background(self, bg: dict, cache: Path):
        t0 = time.perf_counter()
        if cache.exists():
            alphas = np.load(cache)["alpha"]
        else:
            from . import segment                   # 重依赖，只在需要时导入
            alphas = np.stack([(segment.mask(f) * 255).astype(np.uint8) for f in self.frames])
            np.savez_compressed(cache, alpha=alphas)
        back = self._backdrop(bg).astype(np.float32)
        out = []
        for f, a in zip(self.frames, alphas):
            al = (a.astype(np.float32) / 255.0)[..., None]
            out.append((f.astype(np.float32) * al + back * (1 - al)).astype(np.uint8))
        self.frames = out
        log.info("底子换背景 %d 帧 %.1fs（%s）", len(out), time.perf_counter() - t0, "缓存" if cache.exists() else "抠像")

    def _audio_feats(self, wav16k: np.ndarray):
        """16k float32 音频 → (n, 50, 384) fp16 numpy（已加位置编码），n = 帧数。"""
        mx = self.mx
        from musetalk_mlx.whisper.audio2feature import apply_pe, get_whisper_chunk
        from musetalk_mlx.whisper.log_mel import N_SAMPLES, log_mel_spectrogram
        wav = wav16k.astype(np.float32)
        segs = [wav[i:i + N_SAMPLES] for i in range(0, max(len(wav), 1), N_SAMPLES)]
        feats = [self.pipe.whisper_encoder(log_mel_spectrogram(mx.array(s))) for s in segs]
        chunks = get_whisper_chunk(mx.concatenate(feats, axis=1), len(wav), fps=self.fps)
        pe = apply_pe(chunks).astype(mx.float16)
        mx.eval(pe)
        return np.array(pe)

    def _latents_np(self, idxs: list[int]) -> np.ndarray:
        mx = self.mx
        x = self.lat[idxs]
        mx.eval(x)
        return np.array(x)

    def _decode(self, ys: np.ndarray) -> np.ndarray:
        """(b,4,32,32) fp16 → (b, 256-8*r0, 256, 3) uint8 BGR。补齐到 DEC_BS 避免重编译。"""
        mx = self.mx
        b = ys.shape[0]
        if b < DEC_BS:
            ys = np.concatenate([ys, np.zeros((DEC_BS - b, *ys.shape[1:]), ys.dtype)])
        img = self._dec(mx.array(ys)[:, :, self.r0:, :])
        out = np.array(img.astype(mx.uint8) if hasattr(mx, "uint8") else img)
        return out[:b, :, :, ::-1]

    # ───────────── ANE 线程（只做管道收发，不持 GIL）─────────────
    def _ane_loop(self, xs: np.ndarray, pes: np.ndarray, q: Queue, stop: threading.Event):
        sent_stop = False
        for ys in self.unet.run(xs, pes):
            if stop.is_set() and not sent_stop:
                self.unet.stop()
                sent_stop = True
            q.put(ys)
        q.put(None)
        self.stats["ane_ms"] = float(self.unet.last[1])

    # ───────────── 对外 ─────────────
    def render(self, wav16k: np.ndarray, stop: threading.Event | None = None,
               skip_frames: int = 0, stride: int = 1) -> Iterator[np.ndarray]:
        """音频 → 逐帧生成 BGR 画面（self.fps，按序），三级流水线。stop 置位后尽快结束。

        skip_frames：音频开头这些帧只当 Whisper 的左侧上下文，不输出（流式分段时避免切口伪影）。
        stride：每 stride 帧只算一帧（来不及时降到 fps/stride 的有效帧率，浏览器保持上一帧）。
        """
        stop = stop or threading.Event()
        t_start = time.perf_counter()
        pes = self._mlx(self._audio_feats, wav16k).result()
        pes = pes[skip_frames::stride]
        n = pes.shape[0]
        idxs = [self.order[(self._cursor + i * stride) % len(self.order)] for i in range(n)]
        self._cursor = (self._cursor + n * stride) % len(self.order)
        xs = self._mlx(self._latents_np, idxs).result()
        t_feat = time.perf_counter() - t_start

        q: Queue = Queue(maxsize=4)
        th = threading.Thread(target=self._ane_loop, args=(xs, pes, q, stop), daemon=True, name="ane")
        th.start()
        pending: deque[tuple[Future, int]] = deque()
        done_frames, gpu_t, paste_t = 0, [], []
        off = int(round(8 * self.r0))                   # 裁剪坐标里解码区域的起点

        def paste(i: int, patch_crop: np.ndarray) -> np.ndarray:
            xi, yi, si, pad = self.metas[idxs[i]]
            f = self.frames[idxs[i]].copy()
            if pad:
                f = cv2.copyMakeBorder(f, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
            y0 = yi + int(round(off * si / CROP))
            h = si - (y0 - yi)
            patch = cv2.resize(patch_crop, (si, h)).astype(np.float32)
            mk = self.masks[idxs[i]] if self.masks is not None else self.mask
            mm = cv2.resize(mk, (si, h))[..., None]
            dst = f[y0:y0 + h, xi:xi + si].astype(np.float32)
            f[y0:y0 + h, xi:xi + si] = (patch * mm + dst * (1 - mm)).astype(np.uint8)
            if pad:
                f = f[pad:pad + self.H, pad:pad + self.W]
            return self._view(f)

        def drain_one():
            nonlocal done_frames
            fut, start = pending.popleft()
            t = time.perf_counter()
            imgs = fut.result()
            gpu_t.append(time.perf_counter() - t)
            for j in range(imgs.shape[0]):
                t = time.perf_counter()
                fr = paste(start + j, imgs[j])
                paste_t.append(time.perf_counter() - t)
                done_frames += 1
                yield fr

        nxt = 0
        sent_stop_main = [False]
        got_end = False
        while True:
            ys = q.get()
            if ys is None:
                got_end = True
                break
            pending.append((self._mlx(self._decode, ys), nxt))
            nxt += ys.shape[0]
            if len(pending) >= 2:                       # GPU 解码与贴回重叠
                yield from drain_one()
            if stop.is_set():
                if not sent_stop_main[0]:
                    self.unet.stop()
                    sent_stop_main[0] = True
        while pending and not stop.is_set():
            yield from drain_one()
        if not got_end:                                 # 中断提前退出：清空残留批直到哨兵
            while q.get() is not None:
                pass
        th.join(timeout=10)
        total = time.perf_counter() - t_start
        self.stats.update(frames=done_frames, total_s=total, fps=done_frames / total if total else 0,
                          feat_ms=t_feat * 1000,
                          gpu_wait_ms=float(np.mean(gpu_t) * 1000 / DEC_BS) if gpu_t else 0,
                          paste_ms=float(np.mean(paste_t) * 1000) if paste_t else 0)

    def _view(self, frame: np.ndarray) -> np.ndarray:
        vx, vy, vw, vh = self.view
        return cv2.resize(frame[vy:vy + vh, vx:vx + vw], (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)

    def _prepare_idle(self) -> None:
        """静默必须闭嘴；原片可能仍在说话，不能把整段底子当作倾听循环。"""
        if (isinstance(self.idle_frame, bool) or not isinstance(self.idle_frame, int)
                or not 0 <= self.idle_frame < len(self.frames)):
            raise ValueError(f"idle_frame must be an integer in 0..{len(self.frames) - 1}")
        self.idle_jpeg = self.jpeg(self._view(self.frames[self.idle_frame]))
        log.info("静默待机固定第%d帧（%.2fs），不播放原片口型", self.idle_frame, self.idle_frame / FPS)

    def next_idle_jpeg(self) -> bytes:
        """只返回确认闭嘴的中性画面，待机不推进说话素材游标。"""
        return self.idle_jpeg

    def jpeg(self, frame: np.ndarray) -> bytes:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        return buf.tobytes() if ok else b""

    def close(self):
        try:
            self.unet.close()
        except Exception:
            pass
        if self._ex is not None:
            self._ex.shutdown(wait=False)

    def render_to_file(self, wav_path: str, out: str) -> dict:
        import soundfile as sf
        wav, sr = sf.read(wav_path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(1)
        if sr != 16000:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(sr, 16000)
            wav = resample_poly(wav, 16000 // g, sr // g).astype(np.float32)
        tmp = out + ".silent.mp4"
        vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (self.out_w, self.out_h))
        for fr in self.render(wav):
            vw.write(fr)
        vw.release()
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-i", wav_path, "-c:v", "libx264",
                        "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", "-movflags", "+faststart", out],
                       check=True)
        Path(tmp).unlink(missing_ok=True)
        return dict(self.stats)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--r0", type=int, default=9)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--runs", type=int, default=2)
    a = ap.parse_args()
    r = FaceRenderer(a.video, r0=a.r0, fps=a.fps)
    for k in range(a.runs):
        st = r.render_to_file(a.audio, a.out or "/dev/null") if a.out else None
        if not a.out:
            import soundfile as sf
            wav, _ = sf.read(a.audio, dtype="float32")
            for _ in r.render(wav):
                pass
            st = r.stats
        print(f"第{k+1}轮: {st['frames']} 帧 {st['total_s']:.2f}s = {st['fps']:.1f} fps "
              f"（音频 {st['frames']/r.fps:.1f}s，{'✓ 快于实时' if st['fps'] >= r.fps else '✗ 慢于实时'}）"
              f"  ANE {st['ane_ms']:.1f}ms/帧 · GPU等待 {st['gpu_wait_ms']:.1f}ms/帧 · 贴回 {st['paste_ms']:.1f}ms/帧 · 音频特征 {st['feat_ms']:.0f}ms")
    if a.out:
        print(f"→ {a.out}")
