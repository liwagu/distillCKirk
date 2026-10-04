"""人像抠像（离线，用于底子帧换背景）。来源：本项目 agent 原型（scratchpad/scan/bg/segment.py）。

两级：MODNet 人像 matting（细边缘，CPU ONNX）× YOLO11n-seg 主体门控（去掉身后的其他人）。
实测 ~90 ms/帧 @1080p，只在启动时对底子循环跑一遍并缓存。
注意：yolo11n-seg 权重来自 Ultralytics，AGPL-3.0——仅本地个人使用；若要分发需换模型。
模型文件在 assets/models/，可用 MODNET_PATH / YOLO_PATH / SEG_MODEL 覆盖。
"""
"""Person/background segmentation for compositing the speaker over a studio backdrop.

    from segment import mask
    alpha = mask(frame_bgr)      # float32 HxW in [0, 1], 1 = main person

Two stages per frame:

  1. fine matte  -- soft person-vs-background alpha
       "modnet"    (default) MODNet portrait matting, ONNX from HF Xenova/modnet,
                   onnxruntime CPU. Crisp hair/mic edges. ~40-100 ms at 640x480
                   depending on SEG_MATTE_SIZE (384 / 512 longer side).
       "mediapipe" MediaPipe selfie_segmenter (256x256 tflite, XNNPACK CPU) +
                   guided-filter edge snap. ~10-20 ms, blobbier edges.
  2. main-subject gate -- YOLO11n-seg (ONNX, onnxruntime CoreML EP, ~4 ms) finds
       every person instance; the largest confident one is the speaker. The gate
       is that instance's mask (hole-filled, dilated, feathered) with pixels that
       confidently belong to *another* person cut out, so bystanders that touch
       the speaker's silhouette are dropped. A plain person/background model keeps
       everyone in the shot; this is what removes the men standing behind him.
       If YOLO finds nobody, fall back to "largest blob touching the bottom edge".

Env knobs:
  SEG_MATTE       modnet | mediapipe            (default modnet)
  SEG_MATTE_SIZE  MODNet input, longer side px  (default 512; 384 is ~2x faster)
  SEG_GATE        0 disables the YOLO gate      (default 1)
  SEG_REFINE      auto | 0 | 1  guided filter   (auto = only for mediapipe)
  SEG_THREADS     onnxruntime intra-op threads  (default 8)
  SEG_COREML      0 disables CoreML EP for YOLO (default 1)

Note: MediaPipe's Metal GPU delegate hard-crashes the process on macOS with
mediapipe 1.0.1 (CHECK failure in gpu_buffer_storage_cv_pixel_buffer), so the
mediapipe path is CPU-only.
"""
import os
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
MATTE = os.environ.get("SEG_MATTE", "modnet")
MATTE_SIZE = int(os.environ.get("SEG_MATTE_SIZE", "512"))
GATE = os.environ.get("SEG_GATE", "1") != "0"
REFINE = os.environ.get("SEG_REFINE", "auto")
THREADS = int(os.environ.get("SEG_THREADS", "8"))
COREML = os.environ.get("SEG_COREML", "1") != "0"

MODNET_PATH = os.environ.get("MODNET_PATH", os.path.join(HERE, "..", "assets", "models", "modnet.onnx"))
YOLO_PATH = os.environ.get("YOLO_PATH", os.path.join(HERE, "..", "assets", "models", "yolo11n-seg.onnx"))
MP_PATH = os.environ.get("SEG_MODEL", os.path.join(HERE, "..", "assets", "models", "selfie_segmenter.tflite"))

_cache = {}


# ----------------------------------------------------------------------------- models
def _ort_session(path, coreml):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = THREADS
    so.log_severity_level = 3
    provs = []
    if coreml:
        provs.append(("CoreMLExecutionProvider", {
            "ModelFormat": "MLProgram", "MLComputeUnits": "ALL",
            "ModelCacheDirectory": os.path.join(HERE, "coreml_cache"),
        }))
    provs.append("CPUExecutionProvider")
    try:
        return ort.InferenceSession(path, so, providers=provs)
    except Exception:
        if coreml:  # older ORT without ModelCacheDirectory, or CoreML compile failure
            return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        raise


def _modnet():
    if "modnet" not in _cache:
        _cache["modnet"] = _ort_session(MODNET_PATH, coreml=False)
    return _cache["modnet"]


def _yolo():
    if "yolo" not in _cache:
        _cache["yolo"] = _ort_session(YOLO_PATH, coreml=COREML)
    return _cache["yolo"]


def _mediapipe():
    if "mp" not in _cache:
        from mediapipe.tasks.python import BaseOptions, vision
        opts = vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=MP_PATH, delegate=BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.IMAGE,
            output_category_mask=False, output_confidence_masks=True)
        _cache["mp"] = vision.ImageSegmenter.create_from_options(opts)
    return _cache["mp"]


# ----------------------------------------------------------------------------- stage 1: matte
def modnet_matte(bgr, size=None):
    """MODNet soft alpha at frame resolution (float32 HxW)."""
    size = size or MATTE_SIZE
    h, w = bgr.shape[:2]
    if w >= h:
        iw, ih = size, max(32, int(round(h * size / w / 32)) * 32)
    else:
        ih, iw = size, max(32, int(round(w * size / h / 32)) * 32)
    rgb = cv2.cvtColor(cv2.resize(bgr, (iw, ih), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    x = ((rgb.astype(np.float32) / 255.0) - 0.5) / 0.5
    x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
    y = _modnet().run(None, {"input": x})[0][0, 0]
    return cv2.resize(y, (w, h), interpolation=cv2.INTER_LINEAR)


def mediapipe_matte(bgr):
    """MediaPipe selfie segmenter person probability at frame resolution."""
    import mediapipe as mp
    h, w = bgr.shape[:2]
    infer = (256, 144) if "landscape" in os.path.basename(MP_PATH) else (256, 256)
    rgb = cv2.cvtColor(cv2.resize(bgr, infer, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    res = _mediapipe().segment(img)
    m = res.confidence_masks[0].numpy_view()[..., 0].copy()  # view dies with `res`: copy
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)


def refine(alpha, bgr, radius=None, eps=12.0 ** 2, scale=0.5):
    """Fast guided filter: snap a blurry upsampled mask to the frame's edges."""
    h, w = alpha.shape
    radius = radius or max(4, int(round(min(h, w) / 80)))
    return cv2.ximgproc.guidedFilter(bgr, alpha, radius, eps, dDepth=-1, scale=scale)


# ----------------------------------------------------------------------------- stage 2: gate
def _fill_holes(b):
    """Fill enclosed holes in a binary uint8 mask (e.g. the lapel mic)."""
    h, w = b.shape
    ff = b.copy()
    m = np.zeros((h + 2, w + 2), np.uint8)
    seeded = False
    for x, y in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        if ff[y, x] == 0:
            cv2.floodFill(ff, m, (x, y), 1)
            seeded = True
    if not seeded:
        return b
    return np.maximum(b, (ff == 0).astype(np.uint8))


def yolo_persons(bgr, conf_other=0.15, iou=0.6):
    """Run YOLO11n-seg. Returns (list of (score, xyxy_in_frame, prob160), letterbox info)
    where prob160 is the instance mask probability on the 160x160 letterbox grid."""
    h, w = bgr.shape[:2]
    S = 640
    r = S / max(h, w)
    nw, nh = int(round(w * r)), int(round(h * r))
    px, py = (S - nw) // 2, (S - nh) // 2
    lb = np.full((S, S, 3), 114, np.uint8)
    lb[py:py + nh, px:px + nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    x = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
    out0, out1 = _yolo().run(None, {"images": x})
    pred = out0[0].T                      # 8400 x (4 box + 80 cls + 32 mask coef)
    score = pred[:, 4]                    # class 0 = person
    keep = score > conf_other
    info = (r, px, py, nw, nh)
    if not keep.any():
        return [], info
    b, sc, mc = pred[keep, :4], score[keep], pred[keep, 84:]
    xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                     b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
    idx = cv2.dnn.NMSBoxes(np.stack([xyxy[:, 0], xyxy[:, 1], b[:, 2], b[:, 3]], 1).tolist(),
                           sc.tolist(), conf_other, iou)
    idx = np.array(idx).reshape(-1)
    proto = out1[0].reshape(32, -1)       # 32 x (160*160)
    persons = []
    for i in idx:
        p = 1.0 / (1.0 + np.exp(-(mc[i] @ proto))).reshape(160, 160)
        x0, y0, x1, y1 = np.clip(np.round(xyxy[i] / 4).astype(int), 0, 160)
        box = np.zeros((160, 160), np.float32)
        box[y0:y1 + 1, x0:x1 + 1] = 1
        frame_box = ((xyxy[i] - [px, py, px, py]) / r).tolist()
        persons.append((float(sc[i]), frame_box, p * box))
    return persons, info


def main_person_gate(bgr, conf_main=0.4, grow=9, feather=3.0, rule="argmax", largest_only=False):
    """Soft 0..1 gate (HxW float32) around the main person, None if nobody found.

    rule="front":  contested pixels (main and another person both > 0.5) go to the
                   main person -- the speaker is always the front-most person, so
                   where instance masks overlap the visible pixel is his.
    rule="argmax": contested pixels go to whichever instance is more confident
                   (cuts notches into the speaker where a bystander's coarse mask
                   overlaps his shoulder / mic hand).
    largest_only:  keep only the biggest connected blob of the gate, dropping
                   bleed fragments over a bystander's body."""
    h, w = bgr.shape[:2]
    persons, (r, px, py, nw, nh) = yolo_persons(bgr)
    if not persons:
        return None
    confident = [p for p in persons if p[0] >= conf_main] or persons
    main = max(confident, key=lambda p: (p[1][2] - p[1][0]) * (p[1][3] - p[1][1]))
    pm = main[2]
    po = np.zeros_like(pm)
    for p in persons:
        if p is not main:
            po = np.maximum(po, p[2])
    if rule == "argmax":
        main_bin = ((pm > 0.5) & (pm >= po)).astype(np.uint8)
        other_bin = ((po > 0.5) & (po > pm)).astype(np.uint8)
    else:
        main_bin = (pm > 0.5).astype(np.uint8)
        other_bin = ((po > 0.5) & (pm < 0.5)).astype(np.uint8)
    main_bin = _fill_holes(main_bin)
    if largest_only:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(main_bin, connectivity=8)
        if n > 2:
            best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            main_bin = (lab == best).astype(np.uint8)

    def to_frame(m160):
        m = cv2.resize(m160.astype(np.float32), (640, 640), interpolation=cv2.INTER_LINEAR)
        m = m[py:py + nh, px:px + nw]
        return cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)

    gate = (to_frame(main_bin) > 0.5).astype(np.uint8)
    other = to_frame(other_bin) > 0.5
    k = max(3, int(round(grow * max(h, w) / 640)))
    gate = cv2.dilate(gate, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    gate[other] = 0                       # never let a confidently-other-person pixel through
    gate = gate.astype(np.float32)
    if feather:
        gate = cv2.GaussianBlur(gate, (0, 0), feather)
    return gate


def keep_main_subject(alpha, thresh=0.5, grow=7):
    """Fallback gate without YOLO: keep the largest blob touching the bottom edge."""
    binm = (alpha > thresh).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
    if n <= 2:
        return alpha
    touching = set(np.unique(lab[-1, :]).tolist()) - {0}
    cands = touching if touching else set(range(1, n))
    best = max(cands, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    keep = cv2.dilate((lab == best).astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))
    return alpha * keep


# ----------------------------------------------------------------------------- public
def mask(frame_bgr, matte=None, gate=None):
    """float32 HxW alpha in [0,1]; 1 = the main person (speaker)."""
    matte = matte or MATTE
    gate = GATE if gate is None else gate
    m = modnet_matte(frame_bgr) if matte == "modnet" else mediapipe_matte(frame_bgr)
    if REFINE == "1" or (REFINE == "auto" and matte == "mediapipe"):
        m = refine(m, frame_bgr)
    g = main_person_gate(frame_bgr) if gate else None
    m = m * g if g is not None else keep_main_subject(m)
    return np.clip(m, 0.0, 1.0).astype(np.float32)


if __name__ == "__main__":
    import glob, time, sys
    frames = [cv2.imread(f) for f in sorted(glob.glob(os.path.join(HERE, "frames", "*.png")))]
    test = cv2.resize(frames[0], (640, 480))
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40

    def bench(label, fn, warm=5):
        for _ in range(warm):
            fn()
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1000)
        ts = np.array(ts)
        print(f"{label:44s} @640x480: median {np.median(ts):6.2f} ms  min {ts.min():6.2f}  "
              f"p90 {np.percentile(ts, 90):6.2f}")

    bench("yolo gate only (CoreML EP)", lambda: main_person_gate(test))
    bench(f"modnet matte only (size {MATTE_SIZE})", lambda: modnet_matte(test))
    bench("modnet matte only (size 384)", lambda: modnet_matte(test, 384))
    bench("mediapipe matte only", lambda: mediapipe_matte(test))
    bench("mediapipe matte + guided filter", lambda: refine(mediapipe_matte(test), test))
    bench(f"mask(modnet {MATTE_SIZE} + gate)  [default]", lambda: mask(test, matte="modnet"))
    bench("mask(mediapipe + refine + gate)", lambda: mask(test, matte="mediapipe"))
