"""ANE UNet 权重压缩实验：int8 线性量化（快）与 6/4-bit k-means 调色板。看延迟是否随权重字节数下降。"""
import statistics, sys, time
from pathlib import Path
import numpy as np, coremltools as ct
from coremltools.optimize.coreml import (OpLinearQuantizerConfig, OpPalettizerConfig, OptimizationConfig,
                                         linear_quantize_weights, palettize_weights)
ROOT = Path(__file__).resolve().parent.parent
R = Path(sys.argv[1])
x, pe, y = (np.load(R / f"unet_{n}.npy") for n in ("x", "pe", "y"))
enc = np.ascontiguousarray(pe.transpose(0, 2, 1)[:, :, None, :])
feed = {"x": x.astype(np.float16), "enc": enc.astype(np.float16)}
cos = lambda p, q: float((p.ravel() @ q.ravel()) / (np.linalg.norm(p) * np.linalg.norm(q) + 1e-9))
def timeit(m, n=120):
    for _ in range(5): m.predict(feed)
    ts = []
    for _ in range(n):
        t = time.perf_counter(); m.predict(feed); ts.append((time.perf_counter() - t) * 1000)
    return f"{statistics.median(ts[:40]):.1f}/{statistics.median(ts[-40:]):.1f} ms"
base = ct.models.MLModel(str(ROOT / "assets/coreml/musetalk_unet_ane_b1.mlpackage"), compute_units=ct.ComputeUnit.CPU_AND_NE)
print("基线 fp16:", timeit(base), flush=True)
jobs = [("int8 线性(per_block 32)", lambda: linear_quantize_weights(base, OptimizationConfig(global_config=OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8", granularity="per_block", block_size=32)))),
        ("6bit kmeans 调色板(per_grouped_channel 16)", lambda: palettize_weights(base, OptimizationConfig(global_config=OpPalettizerConfig(mode="kmeans", nbits=6, granularity="per_grouped_channel", group_size=16)))),
        ("4bit kmeans 调色板(per_grouped_channel 16)", lambda: palettize_weights(base, OptimizationConfig(global_config=OpPalettizerConfig(mode="kmeans", nbits=4, granularity="per_grouped_channel", group_size=16))))]
for label, fn in jobs:
    t0 = time.perf_counter()
    try:
        q = fn()
    except Exception as e:
        print(f"{label}: 失败 {repr(e)[:160]}", flush=True); continue
    p = R / (label.split()[0] + ".mlpackage"); q.save(str(p))
    q = ct.models.MLModel(str(p), compute_units=ct.ComputeUnit.CPU_AND_NE)
    yc = q.predict(feed)["y"].astype(np.float32)
    print(f"{label}: {timeit(q)}  cos {cos(yc, y):.5f} max|Δ| {np.abs(yc-y).max():.3f}  (压缩耗时 {time.perf_counter()-t0:.0f}s)", flush=True)
