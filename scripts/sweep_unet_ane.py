"""ANE UNet 提速扫描：权重调色板量化(8/6/4 bit)、批大小、部署目标。
每项打印：单帧延迟（前 40 次中位 / 持续中位）、与 MLX 参考的余弦。
用法: .venv-coreml/bin/python scripts/sweep_unet_ane.py --ref-dir DIR
"""
import argparse, statistics, sys, time
from pathlib import Path
import numpy as np, torch, coremltools as ct
from coremltools.optimize.coreml import OpPalettizerConfig, OptimizationConfig, palettize_weights
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ref-ml-stable-diffusion")); sys.path.insert(0, str(ROOT / "scripts"))
import export_unet_ane as E

def timeit(m, feed, n=120):
    for _ in range(5): m.predict(feed)
    ts = []
    for _ in range(n):
        t = time.perf_counter(); m.predict(feed); ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts[:40]), statistics.median(ts[-40:])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ref-dir", required=True); a = ap.parse_args()
    R = Path(a.ref_dir); x, pe, y = (np.load(R / f"unet_{n}.npy") for n in ("x", "pe", "y"))
    enc = np.ascontiguousarray(pe.transpose(0, 2, 1)[:, :, None, :])
    cos = lambda p, q: float((p.ravel() @ q.ravel()) / (np.linalg.norm(p) * np.linalg.norm(q) + 1e-9))
    base = ct.models.MLModel(str(ROOT / "assets/coreml/musetalk_unet_ane_b1.mlpackage"), compute_units=ct.ComputeUnit.CPU_AND_NE)
    feed = {"x": x.astype(np.float16), "enc": enc.astype(np.float16)}
    print(f"基线 fp16 B=1: {timeit(base, feed)} ms  cos {cos(base.predict(feed)['y'].astype(np.float32), y):.5f}", flush=True)
    # 1) 调色板量化
    for nbits in (8, 6, 4):
        t0 = time.perf_counter()
        cfg = OptimizationConfig(global_config=OpPalettizerConfig(mode="kmeans", nbits=nbits, granularity="per_grouped_channel", group_size=16))
        try:
            q = palettize_weights(base, cfg)
        except Exception as e:
            print(f"{nbits}bit 量化失败: {repr(e)[:150]}", flush=True); continue
        p = R / f"unet_ane_b1_p{nbits}.mlpackage"; q.save(str(p))
        q = ct.models.MLModel(str(p), compute_units=ct.ComputeUnit.CPU_AND_NE)
        yc = q.predict(feed)["y"].astype(np.float32)
        print(f"{nbits}bit 调色板 B=1: {timeit(q, feed)} ms  cos {cos(yc, y):.5f}  max|Δ| {np.abs(yc-y).max():.3f}  (量化 {time.perf_counter()-t0:.0f}s)", flush=True)
    # 2) 批大小
    u = E.ane_unet.UNet2DConditionModel(**E.UNET_CFG).eval(); u.load_state_dict(E.mlx_state_dict(), strict=False)
    mod = E.Wrap(u)
    for B in (2, 4):
        xb = np.repeat(x, B, 0); eb = np.repeat(enc, B, 0)
        ex = torch.jit.trace(mod, (torch.from_numpy(xb), torch.from_numpy(eb)))
        m = ct.convert(ex, inputs=[ct.TensorType(name="x", shape=xb.shape, dtype=np.float16), ct.TensorType(name="enc", shape=eb.shape, dtype=np.float16)],
                       outputs=[ct.TensorType(name="y", dtype=np.float16)], convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                       minimum_deployment_target=ct.target.macOS15, compute_units=ct.ComputeUnit.CPU_AND_NE)
        p = R / f"unet_ane_b{B}.mlpackage"; m.save(str(p)); m = ct.models.MLModel(str(p), compute_units=ct.ComputeUnit.CPU_AND_NE)
        fb = {"x": xb.astype(np.float16), "enc": eb.astype(np.float16)}
        a1, a2 = timeit(m, fb, 60)
        print(f"fp16 B={B}: {a1/B:.1f} / {a2/B:.1f} ms/帧", flush=True)
    # 3) 部署目标 macOS26
    try:
        ex = torch.jit.trace(mod, (torch.from_numpy(x), torch.from_numpy(enc)))
        m = ct.convert(ex, inputs=[ct.TensorType(name="x", shape=x.shape, dtype=np.float16), ct.TensorType(name="enc", shape=enc.shape, dtype=np.float16)],
                       outputs=[ct.TensorType(name="y", dtype=np.float16)], convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                       minimum_deployment_target=ct.target.macOS26, compute_units=ct.ComputeUnit.CPU_AND_NE)
        p = R / "unet_ane_b1_os26.mlpackage"; m.save(str(p)); m = ct.models.MLModel(str(p), compute_units=ct.ComputeUnit.CPU_AND_NE)
        print(f"fp16 B=1 target=macOS26: {timeit(m, feed)} ms  cos {cos(m.predict(feed)['y'].astype(np.float32), y):.5f}", flush=True)
    except Exception as e:
        print("macOS26 目标失败:", repr(e)[:150])

if __name__ == "__main__":
    main()
