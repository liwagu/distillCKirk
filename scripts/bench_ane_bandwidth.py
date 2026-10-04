"""自查：ANE 的 1.00x 结论是否只在「算力型争抢」下成立？

上一轮用 4096² matmul 打满 GPU —— 那是算力密集型。
真实 LLM 解码是**内存带宽密集**型（持续从 DRAM 流式读权重），
而 ANE 与 GPU 共享同一条 DRAM 带宽。若瓶颈在带宽，ANE 未必免疫。

本脚本对比三种争抢：
  1. 无     2. 算力型(matmul)     3. 带宽型(大向量流式读写，模拟 LLM 解码)
再加一个 CPU 打满条件 —— 因为 CoreML predict 走进程间通信，CPU 饱和可能放大开销。
"""
import sys, time, threading, statistics, os
from pathlib import Path
import numpy as np, torch, coremltools as ct
sys.path.insert(0, str(Path(__file__).parent))
import coremltools.converters.mil.frontend.torch.ops as _ops
from coremltools.converters.mil.mil import Builder as _mb
def _cast_fix(context, node, dtype, dtype_str):
    x = context[node.inputs[0]]
    if x.val is not None:
        v = x.val
        if hasattr(v, "ndim") and v.ndim > 0: v = v.reshape(-1)[0]
        res = _mb.const(val=dtype(v), name=node.name)
    else:
        res = _mb.cast(x=x, dtype=dtype_str, name=node.name)
    context.add(res)
_ops._cast = _cast_fix
from real_ul_unet import Model

XS, ASHAPE = (1, 6, 160, 160), (1, 16, 32, 32)


class Load(threading.Thread):
    def __init__(self, kind):
        super().__init__(daemon=True); self.kind = kind; self.stop = threading.Event(); self.n = 0
    def run(self):
        import mlx.core as mx
        mx.set_default_device(mx.gpu)
        if self.kind == "compute":
            a = (mx.random.normal((4096, 4096))*0.01).astype(mx.float16)
            x = (mx.random.normal((4096, 4096))*0.01).astype(mx.float16)
            while not self.stop.is_set():
                for _ in range(28): x = (a @ x)*0.01
                mx.eval(x); self.n += 1
        elif self.kind == "bandwidth":
            # 模拟 LLM 解码：反复流式读写 ~8GB 权重（远超缓存，纯 DRAM 带宽）
            bufs = [mx.random.normal((256*1024*1024,)).astype(mx.float16) for _ in range(8)]
            mx.eval(*bufs)
            while not self.stop.is_set():
                s = None
                for b in bufs:
                    v = mx.sum(b)                      # 全量读
                    s = v if s is None else s + v
                mx.eval(s); self.n += 1
        elif self.kind == "cpu":
            import numpy as _np
            while not self.stop.is_set():
                a = _np.random.rand(1200, 1200); _ = a @ a; self.n += 1


def measure(m, xin, iters=120):
    for _ in range(15): m.predict(xin)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); m.predict(xin); ts.append((time.perf_counter()-t0)*1000)
    ts.sort()
    return statistics.median(ts), ts[int(len(ts)*0.99)]


def main():
    net = Model(6, "hubert").eval()
    ex = torch.jit.trace(net, (torch.randn(*XS), torch.randn(*ASHAPE)))
    xin = {"x": np.random.randn(*XS).astype(np.float32),
           "a": np.random.randn(*ASHAPE).astype(np.float32)}
    models = {}
    for nm, cu in [("ANE", ct.ComputeUnit.CPU_AND_NE), ("GPU", ct.ComputeUnit.CPU_AND_GPU)]:
        models[nm] = ct.convert(ex, inputs=[ct.TensorType(name="x", shape=XS),
                                            ct.TensorType(name="a", shape=ASHAPE)],
                                convert_to="mlprogram", compute_units=cu,
                                compute_precision=ct.precision.FLOAT16,
                                minimum_deployment_target=ct.target.macOS15)
    ncpu = os.cpu_count()
    print(f"{'争抢类型':<28} {'ANE ms':>10} {'ANE p99':>9} {'GPU ms':>10} {'GPU p99':>9}")
    print("-"*70)
    base = {}
    for kind in [None, "compute", "bandwidth", "cpu"]:
        threads = []
        if kind == "cpu":
            threads = [Load("cpu") for _ in range(max(1, ncpu-2))]
        elif kind:
            threads = [Load(kind)]
        for t in threads: t.start()
        if threads: time.sleep(4)
        row = {}
        for nm, m in models.items():
            md, p99 = measure(m, xin); row[nm] = (md, p99)
        for t in threads: t.stop.set()
        for t in threads: t.join(timeout=15)
        label = {None: "无（基线）", "compute": "算力型 (matmul)",
                 "bandwidth": "带宽型 (模拟LLM解码)", "cpu": f"CPU 打满 ({max(1,ncpu-2)}线程)"}[kind]
        if kind is None: base = {k: v[0] for k, v in row.items()}
        sfx = "" if kind is None else f"   [ANE {row['ANE'][0]/base['ANE']:.2f}x, GPU {row['GPU'][0]/base['GPU']:.2f}x]"
        print(f"{label:<28} {row['ANE'][0]:>10.2f} {row['ANE'][1]:>9.2f} "
              f"{row['GPU'][0]:>10.2f} {row['GPU'][1]:>9.2f}{sfx}")
    print("\n25fps 预算 = 40 ms/帧")


if __name__ == "__main__":
    main()
