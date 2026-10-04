"""验证承重墙：Apple 神经引擎(ANE) 能否让口型渲染器完全避开 Metal GPU 争抢。

对比 4 种组合：{ANE, GPU} x {空闲, GPU被打满}。
模型 = Ultralight-Digital-Human 真实拓扑（MobileNetV2 InvertedResidual U-Net，
随机权重——只测时序，不测画质）。

判据：25fps 预算 = 40 ms/帧。关键看「被打满时」那一列的劣化倍数。
"""
import sys, time, threading, statistics
from pathlib import Path
import numpy as np, torch, coremltools as ct

sys.path.insert(0, str(Path(__file__).parent))

# coremltools 对 torch 2.14 的 cast op 有兼容问题，按 agent 的方式打补丁
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

XS, ASHAPE = (1, 6, 160, 160), (1, 16, 32, 32)   # hubert 模式
TARGET_MS = 40.0


class GPUFlood:
    """用 MLX 把 Metal GPU 打满（依赖链，防惰性求值消除）。"""
    def __init__(self): self.stop = threading.Event(); self.steps = 0
    def _run(self):
        import mlx.core as mx
        mx.set_default_device(mx.gpu)
        a = (mx.random.normal((4096, 4096)) * 0.01).astype(mx.float16)
        x = (mx.random.normal((4096, 4096)) * 0.01).astype(mx.float16)
        while not self.stop.is_set():
            for _ in range(28): x = (a @ x) * 0.01
            mx.eval(x); self.steps += 1
    def __enter__(self):
        self.t = threading.Thread(target=self._run, daemon=True); self.t.start()
        time.sleep(4.0); return self
    def __exit__(self, *a): self.stop.set(); self.t.join(timeout=15)


def measure(m, xin, iters=150):
    for _ in range(15): m.predict(xin)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); m.predict(xin); ts.append((time.perf_counter()-t0)*1000)
    ts.sort()
    return statistics.median(ts), ts[int(len(ts)*0.99)]


def main():
    net = Model(6, "hubert").eval()
    nparam = sum(p.numel() for p in net.parameters())
    macs = [0]; hs = []
    for mod in net.modules():
        if isinstance(mod, torch.nn.Conv2d):
            def hk(m_, i_, o_, _m=macs):
                _m[0] += m_.in_channels//m_.groups*m_.out_channels*m_.kernel_size[0]*m_.kernel_size[1]*o_.shape[2]*o_.shape[3]
            hs.append(mod.register_forward_hook(hk))
    with torch.no_grad(): net(torch.randn(*XS), torch.randn(*ASHAPE))
    for h in hs: h.remove()
    gflop = 2*macs[0]/1e9
    print(f"模型: Ultralight-Digital-Human 真实拓扑 160x160")
    print(f"      {nparam/1e6:.2f}M 参数, {gflop:.2f} GFLOP/帧\n")

    ex = torch.jit.trace(net, (torch.randn(*XS), torch.randn(*ASHAPE)))
    xin = {"x": np.random.randn(*XS).astype(np.float32),
           "a": np.random.randn(*ASHAPE).astype(np.float32)}

    models = {}
    for nm, cu in [("ANE (CPU_AND_NE)", ct.ComputeUnit.CPU_AND_NE),
                   ("GPU (CPU_AND_GPU)", ct.ComputeUnit.CPU_AND_GPU)]:
        models[nm] = ct.convert(ex, inputs=[ct.TensorType(name="x", shape=XS),
                                            ct.TensorType(name="a", shape=ASHAPE)],
                                convert_to="mlprogram", compute_units=cu,
                                compute_precision=ct.precision.FLOAT16,
                                minimum_deployment_target=ct.target.macOS15)

    print(f"{'放置':<20} {'空闲 (ms)':>14} {'GPU打满 (ms)':>16} {'劣化':>8} {'打满时fps':>11}")
    print("-"*74)
    idle = {}
    for nm, m in models.items():
        md, p99 = measure(m, xin); idle[nm] = md
    with GPUFlood() as f:
        for nm, m in models.items():
            md, p99 = measure(m, xin)
            deg = md/idle[nm]; fps = 1000/md
            flag = "OK" if md < TARGET_MS else "MISS"
            print(f"{nm:<20} {idle[nm]:>11.2f}    {md:>11.2f} (p99 {p99:.1f}) {deg:>6.2f}x {fps:>8.0f} [{flag}]")
        print(f"\n(争抢线程完成 {f.steps} 步 —— 确认 GPU 确实被打满)")
    print(f"\n25fps 预算 = {TARGET_MS:.0f} ms/帧")


if __name__ == "__main__":
    main()
