"""把 MuseTalk UNet（真实权重）导出为 Core ML，跑在神经引擎上；并与 MLX 参考输出核对。

用法:  .venv-coreml/bin/python scripts/export_unet_coreml.py --ref-dir DIR [--out assets/coreml/musetalk_unet_b1.mlpackage]
ref-dir 里需有 unet_x.npy / unet_pe.npy / unet_y.npy（由 MLX 侧生成）。
"""
import argparse, glob, os, statistics, threading, time
from pathlib import Path
import numpy as np, torch, coremltools as ct
from safetensors.numpy import load_file

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

from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor

UNET_CFG = dict(sample_size=32, in_channels=8, out_channels=4, layers_per_block=2,
                block_out_channels=[320, 640, 1280, 1280], attention_head_dim=8, cross_attention_dim=384,
                down_block_types=["CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"],
                up_block_types=["UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"],
                norm_num_groups=32, norm_eps=1e-5, act_fn="silu", flip_sin_to_cos=True, freq_shift=0)


class ANEGroupNorm(torch.nn.Module):
    """与 nn.GroupNorm 等价，但方差用 d*d 而不是 square：
    Core ML 计算计划显示 61 个 `square` 没有分到 ANE，会把整张图切成几十段来回搬运。"""
    def __init__(self, gn: torch.nn.GroupNorm):
        super().__init__()
        self.g, self.eps = gn.num_groups, gn.eps
        self.weight, self.bias = gn.weight, gn.bias
    def forward(self, x):
        B, C, H, W = x.shape
        xg = x.reshape(B, self.g, -1)
        m = xg.mean(-1, keepdim=True)
        d = xg - m
        v = (d * d).mean(-1, keepdim=True)
        y = (d * torch.rsqrt(v + self.eps)).reshape(B, C, H, W)
        return y * self.weight.view(1, C, 1, 1) + self.bias.view(1, C, 1, 1)


def replace_groupnorm(mod: torch.nn.Module) -> int:
    n = 0
    for name, child in list(mod.named_children()):
        if isinstance(child, torch.nn.GroupNorm):
            setattr(mod, name, ANEGroupNorm(child)); n += 1
        else:
            n += replace_groupnorm(child)
    return n


class Unet1(torch.nn.Module):
    def __init__(self, unet): super().__init__(); self.unet = unet
    def forward(self, x, enc):
        return self.unet(x, torch.zeros(1, dtype=torch.float32), encoder_hidden_states=enc).sample


def load_mlx_weights(unet):
    snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--mlx-community--MuseTalk-1.5-fp16/snapshots/*/unet.safetensors"))[0]
    w = load_file(snap); sd = unet.state_dict(); new = {}; bad = []
    for k, v in sd.items():
        if k not in w: bad.append(("missing", k)); continue
        a = w[k].astype(np.float32)
        if a.ndim == 4: a = a.transpose(0, 3, 1, 2)      # MLX (O,H,W,I) -> torch (O,I,H,W)
        if tuple(a.shape) != tuple(v.shape): bad.append(("shape", k, a.shape, tuple(v.shape))); continue
        new[k] = torch.from_numpy(a)
    extra = [k for k in w if k not in sd]
    print(f"权重映射: {len(new)}/{len(sd)} 载入, 问题 {bad[:4]}, MLX 多余 {extra[:4]} ({len(extra)})")
    unet.load_state_dict(new, strict=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--out", default="assets/coreml/musetalk_unet_b1.mlpackage")
    ap.add_argument("--ane-gn", action="store_true", help="把 GroupNorm 换成 ANE 友好实现")
    a = ap.parse_args()
    R = Path(a.ref_dir)
    x, pe, y = (np.load(R / f"unet_{n}.npy") for n in ("x", "pe", "y"))
    unet = UNet2DConditionModel(**UNET_CFG); unet.set_attn_processor(AttnProcessor()); unet.eval()
    load_mlx_weights(unet)
    if a.ane_gn:
        print(f"替换 GroupNorm: {replace_groupnorm(unet)} 处")
    mod = Unet1(unet)
    with torch.no_grad(): yt = mod(torch.from_numpy(x), torch.from_numpy(pe)).numpy()
    cos = lambda p, q: float((p.ravel() @ q.ravel()) / (np.linalg.norm(p) * np.linalg.norm(q) + 1e-9))
    print(f"torch fp32 vs MLX: cos {cos(yt, y):.5f}  max|Δ| {np.abs(yt-y).max():.4f}  (|y| 均值 {np.abs(y).mean():.4f})")

    t0 = time.perf_counter()
    ex = torch.jit.trace(mod, (torch.from_numpy(x), torch.from_numpy(pe)))
    m = ct.convert(ex, inputs=[ct.TensorType(name="x", shape=x.shape, dtype=np.float16),
                               ct.TensorType(name="enc", shape=pe.shape, dtype=np.float16)],
                   outputs=[ct.TensorType(name="y", dtype=np.float16)], convert_to="mlprogram",
                   compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS15,
                   compute_units=ct.ComputeUnit.CPU_AND_NE)
    m.save(a.out)
    print(f"转换+保存 {time.perf_counter()-t0:.0f}s → {a.out}")
    feed = {"x": x.astype(np.float16), "enc": pe.astype(np.float16)}
    yc = m.predict(feed)["y"].astype(np.float32)
    print(f"CoreML(ANE) vs MLX: cos {cos(yc, y):.5f}  max|Δ| {np.abs(yc-y).max():.4f}")
    for _ in range(5): m.predict(feed)
    ts = []
    for _ in range(200):
        t = time.perf_counter(); m.predict(feed); ts.append((time.perf_counter()-t)*1000)
    print(f"ANE 单帧: 前40次中位 {statistics.median(ts[:40]):.1f} ms · 第160-200次中位 {statistics.median(ts[160:]):.1f} ms · p90 {sorted(ts)[180]:.1f} ms")
    # 计算计划：还有多少算子不在 ANE
    try:
        from coremltools.models.compute_plan import MLComputePlan
        import collections
        plan = MLComputePlan.load_from_path(ct.models.utils.compile_model(a.out), compute_units=ct.ComputeUnit.CPU_AND_NE)
        cnt = collections.Counter()
        def walk(block):
            for op in block.operations:
                if op.operator_name == "const": continue
                u = plan.get_compute_device_usage_for_mlprogram_operation(op)
                cnt[type(u.preferred_compute_device).__name__ if u else f"None:{op.operator_name}"] += 1
                for b in op.blocks: walk(b)
        walk(plan.model_structure.program.functions["main"].block)
        print("计算计划（非 const 算子按设备）:", dict(cnt))
    except Exception as e:
        print("计算计划读取失败:", repr(e)[:120])
    # GIL 测试：predict 在子线程跑时，主线程还能不能干活
    stop = threading.Event(); n = [0]
    def worker():
        while not stop.is_set(): m.predict(feed); n[0] += 1
    th = threading.Thread(target=worker, daemon=True); th.start()
    cnt = 0; t = time.perf_counter()
    while time.perf_counter() - t < 2.0: cnt += 1; _ = sum(range(1000))
    stop.set(); th.join()
    print(f"GIL: 子线程 predict {n[0]} 次/2s 期间主线程完成 {cnt} 次小循环（{'释放 GIL ✓' if cnt > 20000 else '持有 GIL ✗ 需要子进程'}）")


if __name__ == "__main__":
    main()
