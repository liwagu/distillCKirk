"""用 Apple ml-stable-diffusion 的 ANE 友好 UNet 结构导出 MuseTalk UNet（真实权重）。

与 export_unet_coreml.py 的区别：那个用 diffusers 原版结构（Linear、rank-3 张量），
这个用 Apple 为神经引擎重写的结构（1x1 Conv 代替 Linear、(B,C,1,S) 布局、SPLIT_EINSUM 注意力）。
Apple 的说明：同样的 SD UNet，ANE 友好结构在神经引擎上快数倍。这里验证在 MuseTalk 上是否成立。

用法:  .venv-coreml/bin/python scripts/export_unet_ane.py --ref-dir DIR [--out assets/coreml/musetalk_unet_ane_b1.mlpackage]
"""
import argparse, collections, glob, os, statistics, sys, time
from pathlib import Path
import numpy as np, torch, coremltools as ct
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ref-ml-stable-diffusion"))

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

from python_coreml_stable_diffusion import unet as ane_unet

UNET_CFG = dict(in_channels=8, out_channels=4, layers_per_block=2,
                block_out_channels=(320, 640, 1280, 1280), attention_head_dim=8, cross_attention_dim=384,
                down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
                up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
                norm_num_groups=32, norm_eps=1e-5, act_fn="silu", flip_sin_to_cos=True, freq_shift=0)


class Wrap(torch.nn.Module):
    def __init__(self, u): super().__init__(); self.u = u
    def forward(self, x, enc):
        return self.u(x, torch.zeros(1, dtype=torch.float32), enc)[0]


def mlx_state_dict():
    snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--mlx-community--MuseTalk-1.5-fp16/snapshots/*/unet.safetensors"))[0]
    sd = {}
    for k, v in load_file(snap).items():
        a = v.astype(np.float32)
        if a.ndim == 4: a = a.transpose(0, 3, 1, 2)
        sd[k] = torch.from_numpy(a)
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--out", default="assets/coreml/musetalk_unet_ane_b1.mlpackage")
    ap.add_argument("--attn", default="SPLIT_EINSUM", choices=["ORIGINAL", "SPLIT_EINSUM", "SPLIT_EINSUM_V2"])
    a = ap.parse_args()
    ane_unet.ATTENTION_IMPLEMENTATION_IN_EFFECT = ane_unet.AttentionImplementations[a.attn]
    R = Path(a.ref_dir)
    x, pe, y = (np.load(R / f"unet_{n}.npy") for n in ("x", "pe", "y"))
    u = ane_unet.UNet2DConditionModel(**UNET_CFG).eval()
    res = u.load_state_dict(mlx_state_dict(), strict=False)
    print(f"权重载入: missing {len(res.missing_keys)} {res.missing_keys[:3]} | unexpected {len(res.unexpected_keys)} {res.unexpected_keys[:3]}")
    enc = np.ascontiguousarray(pe.transpose(0, 2, 1)[:, :, None, :])          # (1,50,384) → (1,384,1,50)
    mod = Wrap(u)
    with torch.no_grad(): yt = mod(torch.from_numpy(x), torch.from_numpy(enc)).numpy()
    cos = lambda p, q: float((p.ravel() @ q.ravel()) / (np.linalg.norm(p) * np.linalg.norm(q) + 1e-9))
    print(f"torch(ANE结构) fp32 vs MLX: cos {cos(yt, y):.5f}  max|Δ| {np.abs(yt-y).max():.4f}")
    t0 = time.perf_counter()
    ex = torch.jit.trace(mod, (torch.from_numpy(x), torch.from_numpy(enc)))
    m = ct.convert(ex, inputs=[ct.TensorType(name="x", shape=x.shape, dtype=np.float16),
                               ct.TensorType(name="enc", shape=enc.shape, dtype=np.float16)],
                   outputs=[ct.TensorType(name="y", dtype=np.float16)], convert_to="mlprogram",
                   compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS15,
                   compute_units=ct.ComputeUnit.CPU_AND_NE)
    m.save(a.out)
    print(f"转换+保存 {time.perf_counter()-t0:.0f}s → {a.out}")
    feed = {"x": x.astype(np.float16), "enc": enc.astype(np.float16)}
    yc = m.predict(feed)["y"].astype(np.float32)
    print(f"CoreML(ANE) vs MLX: cos {cos(yc, y):.5f}  max|Δ| {np.abs(yc-y).max():.4f}")
    for _ in range(5): m.predict(feed)
    ts = []
    for _ in range(200):
        t = time.perf_counter(); m.predict(feed); ts.append((time.perf_counter()-t)*1000)
    print(f"ANE 单帧: 前40次中位 {statistics.median(ts[:40]):.1f} ms · 第160-200次中位 {statistics.median(ts[160:]):.1f} ms · p90 {sorted(ts)[180]:.1f} ms")
    try:
        from coremltools.models.compute_plan import MLComputePlan
        plan = MLComputePlan.load_from_path(ct.models.utils.compile_model(a.out), compute_units=ct.ComputeUnit.CPU_AND_NE)
        cnt = collections.Counter()
        def walk(block):
            for op in block.operations:
                if op.operator_name == "const": continue
                usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
                cnt[type(usage.preferred_compute_device).__name__ if usage else f"None:{op.operator_name}"] += 1
                for b in op.blocks: walk(b)
        walk(plan.model_structure.program.functions["main"].block)
        print("计算计划（非 const 算子按设备）:", dict(cnt))
    except Exception as e:
        print("计算计划读取失败:", repr(e)[:120])


if __name__ == "__main__":
    main()
