"""MuseTalk 两个重模块能不能搬上神经引擎(ANE)？

背景：GPU 持续负载态下 UNet 27.7 + VAE解码 40 ≈ 69 ms/帧 → 14.5 fps，达不到 25。
若 VAE 解码（甚至 UNet）能放到 ANE 并行，GPU 只剩 UNet → 36 fps。

拓扑与 mlx-community/MuseTalk-1.5-fp16 完全一致（diffusers 命名，权重随机——只测时序）：
  VAE 解码器 = sd-vae-ft-mse AutoencoderKL.decoder (+post_quant_conv)，(B,4,32,32)→(B,3,256,256)
  UNet       = SD1.x UNet2DConditionModel(in=8,out=4,cross_attn=384)，单步 t=0

用法（须用带 torch/coremltools 的环境）:
    .venv-coreml/bin/python scripts/bench_ane_musetalk.py [--skip-unet] [--out DIR]
"""
import argparse, statistics, sys, threading, time
from pathlib import Path
import numpy as np, torch, coremltools as ct

# coremltools 9 对 torch 2.14 cast op 的兼容补丁（同 bench_ane.py）
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

from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor

VAE_CFG = dict(in_channels=3, out_channels=3, down_block_types=["DownEncoderBlock2D"] * 4,
               up_block_types=["UpDecoderBlock2D"] * 4, block_out_channels=[128, 256, 512, 512],
               layers_per_block=2, act_fn="silu", latent_channels=4, norm_num_groups=32, sample_size=256)
UNET_CFG = dict(sample_size=32, in_channels=8, out_channels=4, layers_per_block=2,
                block_out_channels=[320, 640, 1280, 1280], attention_head_dim=8, cross_attention_dim=384,
                down_block_types=["CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"],
                up_block_types=["UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"],
                norm_num_groups=32, norm_eps=1e-5, act_fn="silu", flip_sin_to_cos=True, freq_shift=0)


class Dec(torch.nn.Module):
    def __init__(self, vae): super().__init__(); self.vae = vae
    def forward(self, z): return self.vae.decoder(self.vae.post_quant_conv(z))


class Unet1(torch.nn.Module):
    def __init__(self, unet): super().__init__(); self.unet = unet
    def forward(self, x, enc):
        return self.unet(x, torch.zeros(1, dtype=torch.float32), encoder_hidden_states=enc).sample


class GPUFlood:
    """用 MLX 把 Metal GPU 打满（依赖链）。"""
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
        self.t = threading.Thread(target=self._run, daemon=True); self.t.start(); time.sleep(4.0); return self
    def __exit__(self, *a): self.stop.set(); self.t.join(timeout=15)


def convert(mod, inputs, path, units):
    t0 = time.perf_counter()
    ex = torch.jit.trace(mod.eval(), tuple(torch.randn(*s) for s in inputs))
    m = ct.convert(ex, inputs=[ct.TensorType(shape=s, dtype=np.float16) for s in inputs],
                   outputs=[ct.TensorType(dtype=np.float16)], convert_to="mlprogram",
                   compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS15,
                   compute_units=units)
    m.save(str(path))
    print(f"    转换 {time.perf_counter()-t0:.0f}s → {path.name}", flush=True)
    return m


def measure(m, feed, iters=30, warm=5):
    for _ in range(warm): m.predict(feed)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); m.predict(feed); ts.append((time.perf_counter() - t0) * 1000)
    ts.sort(); return statistics.median(ts), ts[int(len(ts) * 0.9)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-unet", action="store_true")
    ap.add_argument("--out", default="/tmp/ane_musetalk")
    ap.add_argument("--batches", default="1,4")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    batches = [int(b) for b in a.batches.split(",")]
    NE, CPU = ct.ComputeUnit.CPU_AND_NE, ct.ComputeUnit.CPU_ONLY

    vae = AutoencoderKL(**VAE_CFG); vae.set_attn_processor(AttnProcessor())
    unet = UNet2DConditionModel(**UNET_CFG); unet.set_attn_processor(AttnProcessor())
    npar = lambda m: sum(p.numel() for p in m.parameters()) / 1e6
    print(f"VAE 解码器 {npar(vae.decoder):.1f}M 参数 · UNet {npar(unet):.1f}M 参数")
    # 形状核对：与 MLX 权重一致（MLX 卷积为 O,H,W,I）
    print("  torch conv_in", tuple(vae.decoder.conv_in.weight.shape), "| unet attn2.to_k",
          tuple(unet.down_blocks[0].attentions[0].transformer_blocks[0].attn2.to_k.weight.shape),
          "(MLX: (512,3,3,4) / (320,384))")

    jobs = [("VAE解码", Dec(vae), lambda B: [(B, 4, 32, 32)], lambda B: {"z": np.random.randn(B, 4, 32, 32).astype(np.float16)})]
    if not a.skip_unet:
        jobs.append(("UNet", Unet1(unet), lambda B: [(B, 8, 32, 32), (B, 50, 384)],
                     lambda B: {"x": np.random.randn(B, 8, 32, 32).astype(np.float16),
                                "enc": np.random.randn(B, 50, 384).astype(np.float16)}))
    best = {}
    for name, mod, shapes, feed in jobs:
        print(f"\n== {name}")
        for B in batches:
            p = out / f"{name}_b{B}.mlpackage"
            m = convert(mod, shapes(B), p, NE)
            f = feed(B)
            # Core ML 输入名按转换结果
            names = [i.name for i in m.get_spec().description.input]
            f = dict(zip(names, f.values()))
            med, p90 = measure(m, f)
            print(f"    B={B} ANE(CPU_AND_NE): {med/B:6.1f} ms/帧 (p90 {p90/B:.1f})  → {1000/(med/B):5.1f} fps", flush=True)
            if B == batches[0]:
                mc = ct.models.MLModel(str(p), compute_units=CPU)
                medc, _ = measure(mc, f, iters=5, warm=1)
                print(f"    B={B} CPU_ONLY 对照:     {medc/B:6.1f} ms/帧  （ANE 快 {medc/med:.1f}x → {'确在 ANE' if medc/med > 3 else '疑似未上 ANE'}）", flush=True)
            best[name] = (m, f, B, med)
    # GPU 打满时是否免疫
    print("\n== GPU 被 MLX 打满时（真实工作条件）")
    with GPUFlood() as g:
        for name, (m, f, B, med0) in best.items():
            med, p90 = measure(m, f, iters=20)
            print(f"    {name} B={B}: {med/B:6.1f} ms/帧 (p90 {p90/B:.1f})  劣化 {med/med0:.2f}x", flush=True)
        print(f"    争抢线程 {g.steps} 步")


if __name__ == "__main__":
    main()
