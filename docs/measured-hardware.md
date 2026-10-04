> Historical engineering record. For the current released architecture and setup, read [ARCHITECTURE.md](ARCHITECTURE.md) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md). Private logs and local session archives referenced below are not distributed.

# M5 Max 实测基线（2026-09-09）

在目标机器上实测，非厂商标称、非 CUDA 外推。所有循环使用依赖链，
避免 MLX 惰性求值把重复运算优化掉（第一次测量因此虚高 4 倍，已作废）。

| 指标 | 实测值 | 测法 |
|---|---|---|
| bf16 matmul | 20.0 TFLOPS | 4096³ 链式 ×50 |
| fp16 matmul | 14.4 TFLOPS | 同上 |
| fp32 matmul | 10.5 TFLOPS | 同上 |
| 有效内存带宽 | 426 GB/s | 1GB 缓冲链式读写 ×30 |

MLX 0.32.2 / mlx-lm 0.31.3 / Python 3.12.9

## 推论 1：扩散数字人不可能实时

RTX 4090 bf16 稠密张量核 ≈ 165 TFLOPS。比值 165/20 ≈ **8.3x**。

VoxEMW 作者在 4090 实测：SoulX-FlashHead Lite 生成 0.96s 画面耗时 0.27s。
换算到本机：0.27 × 8.3 ≈ **2.2s / 0.96s 画面 ≈ 0.43x 实时**，慢 2.3 倍。

且这是乐观上界——SoulX 走 PyTorch MPS 而非 MLX，扩散负载在 MPS 上的
实际效率远低于 matmul 峰值，真实数字预计在 3–5s/chunk。结论稳固：
**写实扩散数字人在本机做不到实时**，差的不是一点半点。

## 推论 2：本地大脑要选 MoE，不要选稠密

解码受内存带宽限制：tok/s ≈ 带宽 ÷ 每 token 激活字节数。按 426 GB/s：

| 模型形态 | 4bit 每 token 激活 | 理论上限 | 现实预期 |
|---|---|---|---|
| ~5B 激活 MoE (如 gpt-oss-120b) | ~2.9 GB | ~145 tok/s | 60–90 tok/s |
| ~22B 激活 MoE | ~12.4 GB | ~34 tok/s | 20–25 tok/s |
| 70B 稠密 | ~40 GB | ~10 tok/s | 6–8 tok/s |

语音对话只需跑赢说话速度（英语 ~3 词/秒 ≈ 4 tok/s）即可，但首句延迟才是
体感关键。70B 稠密 6–8 tok/s 边缘可用但没有余量；**MoE 是明确正解**——
128GB 装得下权重，而每 token 只激活一小部分，带宽压力小一个数量级。
