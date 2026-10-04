> Historical engineering record. For the current released architecture and setup, read [ARCHITECTURE.md](ARCHITECTURE.md) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md). Private logs and local session archives referenced below are not distributed.

# ANE 验证：口型渲染器可完全避开 GPU 争抢（2026-09-09 本机实测）

脚本 `scripts/bench_ane.py`。模型为 Ultralight-Digital-Human 真实拓扑
（MobileNetV2 InvertedResidual U-Net，12.16M 参数，6.33 GFLOP/帧 @160×160，
随机权重——只测时序不测画质）。Core ML mlprogram fp16，batch 1，150 次中位数。

| 放置 | 空闲 | GPU 被 MLX 打满 | 劣化 | 打满时 fps |
|---|---|---|---|---|
| **ANE (CPU_AND_NE)** | 1.49 ms | **1.49 ms** (p99 1.7) | **1.00x** | **673** |
| Metal GPU (CPU_AND_GPU) | 1.25 ms | 17.92 ms (p99 37.8) | 14.28x | 56 |

## 结论

**ANE 对 Metal GPU 争抢完全免疫**，因为它是物理独立的加速器。
25fps 下渲染器只占 ANE 时间的 3.7%（25 × 1.49ms = 37ms/秒），等同于免费。

有效算力 6.33 GFLOP ÷ 1.49 ms = **4.25 TFLOPS**。

对照：同一模型放 Metal GPU 上，被打满时劣化 14.28x、p99 37.8ms 已贴到
40ms 预算边缘。这与 MuseTalk 实测（争抢下掉到 1.7–3.4 fps）方向一致——
**任何走 Metal 的渲染方案在本项目里都不成立**。

## 这条结论改写的架构约束

原以为的约束：「给渲染器分配 30–50% GPU」。
**正确的约束：播放期间渲染器不得消耗 Metal 计算单元。**

原因是音频而非视频：实测 TTS 在 LLM 负载下 RTF 已达 0.92，距离 1.0 只剩 8%。
再叠加一个 GPU 渲染器就会越过 1.0 → **音频断流**。
掉一帧画面看不见，辩论中途音频断一下是致命的。

因此渲染器上 ANE 不是优化，是**可行性前提**。

## 待解决的真正风险：音频前瞻窗口

不是速度。计算问题已经解决（23 倍余量）。

FeatherTalk 的 `face_utils.py` 设 `AUDIO_HALF_WINDOW = 10`，注释写明
「取 [i-10, i+9] 共20个视频帧」——对称窗口需要约 9 个未来帧 ≈ **360–400 ms
未来音频**才能渲染第 i 帧。

好消息：本架构中音频是 TTS **提前生成好**的（RTF 0.35 < 1），未来音频天然存在，
所以这不是可行性问题，只是延迟成本。且模型逐帧无状态，**打断时一帧即停**。

缓解手段（ITU-R BT.1359-1）：音频**滞后**画面 125 ms 内人眼察觉不到，
但音频**超前**画面 45 ms 就能察觉。所以应当延后音频播出来对齐，
免费买回约 125 ms 的前瞻预算。**永远不要让声音跑在嘴前面。**

剩余方案：重训一个因果/短右窗版本（没人做过），或退回预计算口型库。

---

# 第二轮自查：换三种争抢类型复测（2026-09-10）

第一轮只用 4096² matmul 打满 GPU —— 那是**算力密集**型。但真实 LLM 解码是
**带宽密集**型，且 ANE 与 GPU 共享同一条 DRAM 带宽。原测试可能恰好避开了
真正要命的争抢模式。脚本 `scripts/bench_ane_bandwidth.py`。

| 争抢类型 | ANE 中位 | ANE p99 | GPU 中位 | GPU p99 |
|---|---|---|---|---|
| 无（基线） | 1.81 | 2.08 | 1.45 | 1.81 |
| 算力型 (matmul) | 2.49 (1.38x) | 2.81 | 77.24 (53.2x) | 137.79 |
| 带宽型（模拟 LLM 解码） | 2.03 (1.12x) | 3.94 | 3.89 (2.68x) | 23.13 |
| CPU 打满（16 线程） | 2.77 (1.53x) | **14.17** | 8.58 (5.91x) | 94.00 |

## 对第一轮结论的三处修正

1. **ANE 优势成立但没那么绝对**：不是 1.00x，实际 1.12–1.53x。
   最差中位 2.77 ms，对 40 ms 预算仍有 14 倍余量。结论方向不变。

2. **新增硬性约束：必须给 CPU 留余量。**
   CPU 打满时 ANE 的 p99 从 2.08 飙到 14.17 ms（约 7 倍）。原因是 CoreML
   `predict()` 走进程间通信，CPU 被榨干时 IPC 尾延迟爆炸。中位数看不出来，
   但视频流畅度取决于 p99。
   → 编排层必须限制并发线程数，**预留 2–4 个核心**给 CoreML 的 IPC 路径。
   → ASR/VAD/SmartTurn 都跑 CPU，需统一做线程预算，不能各自 `os.cpu_count()`。

3. **第一轮的「GPU 劣化 14.28x」不稳健。**
   实际随负载类型在 2.68x–53.2x 间大幅波动，单点测量不足以支撑结论。
   但即使取最温和的 2.68x，叠加 MuseTalk 实测（争抢下 3.4 fps），
   走 Metal 的渲染方案依然出局。结论不变，但依据要换成 MuseTalk 那组直接实测，
   不要引用这个波动很大的比值。
