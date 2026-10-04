> Historical engineering record. For the current released architecture and setup, read [ARCHITECTURE.md](ARCHITECTURE.md) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md). Private logs and local session archives referenced below are not distributed.

# MuseTalk-MLX 本机实测（2026-09-09，M5 Max 128GB）

脚本：`scripts/bench_musetalk.py`。权重 `mlx-community/MuseTalk-1.5-fp16`。
25fps 预算 = 40 ms/帧。

## 结果

| 阶段 | GPU 空闲 | 有 LLM 级负载争抢 | 劣化 |
|---|---|---|---|
| VAE 编码（**可离线预计算**） | 47.0 ms/帧 | 327.4 ms/帧 | 7.0x |
| Whisper 编码（摊薄到每帧） | 0.006 ms | 0.065 ms | 可忽略 |
| UNet+VAE解码 bs=1 | 54.0 ms → 18.5 fps ❌ | 553.5 ms → 1.8 fps ❌ | 10.2x |
| UNet+VAE解码 bs=2 | 40.1 ms → 24.9 fps ❌ | 345.7 ms → 2.9 fps ❌ | 8.6x |
| **UNet+VAE解码 bs=4** | **32.6 ms → 30.7 fps ✅** | **294.2 ms → 3.4 fps ❌** | **9.0x** |
| UNet+VAE解码 bs=8 | 33.2 ms → 30.1 fps ✅ | 531.6 ms → 1.9 fps ❌ | 16.0x |
| UNet+VAE解码 bs=16 | 30.4 ms → 32.8 fps ✅ | 578.1 ms → 1.7 fps ❌ | 19.0x |

峰值显存 9.1 GB（128GB 下无压力）。

## 结论

1. **空闲时可用**：bs=4 起达 30.7 fps，超过 25fps 目标约 20%。作者声称的 34 fps 基本属实。
2. **必须批处理**：bs=1 只有 18.5 fps，达不到。bs=4 是甜点（bs 再大无收益，且增加打断延迟）。
3. **与 LLM 并发时彻底崩溃**：掉到 1.7–3.4 fps，劣化 9–19 倍。

### 关于劣化幅度的诚实说明

测试用的争抢负载是 28 层 4096² fp16 矩阵链、背靠背无间隙，属于**最坏情况**。
真实的 MoE 解码受内存带宽限制、有更多空隙，激活参数量也远小于此。真实劣化
应在 2x 到 10x 之间。**但即使按最乐观的 2x 算，30.7 fps → 15 fps，仍然达不到 25fps。**
结论不依赖于争抢强度的精确值。

## 由此得出的架构决策：不要句级流式，改为整句生成后再开口

这是本次实测最有价值的产出。

VoxEMW 用 `stream_batch_sentences: 1` 做句级流式——LLM 边生成边喂 TTS。
那是因为**他的 LLM 在云端**，本地 GPU 根本不与之争抢，且 API 首字延迟高，
流式能省时间。

本地部署下这个优化**适得其反**：它制造了 LLM 解码与数字人渲染的持续重叠，
而实测证明两者不能共存。

正确做法是让二者**时间上分离**：

```
用户说完 → ASR(GPU) → LLM 生成完整回复(GPU 独占) → TTS+数字人渲染(GPU 独占) → 播放
                        ↑ 数字人放预渲染待机循环，0 GPU
```

代价很小，因为辩论回复很短：40 token @ ~70 tok/s ≈ 570 ms。

估算首音延迟：

| 阶段 | 耗时 |
|---|---|
| VAD + SmartTurn 判停 | 200 ms（CPU） |
| ASR（此时 GPU 独占，用空闲档数字） | 270 ms |
| LLM 生成完整回复（GPU 独占） | ~570 ms |
| TTS 首音（GPU 与数字人共享，但无 LLM） | 160–360 ms |
| **合计** | **≈ 1.2–1.4 s** |

仍在 2 秒目标内，且**每个阶段都独占 GPU**，没有争抢。

### 遗留待验证

- TTS 与数字人渲染仍会同时跑（都在"说话"阶段）。需实测这两者并发的代价——
  但 TTS 只占 3.94 GB、RTF 0.35，比 LLM 轻得多，预计可接受。
- bs=4 意味着缓冲 160 ms 音频。打断时需丢弃在途批次，实际停嘴延迟 ≤160 ms。

---

## 2026-09-25 追加：预计算版真实运行时（`voxck/face.py`）

底子帧的人脸检测、裁剪、VAE 编码全部离线预计算后，运行时只剩依赖音频的部分。
真实潜变量 + 真实音频 + 服务未运行（GPU 空闲），batch=4，fp16：

| 阶段 | 每帧 | 折合 |
|---|---|---|
| Whisper 编码 | ≈0 | 可忽略 |
| **UNet 单步** | **39.6 ms** | 25.2 fps |
| **VAE 解码 256²** | **38.2 ms** | 26.2 fps |
| UNet+VAE 合计 | 68.9 ms | **14.5 fps** |
| 贴回（CPU） | 2.1 ms | 468 fps |
| 端到端 `render()`（含服务运行时的心跳争抢） | | 14.1 fps |

**修正前一节的 30.7 fps。** 那个数是随机潜变量微基准，MLX 惰性求值把重复计算
合并了（本项目 hardware bench 里踩过的同一个坑）。加依赖链后真实值是 14.5 fps。

结论：
1. 瓶颈是 UNet 和 VAE 解码各一半，不是单点。
2. 只换 VAE 为轻量解码器（TAESD，SD1.5 潜空间兼容）最多到 ≈24 fps，仍无余量。
3. 要在本机跑，必须**降到 12.5 fps 渲染、逐帧复制到 25 fps 播放**，并且渲染期间
   GPU 不能有 LLM 争抢（大脑上云可满足）。TTS 争抢（RTF 0.35）仍会压掉约三分之一。

---

# 2026-09-25 追加：九月的 30.7 fps 是爆发态数字；真正可行的是「UNet 上神经引擎」

## 1. 发现：GPU 持续负载约 1 秒后降到爆发态的 1/3

同一段 `run_batched(bs=4)` 逐次计时（ms/帧）：

```
38 32 33 34 37 44 58 123 273 91 73 68 62 61 61 62 63 64 66 67 68 71 71 72 74 …
```

前 5 次 32ms（= 九月记录的 32.6），随后稳定在 **60–75ms**。裸 bf16 矩阵乘同样：
42 TFLOPS → 13 TFLOPS，且 2 秒空闲不能恢复；按 25fps 节奏间歇跑也一样。
所以 docs/measured-hardware.md 的 20 TFLOPS 是衰减中途的数字，**持续态约 13 TFLOPS**。

推论：任何"跑 10 次取中位"的 GPU 基准都虚高 2–3 倍。实时渲染只能按持续态算。

持续态分项（bs=8）：UNet 27.7 · VAE 解码 40 → 69 ms/帧 → **14.5 fps**，达不到 25。
mx.compile 让 UNet 降到 19.3，VAE 解码不变。

## 2. 神经引擎实测（Core ML，真实权重，与 MLX 输出余弦 0.99999）

| 模型 | 结构 | 单帧（空闲） | 备注 |
|---|---|---|---|
| UNet | diffusers 原版 | 33–34 ms | 编译计划全在 ANE，但 E5RT 报 ANECCompile 失败后分段 |
| UNet | **Apple ml-stable-diffusion ANE 结构** | **29 ms** | 3046 个算子全在 ANE，无编译告警 |
| VAE 解码器 | diffusers | 47 ms | 不如 GPU（持续态 40，半脸解码 16–29） |

没用的：batch 2/4（28.6/29.4）、macOS26 部署目标（28）、GroupNorm 手写替换（42，更慢）、
int8 线性量化（**110 ms**，慢 4 倍——ANE 不擅长按块反量化）。

**分工**：UNet → ANE；Whisper 编码 + 只解码下半张脸的 VAE → GPU；贴回 → CPU。三级流水线。

## 3. 两个工程陷阱

1. **coremltools 的 predict 持有 GIL 约 75%**。放主进程线程里，贴回从 1.4ms 变 47ms，
   asyncio 循环和 MLX 线程也会被卡。→ Core ML 必须跑在 spawn 子进程（voxck/ane_worker.py）。
2. **ANE 推理对父进程的 CPU 活动敏感**：子进程独跑 29–36ms，父进程有 cv2/numpy 工作时
   38–47ms；QoS 提到 user-interactive 无效。原因是 Core ML→aned→ANE 的进程间往返受调度延迟影响。

## 4. 端到端（voxck/face.py，/tmp/base.mp4 底子，10.4s 音频，260 帧）

| 配置 | fps |
|---|---|
| 全 GPU（九月方案，持续态） | 12–14 |
| ANE UNet(diffusers) 子进程 + GPU 半脸解码 | 17.3 |
| **ANE UNet(Apple 结构) 子进程 + GPU 半脸解码** | **21.5** |

25fps 撑不住，**20fps 有 8% 余量**——配置里 `face.fps = 20`，Whisper 特征按 20fps 切块，
浏览器按 pts 显示，与音频对齐不受帧率影响。

## 5. 画质诊断（真实输入，对照表 diag.jpg）

VAE 自重建清晰；MLX UNet / ANE UNet / 全解码 / 半脸解码 四者一致 → 流水线正确。
模糊来自 MuseTalk 本身在这段 480p、四分之三侧脸、话筒遮挡底子上的表现（下半张脸整体重绘）。
改善靠底子：高清、正脸、无遮挡；以及把遮罩收紧到嘴部。

## 6. 2026-09-25 下午：糊的根因是裁剪框，不是模型

`expand`（人脸框外扩比例）之前一直是 0.55。MuseTalk 的训练裁剪是**紧贴人脸的框**（脸填满 256），
外扩 55% 后嘴在裁剪里只剩二十几像素，UNet 输出一团糊、几乎不张嘴——三条路径（ANE 流水线 /
MLX UNet / 原版全解码+下半脸遮罩）结果一致，说明不是流水线 bug；静音中性化底子也无效。

| expand | 嘴内暗区均值（张嘴程度代理） | 观感 |
|---|---|---|
| 0.55 | 0.012 | 无嘴唇结构，模糊闭合 |
| 0.25 | 0.082 | 有嘴唇，开合可见 |
| **0.10** | **0.122** | 嘴唇、牙齿、开合清楚 |

对比图 expand_ab.jpg。默认改为 0.1（configs `face.expand`）。之前所有"MuseTalk 在这台机器上就是糊"的判断作废。

其它同日确认：嘴周椭圆遮罩（按关键点）优于整个下半脸遮罩（脸颊/下颌/话筒保留原片，无背景文字被抹的痕迹）；
底子先抠像（MODNet + YOLO 门控，voxck/segment.py）合成演播室背景后再编码，贴回无原背景光晕。
