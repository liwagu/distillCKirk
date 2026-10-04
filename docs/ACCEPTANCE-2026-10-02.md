> Historical engineering record. For the current released architecture and setup, read [ARCHITECTURE.md](ARCHITECTURE.md) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md). Private logs and local session archives referenced below are not distributed.

# 浏览器链路验收：2026-10-02

本文以下记录此前本地 Qwen 阶段。当前网页已改为官方 `deepseek-v4-pro`，thinking 关闭；Pro 三轮模拟麦克风浏览器检查 29/29 通过，人设修正后的单轮复验 13/13 通过。当前配置、对话内容与限制见 [DEEPSEEK-SWITCH-2026-10-02.md](DEEPSEEK-SWITCH-2026-10-02.md)。

本次已修复浏览器播放故障并实际启动全套服务。验收使用本机 Chromium 和明确标记的模拟麦克风，走原页面的 MediaStream、AudioWorklet 和 WebSocket；不是 Python 收包测试。真人麦克风、扬声器回声消除、声音相似度与嘴型观感尚需实际试听。

## 现场入口

- 地址：<http://127.0.0.1:8000>。点击 Start、允许麦克风、用英语说话；直接开口可打断。
- 本阶段大脑：本地 `mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit`。
- ASR：Qwen3-ASR-1.7B；TTS：Qwen3-TTS-12Hz-1.7B-Base-8bit，沿用参考音 A。
- 嘴型：MuseTalk，Core ML/ANE UNet + MLX VAE，640×480 输出，目标 20fps，沿用 Fox 倾听底子。
- 已保留 `.env.local` 的 DeepSeek 中转设置。中转的模型列表请求可达，但回复请求在直连、代理两种路径均超过 20 秒未返回，因此本轮用 `--local-llm`。
- 首次完整验收中，TypeSafe 直连有 ConnectError、重试、等待超时；内容按现有 fail-open 策略继续播放。随后新增只作用于 TypeSafe SDK 的 `TYPESAFE_PROXY`，配置本地 7897 代理并修正 SDK 的 `aclose()`；重启后预热 6.4s、HTTP 200。不得将链路成功理解为逐句引用验证成功，亦未重新验证全部人物引用。

## 本次修复

1. 一字节 `A` 包头导致 PCM 偏移为 1，旧 `Int16Array` 解码必然抛 RangeError。改用 DataView 读取 little-endian int16，并验证包长度。
2. `generation_done` 与浏览器 `playback_done` 分离。生成完仍播放时服务端保持 speaking，插话会立即清旧声音和画面。
3. 浏览器记录实际播放样本时间；网络等待不算听过。打断后的历史只提交已完整播完的合成段，旧回合 ACK 不会结束新回合。
4. Stop、断线、重连统一停止麦克风与播放队列；旧音频 onended 和迟到图片解码通过回合/epoch 作废。
5. 输出 AudioContext 在 Start 手势内解锁；脸加载失败显式显示，不再用预录说话视频掩盖。
6. 增加项目专用后台服务管理与明确的本地大脑启动参数，保留日志、校验进程身份后才停止。

旧文件备份在 `.backups/codex-20261002-035539/`。此目录不是 Git 仓库。

## 实测结果

报告：report.json（private local evidence; not distributed）。**22/22 浏览器检查通过，控制台和 JavaScript 错误为 0。**

同一 WebSocket 会话中，正常完成两轮英文辩论，再在第三轮生成结束但仍有 772ms 音频缓冲时开口，旧三个音频源被停止、帧队列清空；第四轮正确接住新问题并完整播放。Stop 检查将本次收到的真实 PCM/JPEG 排入播放器后点击实际按钮，确认声音归零、未来帧清空，不额外调用模型。

| 回合 | 内容 | 实际排程首声延迟 | 实际显示画面 | 音频时长 |
|---|---|---:|---:|---:|
| 1 | 工资差距能否证明歧视 | 2.221s | 496 帧，18.1fps | 26.96s |
| 2 | 追问职业选择与歧视的解释 | 2.184s | 454 帧，18.0fps | 25.36s |
| 3 | 同一议题，生成完后插话 | 被打断 | 收到 29 帧 | 收到 1.52s，未全部播放 |
| 4 | “What evidence would change your mind?” | 2.678s | 407 帧，16.5fps | 24.72s |

表中延迟从模拟输入的 ended 回调，到首个 AudioBufferSource 的计划播放时间，包含 VAD/回合判断、ASR、LLM、TTS 与等首帧；不是后端音频首包，也不含实体扬声器输出延迟。首轮输入 ended 回调晚了约 397ms，按输入起点加波形时长估算，首轮延迟为 **2.618s**；第二轮和第四轮的两种算法接近。画面 fps 由 canvas 实际绘制跨度计算。完整计算见 timing-summary.json（private local evidence; not distributed）。

仍有性能余量问题：第一轮音频排程出现 609ms 空隙，canvas 最长一次绘制间隔约 1.325s；第二轮和第四轮没有音频排程空隙。第一轮最晚画面相对样本时钟落后约 245ms。通常画面按前端策略领先样本时钟约 96ms；这不等于测过实体设备的感知同步。语义服务不可达带来的等待和本地推理竞争值得继续优化。

首轮 预览视频（private local evidence; not distributed） 由收到的 PCM 和 JPEG 按原始 PTS 重建，带常驻 AI 标识；它不是浏览器录屏，不能代替上面的播放器事件证据。

## TypeSafe 代理修复后的复验

当前运行服务采用 `TYPESAFE_PROXY=http://127.0.0.1:7897`，仅传入 TypeSafe SDK；本地 Qwen、ASR、TTS 和口型路径不走该代理。单回合 复验报告（private local evidence; not distributed） **13/13 通过，浏览器错误 0**。收到 26.88s 音频，全部排程、自然播完并 ACK；488 帧收到、479 帧实际显示，约 17.5fps。Stop 从非静音输出中主动取消音频源并清未来帧。

输入 ended 回调到输出排程为 1.909s，到实际非零 analyser 采样为 1.914s；首轮回调有约 231ms 偏差，按输入起点加波形时长估算排程延迟为 2.140s。该回合仍有 665ms 音频排程空隙、最长 1.409s 画面绘制间隔，尚未做到全程无停顿。

新服务日志记录 TypeSafe 预热 6.386s、首句审计 6.350s、后续 374–411ms，共 8 次 HTTP 200、0 次 ConnectError。首两段等待超过 1.5s 预算时仍先播放，后来返回审查结果；这是现有策略，未改为严格拦截。后续性能优化应同时看审查排队、语音断流与画面延迟。

## 可复跑检查

```bash
cd .
.venv/bin/python scripts/service.py start --local-llm
.venv/bin/python -m unittest discover -s tests -v
node --check web/app.js
node scripts/browser_smoke.mjs --out logs/browser-check
```

回归共 8/8 通过：后端回合生命周期 6 项覆盖实际播放 ACK、自然完成的浮点误差、插话、迟到 ACK、旧帧作废与无脸模式；TypeSafe 客户端 2 项检查可选代理仅传 SDK、默认行为保留及正确关闭。浏览器脚本保存输入、返回声音、各帧 PTS、截图和细粒度音频事件。`scripts/e2e_test.py` 也已改为必须收到有效动态视频，成功文案明确为后端 fixture 通过。

接手阶段的静态历史记录仍保留在 [CODEX-HANDOFF-2026-10-02.md](CODEX-HANDOFF-2026-10-02.md)，本文件描述本轮实现与实测，不覆盖旧归档。
