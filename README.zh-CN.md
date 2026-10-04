# distillCKirk：MacBook Pro 上的 AI 辩论数字人

在浏览器里用英语与以 Charlie Kirk 公开言论为基础的 AI 辩论伙伴对话，支持流式语音、实时口型和插话打断。开发及演示机器是 **M5 Max MacBook Pro，128 GB 统一内存**。

**成片实际采用混合架构：**语音识别、语音合成、数字人口型在 Mac 本地推理；对话大脑是远程 **DeepSeek V4 Pro API，关闭 thinking**。本地 Qwen 大脑保留为需要显式选择的替代方案。

**首先致谢原项目：[emwstudio/VoxEMW](https://github.com/emwstudio/VoxEMW)。**它的语音数字人架构、Mac 路线和工程经验是本项目的重要参考。这里改用 MuseTalk / MLX / Core ML 渲染，并实现浏览器音画播放、回合状态与打断处理。详细来源见 [ATTRIBUTION.md](docs/ATTRIBUTION.md)。

[English README](README.md) · [详细架构](docs/ARCHITECTURE.md) · [复现指南](docs/REPRODUCIBILITY.md)

## 成品演示

- [Bilibili 视频](https://www.bilibili.com/video/BV14oHv6qEDb)
- [YouTube 英文视频](https://www.youtube.com/watch?v=DtIkQjkBrZg)

![成片中实际对话的画面](docs/demo/conversation.jpg)

截图取自完成版英文视频的 **03:45**；另有 [05:05](docs/demo/speaking.jpg) 和 [05:45](docs/demo/follow-up.jpg) 两张画面。没有把原始大视频上传到 Git。

## 这次做出来的是什么

完整链路是：麦克风 → 浏览器 AudioWorklet / WebSocket → Silero VAD 与 Smart Turn → Qwen3-ASR → 人设与已播放历史 → DeepSeek → Qwen3-TTS → 音频播放与 MuseTalk 口型。

| 部分 | 采用方案 | 执行位置 |
|---|---|---|
| 收音、字幕、播放、画面时钟 | 浏览器原生 API | 本地浏览器 |
| 语音活动、判断用户是否说完 | Silero VAD v5、Smart Turn v3 | CPU / ONNX Runtime |
| 语音识别 | Qwen3-ASR-1.7B / MLX | GPU |
| 对话推理 | DeepSeek V4 Pro，thinking 关闭 | 远程 API |
| 可选本地大脑 | Qwen3-30B-A3B 4bit / MLX-LM | GPU |
| 参考音色合成 | Qwen3-TTS Base 8bit / MLX | GPU |
| 口型 UNet | MuseTalk 转成 ANE 适配结构 / Core ML | Neural Engine 与 CPU |
| 音频特征、VAE 解码 | MuseTalk MLX | GPU |
| 嘴部贴回、背景合成 | NumPy / OpenCV | CPU |
| 可选语义审查 | TypeSafe | 启用时为远程服务 |

数字人输出为 **640×480，配置目标 20fps**。演示成片是 4K 录制与剪辑输出，不能把两者混为一谈。

人设由 26 条带来源的立场、推理模式注释和 15 条辩论打法组成，作为 system prompt 注入模型。这里的“蒸馏”指材料研究与人设整理，**没有训练 Charlie 专属大模型权重**。生成内容也不是他的真实新发言；页面持续显示 AI 与合成音视频标识。

## 关键工程设计

1. **把生成完与播放完分开。**后端生成完不代表用户已经听完；浏览器按实际播放进度返回 ACK。
2. **只记住用户真正听到的完整合成段。**打断后不把尚未播放的回复算进对话历史。
3. **音频样本时钟驱动画面。**JPEG 带回合号与时间戳，浏览器用音频播放时刻选帧；迟到的旧帧不会继续播。
4. **静默时闭嘴。**静音、播完、Stop 与打断都会恢复已确认闭嘴的静态帧，避免没说话却仍像在演讲。
5. **CPU、GPU、ANE 分工。**VAD 在 CPU，ASR/TTS/VAE 在 GPU，UNet 单独放 Core ML 子进程；MLX 工作集中到同一个执行线程。
6. **关闭 thinking 并检查真正的正文。**SSE 保活与 HTTP 200 不算模型成功回答；没有正文会超时或报错。

更细的回合状态、协议、缓存、边界和代码入口见 [ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 如何运行

这是工作原型的源码发布。密钥、私人录音、原始人物视频、模型权重、Core ML 大文件和运行缓存没有进入仓库。**新克隆需要先下载模型、转换 Core ML、准备自己的参考音视频，不能只点一次运行就复现全部效果。**

```bash
git clone --recurse-submodules https://github.com/liwagu/distillCKirk.git
cd distillCKirk
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
cp .env.example .env.local
```

按 [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) 准备资产，只把自己的密钥填到 `.env.local`。完成后：

```bash
.venv/bin/python scripts/service.py start
.venv/bin/python scripts/service.py status
.venv/bin/python scripts/service.py stop
```

浏览器访问 `http://127.0.0.1:8000`，点击 Start 并允许麦克风。说话时可直接开口打断。Stop/Start 会开新会话；远程大脑失败不会自动切换为本地 Qwen。

## 验证与目前的限制

仓库保留回合生命周期、播放 ACK、静默画面、重复回复和关闭 thinking 等回归检查。详见 [RELEASE-CHECKS.md](docs/RELEASE-CHECKS.md)。历史实测文档保留研究过程与被否定的方案，读取当前架构时应以上方说明为准。

目前仍可能出现识别错误、音频间隙、画面延迟及人设生成不准确；静默画面还没有眨眼动画。可选 TypeSafe 审查超时采用 fail-open，成功播出不代表审查通过。模型与媒体各有许可，不能把源码公开理解成全部素材或依赖已获任意商用授权。

完整来源及许可边界见 [ATTRIBUTION.md](docs/ATTRIBUTION.md)。
