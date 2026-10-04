> Historical engineering record. For the current released architecture and setup, read [ARCHITECTURE.md](ARCHITECTURE.md) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md). Private logs and local session archives referenced below are not distributed.

# distillCKirk 接手上下文（2026-10-02）

本文件由 Codex 根据 Claude 原始会话、当前代码、资产和只读现场检查整理。目的：接续已有工作，并区分“已经写好”“历史后端测试通过”和“用户能实际对话”。本次未修改应用代码，未启动模型或服务，未调用云端接口。

## 1. 用户真正要的结果

- 在 M5 Max / 128GB Mac 上，与屏幕中的 Charlie Kirk 数字人自然地用英语辩论，重点是其公开的性别／女权立场；同时练英语口语。
- 完整体验必须包括声音、脸、真实音频驱动的嘴型、看向镜头、连续回合和随时插话。两段录像切换不能算交付。
- 对标的是最初提供的 VoxEMW / Bilibili 视频。不能用参数、基准或一次单轮后端测试代替实际体验。
- 原始要求全部本地。后续配置将大脑切为 DeepSeek 中转；脸仍必须本地，用户明确拒绝租云端 GPU。本次仅核对现有配置，未调用云端；后续沿用已有选型。
- 用户选定了音色参考 A，现保存为 `assets/ck/ref.wav`；不要重新从零选音色。
- 保留常驻 AI 标识和公开材料引用边界。项目里的“蒸馏”目前是 prompt 人设模拟和参考音克隆，没有 Charlie Kirk 专属的 LLM 权重训练。

## 2. 找回的 Claude 会话

归档根目录：`<private local session archive; not distributed>`。下表日期为北京时间；多个 fork 含继承历史，不能当成互不相关的会话或重复计算工作量。文件 10 月 1 日的修改时间也不代表 10 月 1 日有新实现。

| 会话名 | 本地会话 ID | 内容 |
|---|---|---|
| Charlie Kirk 数字人对话系统 | `b9ecb64d-82dc-4d04-b70e-5f5d09958ff9` | 原始主线，9/9 开始，至 9/24 23:03；与截图中的最初目标及早期工作一致 |
| Charlie Kirk 数字人对话系统 | `758967f5-97f4-4ca4-8b1f-b65ebb90103e` | 含继承历史与后续实施，最新有效回复为 9/26 13:53；**后续实现的主要证据源** |
| Charlie Kirk 数字人对话系统 (fork) | `96db5007-d115-4ab0-85e3-8801c610d8ae` | 含共同历史，后续讨论到 9/23 16:04，包含 JEV 相关提问 |
| tech-coach | `489bdc27-b8a5-4abc-a8c0-bffcee864c0b` | 含共同历史，9/25 01:19 前的重新梳理与技术指导 |
| TypeSafe skill installation | `f68e403a-0043-4279-b3c6-49650ec8c265` | 9/21 安装与确认 TypeSafe；保留其使用偏好，但本轮未安装插件或调用该服务 |

另有 `3094b144-e4ad-4e7b-874c-eb790e4e09b7.jsonl`，未见新的完整 assistant 实施结果，不用它覆盖上面的交付状态。

主要原始证据：最新主会话（private local evidence; not distributed）。重要位置：

- 第 34 行：原始目标和六块技术栈。
- 第 2412 行：用户确认音色相似，但没有脸、像在自己回答自己。
- 第 3322、4038、4387、6151 行：用户持续反馈不自然、未达到预期、什么也没看到。
- 第 4454 行：拒绝云端 GPU；必须利用现有 Mac。
- 第 7788 行：用 macOS `say -v Samantha` 合成 wage-gap 问句作为测试输入。
- 第 7832、7835 行：最后一次素材切换命令与原始后端测试输出。
- 第 7860 行：Claude 最后一次状态回复，要求用户实际聊几轮；未见随后成功验收反馈。

Claude 自己的项目笔记位于 `<private local session archive; not distributed>`。本次只读取，未修改任何记忆文件。

## 3. 已有成果与当前有效配置

当前工作目录 `.` **不是 Git 仓库**；四个 `ref-*` 是参考项目。不得把参考项目的 Git 状态当作主项目的版本记录。

```text
浏览器麦克风 / AudioWorklet → 16kHz int16 PCM → WebSocket
  → Silero VAD + SmartTurn → Qwen3-ASR / MLX
  → 人设 system prompt + LLM 句流 + 内容守卫
  → Qwen3-TTS 参考音克隆 → 下行 A + PCM
  → MuseTalk 音频特征 → Core ML UNet / ANE 子进程
     + MLX 半脸 VAE / GPU + CPU 贴回 → 下行 V + turn/pts/JPEG
  → 浏览器 WebAudio + canvas + 字幕
```

| 成果 | 当前位置与状态 |
|---|---|
| 入口、配置 | [run.py](../run.py)、[assistant.json](../configs/assistant.json)；入口先读 `.env.local`，其配置优先于 JSON |
| 当前大脑 | `.env.local` 指向 `aiopenapi.paycools.com`，模型标识 `deepseek-v4-flash`，密钥已配置；这是中转配置事实，不是本轮对上游模型身份或可达性的验证 |
| 本地备用大脑 | `mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit`；四个权重分片现均存在，共 17.18GB。仅核验存在与大小，未重新加载或校验哈希。截图里的“下载失败在 10%”已过时 |
| 人设 | [ck-gender.md](../personas/ck-gender.md)：26 条立场、13 条打法；旧版 `ck-gender.v1.md` 保留。正文经 `persona.py` 作为每条连接的首个 system 消息注入 |
| 调研 | `corpus/raw/` 共 18 份 JSON，含原始 dossier 和复核；`corpus/sources/` 为空，未保存全部来源网页正文 |
| 语音 | `voxck/vad.py`、`asr.py`、`tts.py`；参考音 `assets/ck/ref.wav` 与 `ref.txt` 均存在 |
| 实时口型 | [face.py](../voxck/face.py)、[ane_worker.py](../voxck/ane_worker.py)；实际使用 `assets/coreml/musetalk_unet_ane_b1.mlpackage`，该文件包与底子缓存均存在 |
| 当前底子 | `web/media/base_fox_listen.mp4`，Fox 远程连线 454–466 秒倾听片段；`pre_crop=[0.5,0,1,0.83]`、嘴周遮罩、`expand=0.1`、目标 20fps、深蓝渐变背景 |
| 浏览器 | `web/app.js` 管理麦克风、音频、帧和字幕；`avatar.js` 与 idle/talk 录像仍保留为旧路径／降级路径 |

大脑、TypeSafe 的密钥不复制到本文件。当前存在两个远程依赖，不能把这套有效配置称为“完全离线”。

## 4. 当前现场与确定的播放断点

2026-10-02 03:41 左右（北京时间）只读检查：8000、8097 均无监听；访问本机 8000 连接失败。未发现正在运行的项目 Python 服务。因此当前页面入口本身不可用，但不能把“服务没启动”当成历史失败的全部解释。

**当前浏览器代码存在一个确定的 PCM 对齐异常：**

1. [orchestrator.py:93](../voxck/orchestrator.py) 发送 `b"A" + pcm`，音频头占 1 字节。
2. [app.js:255](../web/app.js) 用 `u8.subarray(1)` 提取音频，保留原 ArrayBuffer，`byteOffset=1`。
3. [app.js:54](../web/app.js) 直接创建 `Int16Array(..., 1, ...)`，JavaScript 要求起始偏移为 2 的倍数，必然抛异常。

本轮在工具的 V8 JavaScript 环境中最小复现：

```javascript
const packet = new Uint8Array([0x41, 0x00, 0x00, 0xff, 0x7f]);
const pcm = packet.subarray(1);
new Int16Array(pcm.buffer, pcm.byteOffset, pcm.byteLength / 2);
// RangeError: start offset of Int16Array should be a multiple of 2
```

无脸模式第一包即失败；有脸模式暂存后在 `player.release()` 失败。音频无法排程，播放起点保持 0，依赖这个时钟的说话帧也无法正常播放。**这是代码与最小复现确认的阻断点，本轮未启动真实页面重现。** 下一步首先修正 PCM 的对齐／解码。

## 5. Claude 最后一次“全链路通”实际证明了什么

原始归档第 7835 行，2026-09-26 13:52：

```text
face renderer ready in 24.6s（20fps，底子 web/media/base_fox_listen.mp4）
warm in 27.2s
ASR 1088ms · TTFT 1241ms · 首音 2818ms · 端到端 12481ms
43 词 · 缓存 5756/5890
收到画面 243 帧（pts 0.0–15.3s，4.1 MB）
首帧比首音晚 583ms · 帧率 15.8 fps · 待机帧 151
结果: ✓ 全链路通
```

这些是**历史后端收包指标**。输入是 Samantha 合成问句 `/tmp/vadtest/a.pcm`，由 Python 客户端按实时速度喂入 WS；它绕过了真实麦克风和浏览器播放器。

[e2e_test.py:117](../scripts/e2e_test.py) 的成功条件只有识别文字、回答消息和非空音频 bytes；即使没有视频帧也能通过。不检查 JS 异常、实际听见声音、canvas 显示、口型准确度或多轮插话。

“首音”也是生成／发送音频的时间，不是用户听见声音的时间。浏览器还会等第一帧，最多扣住音频 2.5 秒。不能把 2818ms 直接作为真人体验延迟。

原图 `final_sheet.jpg`、临时测试脚本和 `/tmp/voxck-logs` 本轮均未找到；当前会话 scratchpad 为空。主要结果只能从 JSONL 工具输出恢复，不假装图片仍可打开。

归档中也有浏览器检查：第 6104–6130 行只确认页面和播放器对象存在、控制台当时无日志；更早第 3146–3268 行手动调用旧录像状态机的 speak/stop/cut。它们没有覆盖最后版本的实际收音、PCM 播放和自然插话，不能补足上述验收。

## 6. 必须保留的后续修正与待验证问题

- [prefill-measured.md:137](prefill-measured.md) 推翻了截图的滚动窗口结论：30 轮 61% 命中受同一 server/LRU 的测试顺序污染；后续重放给出稳态 38%、需重算 2641 token。旧 `+330ms` 不可沿用，30B 延迟需重新测。
- [musetalk-measured.md:73](musetalk-measured.md) 后续纠正早期 30.7fps 微基准；持续 GPU 负载与 MLX 惰性求值影响结果。最终实施已改为 ANE UNet + GPU 半脸解码，不能停留在“全 GPU MuseTalk 不行”的早期判断。
- [ROADMAP.md](ROADMAP.md) 前部含 9/26 Fox 素材进展，后部仍保留旧街头素材、嘴周遮罩等已做待办；`EXPLAINED.md` 也停留在更早阶段。优先看代码、有效配置和最新归档。
- **打断的状态缺口（静态判断，尚未动态验证）**：服务端生成完即 `speaking=False`，浏览器可能仍在播放缓冲音频；此时 `speech_start` 不会触发清播放队列。新回合与 Stop 按钮也未可靠清旧音频。需覆盖“生成中”和“生成完仍在播放”两个插话窗口。
- **脸加载失败可被掩盖**：启动异常会继续无脸运行，旧 E2E 仍可能通过。不能只凭 `ready` 判断数字人就绪。
- **人设有材料但未完成来源清零**：positions 复核列有 10 项缺陷，部分 paraphrase 在现人设仍像逐字引语；warrant 已注明是分析师重建。不能把“26 条带链接”当成全部事实／引语验证完毕。
- 人设 frontmatter 提到的部分会话计数器未见在编排器实现；TypeSafe 首句只后台审计，后续超时可先播。其 guard 不是逐句引用校验，也不保证无幻觉。
- `scripts/restart.sh` 使用临时日志和较宽的 `pkill` 模式。下次运行应跟踪本任务自己的 PID、把诊断保存在项目 `logs/`，避免杀掉别的 Python／spawn 进程。

## 7. 后续执行顺序与验收线

1. 先修复浏览器 PCM 对齐故障，验证真实 A 包能排程、播放起点推进；保留现成模型、人设、音色和底子。
2. 用现有 `.venv/bin/python run.py` 启动，保留完整启动日志；明确确认 ASR、TTS、脸均加载成功，再打开本机页面。
3. 浏览器 Start 后证明：麦克风有实际信号 → 正确 ASR → 回答接住用户问题 → 听见声音 → canvas 显示真实音频驱动的嘴型与字幕；JS 控制台无异常。
4. 同一连接连续至少 3–5 轮，包含寒暄、性别议题、反驳、思考停顿；不自问自答、不播放固定脚本。
5. 在生成中、生成结束仍播放、以及 Stop 三种情况下验证旧音频和旧帧立即停止，新回合不续播旧回答。
6. 以浏览器实际播放时间记录首声、音视频偏差和掉帧；保存可复查的结果。后端 E2E 只作为辅助烟测。

接手判断：**项目已有可复用的完整链路实现和历史后端收包证据；用户可用的数字人对话尚未验收，当前版本还存在确定的浏览器播放断点。** 下一阶段从这个具体断点和真实浏览器验收开始。
