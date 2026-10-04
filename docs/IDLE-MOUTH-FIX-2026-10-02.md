> Historical engineering record. For the current released architecture and setup, read [ARCHITECTURE.md](ARCHITECTURE.md) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md). Private logs and local session archives referenced below are not distributed.

# 静默时假口型：2026-10-02

用户真人试用发现：双方没有说话时，Charlie的嘴仍在动。根因是把采访原片整体当作待机循环；没有模型生成或音频也会继续播放原片的嘴部动作。

`web/media/base_fox_listen.mp4`共299帧、25fps、11.96秒，前约63帧仍在说话。旧 `FaceRenderer.next_idle_jpeg()` 与说话共用游标，在整段原片上往返；文件名不能证明整段素材都在倾听。原片联系图（private local evidence; not distributed）与第120帧（private local evidence; not distributed）已保存。

旧服务6.4秒静音浏览器采样收到了129张不同JPEG和10个不同canvas像素hash，同时回复音频、非待机视频、用户与assistant消息均为0，播放器没有音频源。报告 `logs/idle-baseline-20261002/report.json`；这一短样本主要是闭嘴轻动，不把像素差异本身当成夸张讲话的证明。原片逐帧检查另行确认了开头说话段。

## 修复

- 配置明确选取从0起算的第120帧（4.80秒）：闭嘴、睁眼、中性表情，沿用同一场景、裁剪和背景。静默固定显示该帧，不再推进原片口型。
- 前端保存一份可同步绘制的静默画面，在开始等回复、音频间隙、音频上下文暂停、播完、打断、Stop与断线时立即恢复；尚未收到静默图时隐藏旧画面。
- 只有实际排程音频段正在播放，才按音频时钟显示生成口型。保留首帧释放音频缓存、回合世代隔离和实际播放ACK。

本次采用固定静默帧，没有加入待机眨眼或呼吸动画。模型、密钥、thinking开关、ASR和TTS配置没有改动。

## 验证

Python回归37项通过，其中3项检查选定静默帧恒定、不随讲话游标变化、错误帧索引拒绝。Node回归6组场景通过，覆盖首音前、音频间隙、暂停、自然播完、打断、Stop和迟到旧帧；JS语法检查通过。

重启后实际静音浏览器复验 **14/14通过**：6.4秒窗口收到129张同hash待机JPEG，10次canvas像素采样及Stop后像素完全一致；回复音频、非待机视频与对话回合均为0，浏览器错误为0。截图目视闭嘴。报告：idle-fixed-verified（private local evidence; not distributed）。首次fixed报告因测试探针只观察旧 `FACE.draw` 而漏掉新的静默绘图路径，被保留在 `logs/idle-fixed-20261002/`；修正为观察实际屏幕canvas的 `drawImage` 后重新验证，没有放宽像素或静音条件。

正常问候语音复验也通过：真实ASR → Pro → TTS返回13词、3.84秒音频，生成动态口型；全部声音自然播放完并ACK，**音频结束时画布像素立即与缓存静默画面一致**，无浏览器错误。报告：idle-speech-regression（private local evidence; not distributed）。输入为明确标记的模拟麦克风，不声明代替真人麦克风或扬声器听感验收。

源码备份：`.backups/codex-idle-20261002/`，没有密钥文件。
